"""EDIT path, with bounded self-repair:

Python file -> user request -> localized context -> GPT-5.4 -> Structured
Delta -> Validate -> Apply -> pytest -> (on failure: classify, repair,
retry, bounded) -> store original + delta + new version in MinIO ->
record tokens/cost/latency.

Self-repair (PHOENIX doc, section 18): failure classified into a fixed
set of classes, repair prompt built from the failed Delta IR + exact
error, bounded retries only -- never unbounded, and STRUCTURED_EDIT
never falls back to full regeneration mid-repair.

Routing to FULL_REGENERATION is decided by the model itself via
generate_delta's escalate field ({"kind":"whole_file"} /
"language_conversion"/...), not a hand-written keyword/language
classifier -- those can't express "reaches outside any symbol's span"
or cover every language.

`run_edit` is the callback-driven core (no printing, no sys.exit) so both
the plain CLI below and `incremental_editing/cli.py` can drive it.
"""

import argparse
import ast
import difflib
import json
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..analyzer.locator import (
    AmbiguousSymbolError,
    defined_symbol_name,
    find_delete_candidates,
    find_multi_delete_targets,
    find_rename_target,
    find_symbol,
    index_symbols,
    is_delete_intent,
    is_whole_file_delete_target,
    looks_like_python,
    rename_with_subword_fallback,
    target_file_for_conversion,
)
from ..analyzer.metadata_builder import delete_file_metadata, extract_symbol_metadata, write_file_metadata
from ..analyzer.module_blocks import locate_module_level_block
from ..analyzer.text_blocks import block_context_window, locate_text_block, locate_text_blocks
from ..apply.edit_applier import ApplyError, apply_delta
from ..benchmark.pricing import estimate_cost
from ..context.context_builder import build_context
from ..delta.schema import DeltaIR
from ..delta.validator import DeltaValidationError, UnseenReplaceTargetError, validate_schema, validate_targets
from ..retrieval.jev_router import DISPATCHABLE_KINDS, classify_request_kind
from ..retrieval.multilang_symbols import detect_language, language_has_symbol_concept
from ..storage.minio_client import get_storage
from ..strategies.binary_artifact import (
    BinaryArtifactError,
    binary_artifact_instructions,
    generate_binary_artifact_base64,
    is_binary_artifact_target,
    write_file_content,
)
from ..strategies.full_regeneration import generate_full_file, generate_full_file_edit
from ..strategies.import_repair import generate_import_fix
from ..strategies.module_block_edit import generate_module_block_replacement
from ..strategies.text_block_edit import generate_block_replacement, generate_multi_block_replacement
from ..strategies.qa import generate_answer
from ..strategies.structured_edit import generate_delta, generate_repair
from ..validation.syntax import SyntaxCheckError, check_python_imports, check_syntax
from ..validation.tests import copy_project_for_validation, run_tests
from ..versioning.version_manager import VersionManager
from . import pending_confirmations

MAX_REPAIR_ATTEMPTS = 2

_ENGINE_REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_dir_is_this_engine(project_dir: Path) -> bool:
    """True when project_dir IS this engine's own repository, or is a
    subdirectory inside it."""
    try:
        project_dir.resolve().relative_to(_ENGINE_REPO_ROOT)
        return True
    except ValueError:
        return False


def _discover_relevant_test_target(project_dir: Path, file: str) -> Optional[str]:
    """Finds an existing test file actually about `file` via the standard
    naming convention (test_<name>.<ext> / <name>_test.<ext>, co-located
    or under a tests/ dir) -- never invents one. Returns the narrowest
    match, or None."""
    rel = Path(file)
    stem = rel.stem
    candidates = [
        rel.parent / f"test_{rel.name}",
        rel.parent / f"{stem}_test{rel.suffix}",
        rel.parent / "tests" / f"test_{rel.name}",
        rel.parent / "tests" / f"{stem}_test{rel.suffix}",
        Path("tests") / rel.parent / f"test_{rel.name}",
        Path("tests") / f"test_{rel.name}",
    ]
    for candidate in candidates:
        if (project_dir / candidate).is_file():
            return str(candidate)
    return None


def _effective_test_target(test_target: str, project_dir: Path, file: str) -> str:
    """Narrows the default test_target (".") so validation doesn't pytest
    the whole project dir (unrelated subprojects, this engine's own repo)
    and time out on content the edit never touched. An explicit, already-
    narrower test_target is left untouched; the default narrows only to a
    discovered test file (_discover_relevant_test_target) or, failing
    that, `file`'s own containing directory."""
    if test_target != ".":
        return test_target
    discovered = _discover_relevant_test_target(project_dir, file)
    if discovered:
        return discovered
    parent = str(Path(file).parent)
    return parent if parent != "." else test_target


class _TestFailure(Exception):
    """Internal signal so a failing pytest run joins the same classify-
    and-repair path as the other failure exceptions below."""


def _classify_failure(e: Exception) -> str:
    """Maps exceptions onto PHOENIX's failure-class taxonomy (section 18)."""
    if isinstance(e, SyntaxCheckError):
        return "SYNTAX_ERROR"
    if isinstance(e, ApplyError):
        return "PATCH_CONFLICT"
    if isinstance(e, _TestFailure):
        return "TEST_FAILURE"
    if isinstance(e, (DeltaValidationError, AmbiguousSymbolError)):
        msg = str(e)
        if "schema validation failed" in msg:
            return "SCHEMA_ERROR"
        return "REFERENCE_ERROR"
    return "UNKNOWN"


_COLLECTION_ERROR_RE = re.compile(r"ERROR collecting (\S+)")


def _unrelated_collection_error_file(failure_detail: str, file: str) -> Optional[str]:
    """Returns the path pytest failed to COLLECT, or None if this isn't a
    collection error or the collected path IS the edited file (then the
    edit itself may be the cause, repair still worth trying). A collection
    error in some OTHER file (e.g. an unrelated stale test picked up by a
    too-broad test_target) can't be fixed by repairing a delta to `file`,
    so retrying would just repeat the same error across all repair
    attempts for nothing."""
    m = _COLLECTION_ERROR_RE.search(failure_detail)
    if not m:
        return None
    collected = m.group(1)
    if collected == file or collected.endswith(f"/{file}") or file.endswith(f"/{collected}"):
        return None
    return collected


def _first_defined_name(content: Optional[str]) -> Optional[str]:
    """Python-only wrapper around analyzer.locator.defined_symbol_name,
    kept under its original name since it's part of this module's tested
    surface."""
    return defined_symbol_name(content, "python")


def _change_ratio(original: str, new: str) -> float:
    orig_lines = original.splitlines()
    new_lines = new.splitlines()
    diff = list(difflib.unified_diff(orig_lines, new_lines))
    changed = sum(1 for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    return changed / max(len(orig_lines), 1)


def _compact_created_file_summary(path: str, content: str) -> str:
    """Structural summary (function/class signatures) of a just-created
    file for the also_link_current_file edit, instead of resending its
    full content -- keeps linking N new files cheap. Falls back to the
    real content when there's no extractable symbol structure (.env
    files need exact key names; binary files have no symbols at all)."""
    if is_binary_artifact_target(path):
        return "(binary file, already created)"
    language = "python" if path.endswith(".py") else detect_language(path)
    if language is None:
        return content
    symbols = extract_symbol_metadata(content, language)
    if not symbols:
        return content
    return "\n".join(f"{s.symbol_type} {s.name}({', '.join(s.parameters)})" for s in symbols)


def _persist_run(storage, project_id, run_id, delta_dict, metadata, attempt=None):
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    if delta_dict is not None:
        suffix = f"-attempt{attempt}" if attempt is not None else ""
        storage.put_json(f"projects/{project_id}/deltas/{run_id}{suffix}.json", delta_dict)


def _fold_prior_generation(metadata: dict, prior_gen: Optional[dict]) -> None:
    """Adds a prior LLM call's usage onto metadata["generation"]'s totals
    -- e.g. the classification call that produced an escalate signal
    before this handler ran, so its real cost isn't dropped. No-op when
    there was no prior call."""
    if not prior_gen:
        return
    g = metadata["generation"]
    g["input_tokens"] += prior_gen["input_tokens"]
    g["output_tokens"] += prior_gen["output_tokens"]
    g["total_tokens"] += prior_gen["total_tokens"]
    g["cached_tokens"] = g.get("cached_tokens", 0) + prior_gen.get("cached_tokens", 0)
    g["latency_ms"] += prior_gen["latency_ms"]
    g["estimated_cost_usd"] = round(
        g["estimated_cost_usd"]
        + estimate_cost(
            prior_gen["model"], prior_gen["input_tokens"], prior_gen["output_tokens"], prior_gen.get("cached_tokens", 0)
        ),
        6,
    )


def _cross_file_rename_impact(project_dir: Path, current_file: str, symbol_names: List[str]) -> Optional[List[str]]:
    """Real callers of `symbol_names` OUTSIDE `current_file`, via Joern's
    CPG -- covers the blind spot rename_with_subword_fallback has (it only
    rewrites occurrences inside the one file it's given). None when Joern
    is unavailable or finds nothing; never raises, never blocks the
    rename."""
    from ..retrieval.joern_graph import build_call_graph_via_joern, is_available
    from ..retrieval.repo_index import build_repo_index

    if not is_available():
        return None
    symbols = build_repo_index(str(project_dir))
    graph = build_call_graph_via_joern(str(project_dir), symbols)
    if not graph:
        return None
    called_by = graph.get("called_by", {})
    external = [
        loc
        for name in symbol_names
        for loc in called_by.get(name, [])
        if loc.split("::", 1)[0] != current_file
    ]
    return external or None


def _locate_import_span(source: str) -> Optional[tuple]:
    """1-indexed (start_line, end_line) span covering every top-level
    import statement, via real AST node positions -- not a naive "lines
    starting with import" scan, so a stray non-import line between
    import groups is never swept in. None if unparseable or no imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    spans = [(node.lineno, node.end_lineno) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def _imported_names(import_block: str) -> set:
    """Every local name an import block binds (alias if aliased, else its
    own name) -- used to check a narrow import-only fix doesn't orphan a
    name still used elsewhere. A block that fails to parse (only possible
    for the model's generated replacement) degrades to "no names," which
    only makes the orphan check stricter, never looser."""
    try:
        tree = ast.parse(import_block)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _run_whole_file_edit(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    original_source: str,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    target_file: Path,
    classification_gen: Optional[dict] = None,
    language: Optional[str] = None,
) -> dict:
    """Whole-file regeneration for an EXISTING file's edit is refused by
    policy: this pipeline only ever changes the symbol(s) a request
    identifies, even when the model's escalate signal says the request
    reaches outside every symbol's own span (module docstring, standalone
    module-level statement, inter-function formatting).

    Before refusing, tries analyzer/module_blocks.py: that "outside any
    symbol" content is often one small, narrowly-locatable region (a
    module-level config list, an `if __name__ == "__main__":` guard), so
    only an unlocatable/ambiguous region falls through to the refusal.
    Requires at least one real function/class symbol elsewhere in the
    file -- with zero symbols the "gap" is the whole file, so locating
    "a" module-level block there would just be a whole-file rewrite by
    another name.

    CREATE and language_conversion never call this function -- both have
    no existing symbol to target in the first place, so whole-file
    generation there is the only mechanism that could ever apply, not a
    fallback from something smaller.

    The refusal path makes no extra generation call: only the
    classification call that discovered the escalation is billed, folded
    in via classification_gen."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    if language is not None and index_symbols(original_source, language):
        module_block = locate_module_level_block(original_source, request, language)
        if module_block is not None:
            on_step(
                "LOCALIZE",
                f"'{module_block.name}' module-level region unambiguous -- narrower than whole-file...",
            )
            return _run_module_block_edit(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                block=module_block,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
                classification_gen=classification_gen,
            )

    on_step(
        "GENERATE",
        "refusing: this reaches outside any single symbol's own span, and whole-file "
        "regeneration is disabled for edits by policy...",
    )
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "FULL_REGENERATION",
        "generation": {
            "model": "n/a (whole-file regeneration disabled for edits)",
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "estimated_cost_usd": 0.0,
        },
        "validation": {},
        "result": {
            "status": "failed",
            "retry_count": 0,
            "fallback_used": False,
            "failure_class": "WHOLE_FILE_BLOCKED",
        },
        "error": (
            "This request reaches outside every function/class's own span (a module "
            "docstring, a standalone module-level statement or comment, blank-line "
            "formatting between symbols) -- whole-file regeneration is disabled for "
            "edits, so nothing was changed. Narrow the request to one function/class "
            "at a time, or split it into separate requests."
        ),
    }
    _fold_prior_generation(metadata, classification_gen)
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    return metadata


def _run_module_block_edit(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    original_source: str,
    block,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    target_file: Path,
    classification_gen: Optional[dict] = None,
) -> dict:
    """Same mechanism as _run_text_block_edit, applied to a module-level
    region (analyzer/module_blocks.py): only that region goes in as
    context and comes back as output, spliced in by line range.

    `block` is the SymbolInfo (symbol_type="module_statement")
    locate_module_level_block already resolved and confirmed non-None."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("GENERATE", f"regenerating only the '{block.name}' module-level region...")

    lines = original_source.splitlines()
    block_content = "\n".join(lines[block.start_line - 1 : block.end_line])
    context_before, context_after = block_context_window(original_source, block)
    gen = generate_module_block_replacement(
        file_path=file,
        block_name=block.name,
        block_content=block_content,
        user_request=request,
        context_before=context_before,
        context_after=context_after,
    )
    new_block_lines = gen["code"].rstrip("\n").split("\n")
    spliced = lines[: block.start_line - 1] + new_block_lines + lines[block.end_line :]
    new_source = "\n".join(spliced) + ("\n" if original_source.endswith("\n") or not original_source else "")

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "MODULE_BLOCK_EDIT",
        "block": block.name,
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }
    _fold_prior_generation(metadata, classification_gen)

    on_step("SYNTAX", "checking regenerated file parses...")
    try:
        check_syntax(new_source, filename=file)
    except SyntaxCheckError as e:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "SYNTAX_ERROR"
        metadata["error"] = str(e)
        metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    metadata["diff"] = "".join(
        difflib.unified_diff(
            original_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
        )
    )
    on_preview({"operations": [], "diff": metadata["diff"], "file": file, "new_file_content": new_source})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        (tmp_project / file).write_text(new_source)

        on_step("TEST", f"running pytest ({test_target})...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        storage.put_text(f"projects/{project_id}/patches/{run_id}.full.py", new_source)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "whole_file_edit",
            project_dir=str(project_dir),
            files={file: new_source},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new version...")
    version_id = vm.create_version(
        files={file: new_source},
        change_request=request,
        strategy="MODULE_BLOCK_EDIT",
        delta_id=f"regen-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    storage.put_text(f"projects/{project_id}/patches/{run_id}.full.py", new_source)
    target_file.write_text(new_source)
    write_file_metadata(project_dir, file, new_source)
    return metadata


def _run_text_block_edit(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    original_source: str,
    language: Optional[str],
    block,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    target_file: Path,
) -> dict:
    """For formats with no function/class concept (markdown, YAML, TOML,
    INI, Dockerfile, plain text...), analyzer/text_blocks.py still finds
    the one relevant section (heading, [section], top-level key, or
    paragraph) so only that block -- not the whole file -- goes in as
    context and comes back as output, spliced in by line range.

    `block` is the SymbolInfo (symbol_type="block")
    text_blocks.locate_text_block already resolved and confirmed
    non-None and unambiguous."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("GENERATE", f"regenerating only the '{block.name}' block...")

    lines = original_source.splitlines()
    block_content = "\n".join(lines[block.start_line - 1 : block.end_line])
    context_before, context_after = block_context_window(original_source, block)
    gen = generate_block_replacement(
        file_path=file,
        block_name=block.name,
        block_content=block_content,
        user_request=request,
        context_before=context_before,
        context_after=context_after,
    )
    new_block_lines = gen["code"].rstrip("\n").split("\n")
    spliced = lines[: block.start_line - 1] + new_block_lines + lines[block.end_line :]
    new_source = "\n".join(spliced) + ("\n" if original_source.endswith("\n") or not original_source else "")

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "TEXT_BLOCK_EDIT",
        "block": block.name,
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }

    on_step("SYNTAX", "checking the regenerated file still parses...")
    try:
        check_syntax(new_source, filename=file)
    except SyntaxCheckError as e:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "SYNTAX_ERROR"
        metadata["error"] = str(e)
        metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    metadata["diff"] = "".join(
        difflib.unified_diff(
            original_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
        )
    )
    on_preview({"operations": [], "diff": metadata["diff"], "file": file, "new_file_content": new_source})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        (tmp_project / file).write_text(new_source)

        on_step("TEST", f"running pytest ({test_target})...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "text_block_edit",
            project_dir=str(project_dir),
            files={file: new_source},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new version...")
    version_id = vm.create_version(
        files={file: new_source},
        change_request=request,
        strategy="TEXT_BLOCK_EDIT",
        delta_id=f"block-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    target_file.write_text(new_source)
    return metadata


def _run_multi_text_block_edit(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    original_source: str,
    blocks,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    target_file: Path,
) -> dict:
    """Same mechanism as _run_text_block_edit, extended to SEVERAL blocks
    in one generation call instead of a single best guess -- e.g. a
    rename spanning a markup block and its inline <script> needs both
    touched together.

    `blocks` is the list analyzer.text_blocks.locate_text_blocks already
    resolved and confirmed has >= 2 entries (a single match goes to
    _run_text_block_edit instead)."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("GENERATE", f"regenerating {len(blocks)} blocks ({', '.join(b.name for b in blocks)})...")

    lines = original_source.splitlines()
    block_contents = [(b.name, "\n".join(lines[b.start_line - 1 : b.end_line])) for b in blocks]
    gen = generate_multi_block_replacement(file_path=file, blocks=block_contents, user_request=request)

    missing = [name for name, content in gen["blocks"].items() if content is None]
    if missing:
        metadata = {
            "run_id": run_id,
            "project_id": project_id,
            "base_version": base_version,
            "user_request": request,
            "strategy": "MULTI_TEXT_BLOCK_EDIT",
            "blocks": [b.name for b in blocks],
            "generation": {
                "model": gen["model"],
                "input_tokens": gen["input_tokens"],
                "cached_tokens": gen.get("cached_tokens", 0),
                "output_tokens": gen["output_tokens"],
                "total_tokens": gen["total_tokens"],
                "latency_ms": gen["latency_ms"],
                "estimated_cost_usd": estimate_cost(
                    gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
                ),
            },
            "validation": {},
            "result": {"status": "failed", "failure_class": "GENERATION_INCOMPLETE", "retry_count": 0, "fallback_used": False},
            "error": f"model's response never addressed block(s): {', '.join(missing)}",
        }
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    new_lines = list(lines)
    for b in sorted(blocks, key=lambda b: b.start_line, reverse=True):
        replacement = gen["blocks"][b.name].split("\n")
        new_lines[b.start_line - 1 : b.end_line] = replacement
    new_source = "\n".join(new_lines) + ("\n" if original_source.endswith("\n") or not original_source else "")

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "MULTI_TEXT_BLOCK_EDIT",
        "blocks": [b.name for b in blocks],
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }

    on_step("SYNTAX", "checking the regenerated file still parses...")
    try:
        check_syntax(new_source, filename=file)
    except SyntaxCheckError as e:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "SYNTAX_ERROR"
        metadata["error"] = str(e)
        metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    metadata["diff"] = "".join(
        difflib.unified_diff(
            original_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
        )
    )
    on_preview({"operations": [], "diff": metadata["diff"], "file": file, "new_file_content": new_source})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        (tmp_project / file).write_text(new_source)

        on_step("TEST", f"running pytest ({test_target})...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "multi_text_block_edit",
            project_dir=str(project_dir),
            files={file: new_source},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new version...")
    version_id = vm.create_version(
        files={file: new_source},
        change_request=request,
        strategy="MULTI_TEXT_BLOCK_EDIT",
        delta_id=f"block-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    target_file.write_text(new_source)
    return metadata


def _run_rename_identifier(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    original_source: str,
    renames: List[Dict[str, str]],
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    target_file: Path,
    classification_gen: Optional[dict] = None,
    use_joern: bool = False,
) -> dict:
    """Mechanical, zero-LLM-cost rename of every whole-word occurrence of
    each (old_name, new_name) pair in `renames`. Reached when a rename
    isn't cleanly addressable by REPLACE (module-level variable, or
    occurrences reaching outside a single function/class body) -- a
    deterministic substitution instead of escalating to full-file LLM
    regeneration. A list, not a single pair, since one request often
    means several related renames applied as one atomic change.

    Word-boundary matched (\\bold_name\\b), never a plain substring
    replace (which would mangle e.g. "column" while renaming "col").

    No repair loop: if the mechanical rename doesn't produce valid,
    passing code, that's a genuine conflict the request needs to
    resolve, not something a corrective prompt could fix.

    `use_joern=True` (opt-in) adds a cross-file caller check via Joern's
    CPG (retrieval/joern_graph.py): rename_with_subword_fallback only
    rewrites occurrences inside `file`, so a caller in a different file
    is otherwise invisible and silently breaks. Real cost (~12-45s+ JVM
    startup), so opt-in only; never blocks the rename, just surfaces
    `metadata["cross_file_impact_warning"]`."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    names_desc = ", ".join(f"'{r['old_name']}' -> '{r['new_name']}'" for r in renames)
    on_step("GENERATE", f"renaming {names_desc} -- mechanical, no LLM call")

    cross_file_impact_warning = None
    if use_joern:
        on_step("VALIDATE", "checking cross-file callers via Joern's real call graph...")
        cross_file_impact_warning = _cross_file_rename_impact(project_dir, file, [r["old_name"] for r in renames])

    new_source = original_source
    for r in renames:
        new_source = rename_with_subword_fallback(new_source, r["old_name"], r["new_name"])

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "MECHANICAL_RENAME",
        "generation": {
            "model": "n/a (mechanical rename, no LLM call)",
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "estimated_cost_usd": 0.0,
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }
    if cross_file_impact_warning:
        metadata["cross_file_impact_warning"] = cross_file_impact_warning
    _fold_prior_generation(metadata, classification_gen)

    if new_source == original_source:
        metadata["result"]["status"] = "success"
        metadata["result"]["no_op"] = True
        metadata["note"] = f"none of [{names_desc}] occur in this file -- nothing to rename"
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    on_step("SYNTAX", "checking renamed file parses...")
    try:
        check_syntax(new_source, filename=file)
    except SyntaxCheckError as e:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "SYNTAX_ERROR"
        metadata["error"] = str(e)
        metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    metadata["diff"] = "".join(
        difflib.unified_diff(
            original_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"a/{file}",
            tofile=f"b/{file}",
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        (tmp_project / file).write_text(new_source)

        try:
            check_python_imports(file, cwd=str(tmp_project))
        except SyntaxCheckError as e:
            on_step(
                "GENERATE",
                f"mechanical rename left an inconsistent reference ({e}) -- trying a narrow import fix first...",
            )
            import_span = _locate_import_span(new_source)
            fixed_source = None
            import_fix_gen = None
            if import_span is not None:
                start, end = import_span
                lines = new_source.splitlines()
                import_block = "\n".join(lines[start - 1 : end])
                import_fix_gen = generate_import_fix(file_path=file, import_block=import_block, error=str(e))
                new_import_block = import_fix_gen["code"].rstrip("\n")
                body_after_span = "\n".join(lines[end:])
                orphaned = _imported_names(import_block) - _imported_names(new_import_block)
                orphaned = {name for name in orphaned if re.search(rf"\b{re.escape(name)}\b", body_after_span)}
                if orphaned:
                    fixed_source = None
                else:
                    new_import_lines = new_import_block.split("\n")
                    candidate = "\n".join(lines[: start - 1] + new_import_lines + lines[end:])
                    candidate += "\n" if new_source.endswith("\n") else ""
                    (tmp_project / file).write_text(candidate)
                    try:
                        check_python_imports(file, cwd=str(tmp_project))
                        fixed_source = candidate
                    except SyntaxCheckError:
                        (tmp_project / file).write_text(new_source)

            if fixed_source is None:
                on_step("GENERATE", "narrow import fix didn't resolve it -- falling back to a full rewrite...")
                return _run_whole_file_edit(
                    project_dir=project_dir,
                    file=file,
                    request=request,
                    test_target=test_target,
                    project_id=project_id,
                    base_version=base_version,
                    original_source=original_source,
                    require_confirmation=require_confirmation,
                    on_step=on_step,
                    on_preview=on_preview,
                    storage=storage,
                    vm=vm,
                    target_file=target_file,
                    classification_gen=classification_gen,
                    language="python",
                )

            new_source = fixed_source
            _fold_prior_generation(metadata, import_fix_gen)
            metadata["diff"] = "".join(
                difflib.unified_diff(
                    original_source.splitlines(keepends=True),
                    new_source.splitlines(keepends=True),
                    fromfile=f"a/{file}",
                    tofile=f"b/{file}",
                )
            )

        on_step("TEST", f"running pytest ({test_target})...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    on_preview({"operations": [], "diff": metadata["diff"], "file": file, "new_file_content": new_source})

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "rename_identifier",
            project_dir=str(project_dir),
            files={file: new_source},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new version...")
    version_id = vm.create_version(
        files={file: new_source},
        change_request=request,
        strategy="MECHANICAL_RENAME",
        delta_id=f"rename-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    target_file.write_text(new_source)
    write_file_metadata(project_dir, file, new_source)
    return metadata


def _run_whole_file_delete(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    storage,
    vm: VersionManager,
    target_file: Path,
    classification_gen: Optional[dict] = None,
) -> dict:
    """Deletes the file itself, not a symbol inside it. Reached two ways:
    zero-LLM-cost, matched structurally against the file's own name
    (is_whole_file_delete_target), only after no real symbol inside the
    file matched the request; or via the model's own
    escalate:{"kind":"delete_file"}, whose already-spent classification
    cost is folded in via classification_gen.

    Either way, still runs the real test suite against a copy with the
    file removed, to catch other code that still depends on it."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("GENERATE", f"unambiguous whole-file delete of '{file}' -- skipping LLM call")

    delta_dict = {
        "schema_version": "1.0",
        "base_version": base_version,
        "operations": [{"operation": "DELETE_FILE", "target": {"file": file}}],
    }
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "FILE_DELETE",
        "operations": delta_dict["operations"],
        "generation": {
            "model": classification_gen["model"] if classification_gen else "n/a (whole-file delete, no LLM call)",
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "estimated_cost_usd": 0.0,
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }
    _fold_prior_generation(metadata, classification_gen)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        (tmp_project / file).unlink()

        on_step("TEST", f"running pytest ({test_target}) with '{file}' removed...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = (
            f"deleting '{file}' breaks the existing test suite: {test_result['output_tail']} -- "
            "other code still depends on it; remove/update the dependents first, then retry"
        )
        _persist_run(storage, project_id, run_id, delta_dict, metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "delete_file",
            project_dir=str(project_dir),
            deleted_files=[file],
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new version...")
    version_id = vm.create_version(
        files={},
        deleted_files=[file],
        change_request=request,
        strategy="FILE_DELETE",
        delta_id=f"delete-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    _persist_run(storage, project_id, run_id, delta_dict, metadata)
    target_file.unlink()
    delete_file_metadata(project_dir, file)
    return metadata


def _run_language_conversion(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str,
    project_id: str,
    original_source: str,
    target_language: str,
    target_extension: str,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    classification_gen: Optional[dict] = None,
) -> dict:
    """"Rewrite this in Python"/"convert to JavaScript" isn't an edit to
    the existing file -- the result can't still parse as the original
    language, so no REPLACE/INSERT/DELETE expresses it. Produces a new
    sibling file in the target language instead, closer to CREATE than
    EDIT (mirrors create_pipeline.run_create's shape, no repair loop: a
    wrong translation needs a different prompt, not a corrective patch).

    target_language/target_extension come from the model's own escalate
    response, not a hardcoded language catalog, so any language the model
    knows works, not just ones listed here in advance."""
    new_file = target_file_for_conversion(file, target_extension)
    new_target_file = project_dir / new_file
    if new_target_file.exists():
        raise FileExistsError(
            f"'{new_target_file}' already exists -- rename it or ask to edit that file directly"
        )

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("GENERATE", f"generating {new_file} ({target_language}) from {file}...")
    gen = generate_full_file_edit(file_path=new_file, original_source=original_source, user_request=request)
    new_source = gen["code"]

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": vm.get_head() or "v0",
        "user_request": request,
        "strategy": "FULL_REGENERATION",
        "source_file": file,
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }
    _fold_prior_generation(metadata, classification_gen)

    on_step("SYNTAX", f"checking generated {new_file} parses...")
    try:
        check_syntax(new_source, filename=new_file)
    except SyntaxCheckError as e:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "SYNTAX_ERROR"
        metadata["error"] = str(e)
        metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    on_preview({"new_file_content": new_source, "file": new_file})

    test_result = None
    if test_target and (project_dir / test_target).exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_project = Path(tmp) / "project"
            copy_project_for_validation(project_dir, tmp_project)
            (tmp_project / new_file).write_text(new_source)

            on_step("TEST", f"running pytest ({test_target})...")
            test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"] if test_result else None,
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")} if test_result else None,
    }

    if test_result is not None and not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        storage.put_text(f"projects/{project_id}/patches/{run_id}.full.py", new_source)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = new_file
        pending_confirmations.stash(
            run_id,
            "language_conversion",
            project_dir=str(project_dir),
            files={new_file: new_source},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new file and committing version...")
    version_id = vm.create_version(
        files={new_file: new_source},
        change_request=request,
        strategy="FULL_REGENERATION",
        delta_id=f"convert-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    storage.put_text(f"projects/{project_id}/patches/{run_id}.full.py", new_source)
    new_target_file.write_text(new_source)
    write_file_metadata(project_dir, new_file, new_source)
    return metadata


def _run_question(
    file: str,
    request: str,
    project_id: str,
    base_version: str,
    original_source: str,
    on_step: Callable[[str, str], None],
    classification_gen: Optional[dict] = None,
) -> dict:
    """"List all algorithms in this file"/"what does handle() do" aren't
    edit instructions -- forcing them through REPLACE/INSERT/DELETE made
    the model fabricate some edit to satisfy the schema instead of
    answering. No Delta IR, no apply/test/versioning/commit -- always
    returns result.status == "answered" directly, scoped to this one
    file only (not repo-wide retrieval)."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    on_step("ANSWER", "answering from the file's real content...")
    gen = generate_answer(file_path=file, source=original_source, question=request)

    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "QUESTION_ANSWERING",
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "result": {"status": "answered"},
        "answer": gen["answer"],
    }
    _fold_prior_generation(metadata, classification_gen)

    storage = get_storage()
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    return metadata


def _run_create_files(
    project_dir: Path,
    current_file: str,
    current_source: str,
    new_paths: List[str],
    also_link_current_file: bool,
    request: str,
    test_target: str,
    project_id: str,
    base_version: str,
    language: str,
    require_confirmation: bool,
    on_step: Callable[[str, str], None],
    on_preview: Callable[[dict], None],
    storage,
    vm: VersionManager,
    classification_gen: Optional[dict] = None,
) -> dict:
    """"Make a new frontend and backend file" while editing chatbot.py
    isn't an edit to chatbot.py -- closer to CREATE, extended to N files
    in one request instead of CREATE mode's one-file-at-a-time flow.

    also_link_current_file covers "make a .env that links to this
    chatbot": creates the new file(s) FIRST, then regenerates the
    currently open file with an instruction referencing what was just
    created (generate_full_file_edit). Both are one atomic accept/reject.

    No repair loop: a wrong file needs a different prompt, not a
    corrective patch.

    A path ending in "/" is a bare folder, not a file -- gets no LLM call,
    just mkdir'd, since there's nothing to generate for an empty
    directory."""
    folder_paths = [p for p in new_paths if p.endswith("/")]
    file_paths = [p for p in new_paths if not p.endswith("/")]

    for path in file_paths:
        target = project_dir / path
        if target.exists():
            raise FileExistsError(f"'{target}' already exists -- rename it or ask to edit that file directly")
    for path in folder_paths:
        target = project_dir / path.rstrip("/")
        if target.exists() and not target.is_dir():
            raise FileExistsError(f"'{target}' already exists as a file, not a directory")

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    files: Dict[str, str] = {}
    gens = []

    for path in file_paths:
        on_step("GENERATE", f"generating {path}...")
        file_request = (
            f"{request}\n\nFor reference, here is the current content of {current_file} "
            f"this new file relates to:\n\n{current_source}"
            if also_link_current_file
            else request
        )
        is_binary = is_binary_artifact_target(path)
        if is_binary:
            file_request += binary_artifact_instructions(path)
        gen = generate_full_file(file_path=path, user_request=file_request)
        if is_binary:
            on_step("GENERATE", f"running generated script to materialize {path}...")
            try:
                files[path] = generate_binary_artifact_base64(gen["code"], path)
            except BinaryArtifactError as e:
                spent = gens + [gen]
                run_gen = {
                    "model": spent[0]["model"],
                    "input_tokens": sum(g["input_tokens"] for g in spent),
                    "cached_tokens": sum(g.get("cached_tokens", 0) for g in spent),
                    "output_tokens": sum(g["output_tokens"] for g in spent),
                    "total_tokens": sum(g["total_tokens"] for g in spent),
                    "latency_ms": sum(g["latency_ms"] for g in spent),
                }
                metadata = {
                    "run_id": run_id,
                    "project_id": project_id,
                    "base_version": base_version,
                    "user_request": request,
                    "strategy": "FULL_REGENERATION",
                    "files": list(files.keys()),
                    "folders": [p.rstrip("/") for p in folder_paths],
                    "generation": {
                        **run_gen,
                        "estimated_cost_usd": sum(
                            estimate_cost(g["model"], g["input_tokens"], g["output_tokens"], g.get("cached_tokens", 0))
                            for g in spent
                        ),
                    },
                    "validation": {"syntax_passed": False, "tests_passed": None},
                    "result": {"status": "failed", "failure_class": "SYNTAX_ERROR", "retry_count": 0, "fallback_used": False},
                    "error": str(e),
                }
                _fold_prior_generation(metadata, classification_gen)
                storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
                return metadata
        else:
            files[path] = gen["code"]
        gens.append(gen)

    if also_link_current_file and files:
        created_summary = "\n\n".join(
            f"{p} now contains:\n{_compact_created_file_summary(p, content)}" for p, content in files.items()
        )
        on_step("GENERATE", f"updating {current_file} to use the new file(s)...")
        link_gen = generate_full_file_edit(
            file_path=current_file,
            original_source=current_source,
            user_request=(
                f"{request}\n\n{created_summary}\n\nUpdate this file ({current_file}) to load/use the above "
                "accordingly. Change only what's needed for that -- preserve all other existing logic exactly."
            ),
        )
        files[current_file] = link_gen["code"]
        gens.append(link_gen)

    run_gen = (
        {
            "model": gens[0]["model"],
            "input_tokens": sum(g["input_tokens"] for g in gens),
            "cached_tokens": sum(g.get("cached_tokens", 0) for g in gens),
            "output_tokens": sum(g["output_tokens"] for g in gens),
            "total_tokens": sum(g["total_tokens"] for g in gens),
            "latency_ms": sum(g["latency_ms"] for g in gens),
        }
        if gens
        else {"model": "n/a (folder-only create, no LLM call)", "input_tokens": 0, "cached_tokens": 0,
              "output_tokens": 0, "total_tokens": 0, "latency_ms": 0}
    )
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "FULL_REGENERATION",
        "files": list(files.keys()),
        "folders": [p.rstrip("/") for p in folder_paths],
        "generation": {
            **run_gen,
            "estimated_cost_usd": sum(
                estimate_cost(g["model"], g["input_tokens"], g["output_tokens"], g.get("cached_tokens", 0))
                for g in gens
            ),
        },
        "validation": {},
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }
    _fold_prior_generation(metadata, classification_gen)

    on_step("SYNTAX", f"checking {len(files)} generated file(s) parse...")
    for path, content in files.items():
        if is_binary_artifact_target(path):
            continue
        try:
            check_syntax(content, filename=path)
        except SyntaxCheckError as e:
            metadata["result"]["status"] = "failed"
            metadata["result"]["failure_class"] = "SYNTAX_ERROR"
            metadata["error"] = f"{path}: {e}"
            metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
            storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
            return metadata

    for path, content in files.items():
        if not is_binary_artifact_target(path):
            on_preview({"new_file_content": content, "file": path})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_project = Path(tmp) / "project"
        copy_project_for_validation(project_dir, tmp_project)
        for path, content in files.items():
            write_file_content(tmp_project / path, path, content)
        for path in folder_paths:
            (tmp_project / path.rstrip("/")).mkdir(parents=True, exist_ok=True)

        on_step("TEST", f"running pytest ({test_target})...")
        test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"],
        "no_tests_collected": test_result["no_tests_collected"],
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
        "regression_detected": not test_result["tests_passed"],
    }

    if not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["result"]["failure_class"] = "TEST_TIMEOUT" if test_result.get("timed_out") else "TEST_FAILURE"
        metadata["error"] = test_result["output_tail"]
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        metadata["file"] = next((p for p in file_paths if not is_binary_artifact_target(p)), None)
        pending_confirmations.stash(
            run_id,
            "create_files",
            project_dir=str(project_dir),
            files=files,
            folders=folder_paths,
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "validation passed, writing new file(s) and committing version...")
    version_id = vm.create_version(
        files=files,
        change_request=request,
        strategy="FULL_REGENERATION",
        delta_id=f"create-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    for path, content in files.items():
        write_file_content(project_dir / path, path, content)
        if not is_binary_artifact_target(path):
            write_file_metadata(project_dir, path, content)
    for path in folder_paths:
        (project_dir / path.rstrip("/")).mkdir(parents=True, exist_ok=True)
    return metadata


def run_edit(
    project_dir: Path,
    file: str,
    request: str,
    test_target: str = ".",
    project_id: Optional[str] = None,
    confirm_symbol: Optional[str] = None,
    confirm_symbol_type: Optional[str] = None,
    confirm_symbol_line: Optional[int] = None,
    confirm_symbols: Optional[List[dict]] = None,
    require_confirmation: bool = False,
    use_hybrid_retrieval: bool = False,
    use_joern: bool = False,
    on_step: Callable[[str, str], None] = lambda tag, msg: None,
    on_preview: Callable[[dict], None] = lambda data: None,
) -> dict:
    """Runs the full edit pipeline, always returns a metadata dict (never
    raises for validation/generation failure -- check
    metadata["result"]["status"]); only raises for programmer errors
    (missing file, bad args).

    `on_preview` fires once, right after APPLY, with {"operations": [...],
    "diff": "...", "file": "...", "new_file_content": "..."} -- before
    TEST/COMMIT -- so a caller can show the code change as soon as it
    exists.

    A delete request matching more than one real symbol returns early
    with result.status == "needs_selection" and a `candidates` list
    instead of guessing. The caller re-runs with either `confirm_symbol`
    (+ `confirm_symbol_type`, + `confirm_symbol_line` to disambiguate a
    duplicate name via find_symbol's prefer_line) to confirm exactly one
    candidate, or `confirm_symbols` (a list of {"name", "symbol_type",
    "start_line"} dicts) to confirm several at once -- both skip
    localization and the LLM call and construct the DELETE(s) directly.

    `require_confirmation=True` (web UI's human-in-the-loop mode; CLI
    never sets this) pauses after tests pass but before COMMIT, returning
    result.status == "awaiting_confirmation" and stashing everything
    needed to finish in `pending_confirmations`. Nothing is written to
    disk until `pending_confirmations.resolve` (accept = commit, reject =
    discard).

    `use_hybrid_retrieval=True` fuses name/docstring symbol match with
    BM25 + vector/semantic retrieval (context_builder.build_context's
    use_hybrid). Off by default: it's a real embeddings-API call, and
    this project's offline test suite relies on it defaulting off to stay
    deterministic and network-free.

    `use_joern=True` (opt-in) adds a cross-file caller check via Joern's
    CPG before a mechanical rename -- rename_with_subword_fallback only
    touches the file it's given, so a caller elsewhere would otherwise
    silently break. Never blocks the rename; surfaced as
    metadata["cross_file_impact_warning"]. Off by default: real
    ~12-45s+ JVM cost."""

    project_dir = Path(project_dir).resolve()
    target_file = project_dir / file
    project_id = project_id or project_dir.name.lower()

    if _project_dir_is_this_engine(project_dir):
        on_step(
            "WARNING",
            f"project directory ({project_dir}) is this engine's own repository (or inside it) -- "
            "test validation will run ITS unrelated test suite, not a real project's own, and will "
            "likely be slow and report unrelated failures. Point PROJECT DIRECTORY at a separate, "
            "dedicated folder for real use.",
        )

    if not target_file.exists():
        raise FileNotFoundError(f"target file not found: {target_file}")
    if target_file.is_dir():
        raise IsADirectoryError(f"'{file}' is a directory, not a file -- name a specific file inside it to edit")

    narrowed_test_target = _effective_test_target(test_target, project_dir, file)
    if narrowed_test_target != test_target:
        on_step(
            "TEST",
            f"test target '.' narrowed to '{narrowed_test_target}' -- validating against this file's own "
            "tests, not the entire project directory",
        )
        test_target = narrowed_test_target

    storage = get_storage()
    vm = VersionManager(storage, project_id)
    base_version = vm.get_head() or "v0"

    try:
        original_source = target_file.read_text()
    except UnicodeDecodeError:
        raise ValueError(
            f"'{file}' is a binary file -- this project can only edit plain-text files "
            "(code, .csv, .sql, config, markdown, ...) right now. Spreadsheets (.xlsx), "
            "databases (.db/.sqlite), and other binary formats aren't supported yet."
        )
    language = "python" if file.endswith(".py") else detect_language(file)
    if language is None and looks_like_python(original_source):
        language = "python"

    if not confirm_symbol and not confirm_symbols:
        rename_target = find_rename_target(original_source, request, language)
        if rename_target:
            old_name, new_name = rename_target
            on_step("LOCALIZE", "target unambiguous -- skipping localization")
            on_step("GENERATE", f"unambiguous rename '{old_name}' -> '{new_name}' -- skipping LLM call")
            return _run_rename_identifier(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                renames=[{"old_name": old_name, "new_name": new_name}],
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
                use_joern=use_joern,
            )

    multi_delete_targets = None
    if not confirm_symbol and not confirm_symbols and is_delete_intent(request):
        multi_delete_targets = find_multi_delete_targets(original_source, request, language)
        if multi_delete_targets is None:
            delete_candidates = find_delete_candidates(original_source, request, language)
            if not delete_candidates and is_whole_file_delete_target(file, request):
                return _run_whole_file_delete(
                    project_dir=project_dir,
                    file=file,
                    request=request,
                    test_target=test_target,
                    project_id=project_id,
                    base_version=base_version,
                    require_confirmation=require_confirmation,
                    on_step=on_step,
                    storage=storage,
                    vm=vm,
                    target_file=target_file,
                )
            if len(delete_candidates) > 1:
                return {
                    "result": {"status": "needs_selection"},
                    "strategy": "STRUCTURED_EDIT",
                    "file": file,
                    "base_version": base_version,
                    "user_request": request,
                    "candidates": [
                        {
                            "name": sym.name,
                            "symbol_type": sym.symbol_type,
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                        }
                        for sym in delete_candidates
                    ],
                }
            if len(delete_candidates) == 1:
                only = delete_candidates[0]
                confirm_symbol = only.name
                confirm_symbol_type = only.symbol_type
                confirm_symbol_line = only.start_line

    if (
        not confirm_symbol
        and not confirm_symbols
        and not multi_delete_targets
        and (language is None or not language_has_symbol_concept(language))
    ):
        if use_hybrid_retrieval:
            on_step("RETRIEVE", "block name-match + BM25 + vector retrieval, then fusing rankings...")
        matched_blocks = locate_text_blocks(original_source, request, language)
        if len(matched_blocks) > 1:
            on_step(
                "LOCALIZE",
                f"{len(matched_blocks)} blocks relevant ({', '.join(b.name for b in matched_blocks)}) -- "
                "skipping structured-edit classification",
            )
            return _run_multi_text_block_edit(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                blocks=matched_blocks,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
            )
        block = locate_text_block(original_source, request, language, use_hybrid=use_hybrid_retrieval)
        if block:
            on_step("LOCALIZE", f"'{block.name}' block unambiguous -- skipping structured-edit classification")
            return _run_text_block_edit(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                language=language,
                block=block,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
            )
        on_step("LOCALIZE", f"'{language}' has no function/class concept -- skipping structured-edit classification")
        return _run_whole_file_edit(
            project_dir=project_dir,
            file=file,
            request=request,
            test_target=test_target,
            project_id=project_id,
            base_version=base_version,
            original_source=original_source,
            require_confirmation=require_confirmation,
            on_step=on_step,
            on_preview=on_preview,
            storage=storage,
            vm=vm,
            target_file=target_file,
            language=language,
        )

    if not confirm_symbol and not confirm_symbols and not multi_delete_targets:
        jev_kind = classify_request_kind(request)
        if jev_kind in DISPATCHABLE_KINDS:
            on_step("LOCALIZE", f"Jev classified this as '{jev_kind}' -- skipping structured-edit classification")
            if jev_kind == "question":
                return _run_question(
                    file=file,
                    request=request,
                    project_id=project_id,
                    base_version=base_version,
                    original_source=original_source,
                    on_step=on_step,
                )
            return _run_whole_file_edit(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
                language=language,
            )

    if confirm_symbol or multi_delete_targets or confirm_symbols:
        if multi_delete_targets:
            names = ", ".join(s.name for s in multi_delete_targets)
            on_step("LOCALIZE", f"{len(multi_delete_targets)} targets unambiguous -- skipping localization")
            on_step("GENERATE", f"unambiguous delete of {names} -- skipping LLM call")
            operations = [
                {"operation": "DELETE", "target": {"file": file, "symbol_type": s.symbol_type, "symbol_name": s.name}}
                for s in multi_delete_targets
            ]
            candidate_symbols = [s.name for s in multi_delete_targets]
            candidate_lines = {s.name: s.start_line for s in multi_delete_targets}
        elif confirm_symbols:
            names = ", ".join(s["name"] for s in confirm_symbols)
            on_step("LOCALIZE", f"{len(confirm_symbols)} targets confirmed -- skipping localization")
            on_step("GENERATE", f"unambiguous delete of {names} -- skipping LLM call")
            operations = [
                {
                    "operation": "DELETE",
                    "target": {
                        "file": file,
                        "symbol_type": s.get("symbol_type") or "function",
                        "symbol_name": s["name"],
                    },
                }
                for s in confirm_symbols
            ]
            candidate_symbols = [s["name"] for s in confirm_symbols]
            candidate_lines = {s["name"]: s["start_line"] for s in confirm_symbols if s.get("start_line") is not None}
        else:
            on_step("LOCALIZE", "target unambiguous -- skipping localization")
            on_step("GENERATE", f"unambiguous delete of '{confirm_symbol}' -- skipping LLM call")
            operations = [
                {
                    "operation": "DELETE",
                    "target": {
                        "file": file,
                        "symbol_type": confirm_symbol_type or "function",
                        "symbol_name": confirm_symbol,
                    },
                }
            ]
            candidate_symbols = [confirm_symbol]
            candidate_lines = {confirm_symbol: confirm_symbol_line} if confirm_symbol_line is not None else {}

        delta_dict = {"schema_version": "1.0", "base_version": base_version, "operations": operations}
        gen = {
            "raw_json": json.dumps(delta_dict),
            "delta_dict": delta_dict,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "model": "n/a (confirmed delete, no LLM call)",
        }
        ctx = {
            "total_lines": len(original_source.splitlines()),
            "context_lines": 0,
            "candidate_symbols": candidate_symbols,
            "used_localization": True,
            "candidate_lines": candidate_lines,
        }
    else:
        if use_hybrid_retrieval:
            on_step("RETRIEVE", "symbol + BM25 + vector retrieval, then fusing rankings...")
        ctx = build_context(original_source, request, language=language, use_hybrid=use_hybrid_retrieval)
        if ctx["candidate_symbols"]:
            no_match_label = None
        elif ctx["context_lines"] < ctx["total_lines"]:
            no_match_label = "none, using imports + name index only"
        else:
            no_match_label = "none, using whole file"
        on_step(
            "LOCALIZE",
            f"context: {ctx['context_lines']}/{ctx['total_lines']} lines "
            f"(symbols={ctx['candidate_symbols'] or no_match_label})",
        )

        on_step("GENERATE", "calling LLM for a structured delta...")
        gen = generate_delta(
            file_path=file,
            context_content=ctx["context"],
            user_request=request,
            base_version=base_version,
        )

        escalate = gen["delta_dict"].get("escalate") or {}
        kind = escalate.get("kind")
        if kind == "whole_file":
            on_step("GENERATE", "model escalated: file-wide change reaches outside any single symbol...")
            return _run_whole_file_edit(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
                classification_gen=gen,
                language=language,
            )
        if kind == "language_conversion" and escalate.get("target_language") and escalate.get("target_extension"):
            on_step("GENERATE", f"model escalated: rewrite into {escalate['target_language']}...")
            return _run_language_conversion(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                original_source=original_source,
                target_language=escalate["target_language"],
                target_extension=escalate["target_extension"],
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                classification_gen=gen,
            )
        if kind == "question":
            on_step("GENERATE", "model escalated: this is a question, not an edit instruction...")
            return _run_question(
                file=file,
                request=request,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                on_step=on_step,
                classification_gen=gen,
            )
        if kind == "create_files" and escalate.get("files"):
            on_step("GENERATE", f"model escalated: create {len(escalate['files'])} new file(s), not an edit...")
            return _run_create_files(
                project_dir=project_dir,
                current_file=file,
                current_source=original_source,
                new_paths=escalate["files"],
                also_link_current_file=bool(escalate.get("also_link_current_file")),
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                language=language,
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                classification_gen=gen,
            )
        if kind == "delete_file":
            on_step("GENERATE", "model escalated: this deletes the whole file, not a symbol in it...")
            return _run_whole_file_delete(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                require_confirmation=require_confirmation,
                on_step=on_step,
                storage=storage,
                vm=vm,
                target_file=target_file,
                classification_gen=gen,
            )
        if kind == "rename_identifier" and escalate.get("renames"):
            on_step("GENERATE", "model escalated: identifier rename reaches outside any single symbol...")
            return _run_rename_identifier(
                project_dir=project_dir,
                file=file,
                request=request,
                test_target=test_target,
                project_id=project_id,
                base_version=base_version,
                original_source=original_source,
                renames=escalate["renames"],
                require_confirmation=require_confirmation,
                on_step=on_step,
                on_preview=on_preview,
                storage=storage,
                vm=vm,
                target_file=target_file,
                classification_gen=gen,
                use_joern=use_joern,
            )

    prefer_lines = ctx.get("candidate_lines") or {}

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "STRUCTURED_EDIT",
        "context": {
            "total_lines": ctx["total_lines"],
            "context_lines": ctx["context_lines"],
            "affected_files": [file],
            "affected_symbols": ctx["candidate_symbols"],
            "used_localization": ctx["used_localization"],
        },
        "generation": {
            "model": gen["model"],
            "input_tokens": gen["input_tokens"],
            "cached_tokens": gen.get("cached_tokens", 0),
            "output_tokens": gen["output_tokens"],
            "total_tokens": gen["total_tokens"],
            "latency_ms": gen["latency_ms"],
            "estimated_cost_usd": estimate_cost(
                gen["model"], gen["input_tokens"], gen["output_tokens"], gen.get("cached_tokens", 0)
            ),
        },
        "validation": {},
        "repair_history": [],
        "result": {"status": "pending", "retry_count": 0, "fallback_used": False},
    }

    attempt = 0
    delta = None
    while True:
        try:
            on_step("VALIDATE", "checking delta schema and target symbol...")
            validate_schema(gen["delta_dict"])
            delta = DeltaIR.from_dict(gen["delta_dict"])
            metadata["operations"] = [op.to_dict() for op in delta.operations]

            if not delta.operations:
                metadata["result"]["status"] = "success"
                metadata["result"]["no_op"] = True
                metadata["result"]["retry_count"] = attempt
                metadata["validation"] = {"patch_applied": False, "no_changes_needed": True}
                metadata["note"] = "model found nothing to change for this request"
                _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata)
                return metadata

            validate_targets(
                delta,
                original_source,
                language=language,
                prefer_lines=prefer_lines,
                content_shown_for=set(ctx["candidate_symbols"]),
            )

            symbols = index_symbols(original_source, language)
            for op_dict, op in zip(metadata["operations"], delta.operations):
                t = op.target
                if op.operation in ("REPLACE", "DELETE"):
                    allow_delegate = op.operation == "REPLACE"
                    sym = find_symbol(
                        symbols, t.symbol_type, t.symbol_name, prefer_lines.get(t.symbol_name), original_source, allow_delegate
                    )
                    op_dict["line_range"] = {"start": sym.start_line, "end": sym.end_line}
                else:
                    if t.anchor:
                        anchor = find_symbol(symbols, t.symbol_type, t.anchor, prefer_lines.get(t.anchor), original_source)
                        op_dict["line_range"] = {"after_line": anchor.end_line}
                    else:
                        op_dict["line_range"] = {"after_line": len(original_source.splitlines())}
                    actual_name = defined_symbol_name(op.content, language)
                    if actual_name and actual_name != t.symbol_name:
                        op_dict["target"]["symbol_name"] = actual_name

            pending_renames = []
            for op_dict, op in zip(metadata["operations"], delta.operations):
                if op.operation != "REPLACE":
                    continue
                old_name = op.target.symbol_name
                new_name = defined_symbol_name(op.content, language)
                if not new_name or new_name == old_name:
                    continue
                span = op_dict["line_range"]
                lines = original_source.splitlines()
                outside = "\n".join(lines[: span["start"] - 1] + lines[span["end"] :])
                if re.search(rf"\b{re.escape(old_name)}\b", outside):
                    pending_renames.append((old_name, new_name))

            on_step("APPLY", "splicing delta into a working copy...")
            new_source = apply_delta(original_source, delta, language=language, prefer_lines=prefer_lines)
            for old_name, new_name in pending_renames:
                on_step(
                    "APPLY",
                    f"'{old_name}' renamed to '{new_name}' -- sweeping remaining references elsewhere in the file...",
                )
                new_source = rename_with_subword_fallback(new_source, old_name, new_name)
            check_syntax(new_source, filename=file)

            metadata["diff"] = "".join(
                difflib.unified_diff(
                    original_source.splitlines(keepends=True),
                    new_source.splitlines(keepends=True),
                    fromfile=f"a/{file}",
                    tofile=f"b/{file}",
                )
            )
            on_preview(
                {
                    "operations": metadata["operations"],
                    "diff": metadata["diff"],
                    "file": file,
                    "new_file_content": new_source,
                }
            )

            with tempfile.TemporaryDirectory() as tmp:
                tmp_project = Path(tmp) / "project"
                copy_project_for_validation(project_dir, tmp_project)
                (tmp_project / file).write_text(new_source)

                on_step("TEST", f"running pytest ({test_target})...")
                test_result = run_tests(test_target, cwd=str(tmp_project))

            metadata["validation"] = {
                "patch_applied": True,
                "syntax_passed": True,
                "tests_passed": test_result["tests_passed"],
                "no_tests_collected": test_result["no_tests_collected"],
                "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")},
                "regression_detected": not test_result["tests_passed"],
            }

            if not test_result["tests_passed"]:
                raise _TestFailure(test_result["output_tail"])

            change_ratio = _change_ratio(original_source, new_source)
            metadata["change_ratio"] = change_ratio

            if change_ratio == 0:
                metadata["result"]["status"] = "success"
                metadata["result"]["retry_count"] = attempt
                metadata["result"]["no_op"] = True
                metadata["note"] = "applied delta produced no actual file change"
                _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata)
                return metadata

            if require_confirmation:
                metadata["result"]["status"] = "awaiting_confirmation"
                metadata["result"]["retry_count"] = attempt
                metadata["file"] = file
                pending_confirmations.stash(
                    run_id,
                    "edit",
                    project_dir=str(project_dir),
                    files={file: new_source},
                    request=request,
                    project_id=project_id,
                    delta_dict=gen["delta_dict"],
                    metadata=metadata,
                )
                return metadata

            on_step("COMMIT", "validation passed, writing new version...")
            version_id = vm.create_version(
                files={file: new_source},
                change_request=request,
                strategy="STRUCTURED_EDIT",
                delta_id=f"delta-{run_id}",
                validation_status="passed",
            )
            metadata["result"]["status"] = "success"
            metadata["result"]["retry_count"] = attempt
            metadata["new_version"] = version_id

            if attempt:
                _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata, attempt=attempt)
            _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata)

            target_file.write_text(new_source)
            write_file_metadata(project_dir, file, new_source)
            return metadata

        except (DeltaValidationError, AmbiguousSymbolError, ApplyError, SyntaxCheckError, _TestFailure) as e:
            failure_class = _classify_failure(e)
            failure_detail = str(e)[:2000]

            _persist_run(storage, project_id, run_id, gen.get("delta_dict"), metadata, attempt=attempt)
            metadata["repair_history"].append(
                {"attempt": attempt, "failure_class": failure_class, "failure_detail": failure_detail}
            )

            if isinstance(e, UnseenReplaceTargetError):
                on_step("GENERATE", "REPLACE target's body was never shown -- escalating to whole-file regeneration...")
                return _run_whole_file_edit(
                    project_dir=project_dir,
                    file=file,
                    request=request,
                    test_target=test_target,
                    project_id=project_id,
                    base_version=base_version,
                    original_source=original_source,
                    require_confirmation=require_confirmation,
                    on_step=on_step,
                    on_preview=on_preview,
                    storage=storage,
                    vm=vm,
                    target_file=target_file,
                    classification_gen=gen,
                    language=language,
                )

            if failure_class == "TEST_FAILURE":
                unrelated_file = _unrelated_collection_error_file(failure_detail, file)
                if unrelated_file:
                    metadata["result"]["status"] = "failed"
                    metadata["result"]["retry_count"] = attempt
                    metadata["result"]["failure_class"] = "UNRELATED_TEST_COLLECTION_ERROR"
                    metadata["error"] = (
                        f"pytest could not even collect '{unrelated_file}' -- a real error in that file, "
                        f"unrelated to this edit of '{file}' -- before any test ran. Retrying won't help: "
                        f"nothing about this delta touches '{unrelated_file}'. Fix or exclude it from "
                        "test_target, or scope test_target to just this project.\n\n" + failure_detail
                    )
                    _persist_run(storage, project_id, run_id, gen.get("delta_dict"), metadata)
                    return metadata

            delete_only = bool(delta and delta.operations and all(op.operation == "DELETE" for op in delta.operations))
            if delete_only and failure_class == "TEST_FAILURE":
                metadata["result"]["status"] = "failed"
                metadata["result"]["retry_count"] = attempt
                metadata["result"]["failure_class"] = failure_class
                deleted_names = ", ".join(op.target.symbol_name for op in delta.operations)
                metadata["error"] = (
                    f"removing '{deleted_names}' breaks the existing test suite: {failure_detail} -- "
                    "deletions aren't retried via repair (nothing to correct, only code elsewhere -- "
                    "a test, a caller -- that still depends on what was asked to be removed); update "
                    "or remove the dependent code/tests first, then retry"
                )
                _persist_run(storage, project_id, run_id, gen.get("delta_dict"), metadata)
                return metadata

            if attempt >= MAX_REPAIR_ATTEMPTS or confirm_symbol:
                metadata["result"]["status"] = "failed"
                metadata["result"]["retry_count"] = attempt
                metadata["result"]["failure_class"] = failure_class
                metadata["error"] = (
                    f"confirmed target '{confirm_symbol}' could not be applied: {failure_detail} "
                    "-- the file may have changed since it was listed; try the request again"
                    if confirm_symbol
                    else failure_detail
                )
                _persist_run(storage, project_id, run_id, gen.get("delta_dict"), metadata)
                return metadata

            attempt += 1
            on_step(
                "REPAIR",
                f"attempt {attempt}/{MAX_REPAIR_ATTEMPTS}: {failure_class} -- {failure_detail[:150]}",
            )
            repair_gen = generate_repair(
                file_path=file,
                context_content=ctx["context"],
                user_request=request,
                base_version=base_version,
                failed_delta=gen.get("delta_dict") or {},
                failure_class=failure_class,
                failure_detail=failure_detail,
            )
            for key in ("input_tokens", "output_tokens", "total_tokens", "latency_ms"):
                metadata["generation"][key] += repair_gen[key]
            metadata["generation"]["cached_tokens"] += repair_gen.get("cached_tokens", 0)
            metadata["generation"]["estimated_cost_usd"] = round(
                metadata["generation"]["estimated_cost_usd"]
                + estimate_cost(
                    repair_gen["model"],
                    repair_gen["input_tokens"],
                    repair_gen["output_tokens"],
                    repair_gen.get("cached_tokens", 0),
                ),
                6,
            )
            gen = repair_gen
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Incremental Editing Engine -- Phase 1 structured-edit pipeline"
    )
    parser.add_argument("--project-dir", required=True, help="Directory containing the project (file + its tests)")
    parser.add_argument("--file", required=True, help="Target file, relative to --project-dir")
    parser.add_argument("--request", required=True, help="Natural-language change request")
    parser.add_argument(
        "--test-target", default=".", help="Path (relative to --project-dir) pytest should run, default whole dir"
    )
    parser.add_argument("--project-id", default=None, help="Storage project id, default = project dir name")
    args = parser.parse_args()

    try:
        metadata = run_edit(
            project_dir=Path(args.project_dir),
            file=args.file,
            request=args.request,
            test_target=args.test_target,
            project_id=args.project_id,
            on_step=lambda tag, msg: print(f"[{tag}] {msg}"),
        )
    except FileNotFoundError as e:
        sys.exit(str(e))

    status = metadata["result"]["status"]
    if status == "success":
        gen = metadata["generation"]
        print(
            f"done. {metadata['base_version']} -> {metadata['new_version']}. "
            f"tokens={gen['total_tokens']} latency_ms={gen['latency_ms']} "
            f"change_ratio={metadata['change_ratio']:.1%} retries={metadata['result']['retry_count']}"
        )
    else:
        print(f"FAILED: {metadata.get('error', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
