"""EDIT path, now with bounded self-repair:

Python file -> user request -> localized context -> GPT-5.4 -> Structured
Delta -> Validate -> Apply -> pytest -> (on failure: classify, repair,
retry, bounded) -> store original + delta + new version in MinIO ->
record tokens/cost/latency.

Self-repair (PHOENIX architecture doc, section 18): a failure is
classified into one of a small set of classes, then a repair prompt is
built containing the previous (failed) Delta IR plus the exact error, and
the model gets a bounded number of targeted retries -- never an
unbounded loop. If the budget runs out, the run is reported failed;
STRUCTURED_EDIT itself still never falls back to full regeneration
mid-repair (that's a separate, larger piece).

What routes to FULL_REGENERATION instead is decided by the *model*, not a
hand-written keyword/language list here: a file-wide structural/hygiene
request (comments, whitespace, docstrings, formatting -- reaching
content outside any single symbol's own span) or a request for a
different programming language entirely are both things no
REPLACE/INSERT/DELETE on a named symbol could ever satisfy, so there's
nothing a repair loop could fix either way -- it needs the whole file as
both input and output from the start. generate_delta's normal call
(same compact context, no extra round-trip) can return
{"operations":[],"escalate":{"kind":"whole_file"}} or {"kind":
"language_conversion","target_language":...,"target_extension":...}
instead of guessing -- see structured_edit.py's prompt. A hand-rolled
regex/keyword classifier for this used to live here and in
analyzer/locator.py; it kept needing a new case for every new phrasing
and could never cover "every language," which the model already
understands natively.

`run_edit` is the callback-driven core (no printing, no sys.exit) so both
the plain CLI below and `incremental_editing/cli.py` (the rich-formatted
CLI) can drive it.
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
from ..analyzer.text_blocks import block_context_window, locate_text_block
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
from ..strategies.text_block_edit import generate_block_replacement
from ..strategies.qa import generate_answer
from ..strategies.structured_edit import generate_delta, generate_repair
from ..validation.syntax import SyntaxCheckError, check_python_imports, check_syntax
from ..validation.tests import copy_project_for_validation, run_tests
from ..versioning.version_manager import VersionManager
from . import pending_confirmations

MAX_REPAIR_ATTEMPTS = 2  # PHOENIX section 18: "bounded repair budget, one or two targeted attempts"

# Real, structural marker for "this IS (or is inside) this engine's own
# repository" -- derived from where this module actually lives on disk
# (never a hardcoded path string), so it stays correct wherever this
# project is installed or cloned. Real, repeated case this warns about:
# a project directory pointed at (or nested inside) this same repo means
# every single edit's test_target validation runs THIS engine's own,
# unrelated test suite (or whatever else happens to live alongside it)
# instead of a real project's own -- slow, and reports failures/
# regressions that have nothing to do with the actual change (see
# UNRELATED_TEST_COLLECTION_ERROR below, which catches one concrete
# symptom of exactly this after the fact; this warns about the cause).
_ENGINE_REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_dir_is_this_engine(project_dir: Path) -> bool:
    """True when project_dir IS this engine's own repository, or is a
    subdirectory inside it."""
    try:
        project_dir.resolve().relative_to(_ENGINE_REPO_ROOT)
        return True
    except ValueError:
        return False


class _TestFailure(Exception):
    """Internal signal only -- lets a failing pytest run join the same
    classify-and-repair path as the other failure exceptions below,
    instead of needing its own separate branch."""


def _classify_failure(e: Exception) -> str:
    """Maps our concrete exceptions onto PHOENIX's failure-class taxonomy
    (section 18) -- a fixed set of strings, not a learned classifier."""
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
        return "REFERENCE_ERROR"  # target/anchor not found, or ambiguous duplicate
    return "UNKNOWN"


_COLLECTION_ERROR_RE = re.compile(r"ERROR collecting (\S+)")


def _unrelated_collection_error_file(failure_detail: str, file: str) -> Optional[str]:
    """The path pytest failed to even COLLECT (a real ImportError/
    SyntaxError inside that file itself, before a single test ever ran),
    or None if this failure isn't a collection error at all -- or if the
    collected path IS the file just edited (then the edit itself might
    really be the cause, and repair is still worth trying).

    Real waste this closes: a collection error in some OTHER, unrelated
    file (a stale/broken test elsewhere in a too-broadly-scoped
    test_target -- observed live: test_target='.' picking up an
    unrelated sample_project/test_calculator.py with its own pre-existing
    ModuleNotFoundError) can never be caused by, or fixed by correcting,
    a delta applied to a completely different file. Retrying repeats the
    identical collection error every time, for real cost -- three bounded
    repair attempts, each a full LLM call, burned on something no
    correction to `file` could ever touch."""
    m = _COLLECTION_ERROR_RE.search(failure_detail)
    if not m:
        return None
    collected = m.group(1)
    if collected == file or collected.endswith(f"/{file}") or file.endswith(f"/{collected}"):
        return None
    return collected


def _first_defined_name(content: Optional[str]) -> Optional[str]:
    """Python-only convenience wrapper -- see analyzer.locator.defined_symbol_name
    (the shared, multi-language implementation this now delegates to;
    kept here under its original name since it's part of this module's
    tested surface)."""
    return defined_symbol_name(content, "python")


def _change_ratio(original: str, new: str) -> float:
    orig_lines = original.splitlines()
    new_lines = new.splitlines()
    diff = list(difflib.unified_diff(orig_lines, new_lines))
    changed = sum(1 for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    return changed / max(len(orig_lines), 1)


def _compact_created_file_summary(path: str, content: str) -> str:
    """A structural summary (function/class signatures) of a just-created
    file, for composing the also_link_current_file edit -- not its full
    text. Real waste this closes: linking 4 substantial new files (a
    real "make a chatbot with frontend/backend/storage/sessions"
    request) re-sent all four files' *complete* content just to write
    one import/usage edit in the current file, dominating that request's
    input tokens. A signature list ("function chat(message)", "function
    save_session(session_id, data)") gives the linking step everything
    it actually needs to write correct calls, at a fraction of the cost.

    Falls back to the real content for anything with no extractable
    symbol structure -- a plain .env (KEY=VALUE lines, no functions at
    all) or a binary artifact's base64 both need the real thing (env
    keys must be named exactly right; a binary file has no symbols to
    summarize in the first place, only "(binary file)")."""
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
        # The raw LLM response text used to be stored alongside this under
        # patches/*.raw.json -- pure duplicate bytes of this exact dict
        # (json.loads(raw_json) == delta_dict), no reader anywhere ever
        # used it. Dropped: same information, half the storage.
        storage.put_json(f"projects/{project_id}/deltas/{run_id}{suffix}.json", delta_dict)


def _fold_prior_generation(metadata: dict, prior_gen: Optional[dict]) -> None:
    """Adds a prior LLM call's usage onto metadata["generation"]'s totals
    -- specifically, the classification call that produced an escalate
    signal (see generate_delta / structured_edit.py's prompt) before
    _run_whole_file_edit/_run_language_conversion ever ran. That call
    cost real tokens too; dropping it would undercount the run's real
    cost. A no-op when there was no prior call (e.g. CREATE mode never
    has one)."""
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
    """Real callers of any name in `symbol_names` OUTSIDE `current_file`,
    per Joern's CPG -- exactly the blind spot rename_with_subword_
    fallback has (it only ever rewrites occurrences inside the one file
    it's given). None when Joern isn't installed, the repo has no
    Joern-supported language present, or no external caller was found --
    every one of those is "nothing to warn about," not an error; this
    never raises and never blocks the rename itself."""
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
    """1-indexed (start_line, end_line) inclusive span covering every
    top-level import statement in `source`, real AST node positions --
    not a naive "lines starting with import/from" scan, so a stray
    non-import line between two import groups is never swept into the
    replaced region. None when the source doesn't parse (shouldn't
    happen here -- check_syntax already ran first) or has no imports at
    all (nothing for a narrow import-only fix to target)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    spans = [(node.lineno, node.end_lineno) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def _imported_names(import_block: str) -> set:
    """Every local name an import block binds (its alias if aliased,
    otherwise its own name) -- used to check a narrow import-only fix
    doesn't silently orphan a name the rest of the file still uses (see
    the fix's own comment at its call site). The OLD block passed at the
    call site is always a real span _locate_import_span already found
    inside an already-ast.parse'd file, so it always parses on its own
    too; only the model's own NEW block could genuinely fail to parse --
    that degrades to "no names," which only makes the orphan check
    stricter (every old name looks orphaned), never looser."""
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
    """Whole-file regeneration for an EXISTING file's edit is refused
    outright, by explicit policy: this pipeline only ever changes the
    specific symbol(s) a request identifies, never the whole file --
    even when the model itself recognizes (via generate_delta's escalate
    signal) that a request reaches outside every symbol's own span (a
    module docstring, a standalone module-level comment or statement,
    blank-line formatting between functions). Note this is genuinely
    narrower than "touches many symbols": the prompt itself now steers
    the model to prefer several REPLACE ops over escalating here, since
    each REPLACE already restates its own complete body regardless of
    how many other symbols also change -- whole_file should only ever
    reach this function for content no REPLACE/INSERT/DELETE could ever
    target at all.

    Before refusing, though, tries analyzer/module_blocks.py: content
    outside every symbol's span is still very often just ONE small,
    narrowly-locatable region (a module-level config list, an
    `if __name__ == "__main__":` guard) -- real, confirmed case: "remove
    gpt" against a file whose only "gpt" text lived inside a
    module-level `MODEL_SLOTS = [...]` list (never inside any function/
    class body) found zero symbol candidates and correctly escalated
    here, but the actual edit needed was a 7-line region, not the whole
    138-line file. Only an unlocatable/ambiguous region falls through to
    the refusal below. Requires at least one real function/class symbol
    elsewhere in the file: with zero symbols, the "gap" IS the whole
    file, and locating "a" module-level block there is just a
    whole-file rewrite wearing a narrower name -- exactly what this
    refusal exists to prevent.

    CREATE (api/create_pipeline.py) and language_conversion
    (_run_language_conversion, elsewhere in this file) are NOT affected
    by this refusal and never call this function -- both structurally
    have no existing symbol to target in the first place (a brand-new
    file, or a rewrite into a different language), so whole-file
    generation there isn't a fallback being chosen over something
    smaller, it's the only mechanism that could ever apply. This
    function is reached only when an EXISTING file's edit escalates past
    REPLACE/INSERT/DELETE.

    Zero additional generation cost for the refusal path: no full-file
    call is made at all -- only whatever classification call already
    ran (which is what discovered the escalation) is billed, folded in
    via classification_gen for honest accounting."""
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
    region (analyzer/module_blocks.py) instead of a whole-file-format
    section -- only that region goes in as context and comes back as
    output, spliced into the file by line range afterward, the exact
    same technique apply_delta's REPLACE already uses for a function.

    `block` is the analyzer.locator.SymbolInfo (symbol_type=
    "module_statement") locate_module_level_block already resolved --
    the caller is expected to have already checked it's non-None."""
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
    """LLM-free *locator*, real (small) generation call -- for a file
    whose format has no function/class concept at all (markdown, YAML,
    TOML, INI, Dockerfile, plain text, ...), analyzer/text_blocks.py
    still finds the one relevant section (a heading, a [section], a
    top-level key, or a blank-line-delimited paragraph) the same way
    locate_candidates finds a function, so only THAT block -- not the
    whole file -- goes in as context and comes back as output. Spliced
    into the file by line range afterward, the exact same mechanical
    technique apply_delta's REPLACE already uses for a function.

    Real waste this avoids on top of already skipping STRUCTURED_EDIT's
    classification call: "add retrieval types in the notes part" against
    a 30-line README needs only its 3-line "### Notes" section as input
    and output, not a full-file regeneration of all 30 lines.

    `block` is the analyzer.locator.SymbolInfo (symbol_type="block")
    text_blocks.locate_text_block already resolved -- the caller is
    expected to have already checked it's non-None and unambiguous."""
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
    """Mechanical, zero-LLM-cost rename of every whole-word occurrence
    of each (old_name, new_name) pair in `renames`. Reached when the
    model recognizes a request means renaming one or more identifiers
    everywhere they're used, but they aren't cleanly addressable by
    REPLACE (a module-level variable, or a name whose occurrences reach
    outside any single function/class's own body). Real case: "replace
    col by column" against sessions_col AND messages_col together --
    both are bare top-level assignments (REPLACE has no target for that
    at all) and both are also used inside ensure_indexes()'s body, so
    the model had escalated all the way to a full-file LLM regeneration
    for a change with a known-correct, deterministic answer: find every
    whole-word occurrence of each real name, substitute. A list, not a
    single pair, because one request routinely means several related
    renames together (sessions_col AND messages_col, not just one) --
    reviewed and applied as one atomic change, not a round-trip per name.

    Word-boundary matched (\\bold_name\\b) per pair, same identifier-
    tokenizing convention this project already uses elsewhere -- never
    a plain substring replace, which would also corrupt an unrelated
    name that merely contains old_name (renaming "col" as a substring
    would mangle "column" itself, or "collection").

    No repair loop: same reasoning the other FULL_REGENERATION-style
    paths already apply -- if the mechanical rename doesn't produce
    valid, passing code (e.g. a new_name collides with something real),
    that's a genuine conflict the request itself needs to resolve, not
    something a corrective prompt could fix.

    `use_joern=True` (opt-in) adds one real safety net this rename can't
    give itself: rename_with_subword_fallback only ever rewrites
    occurrences inside `file` -- a caller of the renamed symbol in a
    DIFFERENT file is invisible to it and silently breaks. Joern's real
    CPG (retrieval/joern_graph.py) resolves cross-file callers by actual
    identity, not name-matching, closing the one blind spot the native
    per-file rename has; a real cost (~12-45s+, JVM startup) as with
    every other Joern use in this project, so this only ever runs when
    explicitly asked for. Never blocks the rename -- surfaced as
    `metadata["cross_file_impact_warning"]` for a human/caller to act on,
    since Joern itself isn't infallible (dynamic dispatch, etc.) and a
    false negative here would be worse than a false positive."""
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

        # ast.parse only proves the renamed file is grammatically valid
        # Python, not that every reference it makes still resolves --
        # real bug this caught: renaming a class reference
        # (ChatOpenAI -> ChatAnthropic) without also renaming the
        # *import statement's own module path* it appeared in produced
        # `from langchain_openai import ChatAnthropic`, a real
        # ImportError the instant anything ran it. A project with no
        # test suite covering this file (true of the real case this was
        # caught against) means pytest below would never have imported
        # it either, silently reporting success on broken code. On
        # failure here, the mechanical rename genuinely wasn't enough --
        # fall back to a real LLM rewrite rather than report a failure
        # with no path forward; this is the one real generation cost
        # this otherwise-free path can incur, and only when it's earned.
        try:
            check_python_imports(file, cwd=str(tmp_project))
        except SyntaxCheckError as e:
            # Before paying for a full rewrite: the mechanical rename got
            # everything else right, so try fixing ONLY the broken import
            # block first -- same "locate the one region actually wrong,
            # regenerate just that, splice back" technique already proven
            # for TEXT_BLOCK_EDIT, applied to whichever line span the
            # import statements themselves occupy (see import_repair.py).
            # Real waste this avoids: a 33-line file whose only real
            # breakage is one import line previously paid for the *entire*
            # file as both input and output to fix it.
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
                # A narrow import-only fix is only actually consistent if
                # every name the OLD (broken) import block bound is either
                # still bound by the new one, or genuinely unused
                # elsewhere in the file -- otherwise the fix just moved
                # the breakage into the body instead of resolving it. Real
                # bug this catches: reverting "from xai import XAI" back
                # to "from openai import OpenAI" is a valid import-line
                # fix in isolation, but a body call site of "XAI(...)"
                # (inside a function, so check_python_imports's own plain
                # `import module` never executes it, and no test suite
                # necessarily covers it either) would silently stay
                # broken -- committed as success, a real observed case.
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
                        (tmp_project / file).write_text(new_source)  # restore before falling back

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
                    language="python",  # this whole function's own import-fix path is Python-only
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

    Zero LLM cost, the common case: matched structurally against the
    file's own real name (is_whole_file_delete_target), never guessed
    from a keyword list, only after find_delete_candidates/
    find_multi_delete_targets already found no real symbol inside the
    file matching the request (a real symbol name always wins). There's
    nothing for a model to generate here -- the whole change is "this
    file no longer exists."

    Via the model's own escalate:{"kind":"delete_file"} otherwise (a
    phrasing the fast path's narrow verb-first gate didn't match, e.g.
    "please delete this file") -- classification_gen carries that
    already-spent call's usage so it's folded into the total rather than
    silently dropped, same pattern _run_whole_file_edit/_run_create_files
    already use for their own escalate arrivals.

    Either way, still runs the real test suite against a copy with the
    file actually removed, exactly the same real-conflict check
    code-level DELETE already gets (see run_edit's delete_only/
    TEST_FAILURE handling) -- other code importing this file is exactly
    the kind of thing this project always verifies rather than assumes
    away."""
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
    the existing file at all -- the result can't still be named SHA.go
    and parse as Go (check_syntax would refuse it), so there's no
    REPLACE/INSERT/DELETE that expresses it. It's a new sibling file in
    the target language, informed by the existing one -- closer to
    CREATE than EDIT, so this mirrors create_pipeline.run_create's shape
    (optional test_target, syntax check, then the confirmation gate)
    rather than run_edit's (no repair loop: a wrong translation needs a
    different prompt, not a small corrective patch).

    target_language/target_extension both come from the model's own
    escalate response (generate_delta, structured_edit.py's prompt), not
    a hardcoded language/extension catalog -- it already knows the
    conventional extension for whatever language it names, for literally
    any language it knows about, not just ones someone thought to list
    here in advance."""
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
    edit instructions at all -- forcing them through REPLACE/INSERT/DELETE
    (the only real failure mode before this existed) meant the model had
    to fabricate *some* edit to satisfy the schema, since nothing else was
    on offer; real case: "list all algo" got turned into a REPLACE on an
    unrelated, ambiguously-duplicated function named "handle", failing for
    a reason that had nothing to do with what was actually asked.

    No Delta IR, no apply/test/versioning/commit -- there's nothing to
    write, so nothing to confirm either; this always returns a terminal
    result.status == "answered" directly. Scope is this one file only
    (not a repo-wide retrieval): cheaper, and correct for the common case
    of asking about the file already open."""
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
    """"Make a new code file of frontend and backend" while editing
    chatbot.py isn't an edit to chatbot.py at all -- real failure this
    replaces: that exact request got forced through the whole_file
    escalate instead, silently regenerating chatbot.py's own content
    (bumped its version, touched nothing new) since that was the only
    "this isn't a symbol-level edit" escape hatch available. Closer to
    CREATE than EDIT, extended to N files in one request instead of one
    (CREATE mode's own file-at-a-time flow needs a manual mode switch
    per file, which "make ... frontend and backend" -- two files, one
    request -- doesn't fit).

    also_link_current_file covers the "make a .env file that links to
    this chatbot" shape: create the new file(s) FIRST (so their real
    content -- e.g. .env's actual key names -- exists to react to), THEN
    regenerate the currently open file (generate_full_file_edit, the same
    proven whole-file mechanism _run_whole_file_edit already uses) with
    an instruction that includes what was just created. Both the new
    file(s) and the modified current file are one atomic accept/reject --
    a human reviewing "created .env" would also want to see the matching
    chatbot.py change in the same review, not as a surprise separate step.

    No repair loop: same reasoning as the other FULL_REGENERATION paths
    -- a wrong file needs a different prompt, not a small corrective patch.

    A path ending in "/" is a bare folder, not a file -- real failure
    this fixes: "make a folder called frontend" had no way to express
    "just a directory, no content" in this schema, so the model invented
    a file named "frontend" (no extension) containing a *Python script
    that creates a folder when run* -- syntactically valid, technically
    "a file was created," and completely wrong. Folder paths get no LLM
    call at all (there's nothing to generate) and are just mkdir'd,
    matching this project's zero-cost-when-there's-nothing-to-decide
    pattern elsewhere (deletes, confirmed picks)."""
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
        # also_link_current_file means the new file's own content should
        # be informed by the file it's meant to work with (e.g. a .env's
        # keys should match what chatbot.py actually reads) -- the current
        # file's real content rides along as reference material, not as
        # something being edited here.
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
            # gen["code"] is a script that BUILDS the real file (e.g.
            # via openpyxl), not the file itself -- see binary_artifact.py.
            # Real failure this fixes: "make new excel file and add the
            # data of Agentic AI" produced a file literally named
            # "Agentic_AI.xlsx" whose actual bytes were that Python
            # script, never executed.
            on_step("GENERATE", f"running generated script to materialize {path}...")
            try:
                files[path] = generate_binary_artifact_base64(gen["code"], path)
            except BinaryArtifactError as e:
                # Every generation spent so far (including this failed
                # one) cost real tokens -- fold all of it in rather than
                # report a cheaper-than-real failure.
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
        # A pure folder-only request (e.g. "make a folder called
        # frontend") never enters the loop above at all -- nothing to
        # generate, so nothing to fold beyond the classification call
        # (_fold_prior_generation below) that already decided to escalate.
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
            continue  # already a real, successfully-materialized binary -- base64 text isn't source to parse
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
        if not is_binary_artifact_target(path):  # no meaningful diff/preview for real binary bytes
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
        # A pure folder-only request has no file for the confirm-bar tab
        # to attach to -- the frontend can't fetch/preview a directory
        # (or a real binary file) the way it does plain-text content.
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
        if not is_binary_artifact_target(path):  # no meaningful per-symbol metadata for a binary artifact
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
    require_confirmation: bool = False,
    use_hybrid_retrieval: bool = False,
    use_joern: bool = False,
    on_step: Callable[[str, str], None] = lambda tag, msg: None,
    on_preview: Callable[[dict], None] = lambda data: None,
) -> dict:
    """Runs the full edit pipeline and always returns a metadata dict
    (never raises for a validation/generation failure -- check
    metadata["result"]["status"]). Only raises for programmer errors
    (missing file, bad args).

    `on_preview` fires once, right after APPLY, with {"operations": [...],
    "diff": "...", "file": "...", "new_file_content": "..."} -- before
    TEST/COMMIT even run -- so a caller (the web UI) can show the actual
    code change as soon as it exists instead of waiting for the whole run
    to finish.

    A "remove/delete X" request whose target word matches more than one
    real symbol -- including the same name defined more than once, each
    occurrence its own separate candidate now -- returns early with
    result.status == "needs_selection" and a `candidates` list instead of
    guessing -- a deletion is destructive and permanent, so ambiguity here
    is worth asking about rather than picking. The caller re-runs with
    `confirm_symbol` (+ `confirm_symbol_type`, + `confirm_symbol_line`
    when the picked candidate's start_line disambiguates a duplicate
    name) set to the one the user picked, which skips localization and
    the LLM call entirely and constructs the DELETE directly -- the shape
    of that delta is already fully determined once the target is
    confirmed, so there's nothing left for a model to decide.
    confirm_symbol_line is what makes a *duplicate*-name confirmation
    actually resolve (find_symbol's prefer_line) instead of failing with
    the same "defined N times" error a bare name confirmation can't get
    past no matter how many times it's retried.

    `require_confirmation=True` (the web UI's human-in-the-loop mode, the
    CLI never sets this) pauses right after tests pass but before COMMIT:
    returns early with result.status == "awaiting_confirmation" and
    stashes everything needed to finish the commit in
    `pending_confirmations`. Nothing is written to disk until the caller
    resolves it via `pending_confirmations.resolve` (accept = commit,
    reject = discard -- there's nothing to undo since the file was never
    touched).

    `use_hybrid_retrieval=True` fuses the normal name/docstring symbol
    match with BM25 keyword and vector/semantic retrieval over the
    file's own symbols (context_builder.build_context's use_hybrid) --
    the same reasoning `iee find`'s repo-wide hybrid retrieval already
    applies, scoped to one file. Off by default: vector retrieval is a
    real embeddings-API call on every request that reaches normal
    localization, not free like the rest of this pipeline's fast paths,
    and every offline test in this project's own suite relies on this
    defaulting off to stay deterministic and network-free. The CLI and
    web UI both opt in explicitly.

    `use_joern=True` (opt-in) adds a real cross-file caller check (via
    Joern's CPG, retrieval/joern_graph.py) before a mechanical rename
    only -- the one case this pipeline can silently miss a caller in
    another file for (rename_with_subword_fallback only ever touches
    the file it's given). Never blocks the rename; surfaced as
    metadata["cross_file_impact_warning"] for a human to act on. Off by
    default: a real ~12-45s+ JVM cost, not something to pay on every
    rename unasked."""

    project_dir = Path(project_dir).resolve()
    target_file = project_dir / file
    # Lowercased so "Core" and "core" (same directory on a case-insensitive
    # filesystem like macOS's default) can't silently fork into two
    # unrelated version histories just because of how it was typed.
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

    storage = get_storage()
    vm = VersionManager(storage, project_id)
    base_version = vm.get_head() or "v0"

    try:
        original_source = target_file.read_text()
    except UnicodeDecodeError:
        # Real crash this replaces: editing a real .xlsx leaked a raw
        # "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc7..."
        # straight to the user. Binary formats (spreadsheets, databases,
        # images, ...) need their own read/write layer this project
        # doesn't have yet -- fail with a clear, honest reason instead of
        # an exception message that looks like an internal bug.
        raise ValueError(
            f"'{file}' is a binary file -- this project can only edit plain-text files "
            "(code, .csv, .sql, config, markdown, ...) right now. Spreadsheets (.xlsx), "
            "databases (.db/.sqlite), and other binary formats aren't supported yet."
        )
    # "python" for .py (the proven, docstring-aware ast path); otherwise
    # whatever Tree-sitter grammar the extension maps to, so the same
    # localize/apply pipeline that used to hard-fail on a non-Python file
    # (ast.parse() on `public class Foo {` raises SyntaxError) now indexes
    # it properly instead. None (unrecognized extension) degrades to "no
    # symbols found" rather than guessing at Python syntax -- UNLESS the
    # content itself is real Python despite the extension (a ".db" file
    # that's actually the Python script that *builds* a database, a real
    # observed case): looks_like_python() is deliberately strict (needs a
    # real def/class/import, not just anything ast.parse() accepts) so a
    # plain CSV's rows -- also syntactically valid Python by coincidence
    # -- never gets misdetected this way.
    language = "python" if file.endswith(".py") else detect_language(file)
    if language is None and looks_like_python(original_source):
        language = "python"

    if not confirm_symbol:
        # "rename X to Y" needs to see neither X's nor any caller's body
        # to be renamed correctly -- the request itself already says
        # what to rename and what to call it. Real waste this avoids:
        # "replace name of function of oauthAuthorizationCodeFlow to
        # oauthAuthorizationFlow" pulled in that function's *and*
        # main()'s full bodies as localized context just to decide
        # something the request's own wording already settled, then
        # paid a real LLM call to restate them. Skips localization,
        # context building, and the classification call entirely --
        # find_rename_target already confirmed old_name is a real,
        # unambiguous symbol before this ever fires.
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
    if not confirm_symbol and is_delete_intent(request):
        # "delete tan, sec and cos" names three distinct, deliberate
        # targets, not one ambiguous one -- find_delete_candidates' broad
        # substring search can't tell the two apart (it matched 10
        # symbols for that exact request). Checked first, strictly: only
        # fires when every named target resolves to exactly one exact,
        # unambiguous symbol; otherwise falls through to the existing
        # single-target path unchanged.
        multi_delete_targets = find_multi_delete_targets(original_source, request, language)
        if multi_delete_targets is None:
            delete_candidates = find_delete_candidates(original_source, request, language)
            if not delete_candidates and is_whole_file_delete_target(file, request):
                # No real symbol inside the file matches -- the request
                # means the file itself. Zero LLM cost either way, so
                # short-circuit here rather than fall through to a
                # symbol-level delta the model has nothing real to build.
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
                # Already fully determined -- exactly one real symbol matches
                # the delete target, nothing left to disambiguate. Real waste
                # observed: "remove the subtract function" against a file
                # where that name is unique still spent a full LLM call (585
                # tokens, real $) reconstructing a DELETE whose shape
                # find_delete_candidates had already pinned down for free.
                # Falls into the exact same zero-LLM path a human's
                # needs_selection pick takes below -- still gated by
                # require_confirmation (the web UI's default) exactly like
                # every other edit, so nothing is written without a human
                # reviewing the actual diff first; this only removes the
                # wasted *generation* call, not the review step.
                only = delete_candidates[0]
                confirm_symbol = only.name
                confirm_symbol_type = only.symbol_type
                confirm_symbol_line = only.start_line

    if not confirm_symbol and not multi_delete_targets and (language is None or not language_has_symbol_concept(language)):
        # This language's own grammar has no function/class concept at
        # all (markdown, json, yaml, csv, html, css, ini, dotenv, ...),
        # OR the file's format wasn't recognized at all (language is
        # None -- e.g. a literal "Dockerfile" with no extension for
        # detect_language to key off of). Either way index_symbols
        # already returns [] for it, so STRUCTURED_EDIT's classification
        # call (its whole premise is "REPLACE/INSERT/DELETE on a
        # function/class, or escalate") could only ever answer
        # "escalate" here, every single time -- skip straight past it.
        # Still worth localizing, just via
        # analyzer/text_blocks.py's LLM-free section locator (markdown
        # headings, [section] headers, YAML top-level keys, or
        # blank-line paragraphs) instead of function/class matching --
        # when it confidently finds ONE relevant block, only that block
        # is sent and regenerated, not the whole file. Real waste this
        # closes: "add retrieval types in the notes part" against a
        # 30-line README used to pay ~1045 tokens for the classification
        # call's system prompt, then regenerate all 30 lines, for a
        # change that only ever touched its 3-line "### Notes" section.
        if use_hybrid_retrieval:
            on_step("RETRIEVE", "block name-match + BM25 + vector retrieval, then fusing rankings...")
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

    if not confirm_symbol and not multi_delete_targets:
        # Optional, opt-in-by-config pre-classification (retrieval/jev_router.py):
        # a fast, type-safe TypeSafe AI Jev call answers "what kind of request is
        # this" from the request's own wording alone, before context-building or
        # STRUCTURED_EDIT's own classification generation ever runs. Only acted
        # on for the two kinds ("question", "whole_file") whose handlers need
        # nothing beyond the request + this file's current source -- exactly
        # what Jev itself saw -- and only above its own confidence bar; anything
        # else (not configured, low confidence, any failure) returns None and
        # falls straight through to the existing pipeline, completely
        # unchanged, same as if this check were never here.
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

    if confirm_symbol or multi_delete_targets:
        # The target(s) are already fully determined -- no ambiguity left
        # to resolve and no wording left for a model to interpret, so
        # skip localization and the LLM call entirely rather than spend
        # tokens asking GPT to reconstruct a delta whose shape is already
        # known.
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
            # Only set when the picked candidate's line disambiguates a
            # name defined more than once -- find_symbol's prefer_line,
            # threaded through validate_targets below exactly like the
            # normal (non-confirmed) path already does for locate_candidates'
            # own picks.
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
            # Same wording `iee find`'s own hybrid retrieval step uses --
            # real gap this closes: build_context runs BM25+vector fusion
            # silently (a pure function, no on_step of its own), so a
            # human watching the step log had no visible confirmation it
            # ran at all, even though it was (use_hybrid_retrieval
            # defaults True for the web UI) -- looked indistinguishable
            # from hybrid never having fired.
            on_step("RETRIEVE", "symbol + BM25 + vector retrieval, then fusing rankings...")
        ctx = build_context(original_source, request, language=language, use_hybrid=use_hybrid_retrieval)
        if ctx["candidate_symbols"]:
            no_match_label = None
        elif ctx["context_lines"] < ctx["total_lines"]:
            # No confident symbol match, but build_context's compact
            # fallback (imports + a bare name index, no bodies) still
            # kicks in -- context_lines < total_lines proves the whole
            # file was NOT sent, even though no candidate was found. The
            # old message here claimed "using whole file" unconditionally
            # whenever no symbol matched, which was true before that
            # fallback existed but has been wrong (and misleading) ever
            # since -- a real run with context_lines=8/total_lines=51
            # logged "using whole file" while actually sending 8 lines.
            no_match_label = "none, using imports + name index only"
        else:
            # Genuine last resort: build_context had nothing at all to
            # build a compact fallback from (no imports, no other named
            # symbols), so it truly fell through to the raw file.
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

        # The model itself recognized this can't be a REPLACE/INSERT/DELETE
        # on a named symbol at all (see structured_edit.py's prompt) --
        # not a keyword/language list here deciding for it. Its usage from
        # *this* call still cost real tokens, so it's threaded through
        # (classification_gen) rather than silently dropped from the final
        # metadata's totals.
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
            # Reached when the request meant the whole file but didn't
            # match is_whole_file_delete_target's narrow, zero-LLM-cost
            # fast path (e.g. "please delete this file" -- the leading
            # word isn't a bare delete verb, or extra wording didn't
            # match the real filename) -- the model itself recognized it
            # instead. Costs real tokens (this call already happened),
            # but still routes through the exact same safe pipeline
            # (test-conflict check, human review) as the free path.
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

    # The exact occurrence locate_candidates already resolved for each
    # name (matters only when a bare name is duplicated, e.g. the same
    # method repeated across classes) -- threaded through every later
    # find_symbol lookup so that resolution can't get re-decided
    # differently, or refused as ambiguous a second time, from the bare
    # name alone.
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
                # The model looked at the real target and decided there's
                # nothing to do -- e.g. asked to remove something that's
                # already gone. That's a legitimate outcome, not a failure:
                # no file change, no new version, but still a success.
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

            # Attach the exact original-file line range each operation
            # touches -- validate_targets already confirmed every target/
            # anchor resolves uniquely, so these lookups can't hit
            # AmbiguousSymbolError here.
            symbols = index_symbols(original_source, language)
            for op_dict, op in zip(metadata["operations"], delta.operations):
                t = op.target
                if op.operation in ("REPLACE", "DELETE"):
                    # DELETE never auto-resolves a delegate pair -- see
                    # find_symbol's docstring. Matches validate_targets'
                    # own choice above so this lookup (already guaranteed
                    # to resolve, per the comment below) can't diverge.
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
                    # INSERT's declared symbol_name is never actually used to
                    # apply the change (only `anchor` is) -- the model can
                    # and sometimes does write it inconsistent with what
                    # `content` really defines. Correct the *displayed* name
                    # to match reality rather than trusting an unused label.
                    actual_name = defined_symbol_name(op.content, language)
                    if actual_name and actual_name != t.symbol_name:
                        op_dict["target"]["symbol_name"] = actual_name

            # A REPLACE that changes a symbol's own def-name (a rename
            # bundled inside a broader edit, not the dedicated mechanical
            # rename path elsewhere in this file) only ever touches that
            # symbol's own span -- any OTHER reference to the old name
            # elsewhere in the file is untouched by apply_delta and would
            # stay stale under the new name. Real bug this closes: "also
            # rename ask_all to ask_everyone" bundled with an unrelated
            # change renamed only the def line, leaving `self.ask_all(...)`
            # elsewhere in the same file calling a name that no longer
            # exists -- syntactically valid, semantically broken, and
            # committed as a plain success because the test suite never
            # exercised that call path. Mechanically sweep every
            # remaining whole-word occurrence to the new name (the same
            # regex the dedicated mechanical rename path already uses)
            # rather than trust generation to have done it.
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
                # The delta declared operations (e.g. a REPLACE that
                # regenerated a symbol byte-identical to what was already
                # there) but the applied result matches the original file
                # exactly -- nothing to version. Same "real change, or not"
                # gate the empty-operations no_op path above already uses.
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
                # Keep the attempt that actually succeeded in the per-attempt
                # history too, but the canonical, unsuffixed delta -- the one
                # versions/{v}.json's delta_id points at -- is always this one.
                _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata, attempt=attempt)
            _persist_run(storage, project_id, run_id, gen["delta_dict"], metadata)

            target_file.write_text(new_source)
            write_file_metadata(project_dir, file, new_source)
            return metadata

        except (DeltaValidationError, AmbiguousSymbolError, ApplyError, SyntaxCheckError, _TestFailure) as e:
            failure_class = _classify_failure(e)
            failure_detail = str(e)[:2000]

            # This attempt's delta is evidence either way -- store it
            # before deciding whether to repair or give up (doc section 3:
            # "always store the edit the same internal way").
            _persist_run(storage, project_id, run_id, gen.get("delta_dict"), metadata, attempt=attempt)
            metadata["repair_history"].append(
                {"attempt": attempt, "failure_class": failure_class, "failure_detail": failure_detail}
            )

            # The compact context (bare names, no bodies) was proven
            # insufficient for this request -- a repair round-trip would
            # only hand the model that same limited context again and
            # hope it picks escalate:{"kind":"whole_file"} on its own.
            # Observed not to: it gave up instead (empty operations,
            # reported as a safe but unhelpful no-op), leaving the user's
            # request unfulfilled. Since the fix is already known
            # mechanically -- send the real file -- escalate straight to
            # whole-file regeneration instead of gambling on a retry.
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

            # A pytest COLLECTION error (interrupted before a single test
            # even ran) in some file other than the one just edited can
            # never be caused by, or fixed by retrying, this delta --
            # short-circuit rather than burn the full repair budget
            # repeating an identical, unrelated failure.
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

            # A delete that breaks an existing test is a genuine conflict,
            # not a fixable mistake -- there's no "wrong content" for a
            # repair prompt to correct, only code elsewhere (a test, a
            # caller) that still depends on what was asked to be removed.
            # Real case: "remove add operation" correctly deleted `add`,
            # which broke test_calculator.py's own test_add() (an
            # ImportError, not a logic bug in the edit) -- repair "fixed"
            # that failure the only way it structurally could: by
            # regenerating `add` right back, silently reverting the
            # user's actual request while still reporting plain success
            # (change_ratio 0, buried in a `note` field). Same reasoning
            # the confirm_symbol branch below already applies to a
            # confirmed delete, extended here to a delete the model chose
            # on its own -- checked generally (every operation in the
            # failed delta is DELETE), not by name or file.
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

            # A confirmed delete's shape is fully determined -- there's
            # nothing an LLM repair attempt could productively change if
            # it fails (e.g. the confirmed symbol no longer exists because
            # the file changed between the disambiguation prompt and this
            # request). Retrying would just spend tokens asking the model
            # to guess at a delta that was never generated by it in the
            # first place, and ctx has no "context" key on this path since
            # localization was skipped -- fail immediately instead.
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
