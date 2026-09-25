"""CREATE path: bootstrap a brand-new file from a prompt.

Not part of the doc's core incremental-editing hypothesis (that's about
*existing* artifacts) -- but you need an artifact before you can
incrementally edit one. Uses FULL_REGENERATION (doc section 10, Approach A)
since there's no existing file to localize context against or diff.

Same callback-driven, non-raising shape as `run_edit` in run_pipeline.py so
cli.py can drive both uniformly.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional

from ..analyzer.metadata_builder import write_file_metadata
from ..benchmark.pricing import estimate_cost
from ..storage.minio_client import get_storage
from ..strategies.binary_artifact import (
    BinaryArtifactError,
    binary_artifact_instructions,
    generate_binary_artifact_base64,
    is_binary_artifact_target,
    write_file_content,
)
from ..strategies.full_regeneration import generate_full_file
from ..validation.syntax import SyntaxCheckError, check_syntax
from ..validation.tests import copy_project_for_validation, run_tests
from ..versioning.version_manager import VersionManager
from . import pending_confirmations
from .run_pipeline import _project_dir_is_this_engine


def _persist_run(storage, project_id, run_id, gen, metadata):
    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    storage.put_text(f"projects/{project_id}/patches/{run_id}.full.py", gen["code"])


def run_create(
    project_dir: Path,
    file: str,
    request: str,
    test_target: Optional[str] = None,
    project_id: Optional[str] = None,
    require_confirmation: bool = False,
    on_step: Callable[[str, str], None] = lambda tag, msg: None,
    on_preview: Callable[[dict], None] = lambda data: None,
) -> dict:
    """Always returns a metadata dict (check metadata["result"]["status"]).
    Only raises for programmer errors (target file already exists, bad args).

    `on_preview` fires once, right after GENERATE, with
    {"new_file_content": "...", "file": "..."} -- before SYNTAX/TEST even run.

    `require_confirmation=True` (web UI only, CLI never sets it) pauses
    right after syntax/tests pass but before the file is written: returns
    early with result.status == "awaiting_confirmation", nothing written
    to disk until `pending_confirmations.resolve` is called -- see that
    module's docstring and run_pipeline.run_edit's matching parameter."""

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

    if target_file.exists():
        raise FileExistsError(f"'{target_file}' already exists -- use edit, not create, to modify it")

    os.makedirs(project_dir, exist_ok=True)

    storage = get_storage()
    vm = VersionManager(storage, project_id)
    base_version = vm.get_head() or "v0"

    is_binary = is_binary_artifact_target(file)
    on_step("GENERATE", "calling LLM for a full file...")
    gen = generate_full_file(
        file_path=file,
        user_request=request + (binary_artifact_instructions(file) if is_binary else ""),
    )

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    metadata = {
        "run_id": run_id,
        "project_id": project_id,
        "base_version": base_version,
        "user_request": request,
        "strategy": "FULL_REGENERATION",
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

    if is_binary:
        # gen["code"] is a script that BUILDS the real file, not the
        # file itself -- see binary_artifact.py. Actually run it here,
        # once, rather than write that script out under the real
        # extension (the exact bug this closes: an "Agentic_AI.xlsx"
        # that was actually Python source).
        on_step("GENERATE", f"running generated script to materialize {file}...")
        try:
            content = generate_binary_artifact_base64(gen["code"], file)
        except BinaryArtifactError as e:
            metadata["result"]["status"] = "failed"
            metadata["error"] = str(e)
            metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
            _persist_run(storage, project_id, run_id, gen, metadata)
            return metadata
    else:
        content = gen["code"]
        on_preview({"new_file_content": content, "file": file})

        on_step("SYNTAX", "checking generated code parses...")
        try:
            check_syntax(content, filename=file)
        except SyntaxCheckError as e:
            metadata["result"]["status"] = "failed"
            metadata["error"] = str(e)
            metadata["validation"] = {"syntax_passed": False, "tests_passed": None}
            _persist_run(storage, project_id, run_id, gen, metadata)
            return metadata

    test_result = None
    if test_target and (project_dir / test_target).exists():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_project = Path(tmp) / "project"
            copy_project_for_validation(project_dir, tmp_project)
            write_file_content(tmp_project / file, file, content)

            on_step("TEST", f"running pytest ({test_target})...")
            test_result = run_tests(test_target, cwd=str(tmp_project))

    metadata["validation"] = {
        "syntax_passed": True,
        "tests_passed": test_result["tests_passed"] if test_result else None,
        "test_summary": {k: test_result[k] for k in ("passed", "failed", "errors")} if test_result else None,
    }

    if test_result is not None and not test_result["tests_passed"]:
        metadata["result"]["status"] = "failed"
        metadata["error"] = test_result["output_tail"]
        _persist_run(storage, project_id, run_id, gen, metadata)
        return metadata

    if require_confirmation:
        metadata["result"]["status"] = "awaiting_confirmation"
        # Real bug this fixes: nulling this out for a binary artifact
        # (kept, presumably, since there's no live text preview/diff to
        # show for one -- see the on_preview skip above) also broke the
        # web UI's OWN tab association: markPendingConfirmation(metadata
        # .files || metadata.file, ...) got null, tagged no tab with
        # this run's id, so a binary artifact's already-open tab (typed
        # into the File field before submitting) never learned this run
        # existed at all -- silently keeping whatever stale placeholder
        # it showed before the file was created, forever. The real path
        # is needed here regardless of is_binary; nothing about showing
        # a preview requires hiding it.
        metadata["file"] = file
        pending_confirmations.stash(
            run_id,
            "create",
            project_dir=str(project_dir),
            files={file: content},
            request=request,
            project_id=project_id,
            metadata=metadata,
        )
        return metadata

    on_step("COMMIT", "writing file and committing version...")
    version_id = vm.create_version(
        files={file: content},
        change_request=request,
        strategy="FULL_REGENERATION",
        delta_id=f"regen-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id

    _persist_run(storage, project_id, run_id, gen, metadata)

    write_file_content(target_file, file, content)
    if not is_binary:  # no meaningful per-symbol metadata for a binary artifact
        write_file_metadata(project_dir, file, content)
    return metadata
