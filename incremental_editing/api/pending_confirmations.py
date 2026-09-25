"""Shared in-memory store for edit/create runs paused right before COMMIT
by `require_confirmation=True` -- the human-in-the-loop accept/reject gate
the web UI uses (the CLI never sets this flag, so its behavior is
unchanged: always auto-commits, exactly as before this existed).

By the time a run reaches this gate, APPLY (or GENERATE for create) and
TEST have already run and succeeded -- none of the files below have been
written yet (that only ever happens in `resolve()` below, or in the
original auto-commit path when `require_confirmation` is False). So
"reject" is just forgetting the pending state; nothing to undo.

In-memory only, deliberately -- lost on a server restart, same POC-scope
tradeoff `local_storage_dir` already makes elsewhere in this project (see
storage/minio_client.py). Not meant to survive a multi-instance or
long-lived deployment; a single `iee serve` process is this project's
only target so far.
"""

from pathlib import Path
from typing import Callable, Dict, List

from ..analyzer.metadata_builder import delete_file_metadata, write_file_metadata
from ..storage.minio_client import get_storage
from ..strategies.binary_artifact import is_binary_artifact_target, write_file_content
from ..versioning.version_manager import VersionManager

_pending: Dict[str, dict] = {}


def stash(run_id: str, kind: str, **fields) -> None:
    """kind is "edit" (STRUCTURED_EDIT, has a delta_dict), "create",
    "whole_file_edit", "language_conversion", "create_files" (all but
    "edit" are FULL_REGENERATION, no delta_dict), or "delete_file" (no
    LLM call at all -- see run_pipeline._run_whole_file_delete) --
    resolve only ever needs to know whether it's the delta-based one,
    the deletion one, or neither.

    `fields` must include `files: Dict[relative_path, new_content]` for
    every kind except "delete_file", which instead carries
    `deleted_files: List[relative_path]` -- one entry for a normal
    single-file edit/create, N entries for a multi-file create ("make a
    new code file of frontend and backend") or a create-then-link
    request (a new .env plus an edit to the file that now loads it) --
    all committed and confirmed together, as one atomic accept/reject
    over every file this run touches. "create_files" may also carry
    `folders: List[relative_path]` -- bare directories with no content
    (e.g. a plain "make a folder called frontend" request), mkdir'd on
    accept alongside whatever real files this run also touches."""
    _pending[run_id] = {"kind": kind, **fields}


def resolve(run_id: str, accept: bool, on_step: Callable[[str, str], None] = lambda tag, msg: None) -> dict:
    pending = _pending.pop(run_id, None)
    if pending is None:
        raise KeyError(f"no pending confirmation for run_id={run_id!r} (already resolved, or server restarted)")

    metadata = pending["metadata"]
    if not accept:
        metadata["result"]["status"] = "rejected"
        return metadata

    project_dir = Path(pending["project_dir"])
    project_id = pending["project_id"]
    storage = get_storage()
    vm = VersionManager(storage, project_id)

    on_step("COMMIT", "human accepted -- writing new version...")

    if pending["kind"] == "delete_file":
        deleted_files: List[str] = pending["deleted_files"]
        version_id = vm.create_version(
            files={},
            deleted_files=deleted_files,
            change_request=pending["request"],
            strategy="FILE_DELETE",
            delta_id=f"delete-{run_id}",
            validation_status="passed",
        )
        metadata["result"]["status"] = "success"
        metadata["new_version"] = version_id
        storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
        for path in deleted_files:
            target = project_dir / path
            if target.exists():
                target.unlink()
            delete_file_metadata(project_dir, path)
        return metadata

    files: Dict[str, str] = pending["files"]
    is_edit = pending["kind"] == "edit"
    version_id = vm.create_version(
        files=files,
        change_request=pending["request"],
        strategy="STRUCTURED_EDIT" if is_edit else "FULL_REGENERATION",
        delta_id=f"delta-{run_id}" if is_edit else f"regen-{run_id}",
        validation_status="passed",
    )
    metadata["result"]["status"] = "success"
    metadata["new_version"] = version_id

    storage.put_json(f"projects/{project_id}/benchmarks/{run_id}.json", metadata)
    if is_edit:
        storage.put_json(f"projects/{project_id}/deltas/{run_id}.json", pending["delta_dict"])
    else:
        for path, content in files.items():
            storage.put_text(f"projects/{project_id}/patches/{run_id}.{path.replace('/', '_')}", content)

    for path, content in files.items():
        write_file_content(project_dir / path, path, content)
        if not is_binary_artifact_target(path):  # no meaningful per-symbol metadata for a binary artifact
            write_file_metadata(project_dir, path, content)
    # "create_files" only: bare folder paths (e.g. "frontend/") with no
    # content to write, never generated/reviewed as a diff -- see
    # run_pipeline._run_create_files' docstring for why a trailing "/"
    # means this instead of a placeholder file.
    for path in pending.get("folders", []):
        (project_dir / path.rstrip("/")).mkdir(parents=True, exist_ok=True)
    return metadata
