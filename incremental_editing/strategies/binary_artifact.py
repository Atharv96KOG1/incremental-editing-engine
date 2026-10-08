"""Real binary file creation (.xlsx, .db, .docx, .pptx, .pdf, ...).

The normal create-file path writes the model's raw text straight to disk,
which only works for plain-text formats -- a model can't emit real OOXML/
SQLite bytes as chat text (observed failure: an "Agentic_AI.xlsx" whose
actual bytes were an unexecuted openpyxl script). Fix: ask the model for
that generator script, actually run it in a disposable temp directory, and
keep the real bytes it produces instead of the script text.
"""

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

BINARY_ARTIFACT_LIBRARIES = {
    ".xlsx": "openpyxl", ".xls": "openpyxl",
    ".docx": "python-docx (import docx)",
    ".pptx": "python-pptx (import pptx)",
    ".pdf": "reportlab",
    ".db": "sqlite3 (standard library)",
    ".sqlite": "sqlite3 (standard library)",
    ".sqlite3": "sqlite3 (standard library)",
}


def is_binary_artifact_target(path: str) -> bool:
    return Path(path).suffix.lower() in BINARY_ARTIFACT_LIBRARIES


def binary_artifact_instructions(path: str) -> str:
    """Appended to the create-file request for a binary target: tells the
    model to write a script that produces the file when run, not the file
    itself (running it is this module's job)."""
    ext = Path(path).suffix.lower()
    library = BINARY_ARTIFACT_LIBRARIES[ext]
    return (
        f"\n\nThe target file '{path}' is a real binary format your text output can't "
        f"literally be. Instead, output a complete, runnable Python script that -- when "
        f"executed with its working directory already set to the right folder -- creates "
        f"the real file at the exact relative path '{path}' (use {library}). The script "
        "must do nothing else: no prints except on error, no other files, no command-line "
        "arguments, no input() calls."
    )


class BinaryArtifactError(Exception):
    pass


def materialize_binary_artifact(script: str, target_relpath: str, timeout: float = 30.0) -> bytes:
    """Runs `script` in a fresh temp directory using this same
    interpreter (sys.executable, not a bare "python3" that could resolve
    to a different environment), then reads back the real bytes at
    `target_relpath`. Raises BinaryArtifactError on a non-zero exit or a
    missing/empty output file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        script_path = tmp_dir / "_generate.py"
        script_path.write_text(script)
        output_path = tmp_dir / target_relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(tmp_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise BinaryArtifactError(f"generating '{target_relpath}' timed out after {timeout}s") from e

        if result.returncode != 0:
            raise BinaryArtifactError(
                f"generating '{target_relpath}' failed (exit {result.returncode}): {result.stderr.strip()[:2000]}"
            )
        if not output_path.exists():
            raise BinaryArtifactError(
                f"generating '{target_relpath}' produced no file at that path -- script output: "
                f"{result.stdout.strip()[:500]}"
            )
        data = output_path.read_bytes()
        if not data:
            raise BinaryArtifactError(f"generating '{target_relpath}' produced an empty file")
        return data


def generate_binary_artifact_base64(script: str, target_relpath: str, timeout: float = 30.0) -> str:
    """Base64-encodes materialize_binary_artifact's bytes so the existing
    `files: Dict[str, str]` pipeline (versioning, patches, confirm/reject
    stash) needs no changes; only write_file_content needs to know the
    difference from literal text."""
    return base64.b64encode(materialize_binary_artifact(script, target_relpath, timeout)).decode("ascii")


def write_file_content(target: Path, path: str, content: str) -> None:
    """Writes `content` to `target`, base64-decoding first when `path` is
    a binary artifact target. Keeps the byte/text distinction isolated to
    this one write site instead of threading a bytes-typed dict through
    every function in between."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_binary_artifact_target(path):
        target.write_bytes(base64.b64decode(content))
    else:
        target.write_text(content)
