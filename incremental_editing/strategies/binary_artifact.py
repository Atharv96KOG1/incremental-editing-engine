"""Real binary file creation (.xlsx, .db, .docx, .pptx, .pdf, ...).

generate_full_file's normal path pastes the model's raw text output
straight onto disk -- correct for any plain-text format (.py, .csv,
.json, .sql, ...), where the model's output literally *is* the file.
It cannot work for a real binary/structured format: an LLM can't emit
raw OOXML zip bytes or a real SQLite page format as chat completion
text. Observed failure this fixes: "make new excel file and add the
data of Agentic AI" produced a file literally named "Agentic_AI.xlsx"
whose actual bytes were a Python script (using openpyxl) that would
create the real file *if executed* -- never executed, so opening it in
Excel fails. Same failure for "make a new sqlite database file".

Fix: for these extensions, ask the model for that same kind of script
(it already tends to write one unprompted) but then actually RUN it in
a disposable temp directory -- same risk class as the pytest subprocess
this project already runs on every edit -- and capture the REAL bytes
it produces instead of the script text. The script itself is never
kept as the file's content.
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
    """Appended to the normal create-file request when the target is a
    binary format -- tells the model what to write instead of a literal
    file body: a script that produces the real thing when run, not the
    real thing itself (that part -- actually running it -- is this
    module's job, not the model's)."""
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
    """Runs `script` (the model's generated Python) in a fresh temp
    directory using this same interpreter (sys.executable -- not a bare
    "python3", which could silently resolve to a different environment
    missing the libraries this project's own venv actually has), then
    reads back the real bytes of whatever it created at
    `target_relpath`. Raises BinaryArtifactError with the script's own
    stderr on a non-zero exit or a missing/empty output file -- there's
    no partial-success case worth papering over here."""
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
    """materialize_binary_artifact, then base64-encodes the real bytes
    into a plain string -- so every existing `files: Dict[str, str]`
    pipeline (versioning checkpoints, patches storage, the confirm/
    reject stash) keeps working completely unchanged. Only the final
    on-disk write (write_file_content, below) needs to know the
    difference between this and literal text."""
    return base64.b64encode(materialize_binary_artifact(script, target_relpath, timeout)).decode("ascii")


def write_file_content(target: Path, path: str, content: str) -> None:
    """Writes `content` to `target` -- base64-decoding it first when
    `path` is a binary artifact target (generate_binary_artifact_base64
    is what produces that encoded form). The one place the byte/text
    distinction actually has to be handled at every real disk-write
    site (the project directory itself, and the disposable temp copy
    the test runner uses) instead of threading a second, bytes-typed
    dict through every function in between."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_binary_artifact_target(path):
        target.write_bytes(base64.b64decode(content))
    else:
        target.write_text(content)
