"""Offline tests for real binary file creation (.xlsx, .db, .docx, ...).

Uses sqlite3 (standard library, always available) for the actual
subprocess-execution tests rather than openpyxl -- deterministic and
dependency-free, while still exercising the exact same code path a real
.xlsx run does. No LLM call anywhere here.
"""

import base64

import pytest

from incremental_editing.strategies.binary_artifact import (
    BinaryArtifactError,
    binary_artifact_instructions,
    generate_binary_artifact_base64,
    is_binary_artifact_target,
    materialize_binary_artifact,
    write_file_content,
)

_SQLITE_SCRIPT = (
    "import sqlite3\n"
    "conn = sqlite3.connect('store.db')\n"
    "conn.execute('CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)')\n"
    "conn.execute(\"INSERT INTO products (name) VALUES ('widget')\")\n"
    "conn.commit()\n"
    "conn.close()\n"
)


def test_is_binary_artifact_target_recognizes_known_extensions():
    for path in ("report.xlsx", "book.xls", "letter.docx", "deck.pptx", "doc.pdf", "store.db", "x.sqlite", "y.sqlite3"):
        assert is_binary_artifact_target(path) is True


def test_is_binary_artifact_target_rejects_plain_text_extensions():
    for path in ("main.py", "data.csv", "notes.md", "config.json", "schema.sql"):
        assert is_binary_artifact_target(path) is False


def test_materialize_binary_artifact_runs_script_and_returns_real_bytes():
    """Real regression this exercises: "make a new sqlite database file"
    used to produce a file whose bytes were the Python script itself,
    never executed -- opening it as a database failed. This must
    actually run the script and hand back the real SQLite file bytes."""
    data = materialize_binary_artifact(_SQLITE_SCRIPT, "store.db")
    assert data.startswith(b"SQLite format 3\x00")  # the real SQLite file magic header


def test_materialize_binary_artifact_raises_on_script_error():
    with pytest.raises(BinaryArtifactError, match="failed"):
        materialize_binary_artifact("raise RuntimeError('boom')\n", "store.db")


def test_materialize_binary_artifact_raises_when_output_file_missing():
    with pytest.raises(BinaryArtifactError, match="no file"):
        materialize_binary_artifact("x = 1\n", "store.db")  # runs fine, creates nothing


def test_generate_binary_artifact_base64_round_trips():
    encoded = generate_binary_artifact_base64(_SQLITE_SCRIPT, "store.db")
    assert base64.b64decode(encoded).startswith(b"SQLite format 3\x00")


def test_write_file_content_decodes_base64_for_binary_target_and_writes_raw_bytes(tmp_path):
    real_bytes = materialize_binary_artifact(_SQLITE_SCRIPT, "store.db")
    encoded = base64.b64encode(real_bytes).decode("ascii")
    target = tmp_path / "store.db"
    write_file_content(target, "store.db", encoded)
    assert target.read_bytes() == real_bytes


def test_write_file_content_writes_plain_text_verbatim_for_a_non_binary_target(tmp_path):
    target = tmp_path / "main.py"
    write_file_content(target, "main.py", "VALUE = 1\n")
    assert target.read_text() == "VALUE = 1\n"


def test_binary_artifact_instructions_names_the_right_library():
    assert "openpyxl" in binary_artifact_instructions("report.xlsx")
    assert "sqlite3" in binary_artifact_instructions("store.db")
