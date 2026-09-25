"""Offline tests for validation/tests.py's run_tests -- real pytest
subprocess behavior (a genuine passing/failing suite, and a genuine
timeout), not mocked, since this module's whole job is parsing/handling
a real subprocess's real behavior.
"""

import textwrap

from incremental_editing.validation.tests import copy_project_for_validation, run_tests


def test_copy_project_for_validation_skips_git_and_caches_but_keeps_real_code(tmp_path):
    """Real cost this closes: every syntax/import/test validation copied
    a project's .git history and tool caches byte-for-byte before ever
    running a single test. Must skip those, but never skip anything a
    project's own code/tests could actually need."""
    src = tmp_path / "proj"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00\x01")
    (src / ".aider-venv").mkdir()
    (src / ".aider-venv" / "bin").mkdir()
    (src / "minio_local_data").mkdir()
    (src / "minio_local_data" / "data.json").write_text("{}")
    (src / "iee_metadata").mkdir()
    (src / "iee_metadata" / "app.py.json").write_text("{}")
    # A real project dependency directory sharing a name pattern with
    # nothing in the ignore list -- must be copied, never assumed unneeded.
    (src / "node_modules").mkdir()
    (src / "node_modules" / "left-pad").mkdir()
    (src / "app.py").write_text("def main():\n    return 1\n")
    (src / "tests").mkdir()
    (src / "tests" / "test_app.py").write_text("def test_main():\n    assert True\n")

    dst = tmp_path / "copy"
    copy_project_for_validation(str(src), str(dst))

    assert not (dst / ".git").exists()
    assert not (dst / "__pycache__").exists()
    assert not (dst / ".aider-venv").exists()
    assert not (dst / "minio_local_data").exists()
    assert not (dst / "iee_metadata").exists()
    assert (dst / "node_modules" / "left-pad").is_dir()  # never assumed unneeded
    assert (dst / "app.py").read_text() == "def main():\n    return 1\n"
    assert (dst / "tests" / "test_app.py").exists()


def test_run_tests_reports_a_passing_suite(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = run_tests(".", cwd=str(tmp_path))
    assert result["tests_passed"] is True
    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result.get("timed_out") is not True


def test_run_tests_reports_a_failing_suite(tmp_path):
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    result = run_tests(".", cwd=str(tmp_path))
    assert result["tests_passed"] is False
    assert result["failed"] == 1
    assert result.get("timed_out") is not True


def test_run_tests_reports_no_tests_collected_as_a_pass(tmp_path):
    result = run_tests(".", cwd=str(tmp_path))
    assert result["no_tests_collected"] is True
    assert result["tests_passed"] is True


def test_run_tests_degrades_gracefully_on_a_real_timeout_instead_of_crashing(tmp_path):
    """Real bug this closes: a slow/hung test suite previously crashed
    the entire run with an uncaught subprocess.TimeoutExpired -- no
    controlled "failed" result, no error message, a raw traceback all
    the way up through run_edit. A genuinely slow test (real sleep, not
    mocked) with a short timeout reproduces this exactly, offline and
    fast."""
    (tmp_path / "test_slow.py").write_text(
        textwrap.dedent(
            """
            import time
            def test_slow():
                time.sleep(5)
            """
        )
    )
    result = run_tests(".", cwd=str(tmp_path), timeout=0.5)
    assert result["timed_out"] is True
    assert result["tests_passed"] is False
    assert result["no_tests_collected"] is False
    assert "timed out" in result["output_tail"]
