"""Runs pytest against a (temp-copy) project and parses the pass/fail summary."""

import re
import shutil
import subprocess
import sys

from ..retrieval.repo_index import project_ignore_dirs

NO_TESTS_COLLECTED = 5

_VALIDATION_COPY_IGNORE_NAMES = (
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".aider-venv", "minio_local_data", "iee_metadata",
)


def copy_project_for_validation(project_dir: str, tmp_project: str) -> None:
    """The one isolated-copy operation every syntax/import/test
    validation step already needed before this existed (each via its own
    bare shutil.copytree) -- now skipping directories no real project's
    test suite ever actually needs. Real cost this closes: pointing a
    project directory at a large repo (this engine's own repo included)
    made every single validation step copy gigabytes of .git history,
    tool caches, and a whole separate virtualenv byte-for-byte before
    ever running a single test -- the dominant cost behind "syntax and
    test validation takes too long" for anything but a tiny project.

    Also skips any directory a project's own `.ieeignore` names (same
    file repo-wide retrieval's own iter_source_files already reads) --
    an unrelated demo/scratch folder living alongside the real project
    shouldn't be walked for tests any more than it should be indexed for
    retrieval."""
    extra = project_ignore_dirs(project_dir)
    ignore = shutil.ignore_patterns(*_VALIDATION_COPY_IGNORE_NAMES, *extra)
    shutil.copytree(project_dir, tmp_project, ignore=ignore)


def run_tests(test_target: str, cwd: str, timeout: float = 120) -> dict:
    """Real bug this closes: a slow or genuinely hung test suite (a huge
    project directory that collects far more than intended, a test with
    a real infinite loop or a hung network call) previously crashed the
    *entire* run with an uncaught subprocess.TimeoutExpired -- no
    "failed" result, no error message, a raw Python traceback instead.
    Every other external call in this project degrades to a controlled
    failure instead of crashing (a broken Jev/vector/Aider call, a
    hung embeddings request) -- this was the one place that didn't."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_target, "-q"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "return_code": None,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "tests_passed": False,
            "no_tests_collected": False,
            "timed_out": True,
            "output_tail": (
                f"pytest '{test_target}' timed out after {timeout}s -- either a genuinely hung test, "
                "or test_target is scoped far wider than this change (e.g. an entire unrelated "
                "repository's own test suite instead of just this project's)"
            ),
        }
    output = proc.stdout + proc.stderr

    def _count(pattern):
        m = re.search(pattern, output)
        return int(m.group(1)) if m else 0

    no_tests_collected = proc.returncode == NO_TESTS_COLLECTED
    return {
        "return_code": proc.returncode,
        "passed": _count(r"(\d+) passed"),
        "failed": _count(r"(\d+) failed"),
        "errors": _count(r"(\d+) error"),
        "tests_passed": proc.returncode == 0 or no_tests_collected,
        "no_tests_collected": no_tests_collected,
        "output_tail": "\n".join(output.strip().splitlines()[-20:]),
    }
