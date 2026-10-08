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
    """Skips .git/caches/venvs and any `.ieeignore`-named dirs, so copying a large
    repo for validation doesn't copy gigabytes of unrelated history/tooling."""
    extra = project_ignore_dirs(project_dir)
    ignore = shutil.ignore_patterns(*_VALIDATION_COPY_IGNORE_NAMES, *extra)
    shutil.copytree(project_dir, tmp_project, ignore=ignore)


def run_tests(test_target: str, cwd: str, timeout: float = 120) -> dict:
    """Catches subprocess.TimeoutExpired so a hung test suite degrades to a
    controlled failure instead of crashing the whole run with a raw traceback."""
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
