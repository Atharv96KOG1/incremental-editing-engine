"""Offline tests for strategies/aider_backend.py -- Aider itself is a
real subprocess call into a separate virtualenv, so every test here
mocks subprocess.run (never actually invokes .aider-venv/bin/aider) and
confirms the usage/cost parsing against real strings copied verbatim
from a live run of Aider against this project's own Bifrost gateway.
"""

from pathlib import Path

import pytest

from incremental_editing.strategies import aider_backend
from incremental_editing.strategies.aider_backend import AiderError, _parse_usage, generate_full_file_edit_via_aider


def test_parse_usage_reads_real_tokens_and_cost_line():
    """Verbatim stdout tail from a real Aider run against this project's
    own gateway (openai/gpt-5.4) -- not a fabricated format."""
    stdout = (
        "A code change is needed to add the new `divide(a, b)` function.\n\n"
        "calc.py\n\n```diff\n+def divide(a, b):\n+    return a / b\n```\n\n"
        "Tokens: 657 sent, 61 received. Cost: $0.0026 message, $0.0026 session.\n"
        "Applied edit to calc.py\n"
    )
    usage = _parse_usage(stdout, "openai/gpt-5.4")
    assert usage["input_tokens"] == 657
    assert usage["output_tokens"] == 61
    assert usage["total_tokens"] == 718
    assert usage["estimated_cost_usd"] == 0.0026


def test_parse_usage_handles_k_suffixed_large_counts():
    """format_tokens (aider/utils.py) switches to "X.Xk"/"Xk" above
    1000/10000 -- must parse both, not just bare small integers."""
    stdout = "Tokens: 1.2k sent, 3k received.\n"
    usage = _parse_usage(stdout, "openai/gpt-5.4")
    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 3000


def test_parse_usage_falls_back_to_project_pricing_when_aider_reports_no_cost():
    """An unrecognized model to Aider (base_coder.py's own early return)
    prints tokens but no "Cost:" line at all -- must fall back to this
    project's own estimate_cost table, not silently report $0."""
    stdout = "Tokens: 1,000 sent, 500 received.\n"
    usage = _parse_usage(stdout, "openai/gpt-5.4")  # a model this project's own pricing table does know
    assert usage["estimated_cost_usd"] > 0


def test_parse_usage_none_when_no_tokens_line_found():
    usage = _parse_usage("some unrelated aider output with no usage report\n", "openai/gpt-5.4")
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}


def test_is_available_false_when_venv_missing(monkeypatch):
    monkeypatch.setattr(aider_backend, "AIDER_BIN", Path("/nonexistent/.aider-venv/bin/aider"))
    assert aider_backend.is_available() is False


def test_generate_full_file_edit_via_aider_raises_when_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(aider_backend, "AIDER_BIN", Path("/nonexistent/.aider-venv/bin/aider"))
    with pytest.raises(AiderError):
        generate_full_file_edit_via_aider(project_dir=tmp_path, file="x.py", user_request="do something")


def test_generate_full_file_edit_via_aider_reads_back_the_edited_file(monkeypatch, tmp_path):
    """The subprocess itself is mocked (never a real Aider/LLM call) --
    confirms this adapter reads the file back off disk afterward rather
    than trying to parse code out of Aider's own chat-formatted stdout,
    and that the parsed usage/cost end up in the returned dict."""
    monkeypatch.setattr(aider_backend, "is_available", lambda: True)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")

    class _FakeCompletedProcess:
        returncode = 0
        stdout = "Tokens: 657 sent, 61 received. Cost: $0.0026 message, $0.0026 session.\nApplied edit to calc.py\n"
        stderr = ""

    def _fake_run(cmd, cwd, capture_output, text, timeout, env):
        # Simulate Aider's own file edit -- the real subprocess would
        # have rewritten calc.py on disk directly.
        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n")
        return _FakeCompletedProcess()

    monkeypatch.setattr(aider_backend.subprocess, "run", _fake_run)

    gen = generate_full_file_edit_via_aider(project_dir=tmp_path, file="calc.py", user_request="add subtract")
    assert "subtract" in gen["code"]
    assert gen["input_tokens"] == 657
    assert gen["output_tokens"] == 61
    assert gen["estimated_cost_usd"] == 0.0026
    assert gen["model"] == "aider/openai/gpt-5.4" or gen["model"].startswith("aider/")


def test_generate_full_file_edit_via_aider_raises_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(aider_backend, "is_available", lambda: True)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")

    class _FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "some real Aider/litellm failure"

    monkeypatch.setattr(
        aider_backend.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess()
    )

    with pytest.raises(AiderError):
        generate_full_file_edit_via_aider(project_dir=tmp_path, file="calc.py", user_request="add subtract")


def test_generate_full_file_edit_via_aider_raises_on_timeout(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(aider_backend, "is_available", lambda: True)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")

    def _raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="aider", timeout=120)

    monkeypatch.setattr(aider_backend.subprocess, "run", _raise_timeout)

    with pytest.raises(AiderError):
        generate_full_file_edit_via_aider(project_dir=tmp_path, file="calc.py", user_request="add subtract")
