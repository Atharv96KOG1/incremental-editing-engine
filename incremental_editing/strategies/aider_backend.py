"""Alternate FULL_REGENERATION executor: Aider (aider-chat), run as a
subprocess against its own isolated virtualenv (.aider-venv/) -- never
imported in-process.

Real reason for the isolation, not a defensive guess: aider-chat pins
tree-sitter-language-pack==0.13.0 exactly. Installed once into this
project's own shared environment, it silently downgraded
tree-sitter-language-pack (this project needs >=1.16.1) and broke
detect_language_from_path for every multi-language test in the suite --
confirmed live, the same session this adapter was built in. Subprocess
isolation is the same pattern this project already uses for external
tools with their own dependency footprint (validation/tests.py's pytest
run, validation/syntax.py's check_python_imports) -- never trust an
external tool's own pinned dependencies to coexist with this project's.

Scope: ONLY ever a code *generator* for the FULL_REGENERATION strategy
-- never a locator. This project's own hybrid retriever (BM25 + vector +
name-match, fused) already does localization more precisely and far
more cheaply than any general-purpose coding agent (verified this same
session: zero-LLM-cost fast paths for rename/delete, and real C/C++/
Rust/Ruby symbol-extraction fixes) -- handing that job to Aider would
throw away everything already built and hardened here.

Currently unwired: whole-file regeneration for an existing file's edit
(the one call site this module's own generator was ever dispatched
from) is refused by policy now, not performed (see run_pipeline.py's
_run_whole_file_edit) -- by explicit request, this module is no longer
called from anywhere in the edit pipeline. Left in place, tested, and
importable on its own rather than deleted, in case whole-file generation
for edits is ever reinstated.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..benchmark.pricing import estimate_cost
from ..config import get_settings

AIDER_BIN = Path(__file__).resolve().parent.parent.parent / ".aider-venv" / "bin" / "aider"

# A real edit call, not a fast path -- generous but bounded so a stuck
# subprocess (a half-broken gateway, a runaway retry loop inside Aider
# itself) fails visibly instead of hanging the whole run indefinitely.
_TIMEOUT_SECONDS = 120.0

# Aider's own tokens_report format (aider/coders/base_coder.py,
# format_tokens): a bare int below 1000, "X.Xk" below 10k, "Xk" (no
# decimal) above that -- parsed here rather than guessed, since this
# was read directly from that function's source in .aider-venv.
_TOKENS_RE = re.compile(r"Tokens:\s*([\d.,]+)(k?)\s*sent.*?,\s*([\d.,]+)(k?)\s*received", re.DOTALL)
_COST_RE = re.compile(r"Cost:\s*\$([\d.]+)\s*message")


class AiderError(Exception):
    """Aider isn't installed (.aider-venv missing), or its subprocess
    failed/timed out -- every caller must catch this and fall back to
    the existing direct-LLM whole-file path, never let it crash a run."""


def is_available() -> bool:
    return AIDER_BIN.exists()


def _parse_count(number: str, suffix: str) -> int:
    value = float(number.replace(",", ""))
    return round(value * 1000) if suffix == "k" else round(value)


def _parse_usage(stdout: str, model: str) -> dict:
    """Real per-call token/cost accounting from Aider's own report, not
    a guess -- falls back to this project's own estimate_cost (same
    table every other strategy already uses) only when Aider didn't
    recognize the model well enough to price it itself (no "Cost:" line
    at all in that case -- see base_coder.py's own early-return when
    main_model.info lacks pricing)."""
    matches = list(_TOKENS_RE.finditer(stdout))
    if not matches:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    m = matches[-1]
    input_tokens = _parse_count(m.group(1), m.group(2))
    output_tokens = _parse_count(m.group(3), m.group(4))

    cost_matches = list(_COST_RE.finditer(stdout))
    cost = float(cost_matches[-1].group(1)) if cost_matches else estimate_cost(model, input_tokens, output_tokens, 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": cost,
    }


def generate_full_file_edit_via_aider(project_dir: Path, file: str, user_request: str, model: Optional[str] = None) -> dict:
    """Runs Aider non-interactively (--message, single reply, then exit)
    against the real file on disk, cwd=project_dir. Aider edits the file
    directly on disk; the new content is read back afterward and handed
    back as "code" -- same convention every other strategy's gen dict
    already uses, so _run_whole_file_edit's own syntax-check/test/
    commit/versioning path needs no changes to accept either generator.

    --no-git: this project has its own VersionManager, not real git --
    Aider must never try to init a repo, require one, or auto-commit.
    --yes-always: no interactive prompts are possible over a subprocess
    anyway. --no-pretty/--no-stream: plain, parseable stdout, no ANSI
    color codes to strip before regex-matching the usage report."""
    if not is_available():
        raise AiderError(
            f"Aider not installed at {AIDER_BIN} -- run: "
            "python3 -m venv .aider-venv && .aider-venv/bin/pip install aider-chat"
        )

    settings = get_settings()
    resolved_model = model or settings.llm_model

    env = os.environ.copy()
    if settings.openai_api_key:
        env["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.openai_base_url:
        env["OPENAI_API_BASE"] = settings.openai_base_url

    cmd = [
        str(AIDER_BIN),
        "--model", resolved_model,
        # A model Aider's own registry doesn't recognize (true for this
        # project's Bifrost-gateway model names) defaults to "whole"
        # edit format -- Aider regenerates and re-sends the *entire*
        # file as output on every call, exactly the whole-file-output
        # cost this project exists to avoid, and gives no verifiable
        # anchor for what actually changed. "diff" (SEARCH/REPLACE
        # blocks) fixes both: measured live on an 89-line/30-function
        # file, the same one-function edit request dropped from 548 to
        # 53 output tokens (total cost $0.01 -> $0.008, output being the
        # far more expensive side of this project's own pricing table)
        # and the SEARCH block itself is the exact, literal matched
        # text -- a real anchor, not "whichever line the model felt
        # like touching."
        "--edit-format", "diff",
        "--no-git",
        "--yes-always",
        "--no-pretty",
        "--no-stream",
        "--no-check-update",
        "--no-analytics",
        "--no-show-model-warnings",
        "--message", user_request,
        file,
    ]

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, env=env)
    except subprocess.TimeoutExpired as e:
        raise AiderError(f"Aider timed out after {_TIMEOUT_SECONDS}s") from e
    latency_ms = round((time.monotonic() - start) * 1000)

    if result.returncode != 0:
        raise AiderError(f"Aider exited {result.returncode}: {(result.stderr or result.stdout)[-2000:]}")

    new_source = (project_dir / file).read_text()
    usage = _parse_usage(result.stdout, resolved_model)

    return {
        "code": new_source,
        "model": f"aider/{resolved_model}",
        "cached_tokens": 0,
        "latency_ms": latency_ms,
        **usage,
    }
