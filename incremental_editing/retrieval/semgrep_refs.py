"""Semgrep as a structural filter, not a semantic retrieval engine. Two
uses:

1. `find_call_sites` -- every real call-site of a symbol, structurally
   verified (an actual call expression, not just a name match in a
   comment/string).
2. `structural_match_score` -- checks whether a candidate's code actually
   does what the request describes (raises, validates types, loops, ...),
   not just word overlap.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List

_INTENT_PATTERNS = {
    ("validate", "validation", "type", "types"): "isinstance($X, ...)",
    ("raise", "raises", "error", "exception", "exceptions"): "raise $EXC(...)",
    ("log", "logging"): "logging.$METHOD(...)",
    ("loop", "iterate", "iteration", "iterating"): "for $X in $Y: ...",
    ("async", "asynchronous"): "async def $F(...): ...",
    ("math", "trig", "trigonometric", "sin", "cos", "tan", "sqrt", "log", "exp"): "math.$FUNC(...)",
    ("decorator", "decorated", "wraps"): "@$DEC\ndef $F(...): ...",
    ("regex", "pattern", "match"): "re.$METHOD(...)",
}


def find_call_sites(project_dir: str, symbol_name: str, language: str = "python", timeout: int = 30) -> List[Dict]:
    """Returns [{"file": ..., "line": ...}, ...] for every structural call
    site of symbol_name. Returns [] if semgrep fails, isn't installed, or
    times out -- best-effort, never blocks the rest of the pipeline."""
    pattern = f"{symbol_name}(...)"
    try:
        proc = subprocess.run(
            ["semgrep", "--lang", language, "--pattern", pattern, "--json", "--quiet", project_dir],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        data = json.loads(proc.stdout or "{}")
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return []

    return [
        {"file": match.get("path"), "line": match.get("start", {}).get("line")}
        for match in data.get("results", [])
    ]


def _patterns_for_request(request: str) -> List[str]:
    words = set(re.findall(r"[a-z0-9]+", request.lower()))
    return [pattern for keywords, pattern in _INTENT_PATTERNS.items() if words & set(keywords)]


def structural_match_score(source_snippet: str, request: str, language: str = "python", timeout: int = 10) -> int:
    """Runs each request-relevant semgrep pattern against just this
    candidate's own source and counts matches. 0 if the request names no
    known intent, or semgrep can't confirm it -- only adds a signal, never
    removes one. Patterns are Python-specific, so this only runs for
    language == "python"."""
    if language != "python":
        return 0
    patterns = _patterns_for_request(request)
    if not patterns:
        return 0

    fd, temp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(source_snippet)

        score = 0
        for pattern in patterns:
            try:
                proc = subprocess.run(
                    ["semgrep", "--lang", "python", "--pattern", pattern, "--json", "--quiet", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                data = json.loads(proc.stdout or "{}")
                if data.get("results"):
                    score += 1
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                continue
        return score
    finally:
        os.unlink(temp_path)
