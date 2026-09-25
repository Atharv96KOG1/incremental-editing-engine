"""Semgrep as a structural filter (PHOENIX doc section 5: "Semgrep should
be treated as a structural verification/filtering layer, not another
semantic retrieval engine"). Two uses:

1. `find_call_sites` -- every real call-site of a specific symbol name,
   structurally verified (an actual call expression, not a name that
   happens to appear in a comment/string the way plain grep would match).

2. `structural_match_score` -- "meaning-wise" matching: checks whether a
   candidate's actual code *does* what the request describes (raises
   exceptions, validates types, loops, calls math functions, ...), not
   just whether words overlap. A request like "validate the input types"
   should favor a function whose body actually contains an isinstance/
   raise TypeError pattern -- grounded in real code structure, which text
   overlap alone can't tell apart from a coincidental word match.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List

# keyword(s) in the request -> a semgrep pattern that structurally confirms
# the candidate's code actually does that, not just that a word matches.
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
    site of symbol_name found by semgrep. Returns [] if semgrep itself
    fails, isn't installed, or times out -- this is a best-effort
    enrichment the rest of the pipeline never blocks on. `language` should
    be the target symbol's own language (semgrep's own lang name, e.g.
    "python"/"java"/"javascript"/"go") -- `foo(...)` is valid call syntax
    across every C-family/Python-family grammar semgrep supports."""
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
    """Runs each request-relevant semgrep pattern against just this one
    candidate's own source (a temp file, not the whole repo) and counts
    how many actually match. 0 if the request names no known structural
    intent, or semgrep can't confirm it -- this only ever adds a signal,
    never removes one a text/vector retriever already found.

    The intent patterns above (isinstance(...), raise $EXC(...), math.
    $FUNC(...), ...) are Python syntax specifically, so this only runs for
    language == "python" -- a real per-language intent-pattern table is
    future work, not a silent guess against the wrong grammar."""
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
