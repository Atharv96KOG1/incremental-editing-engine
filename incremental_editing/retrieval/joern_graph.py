"""Optional Joern integration -- a real Code Property Graph (CPG) call
graph, used in place of the native AST-based one (`dependency_graph.py`)
when cross-file/cross-module accuracy matters more than speed.

This is a deliberate escalation, not a default. Measured cost building a
CPG for a 204-line, single-file project: ~12 seconds, dominated by JVM/
Scala startup, not the actual analysis -- versus the native graph's
near-instant AST walk. A real project will cost more, not less. Requires
`joern`/`joern-parse` on PATH: a separate system install (a JDK plus the
Joern CLI, ~1.7GB), neither of which ships with this project's own
dependencies. Never used on the fast edit-auto-locate path
(`locate_best_file`); only reachable from `iee find` when explicitly
requested via `use_joern=True`, and cached per project directory so the
cost is paid once, not per request.

Where the native graph is genuinely weaker (its own docstring says so):
name-based only, no cross-module resolution -- two unrelated `foo`
functions in different files look identical to it. Joern's CPG resolves
calls against real per-file method identity, so this closes exactly that
gap when it's turned on.
"""

import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .repo_index import RepoSymbol

_CACHE_DIR = os.path.expanduser("~/.cache/iee/joern")

# The query is real Scala, in its own .sc file rather than an inline
# Python string -- Scala's own triple-quoted s"""...""" interpolation
# collides with Python's triple-quote string literals, so embedding it
# as a Python string is a real, easy-to-hit bug, not just untidy.
_QUERY_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "_joern_query.sc")

# `joern-parse` auto-detects a single dominant language per invocation --
# pointed at a directory mixing Python and Java, it silently picks one
# and never looks at the other language's files at all (verified: it
# built a CPG covering only the .py files in a Python+Java sample
# directory, no error, no warning). This project indexes symbols
# per-file by real detected language (`RepoSymbol.language`), so each
# language actually present gets its own explicit `--language` pass
# instead of trusting auto-detection to notice a mixed repo.
_LANGUAGE_TO_JOERN_FLAG = {
    "python": "pythonsrc",
    "java": "javasrc",
    "javascript": "jssrc",
    "typescript": "jssrc",  # verified: jssrc2cpg parses .ts directly, no separate TS frontend needed
    "go": "golang",
    "csharp": "csharpsrc",
    "ruby": "rubysrc",
    "kotlin": "kotlin",
    "php": "php",
    "rust": "rust",
    "swift": "swiftsrc",
    "c": "c",
    "cpp": "newc",
}


def is_available() -> bool:
    return shutil.which("joern-parse") is not None and shutil.which("joern") is not None


def _fingerprint(project_dir: str, language: str, symbols: List[RepoSymbol]) -> str:
    """Cheap (stat-only) signature of every file in this language, so a
    cached CPG is invalidated the moment a file it covers actually
    changes. This tool's entire purpose is editing files that change
    between requests -- a cache keyed on project path and language alone
    would silently keep serving a stale CPG (built before the edit) with
    nothing to ever invalidate it, which is worse than not caching at
    all. Mirrors `repo_index.py`'s own fingerprint approach."""
    entries = []
    for rel_file in sorted({s.file for s in symbols if s.language == language}):
        try:
            stat = os.stat(os.path.join(project_dir, rel_file))
            entries.append((rel_file, stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return hashlib.sha256(repr(entries).encode("utf-8")).hexdigest()[:12]


def _cache_key(project_dir: str, language: str, symbols: List[RepoSymbol]) -> str:
    fingerprint = _fingerprint(project_dir, language, symbols)
    path_hash = hashlib.sha256(os.path.abspath(project_dir).encode("utf-8")).hexdigest()[:16]
    return f"{path_hash}_{language}_{fingerprint}"


def _extract_edges_for_language(project_dir: str, joern_language: str, cache_key: str, timeout: int) -> Optional[list]:
    """Runs one joern-parse + query pass for a single forced language and
    returns its raw edge list, or None if that pass fails -- a failure
    analyzing one language (e.g. an unsupported/broken frontend) must not
    lose the languages that did work. `cache_key` already encodes a
    content fingerprint (see `_fingerprint`), so a hit here means those
    exact files, byte for byte, were already analyzed -- safe to reuse
    without re-running the ~10-40s joern-parse + query pass at all."""
    cpg_path = os.path.join(_CACHE_DIR, f"{cache_key}.cpg")
    out_path = os.path.join(_CACHE_DIR, f"{cache_key}_edges.json")
    script_path = os.path.join(_CACHE_DIR, f"{cache_key}_query.sc")

    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception:
            pass  # cached output is somehow corrupt -- fall through and rebuild

    try:
        subprocess.run(
            ["joern-parse", project_dir, "--language", joern_language, "--output", cpg_path],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        with open(_QUERY_SCRIPT_PATH) as f:
            query = f.read()
        query = query.replace("__CPG_PATH__", cpg_path).replace("__OUT_PATH__", out_path)
        with open(script_path, "w") as f:
            f.write(query)
        subprocess.run(
            ["joern", "--script", script_path],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        with open(out_path) as f:
            return json.load(f)
    except Exception:
        return None


def build_call_graph_via_joern(
    project_dir: str, symbols: List[RepoSymbol], timeout: int = 300
) -> Optional[Dict[str, dict]]:
    """Same {"calls": {...}, "called_by": {...}} shape as
    dependency_graph.build_call_graph(), resolved via a real CPG instead
    of name-only AST matching. Runs one joern-parse pass per distinct
    language actually present among `symbols` (see the module-level note
    on why auto-detection alone isn't safe for a mixed-language repo).
    Returns None only if Joern isn't installed at all, or every language
    present failed to analyze -- a partial result (some languages
    succeeded, others didn't or aren't Joern-supported) is still
    returned rather than discarded."""
    if not is_available():
        return None

    known_names = {s.name for s in symbols}
    languages = {s.language for s in symbols if s.language in _LANGUAGE_TO_JOERN_FLAG}
    if not languages:
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)

    # Each language's build is an independent subprocess (its own JVM) --
    # running them concurrently instead of one after another turns a
    # multi-language wait into roughly the slowest single language's
    # time instead of their sum. A cached language returns near-
    # instantly regardless, so this only matters on a real multi-
    # language cache miss, but that's exactly the case that used to be
    # the slowest (measured: ~44s sequential for two languages).
    with ThreadPoolExecutor(max_workers=max(1, len(languages))) as pool:
        futures = {
            pool.submit(
                _extract_edges_for_language,
                project_dir,
                _LANGUAGE_TO_JOERN_FLAG[language],
                _cache_key(project_dir, language, symbols),
                timeout,
            ): language
            for language in languages
        }
        all_edges = []
        for future in as_completed(futures):
            edges = future.result()
            if edges:
                all_edges.extend(edges)

    if not all_edges:
        return None

    calls: Dict[str, List[str]] = defaultdict(list)
    called_by: Dict[str, List[str]] = defaultdict(list)
    for edge in all_edges:
        callee, caller, file = edge["callee"], edge["caller"], edge["file"]
        if callee not in known_names or callee == caller:
            continue
        key_str = f"{file}::{caller}"
        if callee not in calls[key_str]:
            calls[key_str].append(callee)
        if key_str not in called_by[callee]:
            called_by[callee].append(key_str)

    return {"calls": dict(calls), "called_by": dict(called_by)}
