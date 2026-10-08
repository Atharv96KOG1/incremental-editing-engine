"""Optional Joern integration: a real Code Property Graph (CPG) call graph,
used instead of the native AST-based one (`dependency_graph.py`) when
cross-file/cross-module accuracy matters more than speed.

Deliberate escalation, not default: building a CPG costs ~12s+ (JVM/Scala
startup dominated, a real project costs more) vs the native graph's
near-instant walk, and requires `joern`/`joern-parse` on PATH (separate
JDK + Joern install, ~1.7GB). Only reachable via `iee find`'s explicit
use_joern=True, cached per project directory so the cost is paid once.

Native graph is name-based only, no cross-module resolution (two unrelated
`foo` functions in different files look identical); Joern's CPG resolves
real per-file method identity and closes that gap.
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

_QUERY_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "_joern_query.sc")

_LANGUAGE_TO_JOERN_FLAG = {
    "python": "pythonsrc",
    "java": "javasrc",
    "javascript": "jssrc",
    "typescript": "jssrc",
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
    """Cheap stat-only signature of every file in this language, so a
    cached CPG invalidates the moment a covered file changes -- keying the
    cache on path+language alone would silently serve a stale CPG forever.
    Mirrors `repo_index.py`'s own fingerprint approach."""
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
    """Runs one joern-parse + query pass for a single forced language;
    returns None on failure so one broken language doesn't lose the
    others. `cache_key` encodes a content fingerprint, so a hit means
    those exact files were already analyzed."""
    cpg_path = os.path.join(_CACHE_DIR, f"{cache_key}.cpg")
    out_path = os.path.join(_CACHE_DIR, f"{cache_key}_edges.json")
    script_path = os.path.join(_CACHE_DIR, f"{cache_key}_query.sc")

    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception:
            pass

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
    dependency_graph.build_call_graph(), resolved via a real CPG. Runs one
    joern-parse pass per distinct language present; returns None only if
    Joern isn't installed or every language failed -- partial results are
    returned otherwise."""
    if not is_available():
        return None

    known_names = {s.name for s in symbols}
    languages = {s.language for s in symbols if s.language in _LANGUAGE_TO_JOERN_FLAG}
    if not languages:
        return None

    os.makedirs(_CACHE_DIR, exist_ok=True)

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
