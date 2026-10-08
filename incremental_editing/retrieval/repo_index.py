"""Repository-wide symbol index: walks every source file in a project and
indexes every symbol, so retrieval can answer "which file/symbol" instead
of just "which symbol in this file". Python uses `ast`; other languages go
through `multilang_symbols.py` (Tree-sitter), skipping unrecognized
extensions rather than guessing."""

import ast
import os
from dataclasses import dataclass
from typing import List, Optional

from .multilang_symbols import detect_language, extract_symbols

_IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "dist", "build", ".mypy_cache", ".ruff_cache", "minio_local_data",
    "iee_metadata",
}

_IEEIGNORE_FILENAME = ".ieeignore"


def project_ignore_dirs(project_dir: str) -> set:
    """Extra directories to skip for this project, from a `.ieeignore`
    file at its own root (one name per line, '#' comments skipped) --
    like .gitignore. Opt-in per project rather than hardcoded here, since
    a hardcoded name could wrongly exclude another project's real,
    identically-named folder."""
    path = os.path.join(project_dir, _IEEIGNORE_FILENAME)
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}


@dataclass
class RepoSymbol:
    file: str
    name: str
    symbol_type: str
    start_line: int
    end_line: int
    docstring_first_line: Optional[str]
    source: str
    language: str = "python"


def iter_source_files(project_dir: str):
    ignore_dirs = _IGNORE_DIRS | project_ignore_dirs(project_dir)
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for fname in files:
            yield os.path.join(root, fname)


def _index_python_symbols(rel: str, source: str, lines: List[str]) -> List[RepoSymbol]:
    tree = ast.parse(source)
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            doc = ast.get_docstring(node)
            first_line = doc.strip().splitlines()[0] if doc else None
            snippet = "\n".join(lines[start - 1 : node.end_lineno])
            symbols.append(
                RepoSymbol(rel, node.name, symbol_type, start, node.end_lineno, first_line, snippet, "python")
            )
    return symbols


def _index_multilang_symbols(rel: str, source: str, lines: List[str], language: str) -> List[RepoSymbol]:
    symbols = []
    for raw in extract_symbols(source, language):
        snippet = "\n".join(lines[raw.start_line - 1 : raw.end_line])
        symbols.append(RepoSymbol(rel, raw.name, raw.symbol_type, raw.start_line, raw.end_line, None, snippet, language))
    return symbols


_index_cache: dict = {}


def _fingerprint(project_dir: str) -> tuple:
    """Cheap stat-only signature of every source file under project_dir,
    letting build_repo_index skip re-parsing when nothing changed. Uses a
    real stat() per file, not a directory mtime shortcut, since a stale
    index would be worse than the cost this avoids."""
    entries = []
    for path in iter_source_files(project_dir):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        entries.append((os.path.relpath(path, project_dir), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(entries))


def build_repo_index(project_dir: str, use_cache: bool = True) -> List[RepoSymbol]:
    """Indexes every function/class in every recognized source file under
    project_dir. A file that fails to parse is skipped rather than
    aborting the whole index. Cached per project_dir, invalidated when any
    file's mtime/size changes."""
    if use_cache:
        fingerprint = _fingerprint(project_dir)
        cached = _index_cache.get(project_dir)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    symbols: List[RepoSymbol] = []
    for path in iter_source_files(project_dir):
        rel = os.path.relpath(path, project_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        lines = source.splitlines()

        if rel.endswith(".py"):
            try:
                symbols.extend(_index_python_symbols(rel, source, lines))
            except SyntaxError:
                continue
            continue

        language = detect_language(rel)
        if language is None:
            continue
        try:
            symbols.extend(_index_multilang_symbols(rel, source, lines, language))
        except Exception:
            continue

    if use_cache:
        _index_cache[project_dir] = (fingerprint, symbols)
    return symbols
