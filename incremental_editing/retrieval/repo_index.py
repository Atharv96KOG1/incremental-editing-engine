"""Repository-wide symbol index (PHOENIX doc section 1: "Repository
Ingestion -> Global Code Knowledge Base -> Symbol Index").

The existing `analyzer/locator.py` only ever looks at one file, because
every existing entry point (`iee edit --file ...`) already names the file.
This module is what makes the file itself discoverable: walk every source
file in a project, index every symbol in every one, so retrieval can
answer "which file/symbol" instead of just "which symbol in this file".

Python files use the `ast`-based path (unchanged, proven). Every other
file goes through `multilang_symbols.py` (Tree-sitter): unrecognized
extensions are skipped rather than guessed at, same as
`validation/syntax.py` already does for syntax-checking.
"""

import ast
import os
from dataclasses import dataclass
from typing import List, Optional

from .multilang_symbols import detect_language, extract_symbols

_IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "dist", "build", ".mypy_cache", ".ruff_cache", "minio_local_data",
    # This project's own per-file metadata output (metadata_builder.py's
    # write_file_metadata) -- real code is what's indexed, never a JSON
    # description *of* it.
    "iee_metadata",
}

_IEEIGNORE_FILENAME = ".ieeignore"


def project_ignore_dirs(project_dir: str) -> set:
    """Extra directory names to skip for THIS project only, declared in
    a `.ieeignore` file at its own root (one bare directory name per
    line, '#' comments and blank lines skipped) -- the same idea as
    .gitignore, for when a project directory legitimately contains real
    content that repo-wide retrieval (and validation's own copy, see
    validation/tests.py) shouldn't touch: an unrelated demo/scratch
    folder living alongside the real project, for instance. Deliberately
    never a name hardcoded into this shared module -- that would wrongly
    exclude some *other* project's real, identically-named folder.
    Entirely opt-in, declared by whoever owns that project directory."""
    path = os.path.join(project_dir, _IEEIGNORE_FILENAME)
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}


@dataclass
class RepoSymbol:
    file: str  # relative to project root, e.g. "core/agent.py"
    name: str
    symbol_type: str  # "function" | "class"
    start_line: int
    end_line: int
    docstring_first_line: Optional[str]
    source: str  # the symbol's own source text, sliced from its file
    language: str = "python"  # "python" or a tree-sitter language name, e.g. "java", "go"


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


_index_cache: dict = {}  # project_dir -> (fingerprint, symbols)


def _fingerprint(project_dir: str) -> tuple:
    """Cheap (stat-only, no read/parse) signature of every source file
    under project_dir -- lets build_repo_index skip re-reading and
    re-parsing the entire repo on every call when nothing has changed
    since the last one. A stale in-memory index (edits made outside this
    process while it's running) would be worse than the cost this avoids,
    so it's a real stat() per file, not a directory mtime shortcut."""
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
    project_dir. A file that fails to parse (syntax error, non-UTF8, or an
    unrecognized/unsupported language) is skipped rather than aborting the
    whole index.

    Cached per project_dir, invalidated automatically the moment any
    file's mtime/size changes -- repeat calls in a long-running process
    (`iee serve`) reuse the previous index instead of re-walking and
    re-parsing every file from scratch each time."""
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
            continue  # unrecognized extension -- nothing to index, don't guess
        try:
            symbols.extend(_index_multilang_symbols(rel, source, lines, language))
        except Exception:
            continue  # unparseable/unsupported grammar -- skip this file only

    if use_cache:
        _index_cache[project_dir] = (fingerprint, symbols)
    return symbols
