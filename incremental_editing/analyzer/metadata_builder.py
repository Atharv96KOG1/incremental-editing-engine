"""Static, zero-LLM-call code metadata -- one JSON artifact per project,
covering every function/class in every indexed file: name, type, exact
line span, parameters, decorators, return-type hint, async-ness, a
keyword set, how many other symbols call it, what it calls, and a content
hash. Nothing here ever asks a model anything; it's pure AST/Tree-sitter
extraction, the same techniques already used throughout `analyzer/` and
`retrieval/`, just persisted as one artifact instead of recomputed ad hoc
inside each of locate_candidates/build_context/build_call_graph.

Why this exists: the compact "other symbols" line the LLM sees today
(context_builder.py) shows bare names only -- "divide", not "divide(a,
b)". A request naming a parameter or a framework decorator ("the route
handler for /login") has nothing to match against. Pulling parameters and
decorators into that same line costs a handful of tokens and gives real
disambiguating signal, without ever sending a full body. That's the
token-cost tradeoff this whole project has been built around, applied to
one more input.

Persisted in one visible, browsable folder (~/iee-metadata/), one file
per project named after that project's own folder -- not a dot-prefixed
cache directory like Joern's, since this is meant to be looked at
directly. Fingerprinted the same way repo_index's own cache is: unchanged
files reuse the prior metadata instead of re-parsing on every call.
"""

import ast
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from ..retrieval.dependency_graph import build_call_graph
from ..retrieval.multilang_symbols import detect_language, extract_import_lines, extract_symbols
from ..retrieval.repo_index import RepoSymbol, build_repo_index, iter_source_files
from .locator import _words as _real_words
from .locator import looks_like_python

# A visible, browsable folder -- not a dot-prefixed cache dir. This is
# meant to be looked at directly (a human asking "show me the metadata
# file"), not just machine-internal state like Joern's ~/.cache/iee/joern.
_CACHE_DIR = os.path.expanduser("~/iee-metadata")

# Per-file, per-commit metadata folder -- lives inside the project itself
# (not ~/iee-metadata/, which is the one-JSON-per-*project* snapshot
# above) so it's browsable right next to the code it describes. One JSON
# per source file, mirroring that file's own relative path.
_PER_FILE_METADATA_DIRNAME = "iee_metadata"


@dataclass
class SymbolMetadata:
    name: str
    symbol_type: str  # "function" | "class"
    start_line: int
    end_line: int
    line_count: int
    language: str
    parent_class: Optional[str] = None
    docstring_first_line: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    return_type_hint: Optional[str] = None
    is_async: bool = False
    keywords: List[str] = field(default_factory=list)
    called_by_count: int = 0
    calls: List[str] = field(default_factory=list)
    content_hash: str = ""


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _words(text: str) -> List[str]:
    return sorted(_real_words(text))


def _python_symbol_metadata(node: ast.AST, lines: List[str], language: str, parent_class: Optional[str]) -> SymbolMetadata:
    symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end = node.end_lineno
    doc = ast.get_docstring(node)
    first_line = doc.strip().splitlines()[0] if doc else None

    parameters: List[str] = []
    return_type_hint: Optional[str] = None
    is_async = isinstance(node, ast.AsyncFunctionDef)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        parameters = [p.arg for p in getattr(a, "posonlyargs", [])] + [p.arg for p in a.args]
        if a.vararg:
            parameters.append(f"*{a.vararg.arg}")
        parameters += [p.arg for p in a.kwonlyargs]
        if a.kwarg:
            parameters.append(f"**{a.kwarg.arg}")
        if node.returns is not None:
            try:
                return_type_hint = ast.unparse(node.returns)
            except Exception:
                return_type_hint = None

    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            continue

    snippet = "\n".join(lines[start - 1 : end])
    keywords = sorted(set(_words(node.name.replace("_", " "))) | (set(_words(first_line)) if first_line else set()))

    return SymbolMetadata(
        name=node.name,
        symbol_type=symbol_type,
        start_line=start,
        end_line=end,
        line_count=end - start + 1,
        language=language,
        parent_class=parent_class,
        docstring_first_line=first_line,
        parameters=parameters,
        decorators=decorators,
        return_type_hint=return_type_hint,
        is_async=is_async,
        keywords=keywords,
        content_hash=_content_hash(snippet),
    )


def _index_python_metadata(source: str, lines: List[str]) -> List[SymbolMetadata]:
    tree = ast.parse(source)
    out: List[SymbolMetadata] = []

    def walk(node: ast.AST, parent_class: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.append(_python_symbol_metadata(child, lines, "python", parent_class))
                walk(child, child.name if isinstance(child, ast.ClassDef) else parent_class)
            else:
                walk(child, parent_class)

    walk(tree, None)
    return out


def extract_symbol_metadata(source: str, language: str) -> List[SymbolMetadata]:
    """Single-file entry point -- no project_dir, no disk cache, no
    call-graph merge (that needs the whole repo indexed first, see
    build_project_metadata). Same per-symbol fields except calls/
    called_by_count (always empty/0 here). Costs the same one AST/
    Tree-sitter parse `index_symbols()` already pays per call; this
    reads a couple more fields off the same parse, not a second one."""
    lines = source.splitlines()
    if language == "python":
        try:
            return _index_python_metadata(source, lines)
        except SyntaxError:
            return []
    if language is None:
        return []
    try:
        return _index_multilang_metadata(source, lines, language)
    except Exception:
        return []


def _index_multilang_metadata(source: str, lines: List[str], language: str) -> List[SymbolMetadata]:
    """Non-Python languages get the fields cheaply available from the
    generic Tree-sitter extraction already used elsewhere in this
    project (name, type, line span, keywords, content hash) -- not
    parameters/decorators/return-type/async-ness, which would need a
    per-language grammar query rather than the generic node-type-name
    heuristic this project deliberately uses for breadth over depth
    (same honest limitation as docstring_first_line/parent_class being
    Python-only elsewhere in analyzer/locator.py)."""
    out: List[SymbolMetadata] = []
    for raw in extract_symbols(source, language):
        snippet = "\n".join(lines[raw.start_line - 1 : raw.end_line])
        out.append(
            SymbolMetadata(
                name=raw.name,
                symbol_type=raw.symbol_type,
                start_line=raw.start_line,
                end_line=raw.end_line,
                line_count=raw.end_line - raw.start_line + 1,
                language=language,
                keywords=sorted(set(_words(raw.name.replace("_", " ")))),
                content_hash=_content_hash(snippet),
            )
        )
    return out


def _file_imports(source: str, language: str) -> List[str]:
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        names = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names
    try:
        lines = source.splitlines()
        ranges = extract_import_lines(source, language)
        return ["\n".join(lines[s - 1 : e]) for s, e in ranges]
    except Exception:
        return []


def _fingerprint(project_dir: str) -> tuple:
    entries = []
    for path in iter_source_files(project_dir):
        try:
            stat = os.stat(path)
            entries.append((os.path.relpath(path, project_dir), stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return tuple(sorted(entries))


def _cache_path(project_dir: str) -> str:
    # Human-readable, not just a hash: the project folder's own name comes
    # first so browsing ~/iee-metadata/ shows which file is which project
    # at a glance -- the short hash suffix only exists to keep two
    # different projects that happen to share a folder name from
    # colliding, not to be the primary identifier.
    name = os.path.basename(os.path.abspath(project_dir).rstrip(os.sep)) or "root"
    key = hashlib.sha256(os.path.abspath(project_dir).encode("utf-8")).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"{name}-{key}.json")


def build_project_metadata(project_dir: str, use_cache: bool = True) -> dict:
    """Builds (or reuses a fingerprint-valid cached copy of) the full
    per-file, per-symbol metadata document for every recognized source
    file under project_dir. Zero LLM calls -- every field here comes
    from AST (Python) or Tree-sitter (everything else) plus the existing
    native call-graph for called_by_count/calls."""
    cache_file = _cache_path(project_dir)
    fingerprint = _fingerprint(project_dir)
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            if cached.get("_fingerprint") == list(fingerprint):
                return cached
        except Exception:
            pass

    symbols_by_file: Dict[str, List[SymbolMetadata]] = {}
    file_info: Dict[str, dict] = {}

    for path in iter_source_files(project_dir):
        rel = os.path.relpath(path, project_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        lines = source.splitlines()

        if rel.endswith(".py"):
            language = "python"
            try:
                symbols_by_file[rel] = _index_python_metadata(source, lines)
            except SyntaxError:
                continue
        else:
            from ..retrieval.multilang_symbols import detect_language

            language = detect_language(rel)
            if language is None:
                continue
            try:
                symbols_by_file[rel] = _index_multilang_metadata(source, lines, language)
            except Exception:
                continue

        file_info[rel] = {
            "language": language,
            "file_hash": _content_hash(source),
            "total_lines": len(lines),
            "imports": _file_imports(source, language),
        }

    # Reuse the existing native call graph for called_by_count/calls --
    # same technique dependency_graph.py already uses, applied once here
    # instead of recomputed separately by every caller.
    repo_symbols: List[RepoSymbol] = build_repo_index(project_dir)
    call_graph = build_call_graph(repo_symbols)

    files_out = {}
    for rel, syms in symbols_by_file.items():
        symbol_dicts = []
        for sym in syms:
            key = f"{rel}::{sym.name}"
            sym.calls = call_graph["calls"].get(key, [])
            sym.called_by_count = len(call_graph["called_by"].get(sym.name, []))
            symbol_dicts.append(asdict(sym))
        files_out[rel] = {**file_info[rel], "symbols": symbol_dicts}

    document = {
        "_fingerprint": list(fingerprint),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_dir": os.path.abspath(project_dir),
        "files": files_out,
    }

    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(document, f, indent=2)

    return document


def metadata_cache_path(project_dir: str) -> str:
    """Where build_project_metadata writes/reads its cache for this
    project -- exposed so a caller (CLI) can tell the user exactly where
    to look."""
    return _cache_path(project_dir)


def _detect_language(rel_path: str, source: str) -> Optional[str]:
    """Same detection order run_pipeline.run_edit already applies before
    validate_targets/index_symbols -- kept in one place here too so every
    commit site (edit, create, whole-file rewrite, language conversion,
    multi-file create) gets identical, consistent per-file metadata
    without repeating this three-line rule at each call site."""
    if rel_path.endswith(".py"):
        return "python"
    language = detect_language(rel_path)
    if language is None and looks_like_python(source):
        return "python"
    return language


def per_file_metadata_path(project_dir: str, rel_path: str) -> str:
    """Where write_file_metadata/delete_file_metadata put this one
    file's own metadata JSON -- exposed so callers (and tests) don't
    have to know the folder name or the ".json" suffix convention."""
    return os.path.join(str(project_dir), _PER_FILE_METADATA_DIRNAME, rel_path + ".json")


def write_file_metadata(project_dir: str, rel_path: str, source: str) -> Optional[str]:
    """Regenerates and writes *this one file's* own metadata JSON --
    called at every real commit point (STRUCTURED_EDIT, whole-file
    rewrite, language conversion's new file, create, multi-file create)
    right after the file's real content is written to disk, so
    <project_dir>/iee_metadata/ always reflects what's actually there
    right now, never a stale prior version.

    Every symbol entry carries what a human (or the LLM's own
    localization step) needs to find it again exactly: name, start_line,
    end_line, the real "block" (the exact source lines that span it,
    not just a name), and a keyword index (tokenized name + first
    docstring line) -- the same fields build_project_metadata already
    computes for the whole-project snapshot, just persisted per file
    instead of buried inside one combined document, and refreshed on
    every edit instead of only on an explicit `iee metadata` call.

    Returns None (writes nothing) for a file whose language can't be
    determined at all -- there are no symbols to index, so an empty
    metadata file would just be noise."""
    language = _detect_language(rel_path, source)
    if language is None:
        return None

    lines = source.splitlines()
    symbols = extract_symbol_metadata(source, language)
    symbol_dicts = []
    for sym in symbols:
        d = asdict(sym)
        d["block"] = "\n".join(lines[sym.start_line - 1 : sym.end_line])
        symbol_dicts.append(d)

    document = {
        "file": rel_path,
        "language": language,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_lines": len(lines),
        "file_hash": _content_hash(source),
        "symbols": symbol_dicts,
    }

    out_path = per_file_metadata_path(project_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(document, f, indent=2)
    return out_path


def delete_file_metadata(project_dir: str, rel_path: str) -> None:
    """Mirrors a whole-file delete: once the real file is gone, its
    metadata JSON describes something that no longer exists, so it's
    removed too rather than left behind as a stale artifact."""
    path = per_file_metadata_path(project_dir, rel_path)
    if os.path.exists(path):
        os.remove(path)
