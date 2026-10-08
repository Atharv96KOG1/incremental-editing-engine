"""Builds the minimum context sent to the model: module imports + the
located candidate symbol(s) in full, plus a single compact line naming
every *other* symbol in the file (with params/decorators, no bodies),
deduplicated against what's already shown in full.

With no confident candidate (e.g. "add a new function"), context is
imports + that same name line rather than the whole raw file -- still
enough to avoid a hallucinated anchor or a duplicate symbol, without
paying for every body.
"""

import ast

from ..analyzer.locator import index_symbols, locate_candidates, locate_candidates_by_body
from ..analyzer.metadata_builder import extract_symbol_metadata
from ..analyzer.module_blocks import locate_module_level_block
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.fusion import fuse
from ..retrieval.multilang_symbols import extract_import_lines
from ..retrieval.repo_index import RepoSymbol
from ..retrieval.vector_retriever import VectorRetriever


def _find_module_level_imports(node: ast.AST) -> list:
    """Every Import/ImportFrom at module scope, including ones nested
    inside a module-level try/except or if (a conditional/optional-
    dependency import, e.g. `try: import X except ImportError: ...` --
    common enough that missing it left a request touching that exact
    dependency with no visibility into it at all). Stops at any
    function/class boundary: an import local to a function's own body is
    deliberately scoped there, not a module-level dependency the model
    needs upfront."""
    found = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append(child)
        elif not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.extend(_find_module_level_imports(child))
    return found


def _imports_block(source: str, language: str = "python") -> str:
    if language == "python":
        tree = ast.parse(source)
        lines = source.splitlines()
        import_nodes = sorted(_find_module_level_imports(tree), key=lambda n: n.lineno)
        chunks = ["\n".join(lines[node.lineno - 1 : node.end_lineno]) for node in import_nodes]
        return "\n".join(chunks)

    if language is None:
        return ""
    try:
        ranges = extract_import_lines(source, language)
    except Exception:
        return ""
    lines = source.splitlines()
    return "\n".join("\n".join(lines[start - 1 : end]) for start, end in ranges)


_MAX_SYMBOLS_FOR_PARAM_DETAIL = 20


def _format_symbol_entry(sym, show_params: bool) -> str:
    entry = f"{sym.name}({', '.join(sym.parameters)})" if show_params and sym.symbol_type == "function" else sym.name
    if sym.decorators:
        entry += " " + " ".join(f"@{d}" for d in sym.decorators)
    return entry


def _symbol_names_line(source: str, exclude: set, language: str = "python") -> str:
    """One compact line naming every symbol not in `exclude`, with params
    and decorators for files under _MAX_SYMBOLS_FOR_PARAM_DETAIL symbols.
    Deduplicated -- a name defined twice would otherwise repeat for no
    added information."""
    seen = set()
    metas = []
    for sym in extract_symbol_metadata(source, language):
        if sym.name in exclude or sym.name in seen:
            continue
        seen.add(sym.name)
        metas.append(sym)

    show_params = len(metas) <= _MAX_SYMBOLS_FOR_PARAM_DETAIL
    entries = [_format_symbol_entry(sym, show_params) for sym in metas]
    return f"# other symbols defined in this file: {', '.join(entries)}" if entries else ""


def _to_repo_symbols(symbols, source: str, language: str) -> list:
    """Adapts this file's SymbolInfo list to the RepoSymbol shape
    BM25Retriever/VectorRetriever/fuse expect, reusing those retrievers
    scoped to one file instead of the whole repo. file="" since fuse()'s
    dedup key only needs consistency within one call, not a real path."""
    lines = source.splitlines()
    return [
        RepoSymbol(
            file="",
            name=sym.name,
            symbol_type=sym.symbol_type,
            start_line=sym.start_line,
            end_line=sym.end_line,
            docstring_first_line=sym.docstring_first_line,
            source="\n".join(lines[sym.start_line - 1 : sym.end_line]),
            language=language or "python",
        )
        for sym in symbols
    ]


def _safe_vector_rank(repo_symbols: list, user_request: str, top_k: int) -> list:
    """Vector retrieval is one signal among several; a failure (embeddings
    gateway cold start, network blip) must degrade to no signal rather
    than crash the edit. Duplicated from locate_repo.py's own
    _safe_vector_ranking to keep this module's cross-retrieval dependency
    limited to the small shared pieces (BM25Retriever, fuse, RepoSymbol)."""
    try:
        return VectorRetriever(repo_symbols).rank(user_request, top_k=top_k)
    except Exception:
        return []


def _hybrid_rerank(source: str, user_request: str, language: str, symbol_candidates: list) -> list:
    """Fuses the existing name/docstring symbol match (kept as-is -- its
    disambiguation logic is already tuned) with BM25 and vector retrieval
    over every symbol in the file, since name matching alone misses a
    request sharing only body words or a paraphrase. Adds recall only --
    a symbol needs just one signal to surface, so this can narrow which
    match wins but can't erase a match the symbol pass already found.

    Note: vector retrieval is a real embeddings-API call, added to every
    edit request that reaches here."""
    all_symbols = index_symbols(source, language)
    if not all_symbols:
        return symbol_candidates

    repo_symbols = _to_repo_symbols(all_symbols, source, language)
    by_key = {(rs.name, rs.start_line): orig for rs, orig in zip(repo_symbols, all_symbols)}

    wide_k = max(len(symbol_candidates), 2) * 3
    rankings = [
        [(rs, 0.0) for rs in _to_repo_symbols(symbol_candidates, source, language)],
        BM25Retriever(repo_symbols).rank(user_request, top_k=wide_k),
    ]
    vector_ranking = _safe_vector_rank(repo_symbols, user_request, wide_k)
    if vector_ranking:
        rankings.append(vector_ranking)

    fused = fuse(rankings, top_k=max(len(symbol_candidates), 2))
    reranked = [by_key[(rs.name, rs.start_line)] for rs, _ in fused if (rs.name, rs.start_line) in by_key]
    return reranked or symbol_candidates


def _spans_overlap(a, b) -> bool:
    return a.start_line <= b.end_line and b.start_line <= a.end_line


def _module_level_hint(source: str, user_request: str, language: str, use_hybrid: bool) -> str:
    """A matching module-level region (analyzer/module_blocks.py) shown as
    a labeled, clearly-separate piece of context -- real failure this
    closes: a request whose real target is a bare module-level constant
    (e.g. `MODEL_NAME = "..."`) is invisible to locate_candidates (only
    scores functions/classes), so the model never learns the name even
    exists, can't recognize the request reaches outside any symbol's
    span, and silently guess-inserts a second, conflicting definition via
    a normal function-shaped operation instead of setting escalate:
    {"kind": "whole_file"} (which _run_whole_file_edit already handles
    correctly via this exact same locator, narrowly, no whole-file rewrite
    needed). This only ever adds visibility -- the structured-edit schema
    still can't target this region directly, by design.

    Gated the same way _run_whole_file_edit gates its own call to this
    locator: only when the file has at least one real function/class
    (with zero, the "gap" is the entire file, and showing it here would
    just mean showing the whole file under a different name)."""
    if language is None or not index_symbols(source, language):
        return ""
    block = locate_module_level_block(source, user_request, language, use_hybrid=use_hybrid)
    if block is None:
        return ""
    lines = source.splitlines()
    content = "\n".join(lines[block.start_line - 1 : block.end_line])
    return (
        "# module-level region possibly relevant to this request -- NOT a function/class, so "
        "REPLACE/INSERT/DELETE can't target it directly; if it needs changing, set "
        'escalate: {"kind": "whole_file"} instead:\n' + content
    )


def build_context(source: str, user_request: str, language: str = "python", use_hybrid: bool = False) -> dict:
    total_lines = len(source.splitlines())
    candidates = locate_candidates(source, user_request, language=language)
    if use_hybrid:
        candidates = _hybrid_rerank(source, user_request, language, candidates)

    existing_names = {c.name for c in candidates}
    for extra in locate_candidates_by_body(source, user_request, language=language):
        if extra.name in existing_names:
            continue
        if not any(_spans_overlap(extra, existing) for existing in candidates):
            candidates.append(extra)
            existing_names.add(extra.name)

    imports = _imports_block(source, language)
    module_hint = _module_level_hint(source, user_request, language, use_hybrid)

    if not candidates:
        pieces = [imports] if imports else []
        names_line = _symbol_names_line(source, exclude=set(), language=language)
        if names_line:
            pieces.append(names_line)
        if module_hint:
            pieces.append(module_hint)
        context = "\n\n".join(p for p in pieces if p)
        return {
            "context": context or source,
            "used_localization": False,
            "total_lines": total_lines,
            "context_lines": sum(len(p.splitlines()) for p in pieces if p) or total_lines,
            "candidate_symbols": [],
            "candidate_lines": {},
        }

    lines = source.splitlines()
    pieces = [imports] if imports else []
    pieces.extend("\n".join(lines[sym.start_line - 1 : sym.end_line]) for sym in candidates)
    names_line = _symbol_names_line(source, exclude={s.name for s in candidates}, language=language)
    if names_line:
        pieces.append(names_line)
    if module_hint:
        pieces.append(module_hint)
    context = "\n\n".join(p for p in pieces if p)

    context_lines = sum(len(p.splitlines()) for p in pieces if p)

    return {
        "context": context,
        "used_localization": True,
        "total_lines": total_lines,
        "context_lines": context_lines,
        "candidate_symbols": [s.name for s in candidates],
        "candidate_lines": {s.name: s.start_line for s in candidates},
    }
