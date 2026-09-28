"""Builds the minimum context sent to the model: module imports + the
located candidate symbol(s) in full, plus a single compact line naming
every *other* symbol in the file -- never their bodies, and never repeated
if already shown in full (that duplication wasted tokens for nothing).

When localization has no confident candidate (the common case: "add a new
function", nothing existing to anchor against by name), context used to be
the *entire raw file*. It's now imports + that same compact name line --
still enough to avoid hallucinating an anchor or duplicating an existing
symbol, without paying for every function's body. A REPLACE/DELETE request
essentially always names its target, which locate_candidates catches via
literal name mention -- so this path is overwhelmingly "add something
brand new," where names alone (not full signatures) are enough and stay
maximally compact.

The name line carries each function's real parameters (and any
decorators) alongside its bare name -- "divide(a, b)", not just
"divide" -- pulled from the same AST/Tree-sitter parse `index_symbols()`
already does (metadata_builder.extract_symbol_metadata is a superset of
it, not a second pass), so a request naming a parameter or a framework
decorator ("the route handler for /login") has something real to match
against without ever sending a body. Honest tradeoff: this costs a
handful more characters per name than the bare list did -- the bet is
that avoiding one hallucinated anchor or one wasted repair round-trip is
worth far more than those few extra tokens.
"""

import ast

from ..analyzer.locator import index_symbols, locate_candidates, locate_candidates_by_body
from ..analyzer.metadata_builder import extract_symbol_metadata
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.fusion import fuse
from ..retrieval.multilang_symbols import extract_import_lines
from ..retrieval.repo_index import RepoSymbol
from ..retrieval.vector_retriever import VectorRetriever


def _imports_block(source: str, language: str = "python") -> str:
    if language == "python":
        tree = ast.parse(source)
        lines = source.splitlines()
        chunks = [
            "\n".join(lines[node.lineno - 1 : node.end_lineno])
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
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
    """One compact line naming every symbol NOT in `exclude`. On a file
    small enough for it to stay cheap, each function's real parameters
    (and any decorators) ride along -- still no bodies, just enough more
    than a bare name to disambiguate a request that names a parameter or
    a decorator. See _MAX_SYMBOLS_FOR_PARAM_DETAIL for why that detail is
    capped rather than unconditional.

    Deduplicated: a name defined twice (the ambiguous-symbol case
    `find_symbol` already refuses to edit) would otherwise appear twice in
    this list for zero added information -- pure wasted tokens."""
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
    """Adapts this file's own SymbolInfo list into the RepoSymbol shape
    BM25Retriever/VectorRetriever/fuse already expect (duck-typed: they
    only ever read .file/.name/.symbol_type/.start_line/.end_line/
    .docstring_first_line/.source/.language) -- reuses those retrievers
    exactly as `iee find`'s repo-wide retrieval does, just scoped to one
    file's symbols instead of every file in the repo. file="" for all of
    them since fuse()'s dedup key only needs it to be *consistent*
    within one call, not a real path."""
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
    """Vector retrieval is one signal among several -- a failure here
    (embeddings gateway cold-starting, a network blip) must degrade to
    "this signal didn't fire" and let symbol+BM25 carry the request, not
    crash the edit. Same reasoning retrieval/locate_repo.py's own
    _safe_vector_ranking already applies for `iee find`; duplicated (not
    imported) to keep this module's only cross-retrieval dependency on
    the small, genuinely shared pieces (BM25Retriever, fuse, RepoSymbol)."""
    try:
        return VectorRetriever(repo_symbols).rank(user_request, top_k=top_k)
    except Exception:
        return []


def _hybrid_rerank(source: str, user_request: str, language: str, symbol_candidates: list) -> list:
    """Fuses the name/docstring symbol match already computed above
    (preserved exactly -- its class-mention disambiguation and generic-
    verb suppression are real, already-tuned correctness fixes, not
    something to redo here) with BM25 keyword and vector/semantic
    retrieval over every symbol in this file.

    Real motivation: name/docstring matching alone still misses a
    request that shares only rare body words, or a paraphrase, with the
    right symbol -- BM25 and vector retrieval each catch a different one
    of those two cases (PHOENIX doc section 4), the same reasoning
    `iee find`'s repo-wide hybrid retrieval already applies, just scoped
    to one file instead of the whole repo. A symbol only needs to be
    surfaced by ONE signal to be eligible here (fuse() ranks by combined
    rank, not unanimous agreement), so this only ever adds recall on top
    of the existing symbol match -- it can narrow which of several
    matches wins, but can't make a request that matched nothing at all
    still match nothing.

    Real per-call cost callers should know about: BM25 is free (no
    network), but vector retrieval is a real embeddings-API call, added
    to every edit request that reaches this -- not the zero-LLM-call
    guarantee this project's cheaper fast paths (deletes, renames) make."""
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

    if not candidates:
        pieces = [imports] if imports else []
        names_line = _symbol_names_line(source, exclude=set(), language=language)
        if names_line:
            pieces.append(names_line)
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
