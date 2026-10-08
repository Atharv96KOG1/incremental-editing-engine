"""LLM-free locator for module-level content in a language that has a
function/class concept -- content Delta IR structurally can't reach
(REPLACE/INSERT/DELETE only target a function/class symbol): a
module-level constant, a bare statement, an `if __name__ == "__main__":`
guard, a standalone comment between/around real symbols.

Before this existed, such a request had only the refused whole_file
escalation path (run_pipeline.py's _run_whole_file_edit). Closes the gap
the same way analyzer/text_blocks.py does for languages with no symbol
concept at all: locate the one relevant region, send only that, splice it
back by line range.
"""

from typing import List, Optional

from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.fusion import fuse
from ..retrieval.repo_index import RepoSymbol
from .locator import _GENERIC_LEADING_VERBS, _GENERIC_TYPE_WORDS, SymbolInfo, _words, index_symbols
from .text_blocks import _blocks_from_blank_lines, _safe_vector_rank


def _chunk_gap(lines: List[str], start: int, end: int) -> List[SymbolInfo]:
    """Splits one gap into blank-line-separated blocks (same technique
    text_blocks.py uses file-wide), rebased onto the real file's absolute
    line numbers."""
    gap_lines = lines[start - 1 : end]
    offset = start - 1
    return [
        SymbolInfo(
            name=b.name,
            symbol_type="module_statement",
            start_line=b.start_line + offset,
            end_line=b.end_line + offset,
            docstring_first_line=None,
            indent=0,
        )
        for b in _blocks_from_blank_lines(gap_lines)
    ]


def index_module_level_blocks(source: str, language: str) -> List[SymbolInfo]:
    """Every contiguous non-blank region not covered by a function/class
    span -- preamble, gaps between symbols, trailing content. Each gap is
    chunked into blank-line-separated blocks since one gap can hold
    several independently-targetable statements."""
    lines = source.splitlines()
    if not lines:
        return []
    symbols = index_symbols(source, language)
    covered = set()
    for sym in symbols:
        covered.update(range(sym.start_line, sym.end_line + 1))

    blocks: List[SymbolInfo] = []
    gap_start: Optional[int] = None
    for i in range(1, len(lines) + 1):
        if i not in covered:
            if gap_start is None:
                gap_start = i
        elif gap_start is not None:
            blocks.extend(_chunk_gap(lines, gap_start, i - 1))
            gap_start = None
    if gap_start is not None:
        blocks.extend(_chunk_gap(lines, gap_start, len(lines)))
    return blocks


def _to_repo_symbols(blocks: List[SymbolInfo], source: str) -> List[RepoSymbol]:
    lines = source.splitlines()
    return [
        RepoSymbol(
            file="",
            name=b.name,
            symbol_type="module_statement",
            start_line=b.start_line,
            end_line=b.end_line,
            docstring_first_line=None,
            source="\n".join(lines[b.start_line - 1 : b.end_line]),
            language="text",
        )
        for b in blocks
    ]


def locate_module_level_block(
    source: str, user_request: str, language: str, use_hybrid: bool = False
) -> Optional[SymbolInfo]:
    """Best-matching module-level block for a request, or None on no
    confident match (including a tie) -- an ambiguous match falls through
    to the refused whole-file path rather than guessing.

    Scored by body word overlap (same technique as
    analyzer.locator.locate_candidates_by_body). `use_hybrid=True` fuses
    that with BM25 (and vector retrieval, if available) over every
    module-level block, same as text_blocks.py's hybrid mode."""
    blocks = index_module_level_blocks(source, language)
    if not blocks:
        return None

    lines = source.splitlines()
    target_words = {w for w in _words(user_request) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS if len(w) >= 3}
    if not target_words:
        return None

    scored = []
    for block in blocks:
        body = "\n".join(lines[block.start_line - 1 : block.end_line])
        hits = len(target_words & _words(body))
        if hits:
            scored.append((hits, block))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not use_hybrid:
        if not scored:
            return None
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None
        return scored[0][1]

    repo_blocks = _to_repo_symbols(blocks, source)
    by_key = {(rb.name, rb.start_line): b for rb, b in zip(repo_blocks, blocks)}
    name_repo_key = {(b.name, b.start_line) for _, b in scored}
    name_ranking = [(rb, 0.0) for rb in repo_blocks if (rb.name, rb.start_line) in name_repo_key]

    wide_k = max(len(blocks), 2)
    rankings = [name_ranking, BM25Retriever(repo_blocks).rank(user_request, top_k=wide_k)]
    vector_ranking = _safe_vector_rank(repo_blocks, user_request, wide_k)
    if vector_ranking:
        rankings.append(vector_ranking)

    fused = fuse(rankings, top_k=wide_k)
    if not fused:
        return None
    if len(fused) > 1 and fused[0][1] == fused[1][1]:
        return None
    winner_key = (fused[0][0].name, fused[0][0].start_line)
    return by_key.get(winner_key)
