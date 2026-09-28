"""LLM-free locator for MODULE-LEVEL content in a language that DOES have
a function/class concept -- content Delta IR structurally can't reach at
all (REPLACE/INSERT/DELETE only ever target a function or class symbol),
because it sits *between*, *before*, or *after* every real symbol: a
module-level constant or config list, a bare statement, an
`if __name__ == "__main__":` guard, a standalone comment.

Real gap this closes: before this existed, a request touching only this
kind of content had no narrower path at all -- the model's own
escalate:{"kind":"whole_file"} was the only way to express it, and that
path is refused for edits by policy (run_pipeline.py's
_run_whole_file_edit). Confirmed live: "remove gpt" against a file whose
only "gpt" text lives inside a module-level `MODEL_SLOTS = [...]` list
(never inside any function/class body) found zero symbol candidates,
saw zero bodies, and correctly escalated -- with nothing narrower to
fall back to before this module existed.

Closes it the same way analyzer/text_blocks.py already closes the
equivalent gap for languages with NO symbol concept at all: locate the
one relevant region, send only that as input and expected output,
splice it back by line range -- never the whole file.
"""

from typing import List, Optional

from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.fusion import fuse
from ..retrieval.repo_index import RepoSymbol
from .locator import _GENERIC_LEADING_VERBS, _GENERIC_TYPE_WORDS, SymbolInfo, _words, index_symbols
from .text_blocks import _blocks_from_blank_lines, _safe_vector_rank


def _chunk_gap(lines: List[str], start: int, end: int) -> List[SymbolInfo]:
    """Splits one gap's own lines into blank-line-separated blocks (the
    same universal technique text_blocks.py already uses for a whole
    file), re-based from that slice's own 1-indexed positions back onto
    the real file's absolute line numbers."""
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
    """Every contiguous region of real (non-blank) lines NOT covered by
    any function/class's own span -- the module-level preamble, gaps
    between symbols, and any trailing content after the last one. Each
    gap is itself chunked into blank-line-separated paragraphs, since a
    single gap can hold several unrelated module-level statements (an
    import block, a constant, a `__main__` guard) that should stay
    independently targetable, not one giant region."""
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
    """The single best-matching module-level block for a request, or
    None when nothing scores confidently (including a tie) -- an
    ambiguous match must fall through to the existing (refused)
    whole-file path rather than guessing which region was meant, same
    "refuse over guess" rule every other locator in this project already
    holds itself to.

    Scored by body-content word overlap (the block has no name worth
    matching against the way a markdown heading does -- just its own
    first line, truncated) -- the same technique
    analyzer.locator.locate_candidates_by_body already uses for a
    request word that only appears inside a symbol's body, applied here
    to content that isn't inside any symbol's body at all.

    `use_hybrid=True` fuses that word-overlap ranking with BM25 (and
    vector retrieval, if it's available) over every module-level block
    in the file -- same reasoning text_blocks.py's own hybrid mode
    already applies."""
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
