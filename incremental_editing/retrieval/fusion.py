"""Reciprocal Rank Fusion across the symbol/BM25/vector retrievers. Fuses
*rankings*, not raw scores -- a BM25 score and a cosine similarity live on
incomparable scales.
"""

from typing import List, Tuple

from .repo_index import RepoSymbol

_RRF_K = 60


def _key(sym: RepoSymbol) -> tuple:
    return (sym.file, sym.symbol_type, sym.name, sym.start_line)


def fuse(rankings: List[List[Tuple[RepoSymbol, float]]], top_k: int = 5) -> List[Tuple[RepoSymbol, float]]:
    """`rankings` is one (symbol, score) list per retriever. Fused score is the
    sum of 1/(k + rank) across retrievers, so agreement across retrievers
    outranks a single strong signal."""
    fused_scores = {}
    symbol_by_key = {}

    for ranking in rankings:
        for rank, (sym, _score) in enumerate(ranking, start=1):
            key = _key(sym)
            symbol_by_key[key] = sym
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(symbol_by_key[key], score) for key, score in ordered[:top_k]]
