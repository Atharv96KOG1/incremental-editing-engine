"""Reciprocal Rank Fusion across the symbol/BM25/vector retrievers
(PHOENIX doc section 4: "Retrieval maximizes recall; reranking maximizes
precision").

Any one signal alone can mislead: BM25 misses synonyms and paraphrases,
vector search can drift toward vaguely-similar-sounding but wrong code,
literal name matching only works when the request actually names the
symbol. Fusing *rankings* (not raw scores, which live on incomparable
scales -- a BM25 score and a cosine similarity mean nothing next to each
other) is what makes combining them robust.
"""

from typing import List, Tuple

from .repo_index import RepoSymbol

_RRF_K = 60


def _key(sym: RepoSymbol) -> tuple:
    return (sym.file, sym.symbol_type, sym.name, sym.start_line)


def fuse(rankings: List[List[Tuple[RepoSymbol, float]]], top_k: int = 5) -> List[Tuple[RepoSymbol, float]]:
    """`rankings` is one ranked (symbol, score) list per retriever. Returns
    the fused top_k, each symbol's fused score being the sum of
    1/(k + rank) across every retriever that surfaced it -- so a symbol
    every retriever agrees on outranks one only a single retriever liked."""
    fused_scores = {}
    symbol_by_key = {}

    for ranking in rankings:
        for rank, (sym, _score) in enumerate(ranking, start=1):
            key = _key(sym)
            symbol_by_key[key] = sym
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(symbol_by_key[key], score) for key, score in ordered[:top_k]]
