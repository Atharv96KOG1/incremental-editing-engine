"""BM25 keyword retrieval over the repo-wide symbol index. Each symbol is one
BM25 document (name, docstring, source), tokenized. Complements the AST-based
literal name matching in `analyzer/locator.py` by rewarding shared rare words
even when the symbol's name is never mentioned.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi

from .repo_index import RepoSymbol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return _TOKEN_RE.findall(spaced.lower())


def _document_text(sym: RepoSymbol) -> str:
    parts = [sym.name.replace("_", " "), sym.docstring_first_line or "", sym.source]
    return " ".join(p for p in parts if p)


class BM25Retriever:
    def __init__(self, symbols: List[RepoSymbol]):
        self.symbols = symbols
        self._corpus_tokens = [_tokenize(_document_text(s)) for s in symbols]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def rank(self, query: str, top_k: int = 10) -> List[tuple]:
        """Returns [(RepoSymbol, score), ...] sorted best-first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.symbols, scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked[:top_k] if pair[1] > 0]
