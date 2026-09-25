"""BM25 keyword retrieval over the repo-wide symbol index (PHOENIX doc
section 4: "BM25 is strong for exact identifiers and rare terms").

Each symbol becomes one BM25 "document": its name, docstring, and its own
source text, tokenized. This is complementary to the AST-based literal
name matching in `analyzer/locator.py` -- BM25 additionally rewards a
request that shares *rare, specific* words with a symbol's body even when
it never mentions the symbol's name at all.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi

from .repo_index import RepoSymbol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    # snake_case and camelCase both split into their word parts, so
    # "build_knowledge_base" and "BuildKnowledgeBase" tokenize the same way.
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
