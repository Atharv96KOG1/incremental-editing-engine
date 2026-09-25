"""Literal-name / keyword-overlap retrieval over the repo-wide index --
the same scoring `analyzer/locator.py` already uses for a single file,
applied across every file in the repo (PHOENIX doc section 4: "Symbol
retrieval is strong for definitions and references").
"""

import re
from typing import List, Tuple

from ..analyzer.locator import _words
from .repo_index import RepoSymbol


class SymbolRetriever:
    def __init__(self, symbols: List[RepoSymbol]):
        self.symbols = symbols

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[RepoSymbol, float]]:
        query_lower = query.lower()
        query_words = _words(query)
        scored = []
        for sym in self.symbols:
            score = 0.0
            if re.search(rf"\b{re.escape(sym.name.lower())}\b", query_lower):
                score += 5
            score += len(_words(sym.name.replace("_", " ")) & query_words)
            if sym.docstring_first_line:
                score += len(_words(sym.docstring_first_line) & query_words)
            if score > 0:
                scored.append((sym, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
