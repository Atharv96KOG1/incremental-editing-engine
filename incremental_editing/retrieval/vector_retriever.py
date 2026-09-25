"""Vector/semantic retrieval over the repo-wide symbol index (PHOENIX doc
section 4: "Vector retrieval is strong for semantic intent").

Embeddings come from the same Bifrost gateway already used for GPT-5.4 --
no local embedding model, no vector database. Cosine similarity against
an in-memory numpy array is plenty for a single-repo index; this is
explicitly not built to scale to a multi-million-symbol corpus.
"""

import hashlib
from typing import List

import numpy as np

from ..config import get_settings
from .repo_index import RepoSymbol

# content-hash -> normalized embedding vector, process-lifetime. A repo's
# symbols are mostly unchanged between requests, so a long-running process
# (`iee serve`) that re-embeds the whole index on every single call is
# paying real API latency for the same vectors over and over -- only
# symbols whose own text actually changed need a fresh embedding call.
_embedding_cache: dict = {}


def _document_text(sym: RepoSymbol) -> str:
    parts = [sym.name.replace("_", " "), sym.docstring_first_line or "", sym.source[:2000]]
    return "\n".join(p for p in parts if p)


_EMBED_TIMEOUT_SECONDS = 20.0  # a live edit/search request is waiting on this -- a slow or half-broken
# embeddings endpoint must fail fast, not silently eat a minute-plus (observed: 72s with the client's
# default ~600s timeout + retries) before whatever called this can fall back to a non-vector signal.
# 20s, not tighter: normal calls measured 0.2-0.7s, but a real gateway cold start after idle was
# observed taking just over 8s on its own -- an 8s cap turned that legitimate (if slow) cold start
# into a guaranteed failure instead of only catching genuine multi-minute hangs. Every caller of this
# already treats a failure here as "signal unavailable, degrade gracefully" (see locate_repo.py's
# _safe_vector_ranking and locator.py's _semantic_tiebreak), so the cost of erring generous is a
# slightly slower single call, not a stuck request.


def _embed_raw(texts: List[str]) -> np.ndarray:
    from openai import OpenAI  # lazy import: only needed when a live call is made

    settings = get_settings()
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=_EMBED_TIMEOUT_SECONDS,
        max_retries=0,
    )
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed(texts: List[str]) -> np.ndarray:
    """Same normalized embeddings as before, but only ever calls the API
    for text this process hasn't already embedded."""
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [i for i, h in enumerate(hashes) if h not in _embedding_cache]
    if missing:
        fresh = _embed_raw([texts[i] for i in missing])
        for idx, vec in zip(missing, fresh):
            _embedding_cache[hashes[idx]] = vec
    return np.array([_embedding_cache[h] for h in hashes], dtype=np.float32)


class VectorRetriever:
    def __init__(self, symbols: List[RepoSymbol]):
        self.symbols = symbols
        self._vectors = _embed([_document_text(s) for s in symbols]) if symbols else None

    def rank(self, query: str, top_k: int = 10) -> List[tuple]:
        """Returns [(RepoSymbol, cosine_similarity), ...] sorted best-first."""
        if self._vectors is None or len(self.symbols) == 0:
            return []
        query_vec = _embed([query])[0]
        similarities = self._vectors @ query_vec
        order = np.argsort(-similarities)[:top_k]
        return [(self.symbols[i], float(similarities[i])) for i in order]
