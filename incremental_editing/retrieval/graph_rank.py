"""Personalized PageRank over the repo's own call graph -- a structural
"how central/connected is this symbol" signal, complementing the purely
textual BM25/vector/name-match signals hybrid retrieval already fuses.

Adapted from Aider's own repo-map ranking (`aider/repomap.py`'s
`get_ranked_tags`: a personalized PageRank over a referencer -> definer
graph, weighted by identifier "interestingness" and boosted for names
the request itself mentions) -- reimplemented here at symbol granularity
over this project's own existing call graph (`dependency_graph.py`)
instead of a second tags/graph pass, and with a plain NumPy power
iteration instead of adding `networkx` as a new dependency (NumPy is
already one; this project's own established preference is a small
dependency footprint, see requirements.txt).

Real motivation this closes: a vague request ("fix the auth bug") can
textually match nothing in particular, but the actual, heavily-called
`authenticate()` function is exactly what a human would reach for --
structural centrality (how many other symbols call this one, directly
or transitively) is a real, different signal name/BM25/vector matching
can't provide on its own, and this project already computes the call
graph they'd need anyway (`retrieval/dependency_graph.py`, built for
`iee find`'s own "calls"/"called_by" evidence)."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..analyzer.locator import _GENERIC_LEADING_VERBS, _GENERIC_TYPE_WORDS, _words
from .dependency_graph import build_call_graph
from .repo_index import RepoSymbol


def _personalized_pagerank(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
    personalization: Optional[Dict[str, float]] = None,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Standard personalized-PageRank power iteration. `edges` is
    (referencer, definer, weight): rank flows FROM a referencer TO
    whatever it references -- same direction Aider's own repo-map graph
    uses, so a symbol many things call accumulates rank from all of
    them, not the other way around."""
    n = len(nodes)
    if n == 0:
        return {}
    idx = {name: i for i, name in enumerate(nodes)}
    m = np.zeros((n, n))
    out_weight = np.zeros(n)
    for src, dst, weight in edges:
        if src not in idx or dst not in idx:
            continue
        m[idx[dst], idx[src]] += weight
        out_weight[idx[src]] += weight
    for j in range(n):
        if out_weight[j] > 0:
            m[:, j] /= out_weight[j]
        else:
            m[:, j] = 1.0 / n

    if personalization:
        p = np.array([personalization.get(name, 0.0) for name in nodes])
        p = p / p.sum() if p.sum() > 0 else np.ones(n) / n
    else:
        p = np.ones(n) / n

    rank = np.ones(n) / n
    for _ in range(max_iter):
        new_rank = damping * m.dot(rank) + (1 - damping) * p
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    return {name: float(rank[i]) for i, name in enumerate(nodes)}


def graph_centrality_ranking(
    repo_symbols: List[RepoSymbol],
    user_request: str,
    call_graph: Optional[dict] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[RepoSymbol, float]]:
    """Duck-typed the same way BM25Retriever.rank()/VectorRetriever.rank()
    already are -- [(symbol, score), ...] ranked highest first -- so it
    drops straight into an existing fuse([...]) call alongside them.

    `call_graph` lets a caller that already built one (e.g. locate_repo.
    locate(), which needs it anyway for candidate evidence) pass it in
    rather than paying a second identical AST/Tree-sitter pass.

    Personalization boosts any symbol whose own name is a real content
    word in the request -- the same word-overlap technique locate_
    candidates already uses, not a new heuristic -- the same role
    Aider's own "mentioned_idents" plays in its repo-map ranking: without
    it, PageRank alone would just re-rank by raw call-graph popularity,
    ignoring the request's own wording entirely."""
    if len(repo_symbols) < 2:
        return []

    graph = call_graph if call_graph is not None else build_call_graph(repo_symbols)
    keys = [f"{s.file}::{s.name}" for s in repo_symbols]
    by_key = dict(zip(keys, repo_symbols))
    name_to_keys: Dict[str, List[str]] = {}
    for s in repo_symbols:
        name_to_keys.setdefault(s.name, []).append(f"{s.file}::{s.name}")

    edges: List[Tuple[str, str, float]] = []
    for caller_key, callee_names in graph["calls"].items():
        for callee_name in callee_names:
            for definer_key in name_to_keys.get(callee_name, []):
                if definer_key != caller_key:
                    edges.append((caller_key, definer_key, 1.0))

    target_words = {w for w in _words(user_request) - _GENERIC_LEADING_VERBS - _GENERIC_TYPE_WORDS if len(w) >= 3}
    personalization = {}
    if target_words:
        for key, sym in by_key.items():
            if _words(sym.name.replace("_", " ")) & target_words:
                personalization[key] = 1.0

    ranked = _personalized_pagerank(keys, edges, personalization=personalization or None)
    ordered = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)
    result = [(by_key[key], score) for key, score in ordered if key in by_key]
    return result[:top_k] if top_k else result
