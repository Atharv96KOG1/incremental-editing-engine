"""Ties hybrid retrieval together: repo index -> symbol/BM25/vector/
Semgrep-structural retrieval -> fusion -> risk + confidence + evidence.
Picks which file/symbol; the existing single-file pipeline takes over
unchanged once a file is found.

Semgrep is a meaning check, not a keyword matcher -- it confirms a
candidate's code actually does what the request describes, only
narrowing among candidates another signal already surfaced.
"""

from typing import Optional

from .bm25_retriever import BM25Retriever
from .confidence import retrieval_confidence
from .dependency_graph import build_call_graph
from .fusion import fuse
from .graph_rank import graph_centrality_ranking
from .joern_graph import build_call_graph_via_joern
from .repo_index import build_repo_index
from .risk import classify_risk
from .semgrep_refs import find_call_sites, structural_match_score
from .symbol_retriever import SymbolRetriever
from .vector_retriever import VectorRetriever

_WIDE_MULTIPLIER = 3

_LOW_CONFIDENCE_THRESHOLD = 0.6


def _safe_vector_ranking(symbols, request: str, top_k: int) -> list:
    """Vector retrieval is one signal among several; a failure (embeddings
    gateway cold start -- observed taking 8+s vs a sub-second baseline, or
    a network blip) must degrade to no signal rather than crash the call."""
    try:
        return VectorRetriever(symbols).rank(request, top_k=top_k)
    except Exception:
        return []


def locate(
    project_dir: str,
    request: str,
    use_vector: bool = True,
    use_semgrep: bool = True,
    use_joern=False,
    top_k: int = 5,
) -> dict:
    """`use_joern` is tri-state: False never runs Joern, True always does,
    "auto" only pays Joern's real build cost (~12-45s+) when computed
    confidence is low. Accepts the string forms too, so a CLI flag or web
    select can drive it directly."""
    use_joern = {"true": True, "false": False, "auto": "auto"}.get(str(use_joern).lower(), use_joern)
    symbols = build_repo_index(project_dir)
    if not symbols:
        return {"candidates": [], "confidence": 0.0, "evidence": [], "symbols_indexed": 0}

    wide_k = top_k * _WIDE_MULTIPLIER
    native_call_graph = build_call_graph(symbols)
    rankings = [
        SymbolRetriever(symbols).rank(request, top_k=wide_k),
        BM25Retriever(symbols).rank(request, top_k=wide_k),
    ]
    if use_vector:
        vector_ranking = _safe_vector_ranking(symbols, request, wide_k)
        if vector_ranking:
            rankings.append(vector_ranking)
    graph_ranking = graph_centrality_ranking(symbols, request, call_graph=native_call_graph, top_k=wide_k)
    if graph_ranking:
        rankings.append(graph_ranking)

    if use_semgrep:
        shortlist = fuse(rankings, top_k=wide_k)
        structural_ranking = sorted(
            (
                (sym, structural_match_score(sym.source, request, language=sym.language))
                for sym, _ in shortlist
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        structural_ranking = [(sym, score) for sym, score in structural_ranking if score > 0]
        if structural_ranking:
            rankings.append(structural_ranking)

    fused = fuse(rankings, top_k=top_k)
    confidence = retrieval_confidence(fused)

    should_use_joern = use_joern is True or (use_joern == "auto" and confidence < _LOW_CONFIDENCE_THRESHOLD)
    joern_call_graph = build_call_graph_via_joern(project_dir, symbols) if should_use_joern else None
    used_joern = joern_call_graph is not None
    call_graph = joern_call_graph if used_joern else native_call_graph

    candidates = []
    evidence = []
    for sym, score in fused:
        key = f"{sym.file}::{sym.name}"
        callers = find_call_sites(project_dir, sym.name, language=sym.language) if use_semgrep else []
        candidates.append(
            {
                "file": sym.file,
                "symbol": sym.name,
                "symbol_type": sym.symbol_type,
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "fused_score": round(score, 4),
                "risk": classify_risk(sym),
                "calls": call_graph["calls"].get(key, []),
                "called_by_count": len(call_graph["called_by"].get(sym.name, [])),
                "semgrep_call_sites": callers[:10],
            }
        )
        evidence.append(
            {
                "fact": f"'{sym.name}' ({sym.symbol_type}) defined in {sym.file}, lines {sym.start_line}-{sym.end_line}",
                "source": {"type": "symbol_index", "file": sym.file, "line": sym.start_line},
                "confidence": confidence,
            }
        )

    return {
        "candidates": candidates,
        "confidence": confidence,
        "evidence": evidence,
        "symbols_indexed": len(symbols),
        "used_joern": used_joern,
    }


def locate_best_file(project_dir: str, request: str, use_vector: bool = False, wide_k: int = 5) -> Optional[str]:
    """Fast 'which file' lookup for auto-locating an edit target -- skips
    semgrep and the dependency graph, both pure enrichment for `iee find`'s
    output (semgrep's per-candidate subprocess spawn is the pipeline's
    largest single cost, ~1-2+s each).

    Vector retrieval defaults off too, keeping the hot edit path free of an
    embeddings-API round trip; pass use_vector=True when precision matters
    more than latency."""
    symbols = build_repo_index(project_dir)
    if not symbols:
        return None

    rankings = [
        SymbolRetriever(symbols).rank(request, top_k=wide_k),
        BM25Retriever(symbols).rank(request, top_k=wide_k),
    ]
    if use_vector:
        vector_ranking = _safe_vector_ranking(symbols, request, wide_k)
        if vector_ranking:
            rankings.append(vector_ranking)

    fused = fuse(rankings, top_k=1)
    return fused[0][0].file if fused else None
