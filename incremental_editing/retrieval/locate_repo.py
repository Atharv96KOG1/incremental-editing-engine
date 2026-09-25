"""Ties hybrid retrieval together: repo index -> symbol/BM25/vector/
Semgrep-structural retrieval -> fusion -> risk + confidence + evidence.

This is the "which file/symbol" layer that sits in front of the existing
single-file locator (`analyzer/locator.py`): once this picks a file, the
existing single-file pipeline (context_builder, structured_edit,
edit_applier, ...) takes over exactly as it did before -- nothing about
the actual editing changed, only how the target file gets found when the
caller doesn't already know it.

The Semgrep structural signal is a *meaning* check, not another keyword
matcher: text/vector retrieval can only tell you a candidate's words
resemble the request; Semgrep confirms whether the candidate's code
actually *does* what the request describes (raises an exception,
validates types, calls a math function, ...). It only ever adds
candidates a broader first pass already surfaced -- it can't invent one
BM25/vector/symbol retrieval missed entirely.
"""

from typing import Optional

from .bm25_retriever import BM25Retriever
from .confidence import retrieval_confidence
from .dependency_graph import build_call_graph
from .fusion import fuse
from .joern_graph import build_call_graph_via_joern
from .repo_index import build_repo_index
from .risk import classify_risk
from .semgrep_refs import find_call_sites, structural_match_score
from .symbol_retriever import SymbolRetriever
from .vector_retriever import VectorRetriever

_WIDE_MULTIPLIER = 3  # how much broader the first pass casts before structural re-ranking narrows it

# retrieval_confidence()'s own scale (see confidence.py): 0.5 means the top
# candidate and the runner-up are essentially tied, approaching 0.99 means a
# clear, uncontested winner. Below this line, the top pick only barely beat
# the runner-up -- exactly the situation where the more expensive, more
# accurate evidence (a real Joern CPG instead of the name-only native graph)
# is worth its real build cost.
_LOW_CONFIDENCE_THRESHOLD = 0.6


def _safe_vector_ranking(symbols, request: str, top_k: int) -> list:
    """Vector retrieval is one signal among several -- like Semgrep's
    structural check and the locator's semantic tiebreak, a failure here
    (embeddings gateway cold-starting, a network blip, briefly down) must
    degrade to "this signal didn't fire" and let symbol+BM25 carry the
    request, not crash the whole find/locate call. A real gateway cold
    start was observed taking 8+ seconds against a normal sub-second
    baseline -- exactly the kind of transient failure this must absorb
    rather than propagate."""
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
    """`use_joern` is a tri-state, not a plain bool: False (default,
    unchanged from before) never runs Joern; True forces it every call;
    "auto" is the confidence/need gate -- it only pays Joern's real build
    cost (~12-45s+, see build_call_graph_via_joern) when this call's own
    computed confidence is actually low. Accepting the literal string
    "False"/"True"/"auto" (not just a bool) lets this be driven directly
    by a CLI choice argument or a web select without extra translation."""
    use_joern = {"true": True, "false": False, "auto": "auto"}.get(str(use_joern).lower(), use_joern)
    symbols = build_repo_index(project_dir)
    if not symbols:
        return {"candidates": [], "confidence": 0.0, "evidence": [], "symbols_indexed": 0}

    wide_k = top_k * _WIDE_MULTIPLIER
    rankings = [
        SymbolRetriever(symbols).rank(request, top_k=wide_k),
        BM25Retriever(symbols).rank(request, top_k=wide_k),
    ]
    if use_vector:
        vector_ranking = _safe_vector_ranking(symbols, request, wide_k)
        if vector_ranking:
            rankings.append(vector_ranking)

    if use_semgrep:
        # Only worth checking symbols a text/vector signal already thinks
        # are in the running -- fuse the wide pass first, then verify
        # *meaning* on that shortlist instead of every symbol in the repo.
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

    # Confidence/need gate. Important, honest scope: Joern's call graph
    # was never wired into `rankings`/`fused` above -- it can't change
    # *which* candidate wins, only how trustworthy the evidence attached
    # to the candidates already picked is. What auto-escalation buys is
    # spending that real cost (~12-45s+, see build_call_graph_via_joern)
    # exactly when the ranking itself is uncertain (top pick barely beat
    # the runner-up), and never when it's already confident -- rather
    # than either always skipping it (missing real accuracy when it's
    # actually needed) or always paying it (wasting it when the answer
    # was never in doubt).
    should_use_joern = use_joern is True or (use_joern == "auto" and confidence < _LOW_CONFIDENCE_THRESHOLD)
    joern_call_graph = build_call_graph_via_joern(project_dir, symbols) if should_use_joern else None
    used_joern = joern_call_graph is not None
    call_graph = joern_call_graph if used_joern else build_call_graph(symbols)

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
        # True only when the confidence/need gate (or a forced use_joern=True)
        # actually resolved the call graph via a real CPG for this call --
        # lets a caller report "escalated because confidence was low" rather
        # than silently swallowing whether the gate ever fired.
        "used_joern": used_joern,
    }


def locate_best_file(project_dir: str, request: str, use_vector: bool = False, wide_k: int = 5) -> Optional[str]:
    """Fast 'which file' lookup for auto-locating an edit target -- unlike
    `locate()`, this never touches semgrep or the dependency graph.

    Both are pure enrichment `locate()` attaches for `iee find`'s
    exploratory output (call sites, callers, structural verification) --
    never what decides the single best file -- and semgrep spawning a
    subprocess per shortlisted candidate is by far the largest cost in the
    full pipeline (each spawn: a real 1-2+ second cold start). Building the
    repo-wide call graph costs a full extra parse pass over every symbol
    for the same reason: irrelevant when the caller only wants a file path.

    Vector retrieval defaults off too: symbol+BM25 alone already resolves
    most single-file lookups, and skipping it removes a real embeddings-API
    network round trip from the hot edit-request path. Pass use_vector=True
    when precision matters more than latency for a specific call."""
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
