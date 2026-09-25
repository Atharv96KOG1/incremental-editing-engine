"""Multi-dimension confidence (PHOENIX doc section 22): never one global
score. Retrieval confidence here comes from how far the top fused
candidate's score is ahead of the runner-up -- a clear winner reads as
high confidence, a near-tie reads as low, regardless of the absolute
score value (which the RRF formula makes hard to interpret on its own).
"""

from typing import List, Tuple

from .repo_index import RepoSymbol


def retrieval_confidence(fused: List[Tuple[RepoSymbol, float]]) -> float:
    if not fused:
        return 0.0
    if len(fused) == 1:
        return 0.9
    top, second = fused[0][1], fused[1][1]
    if top <= 0:
        return 0.0
    margin = (top - second) / top
    return round(min(0.5 + margin * 0.5, 0.99), 2)
