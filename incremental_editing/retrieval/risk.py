"""Risk classification (PHOENIX doc section 20): LOW / MEDIUM / HIGH.
Heuristic, not learned -- keyword and structural signals only, used to
decide how much validation a change should require before it's trusted.
"""

import re

from .repo_index import RepoSymbol

_HIGH_RISK_KEYWORDS = re.compile(
    r"\b(auth|password|secret|token|crypto|encrypt|decrypt|serialize|deserialize|"
    r"pickle|thread|lock|concurrent|async|session|permission|admin|sql|query)\b",
    re.IGNORECASE,
)


def classify_risk(sym: RepoSymbol) -> str:
    text = f"{sym.name} {sym.source}"
    if _HIGH_RISK_KEYWORDS.search(text):
        return "HIGH"
    if sym.symbol_type == "class" and not sym.name.startswith("_"):
        return "HIGH"  # public API surface -- doc section 20 lists this as HIGH
    if sym.name.startswith("_"):
        return "LOW"
    return "MEDIUM"
