"""Risk classification: LOW / MEDIUM / HIGH. Heuristic (keyword/structural
signals), not learned -- decides how much validation a change needs.
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
        return "HIGH"
    if sym.name.startswith("_"):
        return "LOW"
    return "MEDIUM"
