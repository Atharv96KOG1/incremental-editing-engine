"""Rule-based filler compression for our own static system-prompt text only --
never code, docstrings, symbol names, or user request text. Protected words
(negations/absolutes) must survive every compression; `compress()` enforces
this by raising rather than silently dropping one."""

import re
from collections import Counter

PROTECTED_WORDS = frozenset(
    {
        "not", "no", "never", "only", "except", "must", "always",
        "none", "unless", "cannot", "n't",
    }
)

_DROP_PATTERNS = [
    re.compile(r"\b(a|an|the)\b\s+", re.IGNORECASE),
    re.compile(r"\b(just|really|basically|actually|simply|essentially)\b\s*", re.IGNORECASE),
]

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _protected_word_counts(text: str) -> Counter:
    words = _WORD_RE.findall(text.lower())
    counts = Counter(w for w in words if w in PROTECTED_WORDS or w.endswith("n't"))
    return counts


def compress(text: str) -> str:
    """Drops filler and collapses whitespace. Raises if any protected word's
    count changed -- a silently dropped "not" is a correctness bug, not an
    optimization."""
    compressed = text
    for pattern in _DROP_PATTERNS:
        compressed = pattern.sub("", compressed)
    compressed = re.sub(r"[ \t]{2,}", " ", compressed).strip()

    before = _protected_word_counts(text)
    after = _protected_word_counts(compressed)
    assert after == before, (
        f"compress() would have changed protected-word counts ({before} -> {after}) -- "
        "refusing to return a prompt that might mean something different"
    )
    return compressed


def words_preserved(original: str, compressed: str) -> bool:
    """True if every protected word appears the same number of times in both
    strings. Exposed separately so tests can assert on it directly."""
    return _protected_word_counts(original) == _protected_word_counts(compressed)
