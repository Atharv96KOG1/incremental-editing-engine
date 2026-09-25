"""Rule-based compression for the STATIC instructional text we author
ourselves (system prompts, wrapper phrases around dynamic content) --
drops filler (articles, hedging adverbs, redundant phrasing) the same way
caveman-style compression does for a human reader, but far more
conservative: this text controls model *correctness*, not just brevity,
so a small, explicit, testable set of rules, never a free-form rewrite.

Where this must NEVER be used: actual code (`content` fields, generated
files), docstrings, symbol/variable names, or the user's own request
text. Those aren't filler-laden instructional prose -- they're either
executable code (compressing it would corrupt the program) or meaning
we don't own and have no business rewording. This module only ever
touches strings we wrote ourselves as instructions to the model.

The one hard rule: a fixed set of "protected" words -- negations and
absolutes that flip an instruction's meaning if dropped -- must survive
every single compression, with the same count, every time. `compress()`
enforces this itself (raises rather than silently violating it), and
`words_preserved()` is exposed separately for tests to assert on.
"""

import re
from collections import Counter

# Removing any of these changes what an instruction *means*, not just its
# length -- "do not X" and "do X" are opposites, not a paraphrase of each
# other. Never add a word here that a real prompt needs shortened instead.
PROTECTED_WORDS = frozenset(
    {
        "not", "no", "never", "only", "except", "must", "always",
        "none", "unless", "cannot", "n't",
    }
)

_DROP_PATTERNS = [
    # standalone articles -- "a partial excerpt" -> "partial excerpt"
    re.compile(r"\b(a|an|the)\b\s+", re.IGNORECASE),
    # hedging/filler adverbs that add no instruction content
    re.compile(r"\b(just|really|basically|actually|simply|essentially)\b\s*", re.IGNORECASE),
]

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _protected_word_counts(text: str) -> Counter:
    words = _WORD_RE.findall(text.lower())
    # "isn't"/"doesn't"/"wasn't" etc. all carry a negation via "n't" --
    # count that suffix once per contraction rather than needing every
    # verb+n't combination listed individually in PROTECTED_WORDS.
    counts = Counter(w for w in words if w in PROTECTED_WORDS or w.endswith("n't"))
    return counts


def compress(text: str) -> str:
    """Drops filler from `text` and collapses the whitespace that leaves
    behind. Raises AssertionError if doing so changed the count of any
    protected word -- a compression that silently drops a "not" is a
    correctness bug, not an optimization, so this fails loudly rather
    than shipping a prompt that quietly means something different."""
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
    """True if every protected word appears the same number of times in
    both strings. Exposed separately (rather than only inside compress())
    so tests can assert on it directly, including against text that
    didn't go through compress() at all."""
    return _protected_word_counts(original) == _protected_word_counts(compressed)
