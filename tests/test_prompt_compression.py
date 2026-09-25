"""Offline tests for the reusable prompt-compression utility. The one
property that actually matters: compression must never change what an
instruction *means* -- every negation/conditional word survives, exact
count, every time. compress() enforces this itself (raises rather than
silently violating it); these tests prove that guarantee actually holds,
including against the real SYSTEM_PROMPT this project ships with.
"""

import pytest

from incremental_editing.optimization.prompt_compression import (
    PROTECTED_WORDS,
    compress,
    words_preserved,
)


def test_compress_drops_articles_and_filler():
    text = "This is just a simple test of the compression logic, really."
    compressed = compress(text)
    assert "just" not in compressed.split()
    assert "really" not in compressed.rstrip(".").split()
    assert " a " not in f" {compressed} "
    assert " the " not in f" {compressed} "


def test_compress_never_drops_protected_words():
    text = "Do not invent a name. Only use one if it is never shown in full, except when it is not needed."
    compressed = compress(text)
    assert words_preserved(text, compressed)
    # spot-check the actual words survived, not just the count comparison
    for word in ("not", "only", "never", "except"):
        assert word in compressed.lower().split() or f"{word} " in compressed.lower()


def test_compress_shrinks_real_system_prompt_and_preserves_meaning():
    """The actual prompt this project sends to the model, not a synthetic
    example -- proves the real, shipped compression is safe."""
    from incremental_editing.strategies.structured_edit import SYSTEM_PROMPT, _SYSTEM_PROMPT_SOURCE

    assert len(SYSTEM_PROMPT) < len(_SYSTEM_PROMPT_SOURCE)
    assert words_preserved(_SYSTEM_PROMPT_SOURCE, SYSTEM_PROMPT)
    # the specific rules this prompt depends on for correctness must be intact
    assert "never invent" in SYSTEM_PROMPT.lower()
    assert "redefin" in SYSTEM_PROMPT.lower()  # rule intact regardless of exact phrasing
    assert "if already satisfied" in SYSTEM_PROMPT.lower()


def test_compress_refuses_a_change_that_would_drop_a_protected_word(monkeypatch):
    """If a compression rule ever got greedy enough to eat a protected
    word, compress() must fail loudly (AssertionError) rather than ship a
    prompt that quietly means something different."""
    import incremental_editing.optimization.prompt_compression as module

    bad_pattern = __import__("re").compile(r"\bnot\b\s*")  # deliberately eats a protected word
    monkeypatch.setattr(module, "_DROP_PATTERNS", [bad_pattern])

    with pytest.raises(AssertionError):
        module.compress("Do not do this.")


def test_protected_words_set_is_not_accidentally_empty():
    assert {"not", "never", "only"} <= PROTECTED_WORDS
