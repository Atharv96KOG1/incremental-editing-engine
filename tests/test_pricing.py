"""Offline tests for cost estimation, including the cached-input token
discount. Verified live against the real gateway (see structured_edit.py)
that an identical system+context prefix across two calls -- exactly what
every bounded self-repair retry sends -- gets served largely from cache
on the second call; these tests lock in that estimate_cost() actually
applies the resulting discount instead of silently ignoring it.
"""

from incremental_editing.benchmark.pricing import estimate_cost


def test_estimate_cost_with_no_cached_tokens_matches_plain_calculation():
    cost = estimate_cost("openai/gpt-5.4", 1000, 100)
    assert cost == round(1 * 0.0025 + 0.1 * 0.015, 6)


def test_estimate_cost_applies_cached_discount_for_gpt_5_4():
    """Real, verified rate for this project's actual model: cached input
    is $0.25/1M vs $2.50/1M uncached (a 90% discount) -- not assumed from
    a different model's rate."""
    all_cached = estimate_cost("openai/gpt-5.4", 1000, 0, cached_tokens=1000)
    all_uncached = estimate_cost("openai/gpt-5.4", 1000, 0, cached_tokens=0)
    assert all_cached < all_uncached
    assert all_cached == round(1 * 0.00025, 6)
    assert all_uncached == round(1 * 0.0025, 6)


def test_estimate_cost_splits_cached_and_uncached_input_correctly():
    # 400 of 1000 input tokens cached, 600 charged at the full rate
    cost = estimate_cost("openai/gpt-5.4", 1000, 0, cached_tokens=400)
    expected = round((600 / 1000) * 0.0025 + (400 / 1000) * 0.00025, 6)
    assert cost == expected


def test_estimate_cost_clamps_cached_tokens_to_input_tokens():
    """A caller-reported cached_tokens greater than input_tokens (should
    never happen, but must not silently produce a negative "uncached"
    count that understates cost) is clamped instead of trusted blindly."""
    clamped = estimate_cost("openai/gpt-5.4", 100, 0, cached_tokens=99999)
    fully_cached = estimate_cost("openai/gpt-5.4", 100, 0, cached_tokens=100)
    assert clamped == fully_cached


def test_estimate_cost_unknown_model_stays_zero_regardless_of_cache():
    assert estimate_cost("some/unknown-model", 1000, 1000, cached_tokens=500) == 0.0
