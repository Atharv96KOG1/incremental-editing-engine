"""Per-1K-token pricing for cost metrics.

Defaults below come from the Bifrost gateway's /v1/models pricing field for
openai/gpt-5.4 (prompt $0.0000025, completion $0.000015 per token). Override
via LLM_PRICING_JSON in .env for other models/gateways, e.g.
LLM_PRICING_JSON='{"openai/gpt-5.4": [0.0025, 0.015, 0.00025]}' (the third,
optional value is the cached-input rate -- see below).

Cached-input pricing: verified live against the real gateway (see
structured_edit.py's `_call_llm`) that an identical system+context prefix
across two calls -- exactly what every bounded self-repair retry sends,
since only the failure-specific suffix changes -- gets served largely
from cache on the second call (measured: ~85% of prompt tokens on one
real test). OpenAI's published cached-input rate for gpt-5.4 specifically
is $0.25/1M vs $2.50/1M uncached, a 90% discount -- verified current as
of this writing, not assumed from an older, less specific model's rate.
Charging every repair attempt as if its whole resent prefix were freshly
computed overstates real cost, sometimes substantially.
"""

import json
from functools import lru_cache

from ..config import get_settings

# (input $/1K, output $/1K, cached-input $/1K)
_DEFAULT_PRICING = {
    "openai/gpt-5.4": (0.0025, 0.015, 0.00025),
    "gpt-5.4": (0.0025, 0.015, 0.00025),
}


@lru_cache
def _pricing_table() -> dict:
    table = dict(_DEFAULT_PRICING)
    raw = get_settings().llm_pricing_json
    if raw:
        try:
            overrides = json.loads(raw)
            for model, prices in overrides.items():
                # A 2-element override (no cached rate given) defaults
                # cached to the same as uncached input -- i.e., assume no
                # discount rather than invent one for a model/gateway
                # this project has no verified rate for.
                cached = float(prices[2]) if len(prices) > 2 else float(prices[0])
                table[model] = (float(prices[0]), float(prices[1]), cached)
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
            pass
    return table


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    input_price, output_price, cached_price = _pricing_table().get(model, (0.0, 0.0, 0.0))
    # Never let a caller-reported cached_tokens exceed input_tokens --
    # would silently produce a negative "uncached" count and understate
    # cost instead of overstating it, the opposite of the bug this fixes.
    cached_tokens = max(0, min(cached_tokens, input_tokens))
    uncached_tokens = input_tokens - cached_tokens
    cost = (
        (uncached_tokens / 1000) * input_price
        + (cached_tokens / 1000) * cached_price
        + (output_tokens / 1000) * output_price
    )
    return round(cost, 6)
