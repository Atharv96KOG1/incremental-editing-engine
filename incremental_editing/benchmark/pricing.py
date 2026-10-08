"""Per-1K-token pricing for cost metrics. Defaults are gpt-5.4's Bifrost gateway
rates; override via LLM_PRICING_JSON in .env, e.g.
LLM_PRICING_JSON='{"openai/gpt-5.4": [0.0025, 0.015, 0.00025]}' (third value is
the optional cached-input rate). Cached-input rate matters because repeated
system+context prefixes (e.g. self-repair retries) get served largely from
cache at a 90% discount -- charging full price would overstate real cost.
"""

import json
from functools import lru_cache

from ..config import get_settings

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
                cached = float(prices[2]) if len(prices) > 2 else float(prices[0])
                table[model] = (float(prices[0]), float(prices[1]), cached)
        except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError):
            pass
    return table


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    input_price, output_price, cached_price = _pricing_table().get(model, (0.0, 0.0, 0.0))
    cached_tokens = max(0, min(cached_tokens, input_tokens))
    uncached_tokens = input_tokens - cached_tokens
    cost = (
        (uncached_tokens / 1000) * input_price
        + (cached_tokens / 1000) * cached_price
        + (output_tokens / 1000) * output_price
    )
    return round(cost, 6)
