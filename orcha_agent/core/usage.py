"""Token usage pricing shared by runtime accounting surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "codex:gpt-5.6-sol": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-luna": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-terra": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.5": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4-mini": {"input": 1.5, "output": 6, "cache_read": 0.15},
    "codex:gpt-5.3-codex-spark": {
        "input": 1.5,
        "output": 6,
        "cache_read": 0.15,
    },
    "anthropic:claude-opus-5": {"input": 15, "output": 75, "cache_read": 1.5},
    "anthropic:claude-sonnet-5": {"input": 3, "output": 15, "cache_read": 0.3},
    "anthropic:claude-haiku-4-5": {"input": 0.8, "output": 4, "cache_read": 0.08},
}


def usage_cost(
    model: str,
    usage: Mapping[str, Any],
    configured: Mapping[str, Mapping[str, float]],
) -> float:
    """Return this usage record's cost in dollars."""

    price = {**DEFAULT_PRICING.get(model, {}), **configured.get(model, {})}
    if not price:
        return 0.0
    inputs = float(usage.get("input_tokens", 0) or 0)
    outputs = float(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_token_details", {})
    if not isinstance(details, Mapping):
        details = {}
    reads = float(
        details.get("cache_read", details.get("cache_read_input_tokens", 0)) or 0
    )
    writes = float(
        details.get(
            "cache_creation",
            details.get("cache_write", details.get("cache_creation_input_tokens", 0)),
        )
        or 0
    )
    uncached = max(0.0, inputs - reads - writes)
    return (
        uncached * float(price.get("input", 0))
        + reads * float(price.get("cache_read", price.get("input", 0)))
        + writes * float(price.get("cache_write", price.get("input", 0)))
        + outputs * float(price.get("output", 0))
    ) / 1_000_000
