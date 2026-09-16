"""Token usage pricing shared by runtime accounting surfaces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class _CatalogPricing(Mapping[str, dict[str, float]]):
    """Compatibility view for plugins that imported the old default price map."""

    def __getitem__(self, key: str) -> dict[str, float]:
        from .catalog import get_model

        entry = get_model(key)
        if entry is None:
            raise KeyError(key)
        return dict(entry.cost)

    def __iter__(self) -> Iterator[str]:
        from .catalog import get_catalog

        return iter(get_catalog())

    def __len__(self) -> int:
        from .catalog import get_catalog

        return len(get_catalog())


DEFAULT_PRICING: Mapping[str, dict[str, float]] = _CatalogPricing()


def pricing_for(
    model: str,
    configured: Mapping[str, Mapping[str, float]],
    *,
    config: Any = None,
) -> dict[str, float]:
    """Resolve catalog prices in USD per million tokens, then explicit overrides."""
    from .catalog import get_model

    entry = get_model(model, config)
    return {**(entry.cost if entry is not None else {}), **configured.get(model, {})}


def usage_cost(
    model: str,
    usage: Mapping[str, Any],
    configured: Mapping[str, Mapping[str, float]],
    *,
    config: Any = None,
) -> float:
    """Return this usage record's cost in dollars."""

    price = pricing_for(model, configured, config=config)
    if not price:
        return 0.0
    inputs = float(usage.get("input_tokens", 0) or 0)
    outputs = float(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_token_details", {})
    if not isinstance(details, Mapping):
        details = {}
    reads = float(details.get("cache_read", details.get("cache_read_input_tokens", 0)) or 0)
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
