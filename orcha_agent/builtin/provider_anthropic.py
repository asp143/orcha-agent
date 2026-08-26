"""Anthropic chat-model provider."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec, ProviderCaps

PLUGIN = PluginSpec(name="provider_anthropic", version="1.0.0")
_INSTALL_HINT = "pip install langchain-anthropic"


def _available() -> str | None:
    return None if find_spec("langchain_anthropic") is not None else _INSTALL_HINT


def _factory(model_name: str, config: Mapping[str, Any]) -> Any:
    from langchain_anthropic import ChatAnthropic

    options = dict(config)
    if options.get("thinking") == "adaptive":
        options["thinking"] = {"type": "adaptive"}
    options["model"] = model_name
    return ChatAnthropic(**options)


def register(api: PluginAPI) -> None:
    api.add_provider(
        "anthropic",
        _factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=True,
            max_context=200_000,
        ),
        env_keys=("ANTHROPIC_API_KEY",),
        foreign_block_types=("thinking",),
        available=_available,
    )
