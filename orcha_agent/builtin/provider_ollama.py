"""Ollama chat-model provider."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec, ProviderCaps

PLUGIN = PluginSpec(name="provider_ollama", version="1.0.0")
_INSTALL_HINT = "pip install langchain-ollama"


def _available() -> str | None:
    return None if find_spec("langchain_ollama") is not None else _INSTALL_HINT


def _factory(model_name: str, config: Mapping[str, Any]) -> Any:
    from langchain_ollama import ChatOllama

    options = dict(config)
    options["model"] = model_name
    return ChatOllama(**options)


def register(api: PluginAPI) -> None:
    api.add_provider(
        "ollama",
        _factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=True,
            max_context=None,
        ),
        env_keys=("OLLAMA_HOST",),
        foreign_block_types=(),
        available=_available,
    )
