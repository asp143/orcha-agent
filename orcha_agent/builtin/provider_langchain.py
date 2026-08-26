"""Generic LangChain chat-model provider."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec, ProviderCaps

PLUGIN = PluginSpec(name="provider_langchain", version="1.0.0")
_INSTALL_HINT = "pip install langchain"


def _available() -> str | None:
    return None if find_spec("langchain") is not None else _INSTALL_HINT


def _factory(model_name: str, config: Mapping[str, Any]) -> Any:
    from langchain.chat_models import init_chat_model

    options = dict(config)
    return init_chat_model(model_name, **options)


def register(api: PluginAPI) -> None:
    api.add_provider(
        "langchain",
        _factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
        env_keys=(),
        foreign_block_types=("thinking", "reasoning", "thought"),
        available=_available,
    )
