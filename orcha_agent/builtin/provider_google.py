"""Google Generative AI chat-model provider."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec, ProviderCaps

PLUGIN = PluginSpec(name="provider_google", version="1.0.0")
_INSTALL_HINT = "pip install langchain-google-genai"


def _available() -> str | None:
    return None if find_spec("langchain_google_genai") is not None else _INSTALL_HINT


def _factory(model_name: str, config: Mapping[str, Any]) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    options = dict(config)
    options["model"] = model_name
    return ChatGoogleGenerativeAI(**options)


def register(api: PluginAPI) -> None:
    api.add_provider(
        "google",
        _factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=True,
            max_context=1_000_000,
        ),
        env_keys=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        foreign_block_types=("thinking", "thought"),
        available=_available,
    )
