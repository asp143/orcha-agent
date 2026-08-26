"""Prompt behavior for repository memory files."""

from __future__ import annotations

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="memory", version="1.0.0")

_MEMORY_PROMPT = (
    "Repository memory files are authoritative working context. Follow their "
    "instructions throughout the session, and when instructions differ, prefer the "
    "file closest to the path being worked on. Do not modify a memory file unless the "
    "user explicitly asks you to."
)


def register(api: PluginAPI) -> None:
    api.system_prompt_fragment(_MEMORY_PROMPT, priority=50)
