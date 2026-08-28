from __future__ import annotations

from orcha_agent.builtin import render_default
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


def test_builtin_registers_native_block_renderers_only() -> None:
    registry = Registry()
    api = PluginAPI(
        name="render_default",
        config={},
        state={},
        registry=registry,
        bus=EventBus(),
        request_rebuild=lambda: None,
    )

    render_default.register(api)

    assert {entry.kind for entry in registry.block_renderers} == {
        "user",
        "assistant",
        "thinking",
        "tool",
        "diff",
        "banner",
        "marker",
        "todo",
        "subagents",
    }
    assert registry.renderers == []
