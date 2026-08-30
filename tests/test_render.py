from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from orcha_agent.builtin import render_default
from orcha_agent.core.events import EventBus, ModelChunk
from orcha_agent.core.plugin import Handled, PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui import app
from orcha_agent.tui.blocks import DEFAULT_THEME
from orcha_agent.tui.console import ConsoleOutput
from orcha_agent.tui.frame import Block


def _api(
    registry: Registry,
    bus: EventBus,
    *,
    config: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> PluginAPI:
    return PluginAPI(
        name="render_default",
        registry=registry,
        bus=bus,
        config={} if config is None else config,
        state={} if state is None else state,
        request_rebuild=lambda: None,
    )


class _CapturingConsole:
    def __init__(self) -> None:
        self.output = StringIO()
        self._delegate = ConsoleOutput(
            Console(
                file=self.output,
                force_terminal=False,
                color_system=None,
                width=120,
            )
        )

    def print(self, *objects: object, **kwargs: Any) -> None:
        self._delegate.print(*objects, **kwargs)

    def error(self, message: str) -> None:
        self._delegate.error(message)


@pytest.mark.asyncio
async def test_thinking_command_persists_session_toggle() -> None:
    registry = Registry()
    bus = EventBus()
    renderer_state: dict[str, Any] = {}
    anthropic_state: dict[str, Any] = {}
    render_default.register(
        _api(
            registry,
            bus,
            config={"thinking": "all"},
            state=renderer_state,
        )
    )
    persisted: list[dict[str, dict[str, Any]]] = []
    rebuilds: list[None] = []

    async def rebuild() -> None:
        rebuilds.append(None)

    ctx = SimpleNamespace(
        registry=registry,
        bus=bus,
        console=_CapturingConsole(),
        plugin_states={
            "render_default": renderer_state,
            "provider_anthropic": anthropic_state,
        },
        persist_plugin_states=lambda: persisted.append(
            {name: dict(state) for name, state in ctx.plugin_states.items()}
        ),
        rebuild=rebuild,
    )

    assert await app.dispatch_command(registry, ctx, "/thinking off") is True
    assert await app.dispatch_command(registry, ctx, "/thinking on") is True

    assert renderer_state == {"thinking": "summary"}
    assert anthropic_state == {"thinking": "summary"}
    assert persisted == [
        {
            "render_default": {"thinking": "off"},
            "provider_anthropic": {"thinking": "off"},
        },
        {
            "render_default": {"thinking": "summary"},
            "provider_anthropic": {"thinking": "summary"},
        },
    ]
    assert rebuilds == [None, None]


def test_builtin_thinking_renderer_observes_plugin_state() -> None:
    registry = Registry()
    bus = EventBus()
    state = {"thinking": "off"}
    render_default.register(_api(registry, bus, state=state))
    renderer = next(entry.render for entry in registry.block_renderers if entry.kind == "thinking")
    value = Block(
        id="thinking",
        kind="thinking",
        data={
            "text": "private plan",
            "role": "main",
            "reasoning_tokens": 12,
            "tokens_per_second": 6,
        },
    )

    hidden = renderer(value, DEFAULT_THEME, 80, 3, False)
    state["thinking"] = "summary"
    visible = renderer(value, DEFAULT_THEME, 80, 3, False)

    assert "private plan" not in str(hidden)
    output = StringIO()
    Console(file=output, force_terminal=False, width=80).print(visible)
    assert "private plan" in output.getvalue()


@pytest.mark.asyncio
async def test_priority_handler_returning_handled_suppresses_rendering() -> None:
    registry = Registry()
    bus = EventBus()
    render_default.register(_api(registry, bus))
    handler_calls: list[str] = []

    async def handle(_event: ModelChunk) -> Handled:
        handler_calls.append("handled")
        return Handled()

    bus.on(ModelChunk, handle, priority=0)
    console = _CapturingConsole()
    ctx = SimpleNamespace(registry=registry, bus=bus, console=console)
    event = ModelChunk(chunk="must not render", role="main")

    was_handled = await app._render(ctx, event)

    assert was_handled is True
    assert handler_calls == ["handled"]
    assert console.output.getvalue() == ""
