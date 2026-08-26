from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.app import dispatch_command
from orcha_agent.tui.console import ConsoleOutput


def _api(registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name="test",
        registry=registry,
        bus=bus,
        config={},
        state={},
        request_rebuild=lambda: None,
    )


def _context() -> tuple[SimpleNamespace, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=100)

    class NoModelCalls:
        async def ainvoke(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("slash command dispatch must not call the model")

    return SimpleNamespace(console=ConsoleOutput(console), agent=NoModelCalls()), output


@pytest.mark.asyncio
async def test_non_command_is_not_dispatched() -> None:
    registry = Registry()
    bus = EventBus()
    called = False

    async def handler(ctx: object, args: str) -> None:
        nonlocal called
        called = True

    _api(registry, bus).add_command("echo", handler, help="echo arguments")
    ctx, _ = _context()

    assert await dispatch_command(registry, ctx, "hello") is False
    assert called is False


@pytest.mark.asyncio
async def test_slash_command_receives_context_and_unsplit_argument_text() -> None:
    registry = Registry()
    bus = EventBus()
    received: list[tuple[object, str]] = []

    async def handler(ctx: object, args: str) -> None:
        received.append((ctx, args))

    _api(registry, bus).add_command("echo", handler, help="echo arguments")
    ctx, _ = _context()

    assert await dispatch_command(registry, ctx, "/echo alpha beta --flag") is True
    assert received == [(ctx, "alpha beta --flag")]


@pytest.mark.asyncio
async def test_slash_command_without_arguments_receives_empty_string() -> None:
    registry = Registry()
    bus = EventBus()
    received: list[str] = []

    async def handler(ctx: object, args: str) -> None:
        received.append(args)

    _api(registry, bus).add_command("echo", handler, help="echo arguments")
    ctx, _ = _context()

    assert await dispatch_command(registry, ctx, "/echo") is True
    assert received == [""]


@pytest.mark.asyncio
async def test_unknown_slash_command_is_handled_without_calling_model() -> None:
    registry = Registry()
    ctx, output = _context()

    assert await dispatch_command(registry, ctx, "/missing some args") is True

    rendered = output.getvalue().lower()
    assert "unknown command" in rendered
    assert "/missing" in rendered
