from io import StringIO
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from rich.console import Console

from orcha_agent.builtin import commands_core
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
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


def _context(*, width: int = 100) -> tuple[SimpleNamespace, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=width)

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


@pytest.mark.asyncio
async def test_providers_reports_key_presence_without_printing_secret_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    bus = EventBus()
    secret = "s17-secret-value-must-not-be-rendered"
    monkeypatch.setenv("ORCHA_TEST_API_KEY", secret)

    api = _api(registry, bus)
    api.add_provider(
        "fake",
        lambda model_name, provider_config: FakeListChatModel(responses=[model_name]),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
        env_keys=("ORCHA_TEST_API_KEY",),
    )
    commands_core.register(api)
    ctx, output = _context(width=240)
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/providers") is True

    rendered = output.getvalue()
    assert "ORCHA_TEST_API_KEY: yes" in rendered
    assert secret not in rendered
