from io import StringIO
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from rich.console import Console

from orcha_agent.builtin import commands_core, commands_model, commands_session
from orcha_agent.core.auth import AuthFlow
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionInfo
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
    return SimpleNamespace(console=ConsoleOutput(console), agent=None), output


def _auth_flow(
    calls: list[tuple[str, object]],
    *,
    status: str = "not logged in",
) -> AuthFlow:
    current = {"status": status}

    async def login(ctx: object) -> None:
        calls.append(("login", ctx))
        current["status"] = "logged in as test@example.com"

    async def logout(ctx: object) -> None:
        calls.append(("logout", ctx))
        current["status"] = "not logged in"

    return AuthFlow(
        login=login,
        logout=logout,
        status=lambda: current["status"],
    )


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
        models=("fake-chat", "fake-reasoner"),
    )
    commands_core.register(api)
    ctx, output = _context(width=240)
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/providers") is True

    rendered = output.getvalue()
    assert "ORCHA_TEST_API_KEY: yes" in rendered
    assert "fake-chat" in rendered
    assert "fake-reasoner" in rendered
    assert secret not in rendered


@pytest.mark.asyncio
async def test_providers_uses_compact_80_column_summary_without_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    secret = "fake-provider-key-that-must-not-be-rendered"
    monkeypatch.setenv("ORCHA_TEST_API_KEY", secret)
    factory_calls: list[tuple[str, object]] = []

    def factory(model_name: str, provider_config: object) -> object:
        factory_calls.append((model_name, provider_config))
        raise AssertionError("/providers must not construct a model")

    api.add_provider(
        "anthropic-enterprise",
        factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=True,
            max_context=200_000,
        ),
        env_keys=("ORCHA_TEST_API_KEY",),
        models=("enterprise-chat", "enterprise-reasoner"),
        available=lambda: "unavailable",
    )
    commands_core.register(api)
    ctx, output = _context(width=80)
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/providers") is True

    rendered = output.getvalue()
    header = next(line for line in rendered.splitlines() if "Prefix" in line)
    assert "Available" in header
    assert "Auth / Keys" in header
    assert "T/S/R/O" in header
    assert "Status" in header
    assert "Models" not in header
    assert "Capabilities" not in rendered
    assert "Environment" not in rendered
    assert "Authentication" not in rendered
    assert "anthropic-enterprise" in rendered
    assert "unavailable" in rendered
    assert "ORCHA_TEST_API_KEY" in rendered
    assert "…" not in rendered
    assert all(len(line) <= 80 for line in rendered.splitlines())
    assert secret not in rendered
    assert factory_calls == []


@pytest.mark.asyncio
async def test_providers_codex_models_move_from_narrow_summary_to_prefix_detail(
) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    factory_calls: list[tuple[str, object]] = []

    def factory(model_name: str, provider_config: object) -> object:
        factory_calls.append((model_name, provider_config))
        raise AssertionError("/providers must not construct a model")

    models = ("gpt-5.6-codex", "gpt-5.6-codex-mini")
    api.add_provider(
        "codex",
        factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=400_000,
        ),
        models=models,
    )
    api.add_auth("codex", _auth_flow([], status="logged in"))
    commands_core.register(api)

    summary_ctx, summary_output = _context(width=80)
    summary_ctx.registry = registry
    assert await dispatch_command(registry, summary_ctx, "/providers") is True
    summary = summary_output.getvalue()
    assert all(model not in summary for model in models)
    assert "logged in" in summary
    assert all(len(line) <= 80 for line in summary.splitlines())

    detail_ctx, detail_output = _context(width=80)
    detail_ctx.registry = registry
    assert await dispatch_command(registry, detail_ctx, "/providers codex") is True
    detail = detail_output.getvalue()
    assert "codex" in detail
    assert "Models" in detail
    assert all(model in detail for model in models)
    assert all(len(line) <= 80 for line in detail.splitlines())
    assert factory_calls == []


@pytest.mark.asyncio
async def test_login_dispatches_to_named_auth_flow_and_updates_provider_status() -> None:
    registry = Registry()
    bus = EventBus()
    api = _api(registry, bus)
    calls: list[tuple[str, object]] = []
    api.add_auth("codex", _auth_flow(calls))
    api.add_provider(
        "codex",
        lambda model_name, provider_config: (model_name, provider_config),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
    )
    commands_core.register(api)
    ctx, output = _context(width=240)
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/login codex") is True
    assert calls == [("login", ctx)]
    assert await dispatch_command(registry, ctx, "/providers") is True
    assert "logged in as test@example.com" in output.getvalue()


@pytest.mark.asyncio
async def test_logout_dispatches_to_named_auth_flow() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    calls: list[tuple[str, object]] = []
    api.add_auth(
        "codex",
        _auth_flow(calls, status="logged in as test@example.com"),
    )
    commands_core.register(api)
    ctx, _ = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/logout codex") is True
    assert calls == [("logout", ctx)]
    assert registry.auth["codex"].flow.status() == "not logged in"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["login", "logout"])
async def test_auth_command_without_prefix_renders_usage(command: str) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, f"/{command}") is True
    assert f"usage: /{command} <prefix>" in output.getvalue().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["login", "logout"])
async def test_auth_command_rejects_unknown_prefix(command: str) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, f"/{command} missing") is True
    rendered = output.getvalue().lower()
    assert "unknown auth prefix" in rendered
    assert "missing" in rendered


@pytest.mark.asyncio
async def test_model_command_normalizes_runtime_fallback_chain() -> None:
    registry = Registry()
    bus = EventBus()
    commands_model.register(_api(registry, bus))
    ctx, _ = _context()
    switched: list[str | list[str]] = []

    async def switch_model(spec: str | list[str]) -> None:
        switched.append(spec)

    ctx.switch_model = switch_model

    assert await dispatch_command(registry, ctx, "/model a:x,b:y") is True
    assert switched == [["a:x", "b:y"]]


@pytest.mark.asyncio
async def test_resume_without_session_id_renders_usage_without_resuming() -> None:
    registry = Registry()
    bus = EventBus()
    commands_session.register(_api(registry, bus))
    ctx, output = _context()

    async def unexpected_resume(_thread_id: str) -> None:
        raise AssertionError("missing session id must not attempt a resume")

    ctx.resume = unexpected_resume

    assert await dispatch_command(registry, ctx, "/resume") is True

    assert "usage: /resume <session-id>" in output.getvalue().lower()


@pytest.mark.asyncio
async def test_sessions_command_renders_registered_session_table() -> None:
    registry = Registry()
    bus = EventBus()
    commands_session.register(_api(registry, bus))
    ctx, output = _context(width=240)
    session = SessionInfo(
        thread_id="saved-session",
        cwd="/work/project",
        model="fake:model",
        created="2026-08-27T10:30:00+00:00",
        title="Investigate parser",
    )
    ctx.session = SimpleNamespace(list=lambda: [session])

    assert await dispatch_command(registry, ctx, "/sessions") is True

    rendered = output.getvalue()
    assert "Sessions" in rendered
    assert "saved-session" in rendered
    assert "Investigate parser" in rendered
    assert "fake:model" in rendered
    assert "/work/project" in rendered
    assert "2026-08-27T10:30:00+00:00" in rendered


@pytest.mark.asyncio
async def test_compact_without_summarizer_renders_error_without_compacting() -> None:
    registry = Registry()
    bus = EventBus()
    commands_session.register(_api(registry, bus))
    ctx, output = _context()
    ctx.summarizer = None

    async def unexpected_compact() -> None:
        raise AssertionError("unavailable summarizer must not start compaction")

    ctx.compact = unexpected_compact

    assert await dispatch_command(registry, ctx, "/compact") is True

    rendered = output.getvalue().lower()
    assert "summarizer" in rendered
    assert "unavailable" in rendered
