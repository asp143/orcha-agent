import json
import re
import stat
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    message_to_dict,
)
from rich.console import Console

from orcha_agent.builtin import commands_core, commands_model, commands_session
from orcha_agent.core.auth import AuthFlow
from orcha_agent.core.events import EventBus
from orcha_agent.core.ledger import (
    CompactionEntry,
    Ledger,
    MessageEntry,
    ResetBoundaryEntry,
)
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionInfo, SessionStore
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


def _context(
    *, width: int = 100, styles: bool = False
) -> tuple[SimpleNamespace, StringIO]:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=styles,
        color_system="standard" if styles else None,
        width=width,
    )
    return SimpleNamespace(console=ConsoleOutput(console), agent=None), output


class _ThemeUI:
    def __init__(self) -> None:
        self.selected: list[str] = []
        self.shown: list[object] = []

    def set_theme(self, name: str) -> SimpleNamespace:
        self.selected.append(name)
        return SimpleNamespace(id=name)

    async def show(self, overlay: object) -> None:
        self.shown.append(overlay)



def _auth_flow(
    calls: list[tuple[object, ...]],
    *,
    status: str = "not logged in",
) -> AuthFlow:
    current = {"status": status}

    async def login(ctx: object, mode: str) -> None:
        calls.append(("login", ctx, mode))
        current["status"] = "logged in as test@example.com"

    async def logout(ctx: object) -> None:
        calls.append(("logout", ctx))
        current["status"] = "not logged in"

    return AuthFlow(
        login=login,
        logout=logout,
        status=lambda: current["status"],
    )


def _provider_caps() -> ProviderCaps:
    return ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=True,
        structured_output=False,
        max_context=None,
    )


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_terminal(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value)


def _clear_output(output: StringIO) -> None:
    output.seek(0)
    output.truncate(0)


@pytest.fixture
def session_command_context(
    tmp_path: Path,
) -> Iterator[tuple[SimpleNamespace, StringIO, SessionInfo, Ledger]]:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(
            tmp_path,
            ["fake:primary", "fake:fallback"],
            thread_id="session-a1b2",
            title="Session command tests",
        )
        ctx, output = _context(width=240)
        ctx.session = store
        ctx.session_id = session.thread_id
        ctx.ledger = Ledger(store)
        yield ctx, output, session, ctx.ledger


def test_add_provider_stores_default_model() -> None:
    registry = Registry()
    api = _api(registry, EventBus())

    api.add_provider(
        "codex",
        lambda model_name, provider_config: (model_name, provider_config),
        capabilities=_provider_caps(),
        models=("gpt-5.6-sol",),
        default_model="gpt-5.6-sol",
    )

    assert registry.providers["codex"].default_model == "gpt-5.6-sol"


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
async def test_theme_command_applies_and_persists_selection() -> None:
    registry = Registry()
    commands_core.register(_api(registry, EventBus()))
    ctx, output = _context()
    ui = _ThemeUI()
    persisted: list[dict[str, dict[str, str]]] = []
    ctx.ui = ui
    ctx.plugin_states = {}
    ctx.persist_plugin_states = lambda: persisted.append(
        {
            plugin: dict(state)
            for plugin, state in ctx.plugin_states.items()
        }
    )

    assert await dispatch_command(registry, ctx, "/theme nord") is True

    assert ui.selected == ["nord"]
    assert ctx.plugin_states["commands_core"]["theme"] == "nord"
    assert persisted == [{"commands_core": {"theme": "nord"}}]
    assert "Theme: nord" in output.getvalue()


@pytest.mark.asyncio
async def test_theme_command_without_argument_delegates_to_overlay() -> None:
    registry = Registry()
    commands_core.register(_api(registry, EventBus()))
    ctx, _output = _context()
    ui = _ThemeUI()
    ctx.ui = ui

    assert await dispatch_command(registry, ctx, "/theme") is True
    assert ui.shown == ["theme"]


@pytest.mark.asyncio
async def test_theme_command_handles_unavailable_theme_cleanly() -> None:
    registry = Registry()
    commands_core.register(_api(registry, EventBus()))
    ctx, output = _context()
    ctx.ui = SimpleNamespace(
        set_theme=lambda _name: (_ for _ in ()).throw(KeyError("missing"))
    )
    ctx.plugin_states = {}

    assert await dispatch_command(registry, ctx, "/theme missing") is True
    assert "Unknown theme: missing" in output.getvalue()
    assert ctx.plugin_states == {}


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
async def test_login_without_mode_dispatches_auto_and_updates_provider_status() -> None:
    registry = Registry()
    bus = EventBus()
    api = _api(registry, bus)
    calls: list[tuple[object, ...]] = []
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
    assert calls == [("login", ctx, "auto")]
    assert await dispatch_command(registry, ctx, "/providers") is True
    assert "logged in as test@example.com" in output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argument", "mode"),
    [
        ("browser", "browser"),
        ("device", "device"),
        ("paste", "paste"),
        ("--browser", "browser"),
        ("--device", "device"),
        ("--paste", "paste"),
    ],
)
async def test_login_dispatches_explicit_mode(argument: str, mode: str) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    calls: list[tuple[object, ...]] = []
    api.add_auth("codex", _auth_flow(calls))
    commands_core.register(api)
    ctx, _ = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, f"/login codex {argument}") is True
    assert calls == [("login", ctx, mode)]


@pytest.mark.asyncio
async def test_login_switches_from_unusable_current_provider_to_codex_default() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    auth_calls: list[tuple[object, ...]] = []
    api.add_auth("codex", _auth_flow(auth_calls))
    api.add_provider(
        "codex",
        lambda model_name, provider_config: (model_name, provider_config),
        capabilities=_provider_caps(),
        models=("gpt-5.6-sol",),
        default_model="gpt-5.6-sol",
    )
    api.add_provider(
        "anthropic",
        lambda model_name, provider_config: (model_name, provider_config),
        capabilities=_provider_caps(),
        models=("claude-opus-5",),
        available=lambda: "set ANTHROPIC_API_KEY",
    )
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry
    ctx.cfg = SimpleNamespace(model="anthropic:claude-opus-5")
    switched: list[str | list[str]] = []

    async def switch_model(spec: str | list[str]) -> None:
        switched.append(spec)

    ctx.switch_model = switch_model

    assert await dispatch_command(registry, ctx, "/login codex") is True
    assert auth_calls == [("login", ctx, "auto")]
    assert switched == ["codex:gpt-5.6-sol"]
    assert output.getvalue().splitlines() == [
        "Switched model to codex:gpt-5.6-sol (use /model to change)"
    ]


@pytest.mark.asyncio
async def test_login_keeps_usable_codex_model_and_prints_default_model_hint() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    auth_calls: list[tuple[object, ...]] = []
    api.add_auth("codex", _auth_flow(auth_calls))
    api.add_provider(
        "codex",
        lambda model_name, provider_config: (model_name, provider_config),
        capabilities=_provider_caps(),
        models=("gpt-5.6-sol", "gpt-5.5"),
        default_model="gpt-5.6-sol",
    )
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry
    ctx.cfg = SimpleNamespace(model="codex:gpt-5.5")

    async def unexpected_switch(_spec: str | list[str]) -> None:
        raise AssertionError("a usable current Codex model must not be replaced")

    ctx.switch_model = unexpected_switch

    assert await dispatch_command(registry, ctx, "/login codex") is True
    assert auth_calls == [("login", ctx, "auto")]
    assert "/model codex:gpt-5.6-sol" in output.getvalue()


@pytest.mark.asyncio
async def test_logout_dispatches_to_named_auth_flow() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    calls: list[tuple[object, ...]] = []
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
async def test_login_without_prefix_renders_mode_aware_usage() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/login") is True
    assert (
        "usage: /login <prefix> [browser|device|paste]"
        in output.getvalue().lower()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    ["codex auto", "codex invalid", "codex --remote", "codex browser extra"],
)
async def test_login_rejects_invalid_mode_usage(args: str) -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    calls: list[tuple[object, ...]] = []
    api.add_auth("codex", _auth_flow(calls))
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, f"/login {args}") is True
    assert (
        "usage: /login <prefix> [browser|device|paste]"
        in output.getvalue().lower()
    )
    assert calls == []


@pytest.mark.asyncio
async def test_logout_without_prefix_renders_usage() -> None:
    registry = Registry()
    api = _api(registry, EventBus())
    commands_core.register(api)
    ctx, output = _context()
    ctx.registry = registry

    assert await dispatch_command(registry, ctx, "/logout") is True
    assert "usage: /logout <prefix>" in output.getvalue().lower()


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
async def test_bare_model_reports_effective_role_models_and_usage() -> None:
    registry = Registry()
    commands_model.register(_api(registry, EventBus()))
    ctx, output = _context(width=120)
    ctx.cfg = SimpleNamespace(
        model="anthropic:claude-opus-5",
        subagent_model=None,
        summarizer_model="codex:gpt-5.6-sol",
    )

    async def unexpected_switch(_spec: str | list[str]) -> None:
        raise AssertionError("bare /model must not switch models")

    ctx.switch_model = unexpected_switch

    assert await dispatch_command(registry, ctx, "/model") is True
    assert output.getvalue().splitlines() == [
        "Current model: anthropic:claude-opus-5",
        "Subagent model: anthropic:claude-opus-5 (inherited)",
        "Summarizer model: codex:gpt-5.6-sol (explicit)",
        "Usage: /model <provider:model>[,<provider:model>...]",
    ]


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
async def test_sessions_command_renders_registered_session_table(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, output, session, ledger = session_command_context
    for _ in range(7):
        ledger.append(session.thread_id, ResetBoundaryEntry())
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, "/sessions") is True

    rendered = output.getvalue()
    assert "Sessions" in rendered
    assert session.thread_id in rendered
    assert session.title is not None
    assert session.title in rendered
    assert re.search(r"fake:primary\s*,\s*fake:fallback", rendered)
    assert "['fake:primary'" not in rendered
    assert "Entries" in rendered
    assert re.search(rf"{re.escape(session.thread_id)}.*\b7\b", rendered)
    assert session.cwd in rendered
    assert session.created in rendered


@pytest.mark.asyncio
async def test_compact_delegates_when_summarizer_is_lazily_uninitialized() -> None:
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    ctx, _output = _context()
    ctx.summarizer = None
    compact_calls = 0

    async def compact() -> None:
        nonlocal compact_calls
        compact_calls += 1

    ctx.compact = compact

    assert await dispatch_command(registry, ctx, "/compact") is True
    assert compact_calls == 1


SESSION_COMMANDS = {
    "tree",
    "branch",
    "fork",
    "new",
    "clear",
    "compact",
    "export",
    "sessions",
    "resume",
}


def test_session_commands_own_the_complete_session_surface() -> None:
    session_registry = Registry()
    commands_session.register(_api(session_registry, EventBus()))
    assert SESSION_COMMANDS <= set(session_registry.commands)

    core_registry = Registry()
    commands_core.register(_api(core_registry, EventBus()))
    assert "clear" not in core_registry.commands


@pytest.mark.asyncio
async def test_tree_default_summarizes_users_and_marks_topology_and_leaf(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, _output, session, ledger = session_command_context
    styled_ctx, output = _context(width=240, styles=True)
    ctx.console = styled_ctx.console
    long_prompt = "u" * 60 + "TRUNCATED-SUFFIX"
    user = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content=long_prompt))),
    )
    first_reply = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="first reply"))),
    )
    ledger.branch(session.thread_id, user.id)
    second_reply = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="second reply"))),
    )
    compacted = ledger.append(
        session.thread_id, CompactionEntry(summary="summary of both replies")
    )
    reset = ledger.append(session.thread_id, ResetBoundaryEntry())

    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    assert await dispatch_command(registry, ctx, "/tree") is True

    rendered = output.getvalue()
    plain = _plain_terminal(rendered)
    assert user.id in plain
    assert long_prompt[:60] in plain
    assert "TRUNCATED-SUFFIX" not in plain
    user_line = next(line for line in plain.splitlines() if user.id in line)
    assert re.search(r"\b2\b", user_line)
    assert first_reply.id not in plain
    assert second_reply.id not in plain
    assert compacted.id in plain and "⊟" in plain
    assert reset.id in plain and "⊠" in plain
    assert "⎇" in plain

    leaf_line = next(line for line in rendered.splitlines() if reset.id in line)
    sgr_parameters = {
        parameter
        for escape in re.findall(r"\x1b\[([0-9;]*)m", leaf_line)
        for parameter in escape.split(";")
    }
    highlight_parameters = {"1", "4", "7"} | {
        str(code) for code in (*range(40, 50), *range(100, 108))
    }
    assert sgr_parameters & highlight_parameters


@pytest.mark.asyncio
async def test_tree_all_renders_every_entry_id_and_payload_kind(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, output, session, ledger = session_command_context
    entries = [
        ledger.append(
            session.thread_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="question"))),
        ),
        ledger.append(
            session.thread_id,
            MessageEntry(message=message_to_dict(AIMessage(content="answer"))),
        ),
        ledger.append(session.thread_id, CompactionEntry(summary="short summary")),
        ledger.append(session.thread_id, ResetBoundaryEntry()),
    ]
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, "/tree --all") is True

    rendered = output.getvalue()
    assert all(entry.id in rendered for entry in entries)
    assert "question" in rendered
    assert "answer" in rendered
    assert "⊟" in rendered
    assert "⊠" in rendered


@pytest.mark.asyncio
async def test_branch_from_user_message_delegates_latest_assistant_reply(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, _output, session, ledger = session_command_context
    target = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="branch here"))),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="first reply"))),
    )
    ledger.branch(session.thread_id, target.id)
    latest_reply = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="latest reply"))),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="later question"))),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="later answer"))),
    )
    branched: list[str] = []

    async def branch(entry_id: str) -> None:
        branched.append(entry_id)

    ctx.branch = branch
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, f"/branch {target.id[:8]}") is True
    assert branched == [latest_reply.id]


@pytest.mark.asyncio
async def test_branch_from_user_message_delegates_final_assistant_of_tool_turn(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, _output, session, ledger = session_command_context
    target = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="use a tool"))),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(
            message=message_to_dict(
                AIMessage(
                    content="calling tool",
                    tool_calls=[
                        {"id": "call-1", "name": "lookup", "args": {"query": "x"}}
                    ],
                )
            )
        ),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(
            message=message_to_dict(
                ToolMessage(content="tool result", tool_call_id="call-1")
            )
        ),
    )
    final_reply = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="final answer"))),
    )
    branched: list[str] = []

    async def branch(entry_id: str) -> None:
        branched.append(entry_id)

    ctx.branch = branch
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, f"/branch {target.id[:8]}") is True
    exact_command = f"/branch --exact {target.id[:8]}"
    assert await dispatch_command(registry, ctx, exact_command) is True
    assert branched == [final_reply.id, target.id]


@pytest.mark.asyncio
async def test_branch_exact_delegates_the_resolved_entry(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, _output, session, ledger = session_command_context
    target = ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="branch here"))),
    )
    ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(AIMessage(content="existing reply"))),
    )
    branched: list[str] = []

    async def branch(entry_id: str) -> None:
        branched.append(entry_id)

    ctx.branch = branch
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    command = f"/branch --exact {target.id[:8]}"
    assert await dispatch_command(registry, ctx, command) is True
    assert branched == [target.id]


@pytest.mark.asyncio
async def test_branch_reports_ambiguous_and_missing_prefixes_without_mutation(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
) -> None:
    ctx, output, session, ledger = session_command_context
    entries = [
        ledger.append(
            session.thread_id,
            MessageEntry(message=message_to_dict(HumanMessage(content=f"entry {index}"))),
        )
        for index in range(17)
    ]
    by_initial: dict[str, list[str]] = {}
    for entry in entries:
        by_initial.setdefault(entry.id[0], []).append(entry.id)
    prefix, candidates = next(
        (initial, ids) for initial, ids in by_initial.items() if len(ids) > 1
    )
    branched: list[str] = []

    async def branch(entry_id: str) -> None:
        branched.append(entry_id)

    ctx.branch = branch
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    leaf_before = ledger.leaf(session.thread_id)

    for exact in (False, True):
        option = "--exact " if exact else ""
        assert (
            await dispatch_command(registry, ctx, f"/branch {option}{prefix}") is True
        )
        ambiguous_output = output.getvalue()
        assert "ambiguous" in ambiguous_output.lower()
        assert all(candidate in ambiguous_output for candidate in sorted(candidates))
        assert branched == []
        assert ledger.leaf(session.thread_id) == leaf_before

        _clear_output(output)
        missing_prefix = "not-a-ledger-id"
        assert (
            await dispatch_command(
                registry, ctx, f"/branch {option}{missing_prefix}"
            )
            is True
        )
        missing_output = output.getvalue().lower()
        assert missing_prefix in missing_output
        assert "unknown" in missing_output or "not found" in missing_output
        assert branched == []
        assert ledger.leaf(session.thread_id) == leaf_before
        _clear_output(output)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "method_name"),
    [
        pytest.param("/fork", "fork", id="fork"),
        pytest.param("/new", "new_session", id="new"),
        pytest.param("/clear", "clear", id="clear"),
        pytest.param("/compact", "compact", id="compact"),
    ],
)
async def test_zero_argument_session_commands_delegate_once(
    command: str, method_name: str
) -> None:
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    ctx, _output = _context()
    calls: list[str] = []

    async def operation() -> None:
        calls.append(method_name)

    setattr(ctx, method_name, operation)
    ctx.summarizer = object()

    assert await dispatch_command(registry, ctx, command) is True
    assert calls == [method_name]


@pytest.mark.asyncio
async def test_export_default_uses_absolute_cwd_path_and_private_mode(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, output, session, ledger = session_command_context
    ledger.append(
        session.thread_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="export me"))),
    )
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    monkeypatch.chdir(tmp_path)

    assert await dispatch_command(registry, ctx, "/export") is True

    default_path = tmp_path / f"{session.thread_id}.jsonl"
    header = json.loads(default_path.read_text().splitlines()[0])
    assert header["id"] == session.thread_id
    assert stat.S_IMODE(default_path.stat().st_mode) == 0o600
    assert str(default_path.resolve()) in output.getvalue()


@pytest.mark.asyncio
async def test_export_expands_home_directory(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, output, session, _ledger = session_command_context
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, "/export ~/saved-session.jsonl") is True

    exported = home / "saved-session.jsonl"
    assert json.loads(exported.read_text().splitlines()[0])["id"] == session.thread_id
    assert stat.S_IMODE(exported.stat().st_mode) == 0o600
    assert str(exported.resolve()) in output.getvalue()


@pytest.mark.asyncio
async def test_export_accepts_an_unquoted_path_containing_spaces(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, output, session, _ledger = session_command_context
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    monkeypatch.chdir(tmp_path)

    command = "/export saved session with spaces.jsonl"
    assert await dispatch_command(registry, ctx, command) is True

    exported = tmp_path / "saved session with spaces.jsonl"
    assert json.loads(exported.read_text().splitlines()[0])["id"] == session.thread_id
    assert stat.S_IMODE(exported.stat().st_mode) == 0o600
    assert str(exported.resolve()) in output.getvalue()


@pytest.mark.asyncio
async def test_export_refuses_to_replace_an_existing_file_without_force(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
) -> None:
    ctx, output, _session, _ledger = session_command_context
    exported = tmp_path / "existing.jsonl"
    exported.write_text("keep this content\n")
    exported.chmod(0o640)
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert await dispatch_command(registry, ctx, f"/export {exported}") is True

    assert exported.read_text() == "keep this content\n"
    assert stat.S_IMODE(exported.stat().st_mode) == 0o640
    rendered = output.getvalue().lower()
    assert "exist" in rendered
    assert "--force" in rendered


@pytest.mark.asyncio
async def test_export_force_replaces_an_existing_file_securely(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
) -> None:
    ctx, output, session, _ledger = session_command_context
    exported = tmp_path / "existing.jsonl"
    exported.write_text("replace this content\n")
    exported.chmod(0o644)
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert (
        await dispatch_command(registry, ctx, f"/export --force {exported}") is True
    )

    header = json.loads(exported.read_text().splitlines()[0])
    assert header["id"] == session.thread_id
    assert stat.S_IMODE(exported.stat().st_mode) == 0o600
    assert str(exported.resolve()) in output.getvalue()


@pytest.mark.asyncio
async def test_export_force_refuses_a_symlink_without_touching_its_target(
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
) -> None:
    ctx, output, _session, _ledger = session_command_context
    target = tmp_path / "target.jsonl"
    target.write_text("target must remain unchanged\n")
    target.chmod(0o640)
    exported = tmp_path / "export.jsonl"
    exported.symlink_to(target)
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))

    assert (
        await dispatch_command(registry, ctx, f"/export --force {exported}") is True
    )

    assert exported.is_symlink()
    assert target.read_text() == "target must remain unchanged\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert "symlink" in output.getvalue().lower()


@pytest.mark.asyncio
async def test_resume_delegates_the_supplied_session_prefix() -> None:
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    ctx, _output = _context()
    resumed: list[str] = []

    async def resume(prefix: str) -> None:
        resumed.append(prefix)

    ctx.resume = resume

    assert await dispatch_command(registry, ctx, "/resume session-a1") is True
    assert resumed == ["session-a1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        pytest.param("/tree --bogus", id="tree-option"),
        pytest.param("/branch", id="branch-missing"),
        pytest.param("/branch one two", id="branch-extra"),
        pytest.param("/fork extra", id="fork"),
        pytest.param("/new extra", id="new"),
        pytest.param("/clear extra", id="clear"),
        pytest.param("/compact extra", id="compact"),
        pytest.param("/export --bogus", id="export-unknown-option"),
        pytest.param("/export path.jsonl --force", id="export-option-order"),
        pytest.param("/export --force --force", id="export-repeated-option"),
        pytest.param("/sessions extra", id="sessions"),
        pytest.param("/resume one two", id="resume"),
    ],
)
async def test_invalid_session_command_usage_does_not_mutate_state(
    command: str,
    session_command_context: tuple[SimpleNamespace, StringIO, SessionInfo, Ledger],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, output, session, ledger = session_command_context
    registry = Registry()
    commands_session.register(_api(registry, EventBus()))
    mutations: list[tuple[object, ...]] = []
    monkeypatch.chdir(tmp_path)
    ledger_before = (ledger.leaf(session.thread_id), ledger.count(session.thread_id))
    sessions_before = ctx.session.list()
    files_before = {path.name for path in tmp_path.iterdir()}

    async def mutate(*args: object) -> None:
        mutations.append(args)

    ctx.branch = mutate
    ctx.fork = mutate
    ctx.new_session = mutate
    ctx.clear = mutate
    ctx.compact = mutate
    ctx.resume = mutate
    ctx.summarizer = object()

    assert await dispatch_command(registry, ctx, command) is True
    assert (ledger.leaf(session.thread_id), ledger.count(session.thread_id)) == ledger_before
    assert ctx.session.list() == sessions_before
    assert {path.name for path in tmp_path.iterdir()} == files_before
    assert mutations == []
    assert "usage:" in output.getvalue().lower()
