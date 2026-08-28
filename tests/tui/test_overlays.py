from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.core.events import AppStart, EventBus, InterruptRaised
from orcha_agent.core.plugin import PluginAPI, Resolved
from orcha_agent.builtin import approval_prompt
from orcha_agent.builtin.commands_core import _help, _theme
from orcha_agent.builtin.commands_model import _model
from orcha_agent.builtin.commands_session import _resume, _sessions, _tree
from orcha_agent.core.registry import Registry
from orcha_agent.tui.overlays import (
    ApprovalOverlay,
    AskOverlay,
    HelpOverlay,
    HistoryOverlay,
    ModelOverlay,
    Overlay,
    SelectList,
    SessionOverlay,
    ThemeOverlay,
    TreeOverlay,
    register_builtin_overlays,
)
from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade


async def _drive_overlay(overlay: Overlay, keys: bytes | str) -> Any:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show(overlay))
        await asyncio.sleep(0.02)
        if isinstance(keys, bytes):
            pipe.send_bytes(keys)
        else:
            pipe.send_text(keys)
        result = await asyncio.wait_for(shown, 1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)
        return result


@pytest.mark.asyncio
async def test_overlay_enter_result_and_escape_cancel_are_headless() -> None:
    selected = SelectList("Pick", ["one", "two"])
    assert await _drive_overlay(selected, b"\x1b[B\r") == "two"

    cancelled = SelectList("Pick", ["one"])
    assert await _drive_overlay(cancelled, b"\x1b") is None


@pytest.mark.asyncio
async def test_select_list_filters_pages_and_returns_multiselect() -> None:
    picker = SelectList(
        "Pick",
        ["alpha", "alpine", "beta", "gamma"],
        multi=True,
        page_size=2,
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0), input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show(picker))
        await asyncio.sleep(0.02)
        pipe.send_text("a")
        pipe.send_bytes(b"\x1b[6~")
        pipe.send_text(" ")
        pipe.send_bytes(b"\x1b[5~")
        pipe.send_text(" ")
        pipe.send_bytes(b"\r")
        assert await asyncio.wait_for(shown, 1) == ["alpha", "beta"]
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert picker.filtered_items == ("alpha", "alpine", "beta", "gamma")
    assert "◉" in picker.render_text()


def _api(name: str, registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name=name,
        registry=registry,
        bus=bus,
        config={},
        state={},
        request_rebuild=lambda: None,
    )


def test_overlay_registry_conflicts_and_replace() -> None:
    registry = Registry()
    first = lambda _ctx: SelectList("First", ["one"])
    second = lambda _ctx: SelectList("Second", ["two"])
    _api("first", registry, EventBus()).add_overlay("picker", first)
    with pytest.raises(ValueError, match="already registered by plugin 'first'"):
        _api("second", registry, EventBus()).add_overlay("picker", second)
    _api("second", registry, EventBus()).add_overlay("picker", second, replace=True)
    assert registry.overlays["picker"].factory is second


@pytest.mark.asyncio
async def test_approval_overlay_shortcuts() -> None:
    for key, expected in (("y", "approve"), ("n", "reject"), ("a", "always")):
        overlay = ApprovalOverlay(
            {"name": "execute", "args": {"command": "pwd"}}
        )
        assert await _drive_overlay(overlay, key) == expected


@pytest.mark.asyncio
async def test_runtime_serializes_overlays_and_toggles_mouse() -> None:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0), input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(runtime.run())
        first = SelectList("First", ["one"])
        second = SelectList("Second", ["two"])
        first_result = asyncio.create_task(runtime.ui.show(first))
        second_result = asyncio.create_task(runtime.ui.show(second))
        await asyncio.sleep(0.02)
        assert runtime.active_overlay is first
        assert runtime.application.mouse_support()
        assert len(runtime.application.layout.container.floats) == 2
        pipe.send_bytes(b"\r")
        assert await asyncio.wait_for(first_result, 1) == "one"
        await asyncio.sleep(0.02)
        assert runtime.active_overlay is second
        pipe.send_bytes(b"\x1b")
        assert await asyncio.wait_for(second_result, 1) is None
        assert runtime.active_overlay is None
        assert not runtime.application.mouse_support()
        assert len(runtime.application.layout.container.floats) == 1
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_model_session_and_tree_overlays_invoke_context_actions() -> None:
    called: list[tuple[str, str]] = []

    async def capture(kind: str, value: str) -> None:
        called.append((kind, value))

    provider = SimpleNamespace(models=("small", "large"), available=lambda: None)
    sessions = [
        SimpleNamespace(
            thread_id="old",
            title="Old work",
            cwd="/tmp/old",
            created="2026-08-28T12:00:00+00:00",
        )
    ]
    entries = [
        SimpleNamespace(id="root", parent_id=None),
        SimpleNamespace(id="leaf", parent_id="root"),
    ]
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(model="demo:small"),
        registry=SimpleNamespace(providers={"demo": provider}),
        switch_model=lambda value: capture("model", value),
        session=SimpleNamespace(list=lambda: sessions),
        ledger=SimpleNamespace(
            all=lambda _session_id: entries,
            leaf=lambda _session_id: "leaf",
            count=lambda _session_id: 4,
        ),
        session_id="current",
        resume=lambda value: capture("session", value),
        branch=lambda value: capture("tree", value),
    )

    assert await _drive_overlay(ModelOverlay(ctx), b"\x1b[B\r") == "demo:large"
    assert await _drive_overlay(SessionOverlay(ctx), b"\r") == "old"
    assert await _drive_overlay(TreeOverlay(ctx), b"\x1b[B\r") == "leaf"
    assert called == [
        ("model", "demo:large"),
        ("session", "old"),
        ("tree", "leaf"),
    ]


@pytest.mark.asyncio
async def test_theme_overlay_previews_reverts_and_persists() -> None:
    applied: list[str] = []
    persisted: list[None] = []
    themes = {
        "dark": SimpleNamespace(id="dark"),
        "light": SimpleNamespace(id="light"),
    }
    ctx = SimpleNamespace(
        ui=SimpleNamespace(
            theme=themes["dark"],
            themes=themes,
            set_theme=lambda name: applied.append(name) or themes[name],
        ),
        plugin_states={},
        persist_plugin_states=lambda: persisted.append(None),
    )
    assert await _drive_overlay(ThemeOverlay(ctx), b"\x1b[B\x1b") is None
    assert applied == ["light", "dark"]

    assert await _drive_overlay(ThemeOverlay(ctx), b"\x1b[B\r") == "light"
    assert ctx.plugin_states["commands_core"]["theme"] == "light"
    assert persisted == [None]


@pytest.mark.asyncio
async def test_history_help_and_ask_overlays_return_specified_shapes(tmp_path: Path) -> None:
    from orcha_agent.tui.history import SQLiteHistory

    history = SQLiteHistory(tmp_path / "history.db")
    history.append_string("older prompt")
    history.append_string("newer prompt")
    ctx = SimpleNamespace(
        ui=SimpleNamespace(
            history=history,
            effective_keys={"submit": ("enter",), "tree": ("escape escape",)},
        ),
        registry=SimpleNamespace(
            commands={
                "help": SimpleNamespace(help="Show help"),
                "model": SimpleNamespace(help="Switch model"),
            }
        ),
    )
    history_overlay = HistoryOverlay(ctx)
    assert await _drive_overlay(history_overlay, "older\r") == "older prompt"

    help_overlay = HelpOverlay(ctx)
    assert "/help" in help_overlay.text
    assert "escape escape" in help_overlay.text
    assert await _drive_overlay(help_overlay, b"\x1b") is None

    questions = [
        {
            "id": "auth",
            "header": "Auth",
            "question": "Choose auth",
            "options": [{"label": "JWT"}, {"label": "Sessions"}],
        },
        {
            "id": "features",
            "question": "Choose features",
            "multi": True,
            "options": [{"label": "Logs"}, {"label": "Metrics"}],
        },
    ]
    ask = AskOverlay(questions)
    result = await _drive_overlay(ask, b"\r\t \x1b[B \r")
    assert result == {
        "kind": "submit",
        "results": [
            {"id": "auth", "selectedOptions": ["JWT"]},
            {"id": "features", "selectedOptions": ["Logs", "Metrics"]},
        ],
    }


@pytest.mark.asyncio
async def test_registered_overlay_name_resolves_through_ui_facade() -> None:
    registry = Registry()
    register_builtin_overlays(registry)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(model="demo:one", cwd=Path.cwd()),
        registry=registry,
        plugin_states={},
        persist_plugin_states=lambda: None,
        ui=UIFacade(),
    )
    registry.providers["demo"] = SimpleNamespace(
        models=("one",), available=lambda: None
    )
    ctx.switch_model = lambda _value: asyncio.sleep(0)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            ctx=ctx,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show("model"))
        await asyncio.sleep(0.02)
        pipe.send_bytes(b"\r")
        assert await asyncio.wait_for(shown, 1) == "demo:one"
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_no_argument_commands_delegate_and_explicit_paths_remain() -> None:
    shown: list[str] = []
    switched: list[str] = []
    resumed: list[str] = []

    async def show(name: str, *_args: Any, **_kwargs: Any) -> None:
        shown.append(name)

    async def switch(value: str) -> None:
        switched.append(value)

    async def resume(value: str) -> None:
        resumed.append(value)

    printed: list[object] = []
    errors: list[str] = []
    ctx = SimpleNamespace(
        ui=UIFacade(show_overlay=show),
        cfg=SimpleNamespace(
            model="demo:one",
            subagent_model=None,
            summarizer_model=None,
        ),
        switch_model=switch,
        resume=resume,
        console=SimpleNamespace(print=printed.append, error=errors.append),
        registry=SimpleNamespace(commands={}),
        session=SimpleNamespace(list=lambda: ()),
        ledger=SimpleNamespace(all=lambda _session_id: (), leaf=lambda _session_id: None),
        session_id="current",
        plugin_states={},
        persist_plugin_states=lambda: None,
    )
    await _model(ctx, "")
    await _sessions(ctx, "")
    await _resume(ctx, "")
    await _tree(ctx, "")
    await _theme(ctx, "")
    await _help(ctx, "")
    assert shown == ["model", "session", "session", "tree", "theme", "help"]

    await _model(ctx, "demo:two")
    await _resume(ctx, "saved")
    assert switched == ["demo:two"]
    assert resumed == ["saved"]


@pytest.mark.asyncio
async def test_theme_command_reports_unavailable_picker() -> None:
    errors: list[str] = []
    ctx = SimpleNamespace(
        ui=UIFacade(),
        console=SimpleNamespace(error=errors.append),
    )
    await _theme(ctx, "")
    assert errors == ["Theme picker is unavailable."]


@pytest.mark.asyncio
async def test_configured_tree_and_empty_question_mark_open_overlays(tmp_path: Path) -> None:
    key_file = tmp_path / "keys.toml"
    key_file.write_text("[bindings]\ntree = \"c-x\"\n", encoding="utf-8")
    shown: list[str] = []

    async def show(name: str) -> None:
        shown.append(name)

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, model="demo:one", models={}),
        plugin_states={},
        persist_plugin_states=lambda: None,
        ui=UIFacade(show_overlay=show),
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=ctx,
            status=lambda: "",
            keybindings_path=key_file,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_bytes(b"\x18")
        pipe.send_text("?")
        await asyncio.sleep(0.05)
        pipe.send_text("x?")
        await asyncio.sleep(0.05)
        assert shown == ["tree", "help"]
        assert runtime.buffer.text == "x?"
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_approval_adapter_uses_ui_and_preserves_always_and_fail_closed() -> None:
    registry = Registry()
    bus = EventBus()
    state: dict[str, Any] = {}
    rebuilt: list[None] = []
    choices = iter(["always", None, ValueError("broken overlay")])
    shown: list[tuple[str, dict[str, Any]]] = []

    async def show(name: str, *, action: dict[str, Any]) -> Any:
        shown.append((name, action))
        choice = next(choices)
        if isinstance(choice, Exception):
            raise choice
        return choice

    api = PluginAPI(
        name="approval_prompt",
        registry=registry,
        bus=bus,
        config={},
        state=state,
        request_rebuild=lambda: rebuilt.append(None),
    )
    approval_prompt.register(api)
    await bus.emit(AppStart(ctx=SimpleNamespace(ui=UIFacade(show_overlay=show))))
    payload = {
        "action_requests": [
            {"name": "execute", "args": {"command": "pwd"}},
            {"name": 42, "args": {}},
        ]
    }
    result = await bus.emit(InterruptRaised(payload=payload))
    assert isinstance(result, Resolved)
    assert result.resume_value == {
        "decisions": [{"type": "approve"}, {"type": "reject"}]
    }
    assert state == {"always_allowed": ["execute"]}
    assert rebuilt == [None]
    assert shown[0][0] == "approval"

    cancelled = await bus.emit(
        InterruptRaised(
            payload={"action_requests": [{"name": "write_file", "args": {}}]}
        )
    )
    assert isinstance(cancelled, Resolved)
    assert cancelled.resume_value == {"decisions": [{"type": "reject"}]}

    failed = await bus.emit(
        InterruptRaised(
            payload={"action_requests": [{"name": "edit", "args": {}}]}
        )
    )
    assert isinstance(failed, Resolved)
    assert failed.resume_value == {"decisions": [{"type": "reject"}]}


@pytest.mark.asyncio
async def test_every_concrete_overlay_cancels_headlessly(tmp_path: Path) -> None:
    from orcha_agent.tui.history import SQLiteHistory

    provider = SimpleNamespace(models=("one",), available=lambda: None)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(model="demo:one"),
        registry=SimpleNamespace(
            providers={"demo": provider},
            commands={"help": SimpleNamespace(help="Show help")},
        ),
        switch_model=lambda _value: asyncio.sleep(0),
        session=SimpleNamespace(list=lambda: ()),
        ledger=SimpleNamespace(
            all=lambda _session_id: (),
            leaf=lambda _session_id: None,
            count=lambda _session_id: 0,
        ),
        session_id="current",
        resume=lambda _value: asyncio.sleep(0),
        branch=lambda _value: asyncio.sleep(0),
        ui=SimpleNamespace(
            theme=SimpleNamespace(id="dark"),
            themes={"dark": SimpleNamespace(id="dark")},
            set_theme=lambda _name: None,
            history=SQLiteHistory(tmp_path / "history.db"),
            effective_keys={},
        ),
        plugin_states={},
        persist_plugin_states=lambda: None,
    )
    overlays = [
        ModelOverlay(ctx),
        SessionOverlay(ctx),
        TreeOverlay(ctx),
        ThemeOverlay(ctx),
        ApprovalOverlay({"name": "execute", "args": {"command": "pwd"}}),
        AskOverlay([{"id": "q", "question": "Choose", "options": ["one"]}]),
        HistoryOverlay(ctx),
        HelpOverlay(ctx),
    ]
    for overlay in overlays:
        assert await _drive_overlay(overlay, b"\x1b") is None


@pytest.mark.asyncio
async def test_ask_other_custom_answer_and_cancel_shape() -> None:
    ask = AskOverlay(
        [{"id": "other", "question": "What?", "options": [{"label": "Known"}]}]
    )
    result = await _drive_overlay(ask, b"\x1b[B\rcustom value\r")
    assert result == {
        "kind": "submit",
        "results": [
            {
                "id": "other",
                "selectedOptions": [],
                "customInput": "custom value",
            }
        ],
    }


@pytest.mark.asyncio
async def test_application_eof_cancels_active_overlay() -> None:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0), input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(
            runtime.ui.show(
                ApprovalOverlay({"name": "execute", "args": {"command": "pwd"}})
            )
        )
        await asyncio.sleep(0.02)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)
        assert await asyncio.wait_for(shown, 1) is None


@pytest.mark.asyncio
async def test_unregistered_overlay_is_an_explicit_error() -> None:
    runtime = ApplicationRuntime(lambda _text: asyncio.sleep(0), output=DummyOutput())
    with pytest.raises(RuntimeError, match="overlay 'missing' is unavailable"):
        await runtime.ui.show("missing")


@pytest.mark.asyncio
async def test_ctrl_r_real_history_overlay_inserts_without_submitting(tmp_path: Path) -> None:
    from orcha_agent.tui.history import SQLiteHistory

    submitted: list[str] = []
    history = SQLiteHistory(tmp_path / "history.db")
    history.append_string("chosen prompt")
    registry = Registry()
    register_builtin_overlays(registry)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, model="demo:one"),
        registry=registry,
        plugin_states={},
        persist_plugin_states=lambda: None,
        ui=UIFacade(),
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda text: asyncio.sleep(0, result=submitted.append(text)),
            registry=registry,
            ctx=ctx,
            history=history,
            status=lambda: "",
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_bytes(b"\x12")
        await asyncio.sleep(0.02)
        pipe.send_text("chosen")
        pipe.send_bytes(b"\r")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "chosen prompt"
        assert submitted == []
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


def test_approval_preview_prefers_specialized_tool_rendering() -> None:
    execute = ApprovalOverlay(
        {
            "name": "execute",
            "args": {"command": "printf ok"},
            "description": "middleware summary",
        }
    )
    edit = ApprovalOverlay(
        {
            "name": "edit",
            "args": {"old_string": "before", "new_string": "after"},
            "description": "middleware summary",
        }
    )
    edit_file = ApprovalOverlay(
        {
            "name": "edit_file",
            "args": {"old_string": "old", "new_string": "new"},
            "description": "middleware summary",
        }
    )
    generic = ApprovalOverlay(
        {
            "name": "custom_tool",
            "args": {"value": 1},
            "description": "middleware summary",
        }
    )

    assert execute.preview_text == "$ printf ok"
    assert edit.preview_text == "- before\n+ after"
    assert edit_file.preview_text == "- old\n+ new"
    assert generic.preview_text == "middleware summary"


def test_tree_overlay_orders_entries_depth_first_by_parent_hierarchy() -> None:
    entries = [
        SimpleNamespace(id="root", parent_id=None),
        SimpleNamespace(id="grandchild", parent_id="child-a"),
        SimpleNamespace(id="child-a", parent_id="root"),
        SimpleNamespace(id="child-b", parent_id="root"),
    ]
    ctx = SimpleNamespace(
        ledger=SimpleNamespace(
            all=lambda _session_id: entries,
            leaf=lambda _session_id: "grandchild",
        ),
        session_id="session",
        branch=lambda _value: asyncio.sleep(0),
    )

    overlay = TreeOverlay(ctx)

    assert [entry.id for entry in overlay.filtered_items] == [
        "root",
        "child-a",
        "grandchild",
        "child-b",
    ]


@pytest.mark.asyncio
async def test_select_list_scrolls_long_navigation_to_selected_row() -> None:
    picker = SelectList("Long list", [f"item-{index:02d}" for index in range(40)])
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0), input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show(picker))
        await asyncio.sleep(0.02)
        pipe.send_bytes(b"\x1b[6~" * 4)
        await asyncio.sleep(0.05)

        info = picker.list_window.render_info
        assert info is not None
        assert picker.index == 32
        assert info.vertical_scroll <= picker.index
        assert picker.index < info.vertical_scroll + info.window_height

        pipe.send_bytes(b"\x1b")
        assert await asyncio.wait_for(shown, 1) is None
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_help_arguments_preserve_printed_command_table() -> None:
    shown: list[str] = []
    printed: list[object] = []
    errors: list[str] = []

    async def show(name: str) -> None:
        shown.append(name)

    ctx = SimpleNamespace(
        ui=UIFacade(show_overlay=show),
        console=SimpleNamespace(print=printed.append, error=errors.append),
        registry=SimpleNamespace(
            commands={"help": SimpleNamespace(help="Show command reference")}
        ),
    )

    await _help(ctx, "commands")

    assert shown == []
    assert errors == []
    assert len(printed) == 1
    assert printed[0].title == "Commands"
    assert printed[0].row_count == 1
