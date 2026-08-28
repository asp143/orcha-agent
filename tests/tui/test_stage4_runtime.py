from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.core.events import EventBus, SessionSwitch, ToolCallEnd, ToolCallStart
from orcha_agent.core.plugin import ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.tui.composer import Composer
from orcha_agent.tui.history import SQLiteHistory
from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade
from orcha_agent.tui.theme import load_themes
from prompt_toolkit.layout.dimension import to_dimension
from prompt_toolkit.layout.menus import CompletionsMenu


def _ctx(tmp_path: Path, registry: Registry | None = None) -> SimpleNamespace:
    registry = registry or Registry()
    state: dict[str, dict[str, object]] = {}
    return SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=tmp_path,
            model="able:test",
            models={},
            providers={},
            thinking="summary",
        ),
        registry=registry,
        plugin_states=state,
        persist_plugin_states=lambda: None,
        ui=UIFacade(),
        bus=EventBus(),
        _bus=EventBus(),
        switch_model=lambda _model: asyncio.sleep(0),
        rebuild=lambda: asyncio.sleep(0),
    )


def test_composer_builds_all_shapes_and_dynamic_mode_styles(tmp_path: Path) -> None:
    theme = load_themes(home=tmp_path)["dark"]
    for shape, chrome in (("box", 2), ("claude", 2), ("borderless", 0)):
        composer = Composer(shape=shape, theme=theme, model=lambda: "model", thinking=lambda: "high")
        assert composer.shape == shape
        assert composer.chrome_lines == chrome
        assert composer.height_for_width(80) == 1 + chrome
        composer.buffer.text = "one\ntwo"
        assert composer.height_for_width(80) == 2 + chrome
        assert composer.border_style == "class:thinkinghigh"
        composer.buffer.text = "!pwd"
        assert composer.border_style == "class:bashmode"


@pytest.mark.asyncio
async def test_headless_submit_newline_continuation_dot_and_bash(tmp_path: Path) -> None:
    submitted: list[str] = []
    events: list[object] = []
    submitted_event = asyncio.Event()
    ctx = _ctx(tmp_path)
    async def capture(event: object) -> None:
        events.append(event)

    ctx._bus.on(object, capture, plugin="test")
    async def submit(text: str) -> None:
        submitted.append(text)
        submitted_event.set()

    def shell(command: str, cwd: Path, timeout: float):
        assert (command, cwd, timeout) == ("printf ok", tmp_path, 60.0)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            ctx=ctx,
            composer_shape="claude",
            shell_runner=shell,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("first\\")
        pipe.send_bytes(b"\r")
        pipe.send_text("second")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(submitted_event.wait(), 1)
        submitted_event.clear()
        pipe.send_text(".")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(submitted_event.wait(), 1)
        submitted_event.clear()
        pipe.send_text("!printf ok")
        pipe.send_bytes(b"\r")
        await asyncio.sleep(0.05)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert submitted == ["first\nsecond", "keep going"]
    assert any(isinstance(event, ToolCallStart) and event.name == "execute" for event in events)
    assert any(isinstance(event, ToolCallEnd) and event.result["stdout"] == "ok" for event in events)


@pytest.mark.asyncio
async def test_headless_alt_enter_and_shift_enter_insert_newlines(tmp_path: Path) -> None:
    submitted: list[str] = []
    done = asyncio.Event()

    async def submit(text: str) -> None:
        submitted.append(text)
        done.set()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("alt")
        pipe.send_bytes(b"\x1b\r")
        pipe.send_text("shift")
        pipe.send_bytes(b"\x1b\n")
        pipe.send_text("end")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(done.wait(), 1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert submitted == ["alt\nshift\nend"]


@pytest.mark.asyncio
async def test_history_navigation_and_overlay_selection(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.append_string("older")
    history.append_string("newer")
    shown: list[object] = []

    async def show(value: object) -> str:
        shown.append(value)
        return "chosen"

    ctx = _ctx(tmp_path)
    ctx.ui = UIFacade(show_overlay=show)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=ctx,
            history=history,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0.05)
        pipe.send_bytes(b"\x1b[A")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "newer"
        pipe.send_bytes(b"\x12")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "chosen"
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert shown == ["history"]


@pytest.mark.asyncio
async def test_stream_queue_batch_dequeue_abort_and_auto_dispatch(tmp_path: Path) -> None:
    submitted: list[str] = []
    started = asyncio.Event()
    releases = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    async def submit(text: str) -> None:
        index = len(submitted)
        submitted.append(text)
        started.set()
        await releases[index].wait()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("active")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(started.wait(), 1)
        pipe.send_text("queued draft")
        pipe.send_bytes(b"\x11")
        await asyncio.sleep(0.02)
        assert runtime.queue.items == ("queued draft",)
        pipe.send_bytes(b"\x1b\x1b[A")
        await asyncio.sleep(0.02)
        assert runtime.buffer.text == "queued draft"
        pipe.send_bytes(b"\x11")
        releases[0].set()
        await asyncio.sleep(0.05)
        assert submitted[:2] == ["active", "queued draft"]
        pipe.send_text("-> third\n-> fourth")
        pipe.send_bytes(b"\r")
        releases[1].set()
        await asyncio.sleep(0.05)
        assert submitted[:3] == ["active", "queued draft", "third"]
        assert runtime.queue.items == ("fourth",)
        pipe.send_bytes(b"\x1b\x1b")
        await asyncio.sleep(0.6)
        assert runtime.buffer.text == "fourth"
        assert not runtime.streaming
        releases[2].set()
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_ctrl_c_ladder_and_double_escape_tree(tmp_path: Path) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    shown: list[object] = []

    async def submit(_text: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def show(value: object) -> None:
        shown.append(value)

    ctx = _ctx(tmp_path)
    ctx.ui = UIFacade(show_overlay=show)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, ctx=ctx, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("draft")
        pipe.send_bytes(b"\x03")
        await asyncio.sleep(0.02)
        assert runtime.buffer.text == ""
        pipe.send_text("go")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(started.wait(), 1)
        pipe.send_bytes(b"\x03")
        await asyncio.wait_for(cancelled.wait(), 1)
        pipe.send_bytes(b"\x1b\x1b")
        await asyncio.sleep(0.6)
        assert shown == ["tree"]
        pipe.send_bytes(b"\x03")
        await asyncio.sleep(0.02)
        pipe.send_bytes(b"\x03")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_external_editor_draft_restore_completion_and_actions(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("", encoding="utf-8")
    registry = Registry()

    async def help_command(_ctx: object, _args: str) -> None:
        pass

    registry._add_command("core", "help", help_command, "help text")
    ctx = _ctx(tmp_path, registry)
    ctx.plugin_states["composer"] = {"draft": "saved"}
    shown: list[object] = []
    registry._add_command("core", "hello", help_command, "hello text")

    async def show(value: object) -> None:
        shown.append(value)

    ctx.ui = UIFacade(show_overlay=show)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=ctx,
            registry=registry,
            editor_runner=lambda text: text + " edited",
            input=pipe,
            output=DummyOutput(),
        )
        assert runtime.buffer.text == "saved"
        assert "draft" not in ctx.plugin_states["composer"]
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_bytes(b"\x07")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "saved edited"
        runtime.buffer.text = "/h"
        runtime.buffer.cursor_position = len(runtime.buffer.text)
        pipe.send_bytes(b"\t")
        await asyncio.sleep(0.05)
        assert runtime.buffer.complete_state is not None
        pipe.send_bytes(b"\x1b")
        await asyncio.sleep(0.6)
        assert runtime.buffer.complete_state is None
        runtime.buffer.text = "@alp"
        runtime.buffer.cursor_position = len(runtime.buffer.text)
        pipe.send_bytes(b"\t")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "@alpha.py"
        runtime.buffer.text = "alp"
        runtime.buffer.cursor_position = len(runtime.buffer.text)
        pipe.send_bytes(b"\t")
        await asyncio.sleep(0.05)
        assert runtime.buffer.text == "alpha.py"
        pipe.send_bytes(b"\x14")
        pipe.send_bytes(b"\x0f")
        pipe.send_bytes(b"\x1bp")
        await asyncio.sleep(0.05)
        assert runtime.ui.thinking_visible is False
        assert runtime.ui.tools_expanded is True
        assert "model" in shown
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert ctx.plugin_states["composer"]["draft"] == "alpha.py"

@pytest.mark.asyncio
async def test_empty_submit_aborts_stream_and_dispatches_queue_head() -> None:
    submitted: list[str] = []
    active_cancelled = asyncio.Event()
    queued_started = asyncio.Event()
    queued_release = asyncio.Event()

    async def submit(text: str) -> None:
        submitted.append(text)
        if text == "active":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                active_cancelled.set()
                raise
        else:
            queued_started.set()
            await queued_release.wait()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("active")
        pipe.send_bytes(b"\r")
        while not runtime.streaming:
            await asyncio.sleep(0)
        runtime.queue.append("next")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(active_cancelled.wait(), 1)
        await asyncio.wait_for(queued_started.wait(), 1)
        queued_release.set()
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)

    assert submitted == ["active", "next"]



@pytest.mark.asyncio
async def test_provider_and_plugin_actions_are_headlessly_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    registry.providers["able"] = SimpleNamespace(
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
        models=("test", "next"),
        available=lambda: None,
    )
    plugin_calls: list[str] = []

    async def plugin_handler(_ctx: object, _event: object) -> None:
        plugin_calls.append("custom")

    registry._add_keybinding("plugin", "custom", plugin_handler, "c-x")
    ctx = _ctx(tmp_path, registry)
    class Resolver:
        def __init__(self, _registry: object, _cfg: object) -> None:
            pass

        def resolve(self, _alias: str, _role: str) -> object:
            return object()

    monkeypatch.setattr("orcha_agent.tui.runtime.ModelResolver", Resolver)
    ctx.cfg.model = "first"
    ctx.cfg.models = {"first": "able:test", "second": "able:next"}
    switched: list[str] = []

    async def switch_model(model: str) -> None:
        switched.append(model)

    ctx.switch_model = switch_model
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=ctx,
            registry=registry,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_bytes(b"\x1b[Z")
        pipe.send_bytes(b"\x10")
        pipe.send_bytes(b"\x18")
        await asyncio.sleep(0.2)
        assert runtime.thinking_level == "low"
        assert ctx.plugin_states["composer"]["thinking_level"] == "low"
        assert switched == ["second"]
        assert plugin_calls == ["custom"]
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_keys_command_prints_effective_map() -> None:
    from orcha_agent.builtin.commands_core import _keys

    printed: list[object] = []
    ctx = SimpleNamespace(
        ui=SimpleNamespace(effective_keys={"submit": ("enter",), "tree": ("escape escape",)}),
        console=SimpleNamespace(print=printed.append),
    )
    await _keys(ctx, "")
    rendered = str(printed[0])
    assert "submit" in rendered and "enter" in rendered
    assert "tree" in rendered and "escape escape" in rendered


@pytest.mark.asyncio
async def test_ctrl_d_aborts_streaming_turn_before_exit() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def submit(_text: str) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("active")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(started.wait(), 1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(cancelled.wait(), 1)
        await asyncio.wait_for(task, 1)


def test_completion_menu_float_uses_theme_selection_style(tmp_path: Path) -> None:
    theme = load_themes(home=tmp_path)["dark"]
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            theme=theme,
            themes={theme.id: theme},
            input=pipe,
            output=DummyOutput(),
        )
        root = runtime.application.layout.container
        assert len(root.floats) == 1
        assert isinstance(root.floats[0].content, CompletionsMenu)
        selected = runtime.application.style.get_attrs_for_style_str(
            "class:completion-menu.completion.current"
        )
        assert selected.bgcolor is not None


@pytest.mark.parametrize("shape", ["box", "claude", "borderless"])
def test_composer_container_has_exact_dynamic_content_height(
    tmp_path: Path,
    shape: str,
) -> None:
    theme = load_themes(home=tmp_path)["dark"]
    composer = Composer(shape=shape, theme=theme)
    composer.buffer.text = "one\ntwo"

    dimension = to_dimension(composer.container.height)

    expected = composer.height_for_width(80)
    assert (dimension.min, dimension.preferred, dimension.max) == (
        expected,
        expected,
        expected,
    )


@pytest.mark.asyncio
async def test_ctrl_d_preserves_draft_and_queue_without_dispatching_queue(
    tmp_path: Path,
) -> None:
    submitted: list[str] = []
    started = asyncio.Event()
    cancelled = asyncio.Event()
    ctx = _ctx(tmp_path)

    async def submit(text: str) -> None:
        submitted.append(text)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            ctx=ctx,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("active")
        pipe.send_bytes(b"\r")
        await asyncio.wait_for(started.wait(), 1)
        runtime.queue.extend(["queued one", "queued two"])
        pipe.send_text("draft")
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(cancelled.wait(), 1)
        await asyncio.wait_for(task, 1)

    assert submitted == ["active"]
    assert ctx.plugin_states["composer"]["draft"] == "draft"
    assert ctx.plugin_states["composer"]["queue"] == ["queued one", "queued two"]


@pytest.mark.asyncio
async def test_headless_exact_height_counts_scrollbar_wrap_column() -> None:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            composer_shape="borderless",
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        runtime.buffer.text = "x" * 80
        runtime.application.invalidate()
        await asyncio.sleep(0)

        assert runtime.composer.height_for_width(80) == 2
        dimension = to_dimension(runtime.composer.container.height)
        assert (dimension.min, dimension.preferred, dimension.max) == (2, 2, 2)

        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


def test_reconstructed_runtime_restores_and_clears_persisted_queue(
    tmp_path: Path,
) -> None:
    first_ctx = _ctx(tmp_path)
    with create_pipe_input() as first_pipe:
        first = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=first_ctx,
            input=first_pipe,
            output=DummyOutput(),
        )
        first.queue.extend(["queued one", "queued two"])
        first.buffer.text = "saved draft"
        event = SimpleNamespace(
            current_buffer=first.buffer,
            app=SimpleNamespace(exit=lambda: None),
        )
        first._exit(event)

    persisted = {
        name: dict(state)
        for name, state in first_ctx.plugin_states.items()
    }
    persisted_calls: list[None] = []
    resumed_ctx = _ctx(tmp_path)
    resumed_ctx.plugin_states = persisted
    resumed_ctx.persist_plugin_states = lambda: persisted_calls.append(None)

    with create_pipe_input() as second_pipe:
        resumed = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=resumed_ctx,
            input=second_pipe,
            output=DummyOutput(),
        )

    assert resumed.buffer.text == "saved draft"
    assert resumed.queue.items == ("queued one", "queued two")
    assert "draft" not in resumed_ctx.plugin_states["composer"]
    assert "queue" not in resumed_ctx.plugin_states["composer"]
    assert persisted_calls == [None]


@pytest.mark.asyncio
async def test_thinking_controls_resolve_alias_persist_display_and_rebuild_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    registry.providers["able"] = SimpleNamespace(
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
        models=(),
        available=lambda: None,
    )
    ctx = _ctx(tmp_path, registry)
    ctx.cfg.model = "fast"
    ctx.cfg.models = {"fast": "able:test", "next": "able:dynamic"}
    ctx.cfg.providers = {"able": {}}
    ctx.plugin_states["render_default"] = {"thinking": "summary"}
    resolved: list[tuple[str, str]] = []

    class Resolver:
        def __init__(self, _registry: object, _cfg: object) -> None:
            pass

        def resolve(self, alias: str, role: str) -> object:
            resolved.append((alias, role))
            return object()

    monkeypatch.setattr("orcha_agent.tui.runtime.ModelResolver", Resolver)
    rebuilt: list[str] = []

    async def rebuild() -> None:
        rebuilt.append(ctx.cfg.providers["able"]["reasoning_effort"])

    ctx.rebuild = rebuild
    switched: list[str] = []

    async def switch_model(model: str) -> None:
        switched.append(model)

    ctx.switch_model = switch_model
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        ctx=ctx,
        registry=registry,
        input=create_pipe_input().__enter__(),
        output=DummyOutput(),
    )
    try:
        runtime._action_handlers()["toggle_thinking"](SimpleNamespace())
        runtime._action_handlers()["cycle_thinking_level"](SimpleNamespace())
        runtime._action_handlers()["cycle_model"](SimpleNamespace())
        await runtime._drain_pending()

        assert ctx.plugin_states["render_default"]["thinking"] == "off"
        assert runtime.thinking_level == "low"
        assert rebuilt == ["low"]
        assert switched == ["next"]
        assert resolved == [("fast", "main"), ("next", "main")]
    finally:
        await runtime.scheduler.aclose()
        runtime.application.input.close()


def test_queue_recovery_merges_older_prompts_before_active_draft(tmp_path: Path) -> None:
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        input=create_pipe_input().__enter__(),
        output=DummyOutput(),
    )
    try:
        runtime.buffer.text = "active draft"
        runtime.queue.extend(["queued one", "queued two"])
        event = SimpleNamespace(current_buffer=runtime.buffer)

        runtime._dequeue(event)
        assert runtime.buffer.text == "queued two\n\nactive draft"

        runtime.queue.clear()
        runtime.queue.extend(["queued one", "queued two"])
        runtime.streaming = True
        runtime._escape_ladder(event)
        assert runtime.buffer.text == (
            "-> queued one\n-> queued two\n\nqueued two\n\nactive draft"
        )
    finally:
        runtime.application.input.close()


@pytest.mark.asyncio
async def test_shell_error_and_cancellation_always_settle_tool_card(tmp_path: Path) -> None:
    events: list[object] = []
    ctx = _ctx(tmp_path)

    async def capture(event: object) -> None:
        events.append(event)

    ctx._bus.on(object, capture, plugin="test")
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        ctx=ctx,
        input=create_pipe_input().__enter__(),
        output=DummyOutput(),
    )
    try:
        task = asyncio.create_task(runtime._run_shell("exec sleep 10"))
        while runtime._shell_process is None:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)

        ends = [event for event in events if isinstance(event, ToolCallEnd)]
        assert len(ends) == 1
        assert ends[0].result["returncode"] == 130
        assert "cancelled" in ends[0].result["stderr"]
        assert runtime._shell_process is None
    finally:
        await runtime.scheduler.aclose()
        runtime.application.input.close()



@pytest.mark.asyncio
async def test_shell_runner_failure_emits_error_tool_end(tmp_path: Path) -> None:
    events: list[object] = []
    ctx = _ctx(tmp_path)

    async def capture(event: object) -> None:
        events.append(event)

    def fail(_command: str, _cwd: Path, _timeout: float) -> object:
        raise OSError("runner broke")

    ctx._bus.on(object, capture, plugin="test")
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        ctx=ctx,
        shell_runner=fail,
        input=create_pipe_input().__enter__(),
        output=DummyOutput(),
    )
    try:
        await runtime._run_shell("broken")
        ends = [event for event in events if isinstance(event, ToolCallEnd)]
        assert len(ends) == 1
        assert ends[0].result["returncode"] == 1
        assert ends[0].result["stderr"] == "OSError: runner broke"
    finally:
        await runtime.scheduler.aclose()
        runtime.application.input.close()


@pytest.mark.asyncio
async def test_session_rebind_saves_outgoing_state_and_restores_all_runtime_scopes(
    tmp_path: Path,
) -> None:
    old_cwd = tmp_path / "old"
    new_cwd = tmp_path / "new"
    old_cwd.mkdir()
    new_cwd.mkdir()
    history = SQLiteHistory(tmp_path / "history.db", cwd=old_cwd, session_id="old")
    registry = Registry()
    registry.providers["able"] = SimpleNamespace(
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
        available=lambda: None,
    )
    ctx = _ctx(old_cwd, registry)
    ctx.session_id = "old"
    rebuilt: list[str] = []
    ctx.rebuild = lambda: asyncio.sleep(
        0,
        result=rebuilt.append(ctx.cfg.providers["able"]["reasoning_effort"]),
    )
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        ctx=ctx,
        registry=registry,
        history=history,
        input=create_pipe_input().__enter__(),
        output=DummyOutput(),
    )
    try:
        runtime.buffer.text = "old draft"
        runtime.queue.append("old queued")
        runtime.prepare_session_switch()
        assert ctx.plugin_states["composer"] == {
            "draft": "old draft",
            "queue": ["old queued"],
            "thinking_level": "off",
        }

        ctx.cfg.cwd = new_cwd
        ctx.session_id = "new"
        ctx.plugin_states["composer"] = {
            "draft": "new draft",
            "queue": ["new queued"],
            "thinking_level": "high",
        }
        await runtime.rebind_session(SessionSwitch(old="old", new="new"))

        assert runtime.buffer.text == "new draft"
        assert runtime.queue.items == ("new queued",)
        assert runtime.thinking_level == "high"
        assert rebuilt == ["high"]
        assert ctx.cfg.providers["able"]["reasoning_effort"] == "high"
        assert runtime.buffer.completer.path_index.cwd == new_cwd.resolve()
        assert (history.cwd, history.session_id) == (str(new_cwd.resolve()), "new")
    finally:
        await runtime.scheduler.aclose()
        runtime.application.input.close()
