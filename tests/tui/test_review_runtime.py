"""Review-round regressions at Rich/PT and terminal transaction boundaries."""

from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text import ANSI, fragment_list_to_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.runtime import ApplicationRuntime, _PaintOutput
from orcha_agent.tui.theme import load_themes


@pytest.mark.parametrize(
    "name,args,result",
    [
        ("read", {"path": "src/example.py:2-4"}, "print('hello')"),
        ("grep", {"pattern": "hello", "path": "src"}, "src/example.py:2:hello"),
        ("ls", {"path": "src"}, {"entries": ["src/example.py"]}),
    ],
)
def test_viewport_strips_osc_links_but_commits_keep_them(tmp_path, name, args, result):
    theme = load_themes(home=tmp_path)["dark"]
    block = Block(
        "linked",
        "tool",
        state=BlockState.SETTLED,
        data={"name": name, "args": args, "result": result},
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0), theme=theme, input=pipe, output=DummyOutput()
        )
        captured = runtime._capture_block(block, 100, 20, force_terminal=True)
        plain = fragment_list_to_text(ANSI(captured).__pt_formatted_text__())
        assert "file://" not in plain
        assert "8;id=" not in plain
        assert "\x1b]8;" not in captured
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, color_system="truecolor")
        runtime._print_block(console, block, 100, 20, viewport=False)
        assert "file://" in stream.getvalue()


def test_idle_probe_never_invalidates_but_recovers_a_skipped_frame():
    calls = []
    app = SimpleNamespace(
        output=DummyOutput(),
        input=None,
        _redraw=lambda done=False: calls.append("draw"),
        invalidate=lambda: calls.append("invalidate"),
    )
    paint = _PaintOutput(app)
    paint.pump = SimpleNamespace(pending=0, error=None)
    for tick in range(8):
        paint._sample_output(now=tick / 4, lag=0)
    assert calls == []
    paint.pump.pending = 300000
    paint.redraw()
    assert calls == []
    paint.pump.pending = 0
    paint._sample_output(now=3, lag=0)
    assert calls == ["invalidate"]
    paint.redraw()
    paint._sample_output(now=3.25, lag=0)
    assert calls == ["invalidate", "draw"]


def test_probe_invalidates_only_on_degraded_transition():
    calls = []
    app = SimpleNamespace(
        output=DummyOutput(),
        input=None,
        _redraw=lambda done=False: None,
        invalidate=lambda: calls.append("invalidate"),
    )
    paint = _PaintOutput(app)
    paint._sample_output(now=0, lag=0.3)
    paint._sample_output(now=0.5, lag=0.3)
    paint._sample_output(now=1, lag=0)
    paint._sample_output(now=1.25, lag=0)
    assert calls == ["invalidate", "invalidate"]


@pytest.mark.asyncio
async def test_commit_brackets_erase_content_and_redraw_as_one_frame(monkeypatch):
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    class TTY(StringIO):
        def isatty(self):
            return True

    tty = TTY()
    output = Vt100_Output(tty, lambda: Size(rows=40, columns=100), term="xterm")
    monkeypatch.setenv("ORCHA_SYNC_OUTPUT", "1")
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0),
            input=pipe,
            output=output,
            console=Console(file=tty, force_terminal=True),
        )
        paint = runtime._paint_output

        async def terminal(callback):
            output.write_raw("ERASE")
            output.flush()
            callback()
            output.write_raw("REDRAW")
            output.flush()

        runtime._run_in_app_terminal = terminal
        block = runtime.frame.add("raw", {"renderable": "SETTLED"}, state=BlockState.COMMITTED)
        try:
            runtime._commit_blocks([block])
            await runtime._drain_pending()
            for _ in range(100):
                if not paint.pump.pending:
                    break
                await asyncio.sleep(0.001)
            rendered = tty.getvalue()
            start = rendered.index("ERASE")
            end = rendered.index("REDRAW") + len("REDRAW")
            frame = rendered[rendered.rfind("\x1b[?2026h", 0, start) : end + len("\x1b[?2026l")]
            assert frame.startswith("\x1b[?2026hERASE")
            assert frame.endswith("REDRAW\x1b[?2026l")
            assert "SETTLED" in frame
            assert frame.count("\x1b[?2026h") == 1
            assert frame.count("\x1b[?2026l") == 1
        finally:
            await paint.close()


def test_mouse_scroll_mode_preserves_native_selection():
    from prompt_toolkit.mouse_events import MouseEventType

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        focused = []
        runtime.application.layout.focus = lambda target: focused.append(target)
        assert runtime._mouse_mode() == "scroll"
        assert not runtime.application.mouse_support()
        runtime._viewport_mouse(SimpleNamespace(event_type=MouseEventType.SCROLL_UP))
        assert runtime._viewport_scroll == 3
        assert (
            runtime._viewport_mouse(SimpleNamespace(event_type=MouseEventType.MOUSE_UP))
            is NotImplemented
        )
        assert focused == []
        runtime._tui_config = SimpleNamespace(mouse="full")
        assert runtime.application.mouse_support()
        runtime._viewport_mouse(SimpleNamespace(event_type=MouseEventType.MOUSE_UP))
        assert focused == [runtime.buffer]
        runtime._tui_config = SimpleNamespace(mouse="off")
        assert not runtime.application.mouse_support()
        assert (
            runtime._viewport_mouse(SimpleNamespace(event_type=MouseEventType.SCROLL_UP))
            is NotImplemented
        )
        assert runtime._viewport_scroll == 3


@pytest.mark.asyncio
async def test_clear_and_session_switch_discard_terminal_replay_and_last_card(monkeypatch):
    from orcha_agent.core.events import SessionSwitch

    calls = []
    monkeypatch.setattr(
        "orcha_agent.tui.runtime.clear_terminal_cache", lambda: calls.append("clear")
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
            console=Console(file=StringIO()),
        )
        for action in (
            runtime._clear_scrollback,
            lambda: runtime.rebind_session(SessionSwitch(old="old", new="new")),
        ):
            runtime._last_tool_card = Block("old", "tool", state=BlockState.COMMITTED)
            runtime._expanded_tool_id = "old"
            await action()
            assert runtime._last_tool_card is None
            assert runtime._expanded_tool_id is None
        assert calls == ["clear", "clear"]


def test_colorblind_live_toggle_restores_original_theme(tmp_path):
    from dataclasses import replace
    from orcha_agent.core.config import TuiConfig

    theme = load_themes(home=tmp_path)["dark"]
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput(), theme=theme
        )
        runtime._tui_config = TuiConfig(colorblind=True)
        runtime._apply_theme(theme)
        assert runtime.theme.colors["toolDiffAdded"] != theme.colors["toolDiffAdded"]
        runtime._tui_config = replace(runtime._tui_config, colorblind=False)
        runtime._apply_theme(runtime.theme)
        assert runtime.theme.colors == theme.colors


def test_real_runtime_brand_observes_current_activity():
    from time import monotonic
    from orcha_agent.tui.statusline import brand_segment

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        ctx = SimpleNamespace(
            ui=runtime.ui, plugin_states={"statusbar": {"_turn_started": monotonic()}}
        )
        runtime.frame.add("thinking", {"text": "checking"})
        assert "thinking" in brand_segment(ctx).text
        tool = runtime.frame.add("tool", {"name": "bash"})
        assert "bash" in brand_segment(ctx).text
        tool.settle()
        assert "bash" not in brand_segment(ctx).text
