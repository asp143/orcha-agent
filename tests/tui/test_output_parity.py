from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest

from orcha_agent.tui.runtime import StdoutStallWatchdog, _TerminalReplies, _TerminalPump
from orcha_agent.tui.title import TerminalTitle


def test_stall_watchdog_counts_drain_progress_and_hysteresis():
    watch = StdoutStallWatchdog(100, 20, 1)
    assert not watch.sample(101, 0)
    assert not watch.sample(80, 0.8)
    assert not watch.sample(70, 1.5)
    assert not watch.sample(80, 2.4)
    assert watch.sample(80, 2.6)
    assert not watch.sample(20, 3)
    assert not watch.armed


def test_fragmented_reports_do_not_leak_to_composer_or_eat_paste():
    feed, reports = [], []
    parser = _TerminalReplies(feed.append, reports.append)
    for chunk in ["abc\x1b[?", "2026;1$", "y\x1b]11;rgb:ffff/", "ffff/ffff\x1b", "\\xyz"]:
        parser(chunk)
    assert "".join(feed) == "abcxyz"
    assert reports == ["\x1b[?2026;1$y", "\x1b]11;rgb:ffff/ffff/ffff\x1b\\"]
    parser("\x1b[200~\x1b[?2026;1$y\x1b[201~")
    assert "".join(feed).endswith("\x1b[200~\x1b[?2026;1$y\x1b[201~")
    parser("\x1b")
    parser.flush()
    assert feed[-1] == "\x1b"


@pytest.mark.asyncio
async def test_output_pump_order_and_drain():
    stream = StringIO()
    pump = _TerminalPump(stream)
    pump.write("first\n")
    pump.write("second\n")
    for _ in range(100):
        if not pump.pending:
            break
        await asyncio.sleep(0.001)
    assert pump.pending == 0
    assert stream.getvalue() == "first\nsecond\n"
    pump.close()


def test_progress_only_emits_on_turn_transition():
    emitted = []
    output = SimpleNamespace(set_title=lambda _: None, write_raw=emitted.append, flush=lambda: None)
    title = TerminalTitle(output)
    title.set_turn(True)
    title.set_spinner("*")
    title.set_turn(True)
    title.set_turn(False)
    assert emitted == ["\x1b]9;4;3\x07", "\x1b]9;4;0\x07"]


@pytest.mark.asyncio
async def test_synchronized_flush_does_not_block_loop_when_tty_stalls(monkeypatch):
    import threading
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output
    from orcha_agent.tui.runtime import _PaintOutput

    gate = threading.Event()

    class StalledTTY(StringIO):
        def isatty(self):
            return True

        def write(self, text):
            gate.wait(2)
            return super().write(text)

    tty = StalledTTY()
    output = Vt100_Output(tty, lambda: Size(rows=40, columns=120), term="xterm")
    frames = []
    app = SimpleNamespace(output=output, input=None, _redraw=lambda done=False: frames.append(done))
    monkeypatch.setenv("ORCHA_SYNC_OUTPUT", "1")
    paint = _PaintOutput(app)
    try:
        output.write_raw("x" * 300000)
        output.flush()
        await asyncio.sleep(0.01)
        assert paint.pump.pending > 262144
        paint.redraw()
        assert frames == []
        gate.set()
        for _ in range(100):
            if not paint.pump.pending:
                break
            await asyncio.sleep(0.01)
        paint.redraw()
        assert frames == [False]
        assert tty.getvalue().startswith("\x1b[?2026h")
        assert tty.getvalue().endswith("\x1b[?2026l")
        paint.report("\x1b[?2026;0$y")
        assert not paint.synchronized
        paint.report("\x1b[?2026;2$y")
        assert paint.synchronized
    finally:
        gate.set()
        await paint.close()


@pytest.mark.asyncio
async def test_production_rich_console_uses_the_ordered_writer():
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from rich.console import Console
    from orcha_agent.tui.runtime import ApplicationRuntime

    class TTY(StringIO):
        def isatty(self):
            return True

    tty = TTY()
    console = Console(file=tty, force_terminal=True)
    output = Vt100_Output(tty, lambda: Size(rows=40, columns=120), term="xterm")
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0), input=pipe, output=output, console=console
        )
        assert runtime._scrollback.file is runtime._paint_output.pump
        console.print("ordered settled output")
        await runtime._paint_output.close()
    assert "ordered settled output" in tty.getvalue()


def test_last_committed_tool_can_be_expanded_without_recommitting():
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from orcha_agent.tui.frame import Block, BlockState
    from orcha_agent.tui.runtime import ApplicationRuntime

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        runtime._last_tool_card = Block(
            "last",
            "tool",
            state=BlockState.COMMITTED,
            data={
                "name": "bash",
                "args": {"command": "demo"},
                "result": {"stdout": "recalled-output", "exit_code": 0},
            },
        )
        runtime._toggle_last_tool()
        assert "recalled-output" in runtime._viewport_text().value
        assert runtime.frame.blocks == []
        runtime._toggle_last_tool()
        assert "recalled-output" not in runtime._viewport_text().value
