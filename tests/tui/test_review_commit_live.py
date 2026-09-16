"""Exercise the real prompt-toolkit suspension/redraw transaction."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from orcha_agent.tui.frame import BlockState
from orcha_agent.tui.runtime import ApplicationRuntime


@pytest.mark.asyncio
async def test_live_commit_keeps_erase_settled_and_composer_in_one_sync_frame(
    monkeypatch, wait_until
):
    class TTY(StringIO):
        def isatty(self):
            return True

    tty = TTY()
    output = Vt100_Output(tty, lambda: Size(rows=40, columns=100), term="xterm", enable_cpr=False)
    monkeypatch.setenv("ORCHA_SYNC_OUTPUT", "1")
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0),
            input=pipe,
            output=output,
            console=Console(file=tty, force_terminal=True),
            status=lambda: "ready",
        )
        runtime.buffer.text = "LIVE-COMPOSER"
        task = asyncio.create_task(runtime.run())
        try:
            await wait_until(
                lambda: runtime.application.is_running and runtime.application.render_counter > 0
            )
            block = runtime.frame.add(
                "raw", {"renderable": "UNIQUE-SETTLED-COMMIT"}, state=BlockState.COMMITTED
            )
            runtime._commit_blocks([block])
            await runtime._drain_pending()
            for _ in range(100):
                if not runtime._paint_output.pump.pending:
                    break
                await asyncio.sleep(0.001)
            rendered = tty.getvalue()
            marker = rendered.index("UNIQUE-SETTLED-COMMIT")
            opening = rendered.rfind("\x1b[?2026h", 0, marker)
            closing = rendered.index("\x1b[?2026l", marker)
            frame = rendered[opening:closing]
            assert "\x1b[J" in frame[: frame.index("UNIQUE-SETTLED-COMMIT")]
            assert "LIVE-COMPOSER" in frame[frame.index("UNIQUE-SETTLED-COMMIT") :]
            assert frame.count("\x1b[?2026h") == 1
            assert "\x1b[?2026l" not in frame
        finally:
            runtime.application.exit()
            await task
