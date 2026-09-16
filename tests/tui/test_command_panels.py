"""Committed command summaries must stay visible without duplicate history."""

from __future__ import annotations

import asyncio
from io import StringIO

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from orcha_agent.tui.frame import BlockState
from orcha_agent.tui.panels import summary_panel, table_panel
from orcha_agent.tui.runtime import ApplicationRuntime


class SizedOutput(DummyOutput):
    def __init__(self, width, height):
        self.size = Size(rows=height, columns=width)

    def get_size(self):
        return self.size


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "height"), [(80, 30), (120, 40)])
async def test_committed_panels_stay_visible_and_retire_once(width, height):
    stream = StringIO()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=SizedOutput(width, height),
            console=Console(file=stream, width=width),
        )
        first = runtime.transcript.print(
            summary_panel("Status", ("Segment", "Value"), [("model", "fixture-model")])
        )
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        assert first.state is BlockState.COMMITTED
        assert first not in runtime.frame.blocks
        viewport = Text.from_ansi(runtime._viewport_text().value).plain
        assert "Status" in viewport and "fixture-model" in viewport
        assert "fixture-model" not in stream.getvalue()
        runtime.transcript.append_banner("background notice", level="info")
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        assert "fixture-model" in Text.from_ansi(runtime._viewport_text().value).plain
        runtime.transcript.print(
            summary_panel("Providers", ("Name", "Auth"), [("codex", "logged in")])
        )
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        assert stream.getvalue().count("fixture-model") == 1
        assert stream.getvalue().index("fixture-model") < stream.getvalue().index(
            "background notice"
        )
        assert "Providers" in Text.from_ansi(runtime._viewport_text().value).plain
        assert "Status" not in Text.from_ansi(runtime._viewport_text().value).plain
        await runtime._clear_scrollback()
        assert not runtime._retained_panels
        await runtime.scheduler.aclose()


@pytest.mark.parametrize("factory", ["summary", "existing"])
def test_command_panel_columns_have_two_column_gutters(factory):
    from rich.table import Table

    if factory == "summary":
        panel = summary_panel(
            "Providers", ("Available", "Auth / Keys", "Flags"), [("yes", "logged in", "T S R O")]
        )
    else:
        table = Table(title="Providers", padding=0)
        for name in ("Available", "Auth / Keys", "Flags"):
            table.add_column(name)
        table.add_row("yes", "logged in", "T S R O")
        panel = table_panel(table)
    stream = StringIO()
    Console(file=stream, width=80, color_system=None).print(panel)
    text = stream.getvalue()
    assert "Available  " in text
    assert "Auth / Keys  " in text
    assert "logged in  " in text
    assert "AvailableAuth" not in text


@pytest.mark.asyncio
async def test_user_output_retires_panel_exactly_once():
    stream = StringIO()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=SizedOutput(80, 30),
            console=Console(file=stream, width=80),
        )
        runtime.transcript.print(summary_panel("Status", ("Field",), [("retained-marker",)]))
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        block = runtime.frame.add("user", {"text": "local fixture only"})
        block.settle()
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        assert not runtime._retained_panels
        assert stream.getvalue().count("retained-marker") == 1
        await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_session_switch_drains_pending_panels_before_rebinding():
    stream = StringIO()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=SizedOutput(80, 30),
            console=Console(file=stream, width=80),
        )
        runtime.transcript.print(summary_panel("Old session", ("Field",), [("old-marker",)]))
        await runtime.rebind_session(None)
        runtime.scheduler.commit_now()
        await runtime._drain_pending()
        assert not runtime._retained_panels
        assert "old-marker" not in Text.from_ansi(runtime._viewport_text().value).plain
        assert stream.getvalue().count("old-marker") == 1
        await runtime.scheduler.aclose()
