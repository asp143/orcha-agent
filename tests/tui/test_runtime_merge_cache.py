"""Rendered-row caching must respect parity's live settings and per-card expansion."""

import asyncio

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.text import Text

from orcha_agent.tui.runtime import ApplicationRuntime


@pytest.mark.asyncio
async def test_cached_rows_follow_same_id_theme_reload_and_individual_expansion():
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
            theme={"id": "custom", "marker": "before"},
        )
        calls = []

        def render(block, theme, width, rows, expanded):
            calls.append((theme["marker"], expanded))
            return Text(f"{theme['marker']} {'expanded' if expanded else 'collapsed'}")

        runtime._block_dispatcher._renderers["tool"] = render
        block = runtime.frame.add("tool", {"name": "read"})

        def capture():
            return runtime._capture_block(block, 80, 20, force_terminal=True)

        try:
            assert "before collapsed" in capture()
            assert "before collapsed" in capture()
            assert len(calls) == 1
            runtime._expanded_tool_id = block.id
            assert "before expanded" in capture()
            replacement = {"id": "custom", "marker": "after"}
            runtime.replace_themes({"custom": replacement}, replacement)
            assert "after expanded" in capture()
            assert len(calls) == 3
        finally:
            await runtime.scheduler.aclose()
