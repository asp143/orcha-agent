from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from rich.console import Console
from rich.text import Text

from orcha_agent.builtin.usage import register
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.core.usage_store import usage_table
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.theme import load_themes
from tests.tui.test_command_panels import SizedOutput


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "height"), [(80, 30), (120, 40)])
@pytest.mark.parametrize("theme_name", ["dark", "light"])
async def test_usage_command_remains_in_viewport(tmp_path, width, height, theme_name):
    registry = Registry()
    register(
        PluginAPI(
            name="usage",
            config={},
            state={},
            registry=registry,
            bus=EventBus(),
            request_rebuild=lambda: None,
        )
    )

    async def submit(_text):
        raise AssertionError("No model request is allowed")

    with SessionStore(tmp_path / "usage.db") as session, create_pipe_input() as pipe:
        info = session.create(tmp_path, "fake:model")
        runtime = ApplicationRuntime(
            submit,
            input=pipe,
            output=SizedOutput(width, height),
            console=Console(file=StringIO(), width=width),
            theme=load_themes(home=tmp_path)[theme_name],
        )
        ctx = SimpleNamespace(
            session=session, session_id=info.thread_id, console=runtime.transcript
        )
        try:
            await registry.commands["usage"].handler(ctx, "all")
            runtime.scheduler.commit_now()
            await runtime._drain_pending()
            viewport = Text.from_ansi(runtime._viewport_text().value).plain
            assert "Usage · all" in viewport
            assert "Total" in viewport
            assert "$0.0000" in viewport
            assert len(viewport.splitlines()) <= height
            assert runtime._retained_panels
            # CLI stats still receives the original bordered Table.
            table = usage_table([], "all")
            assert table.title == "Usage · all (estimated API cost)"
            assert table.box is not None
        finally:
            await runtime.scheduler.aclose()
