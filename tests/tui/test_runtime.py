from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade


@pytest.mark.asyncio
async def test_application_is_headlessly_driveable_through_submit_and_exit() -> None:
    submitted: list[str] = []
    submitted_event = asyncio.Event()

    async def submit(text: str) -> None:
        submitted.append(text)
        submitted_event.set()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("hello\n")
        await asyncio.wait_for(submitted_event.wait(), timeout=1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, timeout=1)

    assert submitted == ["hello"]
    assert runtime.application.full_screen is False


@pytest.mark.asyncio
async def test_ui_facade_controls_runtime_state_and_overlay_results() -> None:
    shown: list[object] = []

    async def show(value: object) -> str:
        shown.append(value)
        return "selected"

    facade = UIFacade(show_overlay=show)
    facade.notify("saved")
    facade.toggle_thinking()
    facade.expand_tools(True)

    assert await facade.show("picker") == "selected"
    assert await facade.ask([{"question": "Choose"}]) == "selected"
    assert shown == ["picker", [{"question": "Choose"}]]
    assert facade.notifications == ["saved"]
    assert facade.thinking_visible is False
    assert facade.tools_expanded is True
