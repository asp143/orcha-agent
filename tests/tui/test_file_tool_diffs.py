from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from rich.console import Console

from orcha_agent.core.events import Event, EventBus, ToolCallStart
from orcha_agent.core.registry import Registry
from orcha_agent.tui import turn
from orcha_agent.tui.blocks import DEFAULT_THEME
from orcha_agent.tui.blocks.tool import render as render_tool
from orcha_agent.tui.frame import Frame
from orcha_agent.tui.transcript import Transcript

def _plain(renderable: object, width: int) -> str:
    output = StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    ).print(renderable)
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "args", "content"),
    [
        (
            "write_file",
            {"file_path": "/demo.py", "content": "value = 'new'\n"},
            "Updated file /demo.py",
        ),
        (
            "edit_file",
            {
                "file_path": "/demo.py",
                "old_string": "old",
                "new_string": "new",
            },
            "Successfully replaced 1 instance(s) of the string in '/demo.py'",
        ),
    ],
)
async def test_real_file_tool_message_reaches_unified_diff_renderer(
    tmp_path: Path,
    name: str,
    args: dict[str, object],
    content: str,
) -> None:
    path = tmp_path / "demo.py"
    path.write_text("value = 'old'\n")
    event = ToolCallStart(
        name=name,
        args=args,
        id=f"{name}-1",
        source_id="main",
    )
    capture = turn._FileDiffCapture(tmp_path)
    capture.start(event)

    path.write_text("value = 'new'\n")
    result = ToolMessage(
        content=content,
        name=name,
        tool_call_id=event.id,
        status="success",
    )
    frame = Frame()
    transcript = Transcript(frame)
    await transcript.handle(event)
    bus = EventBus()
    bus.on(Event, transcript.handle, plugin="<test>")
    ctx = SimpleNamespace(
        _bus=bus,
        bus=bus,
        transcript=transcript,
        registry=Registry(),
    )

    await turn._updates_event(
        ctx,
        {"messages": [result]},
        set(),
        set(),
        file_diffs=capture,
    )

    rendered = _plain(
        render_tool(frame.blocks[0], DEFAULT_THEME, 100, 50, True),
        100,
    )
    assert "/demo.py" in rendered
    assert "new" in rendered
    if name == "edit_file":
        assert "old" in rendered
        assert "-  1│" in rendered
        assert "+  1│" in rendered
    else:
        assert "old" not in rendered
        assert "✎ Write" in rendered


def test_non_filesystem_turn_context_does_not_require_cfg() -> None:
    ctx = SimpleNamespace()
    event = ToolCallStart(
        name="execute",
        args={"command": "true"},
        id="execute-1",
    )

    assert turn._start_file_diff_capture(ctx, event, None) is None
