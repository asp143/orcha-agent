from io import StringIO
from typing import Any

import pytest

from langchain_core.messages import AIMessageChunk, ToolMessage
from rich.console import Console
from rich.panel import Panel

from orcha_agent.builtin import render_default
from orcha_agent.core.events import EventBus, ModelChunk, ToolCallEnd, ToolCallStart
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.console import ConsoleOutput


def _api(registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name="render-default",
        registry=registry,
        bus=bus,
        config={},
        state={},
        request_rebuild=lambda: None,
    )


def _matches(match: Any, event: object) -> bool:
    if callable(match):
        return bool(match(event))
    return match == type(event).__name__ or match == getattr(event, "name", None)


def _render(event: object) -> tuple[list[object], str]:
    registry = Registry()
    bus = EventBus()
    render_default.register(_api(registry, bus))

    output = StringIO()
    console = ConsoleOutput(
        Console(file=output, force_terminal=False, color_system=None, width=120)
    )
    renderables: list[object] = []
    for registration in registry.renderers:
        if not _matches(registration.match, event):
            continue
        rendered = registration.render(event)
        if rendered is not None:
            renderables.append(rendered)
            console.print(rendered)
    return renderables, output.getvalue()


def test_model_chunk_renders_assistant_text_to_string_console() -> None:
    _, rendered = _render(ModelChunk(chunk=AIMessageChunk(content="hello from the agent"), role="main"))

    assert "hello from the agent" in rendered


def test_model_chunk_renders_responding_model_name_when_present() -> None:
    event = ModelChunk(
        chunk=AIMessageChunk(content="fallback response"),
        role="main",
        model_name="fake:fallback",
    )

    _, rendered = _render(event)

    assert event.model_name == "fake:fallback"
    assert "fake:fallback" in rendered
    assert "fallback response" in rendered


def test_tool_call_start_renders_named_panel_and_arguments() -> None:
    renderables, rendered = _render(
        ToolCallStart(
            name="write_file",
            args={"file_path": "/notes.txt", "content": "hello"},
            id="call-1",
        )
    )

    assert any(isinstance(value, Panel) for value in renderables)
    assert "⚙" in rendered
    assert "write_file" in rendered
    assert "/notes.txt" in rendered
    assert "hello" in rendered


@pytest.mark.parametrize("tool_name", ["edit_file", "write_file"])
def test_file_tool_result_renders_unified_diff_text(tool_name: str) -> None:
    diff = """--- /notes.txt
+++ /notes.txt
@@ -1 +1 @@
-old line
+new line"""

    _, rendered = _render(ToolCallEnd(name=tool_name, id="call-2", result=diff))

    assert "--- /notes.txt" in rendered
    assert "+++ /notes.txt" in rendered
    assert "-old line" in rendered
    assert "+new line" in rendered


def test_execute_result_renders_stdout_stderr_and_exit_code() -> None:
    result = ToolMessage(
        content="build output\nbuild warning",
        name="execute",
        tool_call_id="call-3",
        artifact={"exit_code": 2},
        status="success",
    )

    renderables, rendered = _render(ToolCallEnd(name="execute", id="call-3", result=result))

    assert any(isinstance(value, Panel) for value in renderables)
    assert "build output" in rendered
    assert "build warning" in rendered
    assert "2" in rendered


def test_tool_error_text_is_rendered() -> None:
    result = ToolMessage(
        content="Error: file does not exist",
        name="read_file",
        tool_call_id="call-4",
        status="error",
    )

    _, rendered = _render(ToolCallEnd(name="read_file", id="call-4", result=result))

    assert "Error: file does not exist" in rendered
