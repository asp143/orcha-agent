from io import StringIO
from types import SimpleNamespace
from typing import Any
from rich.text import Text

import pytest

from langchain_core.messages import AIMessageChunk, ToolMessage
from rich.console import Console
from rich.panel import Panel

from orcha_agent.builtin import render_default
from orcha_agent.core.events import EventBus, ModelChunk, ToolCallEnd, ToolCallStart
from orcha_agent.core.plugin import Handled, PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui import app
from orcha_agent.tui.console import ConsoleOutput


def _api(
    registry: Registry,
    bus: EventBus,
    *,
    config: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> PluginAPI:
    return PluginAPI(
        name="render-default",
        registry=registry,
        bus=bus,
        config={} if config is None else config,
        state={} if state is None else state,
        request_rebuild=lambda: None,
    )


class _CapturingConsole:
    def __init__(self) -> None:
        self.output = StringIO()
        self.renderables: list[object] = []
        self._delegate = ConsoleOutput(
            Console(
                file=self.output,
                force_terminal=False,
                color_system=None,
                width=120,
            )
        )

    def print(self, *objects: object, **kwargs: Any) -> None:
        self.renderables.extend(objects)
        self._delegate.print(*objects, **kwargs)

    def error(self, message: str) -> None:
        self._delegate.error(message)


async def _render(event: object) -> tuple[list[object], str]:
    return await _render_events(event)


async def _render_events(
    *events: object,
    config: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[list[object], str]:
    registry = Registry()
    bus = EventBus()
    render_default.register(
        _api(registry, bus, config=config, state=state)
    )
    console = _CapturingConsole()
    ctx = SimpleNamespace(registry=registry, bus=bus, console=console)

    for event in events:
        await app._render(ctx, event)

    return console.renderables, console.output.getvalue()


@pytest.mark.asyncio
async def test_thinking_command_persists_session_toggle() -> None:
    registry = Registry()
    bus = EventBus()
    renderer_state: dict[str, Any] = {}
    anthropic_state: dict[str, Any] = {}
    render_default.register(
        _api(
            registry,
            bus,
            config={"thinking": "all", "icons": False},
            state=renderer_state,
        )
    )
    persisted: list[dict[str, dict[str, Any]]] = []
    rebuilds: list[None] = []

    async def rebuild() -> None:
        rebuilds.append(None)

    ctx = SimpleNamespace(
        registry=registry,
        bus=bus,
        console=_CapturingConsole(),
        plugin_states={
            "render_default": renderer_state,
            "provider_anthropic": anthropic_state,
        },
        persist_plugin_states=lambda: persisted.append(
            {
                name: dict(state)
                for name, state in ctx.plugin_states.items()
            }
        ),
        rebuild=rebuild,
    )

    assert await app.dispatch_command(registry, ctx, "/thinking off") is True
    assert await app.dispatch_command(registry, ctx, "/thinking on") is True

    assert renderer_state == {"thinking": "summary"}
    assert anthropic_state == {"thinking": "summary"}
    assert persisted == [
        {
            "render_default": {"thinking": "off"},
            "provider_anthropic": {"thinking": "off"},
        },
        {
            "render_default": {"thinking": "summary"},
            "provider_anthropic": {"thinking": "summary"},
        },
    ]
    assert rebuilds == [None, None]


@pytest.mark.asyncio
async def test_openai_reasoning_streams_once_then_separates_answer() -> None:
    renderables, rendered = await _render_events(
        ModelChunk(
            chunk=AIMessageChunk(
                id="response-1",
                content=[
                    {
                        "id": "rs-1",
                        "type": "reasoning",
                        "index": 0,
                        "summary": [
                            {
                                "type": "summary_text",
                                "index": 0,
                                "text": "Check ",
                            }
                        ],
                    }
                ],
            ),
            role="main",
        ),
        ModelChunk(
            chunk=AIMessageChunk(
                id="response-1",
                content=[
                    {
                        "type": "reasoning",
                        "index": 0,
                        "summary": [
                            {
                                "type": "summary_text",
                                "index": 0,
                                "text": "constraints.",
                            }
                        ],
                    }
                ],
            ),
            role="main",
        ),
        ModelChunk(
            chunk=AIMessageChunk(
                id="response-1",
                content=[{"type": "text", "text": "Final answer"}],
            ),
            role="main",
        ),
        config={"thinking": "summary", "icons": False},
    )

    assert rendered == "[thinking]\nCheck constraints.\n\nFinal answer"
    assert rendered.count("[thinking]") == 1
    thinking_text = [value for value in renderables[:2] if isinstance(value, Text)]
    assert thinking_text
    assert all(
        value.style == "dim italic"
        or any(span.style == "dim italic" for span in value.spans)
        for value in thinking_text
    )


@pytest.mark.asyncio
async def test_anthropic_thinking_stream_uses_icon_header() -> None:
    _, rendered = await _render_events(
        ModelChunk(
            chunk=AIMessageChunk(
                id="message-1",
                content=[
                    {
                        "type": "thinking",
                        "index": 0,
                        "thinking": "Plan the answer.",
                    }
                ],
            ),
            role="main",
        ),
        config={"thinking": "summary", "icons": True},
    )

    assert rendered == "󰟶 thinking\nPlan the answer."


@pytest.mark.asyncio
async def test_thinking_off_renders_no_reasoning_content() -> None:
    renderables, rendered = await _render_events(
        ModelChunk(
            chunk=AIMessageChunk(
                id="response-1",
                content=[
                    {
                        "type": "reasoning",
                        "index": 0,
                        "summary": [
                            {"type": "summary_text", "index": 0, "text": "secret"}
                        ],
                    }
                ],
            ),
            role="main",
        ),
        config={"thinking": "off", "icons": False},
    )

    assert renderables == []
    assert rendered == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("summary", ""), ("all", "[thinking]\nSubagent plan")],
)
async def test_subagent_reasoning_requires_all_mode(mode: str, expected: str) -> None:
    _, rendered = await _render_events(
        ModelChunk(
            chunk=AIMessageChunk(
                id="subagent-1",
                content=[
                    {
                        "type": "thinking",
                        "index": 0,
                        "thinking": "Subagent plan",
                    }
                ],
            ),
            role="subagent",
        ),
        config={"thinking": mode, "icons": False},
    )

    assert rendered == expected


@pytest.mark.asyncio
async def test_thinking_stream_separates_following_tool_call() -> None:
    _, rendered = await _render_events(
        ModelChunk(
            chunk=AIMessageChunk(
                id="response-1",
                content=[
                    {
                        "type": "thinking",
                        "index": 0,
                        "thinking": "Inspect first.",
                    }
                ],
            ),
            role="main",
        ),
        ToolCallStart(name="read_file", args={"path": "README.md"}, id="call-1"),
        config={"thinking": "summary", "icons": False},
    )

    assert "Inspect first.\n\n" in rendered
    assert "read_file" in rendered


@pytest.mark.asyncio
async def test_model_chunk_renders_assistant_text_to_string_console() -> None:
    _, rendered = await _render(
        ModelChunk(chunk=AIMessageChunk(content="hello from the agent"), role="main")
    )

    assert "hello from the agent" in rendered


@pytest.mark.asyncio
async def test_model_chunk_renders_responding_model_name_when_present() -> None:
    event = ModelChunk(
        chunk=AIMessageChunk(content="fallback response"),
        role="main",
        model_name="fake:fallback",
    )

    _, rendered = await _render(event)

    assert "fake:fallback" in rendered
    assert "fallback response" in rendered


@pytest.mark.asyncio
async def test_tool_call_start_renders_named_panel_and_arguments() -> None:
    renderables, rendered = await _render(
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


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["edit_file", "write_file"])
async def test_file_tool_result_renders_unified_diff_text(tool_name: str) -> None:
    diff = """--- /notes.txt
+++ /notes.txt
@@ -1 +1 @@
-old line
+new line"""

    _, rendered = await _render(
        ToolCallEnd(name=tool_name, id="call-2", result=diff)
    )

    assert "--- /notes.txt" in rendered
    assert "+++ /notes.txt" in rendered
    assert "-old line" in rendered
    assert "+new line" in rendered


@pytest.mark.asyncio
async def test_execute_result_renders_stdout_stderr_and_exit_code() -> None:
    result = ToolMessage(
        content="build output\nbuild warning",
        name="execute",
        tool_call_id="call-3",
        artifact={"exit_code": 2},
        status="success",
    )

    renderables, rendered = await _render(
        ToolCallEnd(name="execute", id="call-3", result=result)
    )

    assert any(isinstance(value, Panel) for value in renderables)
    assert "build output" in rendered
    assert "build warning" in rendered
    assert "2" in rendered


@pytest.mark.asyncio
async def test_tool_error_text_is_rendered() -> None:
    result = ToolMessage(
        content="Error: file does not exist",
        name="read_file",
        tool_call_id="call-4",
        status="error",
    )

    _, rendered = await _render(
        ToolCallEnd(name="read_file", id="call-4", result=result)
    )

    assert "Error: file does not exist" in rendered


@pytest.mark.asyncio
async def test_priority_handler_returning_handled_suppresses_renderer() -> None:
    registry = Registry()
    bus = EventBus()
    render_default.register(_api(registry, bus))
    handler_calls: list[str] = []

    async def handle(_event: ModelChunk) -> Handled:
        handler_calls.append("handled")
        return Handled()

    bus.on(ModelChunk, handle, priority=0)
    console = _CapturingConsole()
    ctx = SimpleNamespace(registry=registry, bus=bus, console=console)
    event = ModelChunk(
        chunk=AIMessageChunk(content="must not render"),
        role="main",
    )

    was_handled = await app._render(ctx, event)

    assert was_handled is True
    assert handler_calls == ["handled"]
    assert console.renderables == []
    assert console.output.getvalue() == ""
