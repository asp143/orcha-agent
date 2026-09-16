from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from orcha_agent.core.tools.middleware import NativeOutputMiddleware, rewrite_result


NOTICE = (
    "Tool result too large, the result of this tool call call-1 was saved in the filesystem at this path: /workspace/.orcha/artifacts/large_tool_results/call-1\n"
    "Use read_file with offset and limit.\n"
    "Here is a preview showing the head and tail of the result:\nfirst\nlast"
)


def test_overflow_translation_preserves_messages_commands_and_preview() -> None:
    message = ToolMessage(
        content=NOTICE,
        tool_call_id="call-1",
        id="message-1",
        status="error",
        artifact={"raw": "value"},
    )
    command = Command(update={"messages": [message], "other": 42}, goto="next")
    rewritten = rewrite_result(command)
    result = rewritten.update["messages"][0]
    assert "read_file" not in result.content
    assert ":1+100" in result.content
    assert result.content.endswith("first\nlast")
    assert result.id == message.id
    assert result.status == "error"
    assert result.artifact == message.artifact
    assert rewritten.goto == "next"
    assert rewritten.update["other"] == 42
    ordinary = ToolMessage(content="read_file in user source code", tool_call_id="ordinary")
    assert rewrite_result(ordinary) is ordinary


@pytest.mark.asyncio
async def test_model_retry_boundary_translates_multimodal_notice_without_extra_graph_nodes() -> (
    None
):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    message = ToolMessage(content=[{"type": "text", "text": NOTICE}, image], tool_call_id="call-1")
    middleware = NativeOutputMiddleware()
    request = SimpleNamespace(
        messages=[message], override=lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def handler(value):
        assert "read_file" not in value.messages[0].content[0]["text"]
        assert value.messages[0].content[1] == image
        return "ok"

    async def async_handler(value):
        return handler(value)

    assert middleware.wrap_model_call(request, handler) == "ok"
    assert await middleware.awrap_model_call(request, async_handler) == "ok"
