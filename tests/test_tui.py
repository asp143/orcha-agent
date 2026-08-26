from langchain_core.messages import AIMessageChunk

from orcha_agent.tui.app import _ToolCallBuffer


def test_tool_call_chunks_are_buffered_until_arguments_form_complete_json() -> None:
    buffer = _ToolCallBuffer()
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "write_file",
                "args": '{"file_path":"/notes.txt",',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    second = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": None,
                "args": '"content":"hello"}',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )

    assert buffer.add(first) == []
    events = buffer.add(second)

    assert len(events) == 1
    assert events[0].name == "write_file"
    assert events[0].id == "call-1"
    assert events[0].args == {"file_path": "/notes.txt", "content": "hello"}
