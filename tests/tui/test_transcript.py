from __future__ import annotations

import asyncio
import pytest
from langchain_core.messages import AIMessageChunk
from rich.text import Text

from orcha_agent.core.events import (
    ModelChunk,
    ThreadSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.registry import Registry
from orcha_agent.tui.console import ConsoleOutput
from orcha_agent.tui.frame import BlockState, Frame, FrameScheduler
from orcha_agent.tui.transcript import Transcript


@pytest.mark.asyncio
async def test_transcript_maps_turn_stream_tool_and_thread_events() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(TurnStart(thread_id="thread", text="question"))
    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(
                content=[
                    {"type": "reasoning", "summary": [{"text": "plan"}]},
                    {"type": "text", "text": "answer"},
                ]
            ),
            role="subagent",
            source_id="researcher/one",
        )
    )
    await transcript.handle(
        ToolCallStart(name="read_file", args={"path": "a.py"}, id="call-1", source_id="researcher/one")
    )
    await transcript.handle(ToolCallEnd(name="read_file", id="call-1", result="ok"))
    await transcript.handle(
        ThreadSwitch(session_id="session", old="thread", new="thread.1", reason="branch")
    )
    await transcript.handle(TurnEnd(thread_id="thread"))

    assert [block.kind for block in frame.blocks] == [
        "user",
        "thinking",
        "assistant",
        "tool",
        "marker",
    ]
    assert frame.blocks[0].state is BlockState.COMMITTED
    assert frame.blocks[1].data["text"] == "plan"
    assert frame.blocks[2].data == {
        "text": "answer",
        "role": "subagent",
        "subagent": True,
    }
    assert frame.blocks[3].data["result"] == "ok"
    assert frame.blocks[4].data["text"] == "⎇ branched to thread.1"
    assert all(block.state is not BlockState.ACTIVE for block in frame.blocks)


@pytest.mark.asyncio
async def test_turn_start_resets_source_and_tool_accumulators() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    for text in ("first", "second"):
        await transcript.handle(TurnStart(thread_id="thread", text="question"))
        await transcript.handle(
            ModelChunk(
                chunk=AIMessageChunk(content=text),
                role="main",
                source_id="main",
            )
        )
        await transcript.handle(
            ToolCallStart(name="read", args={}, id="reused", source_id="main")
        )
        await transcript.handle(ToolCallEnd(name="read", id="reused", result=text))
        await transcript.handle(TurnEnd(thread_id="thread"))

    assert [
        block.data["text"]
        for block in frame.blocks
        if block.kind == "assistant"
    ] == ["first", "second"]
    assert [
        block.data["result"]
        for block in frame.blocks
        if block.kind == "tool"
    ] == ["first", "second"]


def test_error_banner_truncation_includes_marker_within_eight_lines() -> None:
    transcript = Transcript(Frame())

    block = transcript.append_banner("\n".join(str(index) for index in range(12)))

    assert block.data["message"].splitlines() == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "…",
    ]


@pytest.mark.asyncio
async def test_console_and_banner_commits_share_the_coalesced_flush() -> None:
    batches: list[list[str]] = []
    frame = Frame()
    scheduler = FrameScheduler(
        frame,
        commit=lambda blocks: batches.append([block.kind for block in blocks]),
        invalidate=lambda: None,
    )
    output = ConsoleOutput(transcript=Transcript(frame, scheduler=scheduler))

    output.print("one")
    output.warning("two")
    await asyncio.sleep(0.06)

    assert batches == [["raw", "banner"]]
    await scheduler.aclose()


def test_console_routes_to_transcript_and_keeps_direct_fallback() -> None:
    frame = Frame()
    transcript = Transcript(frame)
    output = ConsoleOutput(transcript=transcript)

    output.print("plain")
    output.warning("careful")
    output.error("broken")

    assert [block.kind for block in frame.blocks] == ["raw", "banner", "banner"]
    assert [block.data["level"] for block in frame.blocks[1:]] == ["warning", "error"]
    assert all(block.state is BlockState.COMMITTED for block in frame.blocks)

    class RecordingConsole:
        def __init__(self) -> None:
            self.values: list[tuple[object, ...]] = []

        def print(self, *objects: object, **_kwargs: object) -> None:
            self.values.append(objects)

    fallback = RecordingConsole()
    ConsoleOutput(fallback).print("startup")
    assert fallback.values == [("startup",)]


@pytest.mark.asyncio
async def test_legacy_renderer_is_adapted_to_a_committed_raw_block() -> None:
    registry = Registry()
    registry._add_renderer(
        "legacy",
        "ModelChunk",
        lambda _event: Text("legacy output"),
    )
    frame = Frame()
    transcript = Transcript(frame, registry=registry)

    await transcript.handle(
        ModelChunk(chunk=AIMessageChunk(content="new"), role="main", source_id="main")
    )

    assert len(frame.blocks) == 1
    assert frame.blocks[0].kind == "raw"
    assert frame.blocks[0].data["renderable"].plain == "legacy output"
    assert frame.blocks[0].state is BlockState.COMMITTED
