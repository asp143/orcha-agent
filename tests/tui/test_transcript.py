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
async def test_tool_blocks_expose_live_elapsed_then_settled_duration() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(
        ToolCallStart(name="execute", args={"command": "sleep"}, id="timed")
    )
    block = frame.blocks[-1]
    assert block.data["elapsed"] == 0.0

    block.created -= 2.0
    await transcript.handle(ToolCallEnd(name="execute", id="timed", result="ok"))

    assert "elapsed" not in block.data
    assert isinstance(block.data["duration"], float)
    assert block.data["duration"] >= 2.0



@pytest.mark.asyncio
async def test_working_indicator_disappears_on_first_visible_model_output() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(TurnStart(thread_id="thread", text="question"))

    assert [block.kind for block in frame.blocks] == ["user", "working"]
    working = frame.blocks[-1]
    assert working.state is BlockState.ACTIVE
    assert working.data["message"] == "Working… (Esc to interrupt)"

    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(content="answer"),
            role="main",
            source_id="main",
        )
    )

    assert [block.kind for block in frame.blocks] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_working_indicator_disappears_on_first_tool_output() -> None:
    frame = Frame()
    transcript = Transcript(frame)
    await transcript.handle(TurnStart(thread_id="thread", text="question"))
    assert frame.blocks[-1].kind == "working"

    await transcript.handle(ToolCallStart(name="read", args={}, id="call"))

    assert [block.kind for block in frame.blocks] == ["user", "tool"]


@pytest.mark.asyncio
async def test_retry_indicator_uses_warning_countdown_on_shared_ticker() -> None:
    frame = Frame()
    scheduler = FrameScheduler(
        frame,
        commit=lambda _blocks: None,
        invalidate=lambda: None,
    )
    transcript = Transcript(frame, scheduler=scheduler)

    transcript.show_retry(
        attempt=2,
        max_attempts=5,
        delay_seconds=4,
        now=10,
    )
    retry = frame.blocks[-1]
    assert retry.data["message"] == "Retrying (2/5) in 4s…"
    assert retry.data["level"] == "warning"

    scheduler.tick_spinners(now=11.1)

    await scheduler.aclose()
    assert retry.data["message"] == "Retrying (2/5) in 3s…"

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


@pytest.mark.asyncio
async def test_consecutive_same_source_read_starts_form_one_grouped_block() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(
        ToolCallStart(
            name="read_file",
            args={"path": "a.py"},
            id="read-a",
            source_id="researcher",
        )
    )
    await transcript.handle(
        ToolCallStart(
            name="read_file",
            args={"path": "b.py"},
            id="read-b",
            source_id="researcher",
        )
    )
    await transcript.handle(ToolCallEnd(name="read_file", id="read-a", result="a"))
    await transcript.handle(ToolCallEnd(name="read_file", id="read-b", result="b"))

    assert len(frame.blocks) == 1
    assert frame.blocks[0].data["calls"] == [
        {"id": "read-a", "args": {"path": "a.py"}, "result": "a"},
        {"id": "read-b", "args": {"path": "b.py"}, "result": "b"},
    ]
    assert frame.blocks[0].state is BlockState.SETTLED



@pytest.mark.asyncio
async def test_release_committed_drops_tool_and_source_accumulator_references() -> None:
    frame = Frame()
    transcript = Transcript(frame)
    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(content="answer"),
            role="main",
            source_id="main",
        )
    )
    await transcript.handle(
        ToolCallStart(
            name="read_file",
            args={"path": "a.py"},
            id="read-a",
            source_id="main",
        )
    )
    await transcript.handle(
        ToolCallEnd(name="read_file", id="read-a", result="done")
    )
    await transcript.handle(TurnEnd(thread_id="thread"))
    committed = frame.commit_ready()

    transcript.release_committed(committed)

    assert transcript._source_blocks == {}
    assert transcript._tools == {}
    assert transcript._read_groups == {}


@pytest.mark.asyncio
async def test_reasoning_summary_stream_preserves_parts_runs_and_event_order() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    async def reasoning(run: int, part: int, text: str) -> None:
        await transcript.handle(
            ModelChunk(
                chunk=AIMessageChunk(
                    content=[
                        {
                            "type": "reasoning",
                            "index": run,
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "index": part,
                                    "text": text,
                                }
                            ],
                        }
                    ]
                ),
                role="main",
                source_id="main",
            )
        )

    await reasoning(0, 0, "directory")
    await reasoning(0, 1, "Explor")
    await reasoning(0, 1, "ing…")
    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(content=[{"type": "text", "text": "Found it.", "index": 1}]),
            role="main",
            source_id="main",
        )
    )
    await transcript.handle(
        ToolCallStart(name="read_file", args={"path": "a.py"}, id="read-a", source_id="main")
    )
    await transcript.handle(ToolCallEnd(name="read_file", id="read-a", result="ok"))
    await reasoning(2, 0, "Reading…")
    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(content=[{"type": "text", "text": "Done.", "index": 3}]),
            role="main",
            source_id="main",
        )
    )

    assert [block.kind for block in frame.blocks] == [
        "thinking",
        "assistant",
        "tool",
        "thinking",
        "assistant",
    ]
    assert [block.data["text"] for block in frame.blocks if block.kind == "thinking"] == [
        "directory\nExploring…",
        "Reading…",
    ]
    assert [block.data["text"] for block in frame.blocks if block.kind == "assistant"] == [
        "Found it.",
        "Done.",
    ]


@pytest.mark.asyncio
async def test_thinking_usage_and_ticker_metrics_flow_through_transcript() -> None:
    frame = Frame()
    scheduler = FrameScheduler(
        frame,
        commit=lambda _blocks: None,
        invalidate=lambda: None,
    )
    transcript = Transcript(frame, scheduler=scheduler)

    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(
                content=[
                    {
                        "type": "reasoning",
                        "summary": [{"text": "inspect constraints"}],
                    }
                ],
            ),
            role="main",
            source_id="main",
        )
    )
    await transcript.handle(
        ModelChunk(
            chunk=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 8,
                    "total_tokens": 9,
                    "output_token_details": {"reasoning": 8},
                },
            ),
            role="main",
            source_id="main",
        )
    )
    thinking = frame.blocks[0]
    scheduler.tick_spinners(now=thinking.created + 2.0)

    assert thinking.data["reasoning_tokens"] == 8
    assert thinking.data["spinner_frame"] == 1
    assert thinking.data["tokens_per_second"] == 4.0
    assert thinking.revision == 3
    await scheduler.aclose()
