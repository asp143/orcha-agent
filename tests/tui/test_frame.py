from __future__ import annotations

import asyncio

import pytest

from orcha_agent.tui.frame import (
    Block,
    BlockState,
    Frame,
    FrameScheduler,
    ViewportItem,
)


def test_blocks_commit_in_creation_order_after_earlier_active_block_settles() -> None:
    frame = Frame()
    first = frame.add("assistant", {"text": "first"})
    second = frame.add("tool", {"name": "read"})

    frame.settle(second)
    assert frame.commit_ready() == []
    assert second.state is BlockState.SETTLED

    frame.settle(first)
    assert frame.commit_ready() == [first, second]
    assert [block.state for block in frame.blocks] == [
        BlockState.COMMITTED,
        BlockState.COMMITTED,
    ]


def test_active_block_updates_revision_and_rejects_mutation_after_settle() -> None:
    block = Block(id="answer", kind="assistant", data={"text": "a"})

    block.update(text="ab")
    assert block.revision == 1
    assert block.data == {"text": "ab"}

    block.settle()
    with pytest.raises(RuntimeError, match="settled"):
        block.update(text="abc")


def test_viewport_budget_and_tool_degradation_follow_available_rows() -> None:
    assert Frame.row_budget(terminal_rows=12, composer_rows=3, status_rows=1) == 8
    assert Frame.row_budget(terminal_rows=2, composer_rows=3, status_rows=1) == 0
    assert [Frame.tool_rows(rows) for rows in range(5)] == [0, 1, 2, 3, 3]


def test_viewport_planning_counts_wrapped_rows_at_the_current_width() -> None:
    frame = Frame()
    block = frame.add("assistant", {"text": "abcdefghijkl"})

    narrow = frame.viewport_plan(5, width=5)
    wide = frame.viewport_plan(5, width=20)

    assert narrow == [ViewportItem(block, 3)]
    assert wide == [ViewportItem(block, 1)]


@pytest.mark.asyncio
async def test_commit_and_invalidation_requests_are_coalesced() -> None:
    commit_batches: list[list[str]] = []
    invalidations = 0

    def commit(blocks: list[Block]) -> None:
        commit_batches.append([block.id for block in blocks])

    def invalidate() -> None:
        nonlocal invalidations
        invalidations += 1

    frame = Frame()
    scheduler = FrameScheduler(frame, commit=commit, invalidate=invalidate)
    one = frame.add("raw", {"value": "one"})
    two = frame.add("raw", {"value": "two"})
    frame.settle(one)
    frame.settle(two)

    scheduler.request_commit()
    scheduler.request_commit()
    for _ in range(20):
        scheduler.request_invalidate()

    await asyncio.sleep(0.06)
    assert commit_batches == [[one.id, two.id]]
    assert invalidations <= 2

    before = invalidations
    scheduler.render_now()
    assert invalidations == before + 1
    await scheduler.aclose()


@pytest.mark.asyncio
async def test_all_spinners_share_one_ticker_task() -> None:
    ticks: list[int] = []
    scheduler = FrameScheduler(Frame(), commit=lambda _blocks: None, invalidate=lambda: ticks.append(1))

    first = scheduler.start_spinner()
    second = scheduler.start_spinner()
    assert first is second

    await asyncio.sleep(0.09)
    assert ticks
    await scheduler.aclose()


def test_shared_ticker_advances_pending_tool_and_thinking_state() -> None:
    frame = Frame()
    tool = frame.add("tool", {"name": "execute"})
    thinking = frame.add(
        "thinking",
        {"text": "plan", "reasoning_tokens": 20},
    )
    tool.created = 10.0
    thinking.created = 10.0
    scheduler = FrameScheduler(
        frame,
        commit=lambda _blocks: None,
        invalidate=lambda: None,
    )

    scheduler.tick_spinners(now=12.0)

    assert tool.data == {
        "name": "execute",
        "spinner_frame": 1,
        "elapsed": 2.0,
    }
    assert thinking.data == {
        "text": "plan",
        "reasoning_tokens": 20,
        "spinner_frame": 1,
        "tokens_per_second": 10.0,
    }
    assert tool.revision == thinking.revision == 1
