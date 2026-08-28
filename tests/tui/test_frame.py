from __future__ import annotations

import asyncio

import pytest

from orcha_agent.tui.frame import Block, Frame, FrameScheduler, BlockState


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
