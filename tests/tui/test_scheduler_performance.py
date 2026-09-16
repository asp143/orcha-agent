from __future__ import annotations

import asyncio

import pytest

from orcha_agent.tui.frame import Frame, FrameScheduler


@pytest.mark.asyncio
async def test_immediate_redraw_cancels_superseded_trailing_redraw() -> None:
    calls = []
    scheduler = FrameScheduler(Frame(), commit=lambda _: None, invalidate=lambda: calls.append(1))
    scheduler.INVALIDATE_INTERVAL = 0.01
    try:
        scheduler.request_invalidate()
        for _ in range(100):
            scheduler.request_invalidate()
        pending = scheduler._invalidate_task
        assert pending is not None
        scheduler.render_now()
        await asyncio.sleep(0.02)
        assert pending.cancelled()
        assert len(calls) == 2
    finally:
        await scheduler.aclose()
