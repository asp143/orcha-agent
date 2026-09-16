"""The UI toggle gates timers while manual compaction remains available."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from orcha_agent.core.compaction_config import CompactionConfig
from orcha_agent.core.events import AppExit, TurnEnd
from orcha_agent.tui.context import AppContext
from tests.test_tui import _HistoryGraph, _context


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_enabled,policy_enabled", [(False, True), (True, False)])
async def test_disabled_compaction_does_not_schedule_idle(tmp_path, auto_enabled, policy_enabled):
    ctx = _context(tmp_path, agent=_HistoryGraph([]))
    ctx.cfg = replace(
        ctx.cfg,
        auto_compact=auto_enabled,
        compaction=CompactionConfig(enabled=policy_enabled),
    )
    try:
        await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
        assert ctx._idle_compaction is None
    finally:
        ctx.session.close()


@pytest.mark.asyncio
async def test_idle_timer_rechecks_toggle(tmp_path, monkeypatch):
    ctx = _context(tmp_path, agent=_HistoryGraph([]))
    ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(threshold_tokens=1))
    entered, release = asyncio.Event(), asyncio.Event()

    async def wait_for_release(_seconds):
        entered.set()
        await release.wait()

    monkeypatch.setattr("orcha_agent.tui.context.asyncio.sleep", wait_for_release)
    compact = AsyncMock()
    monkeypatch.setattr(AppContext, "compact", compact)
    try:
        await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
        task = ctx._idle_compaction
        assert task is not None
        await asyncio.wait_for(entered.wait(), 10)
        ctx.cfg = replace(ctx.cfg, auto_compact=False)
        release.set()
        await asyncio.wait_for(task, 10)
        compact.assert_not_awaited()
    finally:
        await ctx._compaction_activity(AppExit())
        ctx.session.close()
