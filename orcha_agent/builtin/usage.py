"""Cumulative usage command and cached status totals."""

from __future__ import annotations

import asyncio
from typing import Any

from orcha_agent.core.events import AppStart, SessionSwitch, ThreadSwitch, TurnEnd, TurnStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.core.usage_store import UsageRecorded, UsageStore, usage_table

PLUGIN = PluginSpec(name="usage", version="1.0.0")


def register(api: PluginAPI) -> None:
    context: Any = None

    async def refresh(_event: Any = None) -> None:
        if context is None:
            return
        store = UsageStore(context.session)
        session_rows, today_rows = await asyncio.gather(
            asyncio.to_thread(store.report, "session", context.session_id),
            asyncio.to_thread(store.report, "today"),
        )
        api.state.update(
            session_cost=sum(row["cost"] for row in session_rows),
            today_cost=sum(row["cost"] for row in today_rows),
        )

    async def started(event: AppStart) -> None:
        nonlocal context
        if not hasattr(event.ctx, "session") or not hasattr(event.ctx, "session_id"):
            return
        context = event.ctx
        await refresh()

    async def show(ctx: Any, args: str) -> None:
        period = args.strip() or "session"
        if period not in {"today", "week", "session", "all"}:
            ctx.console.error("Usage: /usage [today|week|session|all]")
            return
        rows = await asyncio.to_thread(UsageStore(ctx.session).report, period, ctx.session_id)
        ctx.console.print(usage_table(rows, period))

    api.on(AppStart, started)
    api.on(TurnEnd, refresh)
    api.on(TurnStart, refresh)
    api.on(UsageRecorded, refresh)
    api.on(SessionSwitch, refresh)
    api.on(ThreadSwitch, refresh)
    api.add_command("usage", show, help="Usage and cost by model: today, week, session, all")
