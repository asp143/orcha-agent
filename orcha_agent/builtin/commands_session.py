"""Session-management slash commands."""

from __future__ import annotations

from typing import Any

from rich.table import Table

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="commands_session", version="1.0.0")


async def _sessions(ctx: Any, _args: str) -> None:
    table = Table(title="Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Model")
    table.add_column("Working directory")
    table.add_column("Created")
    for session in ctx.session.list():
        table.add_row(
            session.thread_id,
            session.title or "",
            session.model,
            session.cwd,
            session.created,
        )
    ctx.console.print(table)


async def _resume(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) != 1:
        ctx.console.error("Usage: /resume <session-id>")
        return
    await ctx.resume(parts[0])


async def _compact(ctx: Any, _args: str) -> None:
    await ctx.compact()


def register(api: PluginAPI) -> None:
    api.add_command("sessions", _sessions, help="List saved sessions")
    api.add_command("resume", _resume, help="Resume a saved session: /resume <session-id>")
    api.add_command("compact", _compact, help="Compact the current conversation")
