"""Saved-session picker."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .select import SelectList


def _age(created: Any) -> str:
    if not isinstance(created, str):
        return "unknown age"
    try:
        then = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=UTC)
        seconds = max(0, int((datetime.now(UTC) - then).total_seconds()))
    except ValueError:
        return created
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86_400}d ago"


class SessionOverlay(SelectList[Any]):
    def __init__(self, ctx: Any) -> None:
        sessions = tuple(ctx.session.list())

        def label(session: Any) -> str:
            title = getattr(session, "title", None) or getattr(session, "thread_id", "")
            session_id = str(getattr(session, "thread_id", ""))
            count = ctx.ledger.count(session_id)
            cwd = getattr(session, "cwd", "")
            return f"{title} · {_age(getattr(session, 'created', None))} · {cwd} · {count} entries"

        async def resume(session: Any) -> str:
            session_id = str(session.thread_id)
            await ctx.resume(session_id)
            return session_id

        super().__init__(
            "Sessions",
            sessions,
            label=label,
            empty_text="No saved sessions",
            on_accept=resume,
        )


__all__ = ["SessionOverlay"]
