"""Saved-session picker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
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


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _message_text(message: object) -> str:
    if not isinstance(message, Mapping):
        return ""
    data = message.get("data")
    payload = data if isinstance(data, Mapping) else message
    role = str(
        message.get("role")
        or message.get("type")
        or payload.get("role")
        or payload.get("type")
        or ""
    ).casefold()
    if role not in {"human", "user"}:
        return ""
    content = payload.get("content")
    if isinstance(content, str):
        return _clean_text(content)
    if not isinstance(content, Sequence):
        return ""
    parts = [
        _clean_text(block.get("text"))
        for block in content
        if isinstance(block, Mapping) and block.get("type") in {None, "text"}
    ]
    return " ".join(part for part in parts if part)


def _first_prompt(ctx: Any, session: Any) -> str:
    for attribute in ("first_prompt", "first_message", "firstMessage"):
        prompt = _clean_text(getattr(session, attribute, None))
        if prompt:
            return prompt

    session_id = str(getattr(session, "thread_id", ""))
    load_entries = getattr(ctx.ledger, "all", None)
    if not callable(load_entries):
        return ""
    for entry in load_entries(session_id):
        prompt = _message_text(getattr(entry, "message", None))
        if prompt:
            return prompt
    return ""


def _shorten_path(value: object, *, max_length: int = 30) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value).expanduser()
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        display = str(path)
    else:
        display = "~" if not relative.parts else f"~/{relative.as_posix()}"
    if len(display) <= max_length:
        return display

    if display.startswith("~/"):
        prefix, parts = "~/", display[2:].split("/")
    elif display.startswith("/"):
        prefix, parts = "/", display[1:].split("/")
    else:
        prefix, parts = "", display.split("/")
    if len(parts) >= 5:
        candidate = prefix + "/".join((*parts[:2], "…", *parts[-2:]))
        if len(candidate) <= max_length:
            return candidate

    left = max(1, (max_length - 1) // 2)
    right = max(1, max_length - left - 1)
    return f"{display[:left].rstrip('/') }…{display[-right:].lstrip('/')}"


class SessionOverlay(SelectList[Any]):
    def __init__(self, ctx: Any) -> None:
        sessions = tuple(ctx.session.list())
        labels: dict[str, str] = {}
        for session in sessions:
            session_id = str(getattr(session, "thread_id", ""))
            title = (
                _clean_text(getattr(session, "title", None))
                or _first_prompt(ctx, session)
                or "Untitled"
            )
            count = ctx.ledger.count(session_id)
            cwd = _shorten_path(getattr(session, "cwd", ""))
            age = _age(getattr(session, "created", None))
            labels[session_id] = f"{title} · {age} · {cwd} · {count} entries"

        def label(session: Any) -> str:
            return labels[str(getattr(session, "thread_id", ""))]

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
