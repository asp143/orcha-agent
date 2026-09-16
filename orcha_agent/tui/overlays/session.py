"""Saved-session picker."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer

from orcha_agent.core.session import SessionStore
from orcha_agent.core.session_picker import session_page

from .select import SelectList, _fuzzy


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
    return f"{display[:left].rstrip('/')}…{display[-right:].lstrip('/')}"


class SessionOverlay(SelectList[Any]):
    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._page_offset = 0
        self._page_limit = 200
        self._has_next = False
        self._labels: dict[str, str] = {}
        self._search_cancel = Event()
        self._search_task: asyncio.Task[None] | None = None
        sessions = self._load_page()

        def label(session: Any) -> str:
            return self._labels[str(getattr(session, "thread_id", ""))]

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

    @staticmethod
    def _read_page(
        ctx: Any,
        page_offset: int,
        page_limit: int,
        query: str,
        cancelled: Event,
    ) -> tuple[tuple[Any, ...], dict[str, str], bool]:
        labels: dict[str, str] = {}
        has_next = False
        if isinstance(ctx.session, SessionStore):
            snapshots = []
            offset = 0 if query else page_offset
            skipped = 0
            while not cancelled.is_set():
                rows = session_page(
                    ctx.session,
                    limit=page_limit + 1,
                    offset=offset,
                    cancelled=cancelled,
                )
                for row in rows:
                    if cancelled.is_set():
                        return (), {}, False
                    session = row.session
                    prompt = _message_text(row.first_message)
                    label = SessionOverlay._label_text(session, row.count, prompt)
                    if query and not SessionOverlay._matches(query, session, label):
                        continue
                    if query and skipped < page_offset:
                        skipped += 1
                        continue
                    snapshots.append((session, row.count, prompt))
                    if len(snapshots) > page_limit:
                        break
                if len(snapshots) > page_limit or len(rows) <= page_limit or not query:
                    break
                offset += len(rows)
            has_next = len(snapshots) > page_limit
            snapshots = snapshots[:page_limit]
        else:
            # Preserve the public duck-typed context protocol used by plugins.
            snapshots = [
                (
                    session,
                    ctx.ledger.count(session.thread_id),
                    ""
                    if _clean_text(getattr(session, "title", None))
                    else _first_prompt(ctx, session),
                )
                for session in ctx.session.list()
            ]
        for session, count, prompt in snapshots:
            labels[str(session.thread_id)] = SessionOverlay._label_text(session, count, prompt)
        return tuple(session for session, _, _ in snapshots), labels, has_next

    def _load_page(self, query: str = "") -> tuple[Any, ...]:
        items, self._labels, self._has_next = self._read_page(
            self._ctx,
            self._page_offset,
            self._page_limit,
            query,
            Event(),
        )
        return items

    def _request_page(self, query: str, index: int = 0) -> None:
        self._search_cancel.set()
        cancelled = self._search_cancel = Event()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.items = self._load_page(query)
            self.index = max(0, min(len(self.items) - 1, index))
            self._changed()
            return
        offset = self._page_offset
        app = get_app()
        # Do not leave old selections actionable while a new filter is loading.
        self.items = ()
        self._labels = {}
        self._has_next = False
        self.index = 0
        self.empty_text = "Searching sessions…"
        self._error = None

        async def search() -> None:
            try:
                result = await asyncio.to_thread(
                    self._read_page,
                    self._ctx,
                    offset,
                    self._page_limit,
                    query,
                    cancelled,
                )
            except asyncio.CancelledError:
                cancelled.set()
                raise
            except Exception as exc:
                if not cancelled.is_set() and not self.done:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self.empty_text = "No saved sessions"
                    app.invalidate()
                return
            if cancelled.is_set() or self.done:
                return
            self.items, self._labels, self._has_next = result
            self.index = max(0, min(len(self.items) - 1, index))
            self.empty_text = "No saved sessions"
            self._changed()
            app.invalidate()

        self._search_task = loop.create_task(search())

    def resolve(self, value: Any) -> None:
        self._search_cancel.set()
        super().resolve(value)

    @staticmethod
    def _label_text(session: Any, count: int, prompt: str) -> str:
        title = _clean_text(getattr(session, "title", None)) or prompt or "Untitled"
        cwd = _shorten_path(getattr(session, "cwd", ""))
        age = _age(getattr(session, "created", None))
        return f"{title} · {age} · {cwd} · {count} entries"

    @staticmethod
    def _matches(query: str, session: Any, label: str) -> bool:
        return any(
            _fuzzy(query, field)
            for field in (label, str(session.thread_id), str(getattr(session, "cwd", "")))
        )

    def _filter_changed(self, buffer: Buffer) -> None:
        if isinstance(self._ctx.session, SessionStore):
            self._page_offset = 0
            self._request_page(buffer.text)
        super()._filter_changed(buffer)

    def _filtered_pairs(self) -> list[tuple[int, Any]]:
        if isinstance(self._ctx.session, SessionStore):
            return list(enumerate(self.items))
        return [
            (offset, session)
            for offset, session in enumerate(self.items)
            if self._matches(self.filter.text, session, self._labels[str(session.thread_id)])
        ]

    def _move(self, delta: int) -> None:
        if isinstance(self._ctx.session, SessionStore):
            target = self.index + delta
            if target >= len(self.items) and self._has_next:
                self._page_offset += self._page_limit
                self._request_page(self.filter.text, target - self._page_limit)
                return
            if target < 0 and self._page_offset:
                self._page_offset = max(0, self._page_offset - self._page_limit)
                self._request_page(self.filter.text, self._page_limit + target)
                return
        super()._move(delta)


__all__ = ["SessionOverlay"]
