"""Bounded session picker snapshots, read under the checkpoint connection lock."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event
from typing import Any

from .session import SessionInfo, SessionStore


@dataclass(frozen=True, slots=True)
class PickerSession:
    session: SessionInfo
    count: int
    first_message: dict[str, Any] | None


def _has_prompt(message: Any) -> bool:
    if not isinstance(message, Mapping):
        return False
    data = message.get("data", message)
    if not isinstance(data, Mapping):
        return False
    content = data.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and any(
        isinstance(part, Mapping)
        and part.get("type") in {None, "text"}
        and isinstance(part.get("text"), str)
        and bool(part["text"].strip())
        for part in content
    )


def _next_prompt(
    store: SessionStore,
    session_id: str,
    cancelled: Event | None,
) -> dict[str, Any] | None:
    """Skip empty/image-only initial messages with bounded reads on the rare path."""
    sequence = -1
    while True:
        with store.saver.lock:
            if cancelled is not None and cancelled.is_set():
                return None
            rows = store._connection.execute(
                """
                SELECT seq, payload FROM entries
                WHERE session_id = ? AND seq > ? AND type = 'message'
                  AND json_valid(payload)
                  AND json_extract(payload, '$.message.type') IN ('human', 'user')
                ORDER BY seq LIMIT 64
                """,
                (session_id, sequence),
            ).fetchall()
        for row in rows:
            message = json.loads(row["payload"]).get("message")
            if _has_prompt(message):
                return message
        if len(rows) < 64:
            return None
        sequence = rows[-1]["seq"]


def session_page(
    store: SessionStore,
    *,
    limit: int = 200,
    offset: int = 0,
    cancelled: Event | None = None,
) -> list[PickerSession]:
    """Fetch a bounded page and its label inputs in one SQL statement.

    Counts and first prompts use the session/sequence index; abandoned message
    payloads are never loaded merely to display a session title.
    """
    with store.saver.lock:
        if cancelled is not None and cancelled.is_set():
            return []
        rows = store._connection.execute(
            """
            SELECT sessions.*, 'user' AS kind,
                (SELECT COUNT(*) FROM entries
                 WHERE entries.session_id = sessions.thread_id) AS entry_count,
                CASE WHEN COALESCE(trim(title), '') = '' THEN
                    (SELECT payload FROM entries
                     WHERE entries.session_id = sessions.thread_id
                       AND type = 'message' AND json_valid(payload)
                       AND json_extract(payload, '$.message.type') IN ('human', 'user')
                     ORDER BY seq LIMIT 1)
                END AS first_payload
            FROM sessions
            WHERE NOT EXISTS (
                SELECT 1 FROM internal_sessions
                WHERE internal_sessions.thread_id = sessions.thread_id
            )
            ORDER BY created DESC, thread_id DESC
            LIMIT ? OFFSET ?
            """,
            (max(1, limit), max(0, offset)),
        ).fetchall()
    result = []
    for row in rows:
        session = store._session(row)
        if session is not None:
            payload = json.loads(row["first_payload"]) if row["first_payload"] else {}
            message = payload.get("message")
            if message is not None and not _has_prompt(message):
                message = _next_prompt(store, session.thread_id, cancelled)
            result.append(PickerSession(session, row["entry_count"], message))
    return result
