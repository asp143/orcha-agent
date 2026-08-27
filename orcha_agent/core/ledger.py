"""Typed append-only session ledger and pure context reconstruction."""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)

from .models import filter_foreign_blocks

if TYPE_CHECKING:
    from .session import SessionStore


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    id: str = ""
    parent_id: str | None = None
    ts: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageEntry(Entry):
    message: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelChangeEntry(Entry):
    model: str | list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeChangeEntry(Entry):
    mode: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionEntry(Entry):
    summary: str
    first_kept_id: str | None = None
    tokens_before: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResetBoundaryEntry(Entry):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class CustomEntry(Entry):
    custom_type: str
    data: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class OpaqueEntry(Entry):
    entry_type: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCallRef:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Context:
    messages: list[BaseMessage] = field(default_factory=list)
    model: str | list[str] | None = None
    mode: str | None = None
    todos: list[Any] = field(default_factory=list)
    files: dict[str, Any] = field(default_factory=dict)
    dangling: list[ToolCallRef] = field(default_factory=list)
    compacted: bool = False


class EntryNotFound(LookupError):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        super().__init__(f"Entry not found: {prefix}")


class AmbiguousEntry(LookupError):
    def __init__(self, prefix: str, candidates: Iterable[str]) -> None:
        self.prefix = prefix
        self.candidates = tuple(sorted(candidates))
        super().__init__(
            f"Ambiguous entry prefix {prefix}: {', '.join(self.candidates)}"
        )


class LedgerCycleError(RuntimeError):
    def __init__(self, entry_id: str) -> None:
        self.entry_id = entry_id
        super().__init__(f"Cycle in ledger parent chain at entry {entry_id}")


_ENTRY_TYPES: dict[str, type[Entry]] = {
    "message": MessageEntry,
    "model_change": ModelChangeEntry,
    "mode_change": ModeChangeEntry,
    "compaction": CompactionEntry,
    "reset_boundary": ResetBoundaryEntry,
    "custom": CustomEntry,
}


def _entry_type_and_payload(entry: Entry) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, MessageEntry):
        return "message", {"message": entry.message}
    if isinstance(entry, ModelChangeEntry):
        return "model_change", {"model": entry.model}
    if isinstance(entry, ModeChangeEntry):
        return "mode_change", {"mode": entry.mode}
    if isinstance(entry, CompactionEntry):
        return "compaction", {
            "summary": entry.summary,
            "first_kept_id": entry.first_kept_id,
            "tokens_before": entry.tokens_before,
        }
    if isinstance(entry, ResetBoundaryEntry):
        return "reset_boundary", {}
    if isinstance(entry, CustomEntry):
        return "custom", {"custom_type": entry.custom_type, "data": entry.data}
    if isinstance(entry, OpaqueEntry):
        return entry.entry_type, entry.payload
    raise TypeError(f"Unsupported ledger entry: {type(entry).__name__}")


def _encode_payload(entry: Entry) -> tuple[str, str]:
    entry_type, payload = _entry_type_and_payload(entry)
    return entry_type, json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_entry(
    entry_type: str,
    payload: Mapping[str, Any],
    *,
    id: str = "",
    parent_id: str | None = None,
    ts: str = "",
) -> Entry:
    common = {"id": id, "parent_id": parent_id, "ts": ts}
    if entry_type == "message":
        return MessageEntry(message=payload["message"], **common)
    if entry_type == "model_change":
        return ModelChangeEntry(model=payload["model"], **common)
    if entry_type == "mode_change":
        return ModeChangeEntry(mode=payload["mode"], **common)
    if entry_type == "compaction":
        return CompactionEntry(
            summary=payload["summary"],
            first_kept_id=payload.get("first_kept_id"),
            tokens_before=payload.get("tokens_before"),
            **common,
        )
    if entry_type == "reset_boundary":
        return ResetBoundaryEntry(**common)
    if entry_type == "custom":
        return CustomEntry(
            custom_type=payload["custom_type"], data=payload["data"], **common
        )
    return OpaqueEntry(entry_type=entry_type, payload=dict(payload), **common)


def _entry_from_row(row: sqlite3.Row) -> Entry:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError(f"Ledger payload for {row['id']} is not an object")
    return _decode_entry(
        row["type"],
        payload,
        id=row["id"],
        parent_id=row["parent_id"],
        ts=row["ts"],
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Ledger:
    """Persist typed entries against a :class:`SessionStore` connection."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def _new_id(self, session_id: str, reserved: set[str]) -> str:
        while True:
            entry_id = secrets.token_hex(4)
            if entry_id in reserved:
                continue
            found = self.store._connection.execute(
                "SELECT 1 FROM entries WHERE session_id = ? AND id = ?",
                (session_id, entry_id),
            ).fetchone()
            if found is None:
                reserved.add(entry_id)
                return entry_id

    def _append_in_transaction(
        self, session_id: str, entries: list[Entry]
    ) -> list[Entry]:
        connection = self.store._connection
        leaf_row = connection.execute(
            "SELECT leaf_id FROM sessions WHERE thread_id = ?", (session_id,)
        ).fetchone()
        if leaf_row is None:
            raise EntryNotFound(session_id)
        parent_id = leaf_row["leaf_id"]
        seq_row = connection.execute(
            "SELECT COALESCE(MAX(seq), -1) AS seq FROM entries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_seq = int(seq_row["seq"]) + 1
        reserved: set[str] = set()
        appended: list[Entry] = []

        for offset, entry in enumerate(entries):
            entry_id = entry.id or self._new_id(session_id, reserved)
            timestamp = entry.ts or _timestamp()
            actual_parent = (
                entry.parent_id if entry.parent_id is not None else parent_id
            )
            persisted = replace(
                entry,
                id=entry_id,
                parent_id=actual_parent,
                ts=timestamp,
            )
            entry_type, payload = _encode_payload(persisted)
            connection.execute(
                """
                INSERT INTO entries(session_id, id, parent_id, seq, type, ts, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    persisted.id,
                    persisted.parent_id,
                    next_seq + offset,
                    entry_type,
                    timestamp,
                    payload,
                ),
            )
            appended.append(persisted)
            parent_id = persisted.id

        if appended:
            connection.execute(
                "UPDATE sessions SET leaf_id = ? WHERE thread_id = ?",
                (appended[-1].id, session_id),
            )
        return appended

    def append(self, session_id: str, entry: Entry) -> Entry:
        return self.append_many(session_id, [entry])[0]

    def append_many(
        self, session_id: str, entries: Iterable[Entry]
    ) -> list[Entry]:
        batch = list(entries)
        if not batch:
            return []
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                appended = self._append_in_transaction(session_id, batch)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return appended

    def append_for_reseed(self, session_id: str, entry: Entry) -> Entry:
        """Append an entry and mark its session as awaiting a seeded thread."""
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                appended = self._append_in_transaction(session_id, [entry])
                connection.execute(
                    "UPDATE sessions SET current_thread = NULL WHERE thread_id = ?",
                    (session_id,),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return appended[0]

    def branch_for_reseed(self, session_id: str, entry_id: str) -> None:
        """Move the leaf and mark its session as awaiting a seeded thread."""
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM entries WHERE session_id = ? AND id = ?",
                    (session_id, entry_id),
                ).fetchone()
                if exists is None:
                    raise EntryNotFound(entry_id)
                connection.execute(
                    """
                    UPDATE sessions
                    SET leaf_id = ?, current_thread = NULL
                    WHERE thread_id = ?
                    """,
                    (entry_id, session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def restore_position(
        self,
        session_id: str,
        *,
        leaf_id: str | None,
        thread_id: str | None,
    ) -> None:
        """Atomically restore a session leaf and active graph thread."""
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                connection.execute(
                    """
                    UPDATE sessions
                    SET leaf_id = ?, current_thread = ?
                    WHERE thread_id = ?
                    """,
                    (leaf_id, thread_id, session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def capture(
        self,
        session_id: str,
        thread_id: str,
        entries: Iterable[Entry],
        *,
        message_count: int,
    ) -> list[Entry]:
        """Append a turn and advance its graph-thread capture cursor atomically."""
        if message_count < 0:
            raise ValueError("message_count must be non-negative")
        batch = list(entries)
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                appended = self._append_in_transaction(session_id, batch)
                cursor = connection.execute(
                    """
                    UPDATE threads
                    SET captured = captured + ?
                    WHERE thread_id = ? AND session_id = ?
                    """,
                    (message_count, thread_id, session_id),
                )
                if cursor.rowcount != 1:
                    raise EntryNotFound(thread_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return appended

    def branch(self, session_id: str, entry_id: str) -> None:
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                exists = connection.execute(
                    "SELECT 1 FROM entries WHERE session_id = ? AND id = ?",
                    (session_id, entry_id),
                ).fetchone()
                if exists is None:
                    raise EntryNotFound(entry_id)
                connection.execute(
                    "UPDATE sessions SET leaf_id = ? WHERE thread_id = ?",
                    (entry_id, session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def leaf(self, session_id: str) -> str | None:
        with self.store.saver.lock:
            row = self.store._connection.execute(
                "SELECT leaf_id FROM sessions WHERE thread_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else row["leaf_id"]

    def get(self, session_id: str, entry_id: str) -> Entry | None:
        with self.store.saver.lock:
            row = self.store._connection.execute(
                """
                SELECT id, parent_id, type, ts, payload
                FROM entries WHERE session_id = ? AND id = ?
                """,
                (session_id, entry_id),
            ).fetchone()
        return None if row is None else _entry_from_row(row)

    def resolve(self, session_id: str, prefix: str) -> Entry:
        with self.store.saver.lock:
            rows = self.store._connection.execute(
                """
                SELECT id, parent_id, type, ts, payload
                FROM entries
                WHERE session_id = ?
                  AND substr(id, 1, length(?)) = ?
                ORDER BY id
                """,
                (session_id, prefix, prefix),
            ).fetchall()
        exact = next((row for row in rows if row["id"] == prefix), None)
        if exact is not None:
            return _entry_from_row(exact)
        if not rows:
            raise EntryNotFound(prefix)
        if len(rows) > 1:
            raise AmbiguousEntry(prefix, (row["id"] for row in rows))
        return _entry_from_row(rows[0])

    @staticmethod
    def _path_from_rows(
        rows: Iterable[sqlite3.Row], leaf_id: str | None
    ) -> list[Entry]:
        if leaf_id is None:
            return []
        by_id = {row["id"]: row for row in rows}
        reverse_path: list[Entry] = []
        seen: set[str] = set()
        current: str | None = leaf_id
        while current is not None:
            if current in seen:
                raise LedgerCycleError(current)
            seen.add(current)
            row = by_id.get(current)
            if row is None:
                raise EntryNotFound(current)
            reverse_path.append(_entry_from_row(row))
            current = row["parent_id"]
        reverse_path.reverse()
        return reverse_path

    def path(self, session_id: str, leaf: str | None = None) -> list[Entry]:
        with self.store.saver.lock:
            connection = self.store._connection
            if leaf is None:
                session = connection.execute(
                    "SELECT leaf_id FROM sessions WHERE thread_id = ?", (session_id,)
                ).fetchone()
                leaf_id = None if session is None else session["leaf_id"]
            else:
                leaf_id = leaf
            rows = connection.execute(
                """
                SELECT id, parent_id, type, ts, payload
                FROM entries WHERE session_id = ?
                """,
                (session_id,),
            ).fetchall()
        return self._path_from_rows(rows, leaf_id)

    def all(self, session_id: str) -> list[Entry]:
        with self.store.saver.lock:
            rows = self.store._connection.execute(
                """
                SELECT id, parent_id, type, ts, payload
                FROM entries WHERE session_id = ? ORDER BY seq
                """,
                (session_id,),
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def fork(self, session_id: str, new_session_id: str) -> None:
        with self.store.saver.lock:
            connection = self.store._connection
            connection.execute("BEGIN")
            try:
                source = connection.execute(
                    "SELECT leaf_id FROM sessions WHERE thread_id = ?", (session_id,)
                ).fetchone()
                target = connection.execute(
                    "SELECT 1 FROM sessions WHERE thread_id = ?", (new_session_id,)
                ).fetchone()
                if source is None:
                    raise EntryNotFound(session_id)
                if target is None:
                    raise EntryNotFound(new_session_id)
                rows = connection.execute(
                    """
                    SELECT id, parent_id, type, ts, payload
                    FROM entries WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
                active_path = self._path_from_rows(rows, source["leaf_id"])
                for seq, entry in enumerate(active_path):
                    entry_type, payload = _encode_payload(entry)
                    connection.execute(
                        """
                        INSERT INTO entries(
                            session_id, id, parent_id, seq, type, ts, payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_session_id,
                            entry.id,
                            entry.parent_id,
                            seq,
                            entry_type,
                            entry.ts,
                            payload,
                        ),
                    )
                connection.execute(
                    "UPDATE sessions SET leaf_id = ? WHERE thread_id = ?",
                    (source["leaf_id"], new_session_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def count(self, session_id: str) -> int:
        with self.store.saver.lock:
            row = self.store._connection.execute(
                "SELECT COUNT(*) AS count FROM entries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"])


def _after_last_reset(path: list[Entry]) -> list[Entry]:
    reset_at = -1
    for index, entry in enumerate(path):
        if isinstance(entry, ResetBoundaryEntry):
            reset_at = index
    return path[reset_at + 1 :]


def _apply_last_compaction(path: list[Entry]) -> tuple[list[Entry], str | None]:
    for compact_at in range(len(path) - 1, -1, -1):
        entry = path[compact_at]
        if not isinstance(entry, CompactionEntry):
            continue
        if entry.first_kept_id is None:
            return path[compact_at + 1 :], entry.summary
        marker_at = next(
            (
                index
                for index, candidate in enumerate(path)
                if candidate.id and candidate.id == entry.first_kept_id
            ),
            None,
        )
        if marker_at is not None:
            return path[marker_at + 1 :], entry.summary
    return path, None


def _message_from_entry(entry: MessageEntry) -> BaseMessage:
    return messages_from_dict([entry.message])[0]


def _remove_dangling_tools(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[ToolCallRef]]:
    assistant_calls: dict[int, list[tuple[str, str]]] = {}
    active_call_owner: dict[str, int] = {}
    matched_calls: dict[int, set[str]] = {}
    tool_owners: dict[int, int | None] = {}

    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            calls: list[tuple[str, str]] = []
            for call in message.tool_calls:
                call_id = call.get("id")
                if not isinstance(call_id, str):
                    continue
                name = call.get("name")
                calls.append((call_id, name if isinstance(name, str) else ""))
                active_call_owner[call_id] = index
            if calls:
                assistant_calls[index] = calls
        elif isinstance(message, ToolMessage):
            owner = active_call_owner.get(message.tool_call_id)
            tool_owners[index] = owner
            if owner is not None:
                matched_calls.setdefault(owner, set()).add(message.tool_call_id)

    dropped_assistants: set[int] = set()
    dangling: list[ToolCallRef] = []
    for assistant_at, calls in assistant_calls.items():
        matched = matched_calls.get(assistant_at, set())
        unmatched = [
            (call_id, name) for call_id, name in calls if call_id not in matched
        ]
        if unmatched:
            dropped_assistants.add(assistant_at)
            dangling.extend(
                ToolCallRef(id=call_id, name=name) for call_id, name in unmatched
            )

    retained = [
        message
        for index, message in enumerate(messages)
        if index not in dropped_assistants
        and (
            not isinstance(message, ToolMessage)
            or (
                tool_owners.get(index) is not None
                and tool_owners[index] not in dropped_assistants
            )
        )
    ]
    return retained, dangling


def build_context(
    path: list[Entry], *, strip: set[str] | frozenset[str] = frozenset()
) -> Context:
    """Reconstruct live agent state from one root-to-leaf entry path."""
    post_reset = _after_last_reset(path)
    message_entries, summary = _apply_last_compaction(post_reset)

    messages: list[BaseMessage] = []
    if summary is not None:
        messages.append(HumanMessage(content=f"[Conversation summary]\n{summary}"))
    messages.extend(
        _message_from_entry(entry)
        for entry in message_entries
        if isinstance(entry, MessageEntry)
    )
    if strip:
        messages = filter_foreign_blocks(messages, strip)
    messages, dangling = _remove_dangling_tools(messages)

    model: str | list[str] | None = None
    mode: str | None = None
    todos: list[Any] = []
    files: dict[str, Any] = {}
    for entry in post_reset:
        if isinstance(entry, ModelChangeEntry):
            model = deepcopy(entry.model)
        elif isinstance(entry, ModeChangeEntry):
            mode = entry.mode
        elif isinstance(entry, CustomEntry) and entry.custom_type == "turn_state":
            data = entry.data
            if isinstance(data, Mapping):
                raw_todos = data.get("todos", [])
                raw_files = data.get("files", {})
                todos = deepcopy(raw_todos) if isinstance(raw_todos, list) else []
                files = deepcopy(raw_files) if isinstance(raw_files, dict) else {}

    return Context(
        messages=messages,
        model=model,
        mode=mode,
        todos=todos,
        files=files,
        dangling=dangling,
        compacted=summary is not None,
    )
