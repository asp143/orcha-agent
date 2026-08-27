"""SQLite-backed agent sessions and plugin state."""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import message_to_dict
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver


@dataclass(frozen=True, slots=True)
class SessionInfo:
    thread_id: str
    cwd: str
    model: str | list[str]
    created: str
    title: str | None
    mode: str = "ask"
    leaf_id: str | None = None
    current_thread: str | None = None
    parent_session: str | None = None


@dataclass(frozen=True, slots=True)
class ThreadInfo:
    thread_id: str
    session_id: str
    seeded_from: str | None = None
    captured: int = 0


class _AsyncSqliteSaver(SqliteSaver):
    """Expose SqliteSaver's synchronized operations to async graph execution."""

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        checkpoints = await asyncio.to_thread(
            lambda: tuple(
                self.list(
                    config,
                    filter=filter,
                    before=before,
                    limit=limit,
                )
            )
        )
        for checkpoint in checkpoints:
            yield checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)


class SessionStore:
    """Own one SQLite connection shared by checkpoints and session metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.saver = _AsyncSqliteSaver(self._connection)
        self.saver.setup()
        try:
            self._migrate_schema()
        except BaseException:
            self._connection.close()
            raise

    def _migrate_schema(self) -> None:
        with self.saver.lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        thread_id TEXT PRIMARY KEY,
                        cwd TEXT NOT NULL,
                        model TEXT NOT NULL,
                        created TEXT NOT NULL,
                        title TEXT,
                        mode TEXT NOT NULL DEFAULT 'ask'
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS plugin_state (
                        thread_id TEXT NOT NULL,
                        plugin TEXT NOT NULL,
                        state TEXT NOT NULL,
                        PRIMARY KEY (thread_id, plugin),
                        FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                    """
                )
                version_row = self._connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                version = 0 if version_row is None else int(version_row["value"])
                if version == 0:
                    self._migrate_v0_to_v1()
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _migrate_v0_to_v1(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(sessions)")
        }
        additions = (
            ("mode", "TEXT NOT NULL DEFAULT 'ask'"),
            ("leaf_id", "TEXT"),
            ("current_thread", "TEXT"),
            ("parent_session", "TEXT"),
        )
        for name, declaration in additions:
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE sessions ADD COLUMN {name} {declaration}"
                )

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                session_id TEXT NOT NULL,
                id TEXT NOT NULL,
                parent_id TEXT,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                ts TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (session_id, id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                seeded_from TEXT,
                captured INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS entries_session_seq ON entries(session_id, seq)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS entries_session_parent ON entries(session_id, parent_id)"
        )

        sessions = self._connection.execute(
            "SELECT thread_id FROM sessions ORDER BY thread_id"
        ).fetchall()
        for session in sessions:
            session_id = session["thread_id"]
            checkpoint_row = self._connection.execute(
                """
                SELECT type, checkpoint
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ''
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if checkpoint_row is None:
                continue

            checkpoint = self.saver.serde.loads_typed(
                (checkpoint_row["type"], checkpoint_row["checkpoint"])
            )
            if not isinstance(checkpoint, Mapping):
                raise TypeError(f"Invalid checkpoint for session {session_id!r}")
            channel_values = checkpoint.get("channel_values", {})
            if not isinstance(channel_values, Mapping):
                raise TypeError(
                    f"Invalid checkpoint channel values for session {session_id!r}"
                )
            raw_messages = channel_values.get("messages", ())
            if raw_messages is None:
                raw_messages = ()
            if not isinstance(raw_messages, Sequence) or isinstance(
                raw_messages, (str, bytes, bytearray)
            ):
                raise TypeError(
                    f"Invalid checkpoint messages for session {session_id!r}"
                )

            parent_id: str | None = None
            for seq, message in enumerate(raw_messages):
                while True:
                    entry_id = secrets.token_hex(4)
                    collision = self._connection.execute(
                        "SELECT 1 FROM entries WHERE session_id = ? AND id = ?",
                        (session_id, entry_id),
                    ).fetchone()
                    if collision is None:
                        break
                serialized = (
                    message
                    if isinstance(message, dict)
                    else message_to_dict(message)
                )
                self._connection.execute(
                    """
                    INSERT INTO entries(
                        session_id, id, parent_id, seq, type, ts, payload
                    ) VALUES (?, ?, ?, ?, 'message', ?, ?)
                    """,
                    (
                        session_id,
                        entry_id,
                        parent_id,
                        seq,
                        datetime.now(UTC).isoformat(timespec="microseconds"),
                        json.dumps(
                            {"message": serialized},
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    ),
                )
                parent_id = entry_id

            captured = len(raw_messages)
            self._connection.execute(
                """
                INSERT INTO threads(thread_id, session_id, seeded_from, captured)
                VALUES (?, ?, NULL, ?)
                """,
                (session_id, session_id, captured),
            )
            self._connection.execute(
                """
                UPDATE sessions
                SET leaf_id = ?, current_thread = ?
                WHERE thread_id = ?
                """,
                (parent_id, session_id, session_id),
            )

        self._connection.execute(
            """
            INSERT INTO meta(key, value) VALUES ('schema_version', '1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _write(self, sql: str, parameters: Sequence[Any]) -> None:
        with self.saver.lock:
            try:
                self._connection.execute(sql, parameters)
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def create(
        self,
        cwd: str | Path,
        model: str | list[str],
        mode: str = "ask",
        *,
        title: str | None = None,
        thread_id: str | None = None,
        parent_session: str | None = None,
    ) -> SessionInfo:
        created = datetime.now(UTC).isoformat(timespec="microseconds")
        identifier = thread_id or (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        )
        current_thread = f"{identifier}.0"
        model_text = self._encode_model(model)
        stored_model = model if isinstance(model, str) else list(model)
        info = SessionInfo(
            identifier,
            str(cwd),
            stored_model,
            created,
            title,
            mode,
            current_thread=current_thread,
            parent_session=parent_session,
        )
        with self.saver.lock:
            try:
                self._connection.execute(
                    """
                    INSERT INTO sessions(
                        thread_id, cwd, model, created, title, mode,
                        leaf_id, current_thread, parent_session
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        info.thread_id,
                        info.cwd,
                        model_text,
                        info.created,
                        info.title,
                        info.mode,
                        info.current_thread,
                        info.parent_session,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO threads(thread_id, session_id, seeded_from, captured)
                    VALUES (?, ?, NULL, 0)
                    """,
                    (current_thread, identifier),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return info

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all graph/checkpoint state owned by it."""
        with self.saver.lock:
            rows = self._connection.execute(
                "SELECT thread_id FROM threads WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        for row in rows:
            self.saver.delete_thread(row["thread_id"])
        with self.saver.lock:
            try:
                self._connection.execute("BEGIN")
                self._connection.execute(
                    "DELETE FROM entries WHERE session_id = ?",
                    (session_id,),
                )
                self._connection.execute(
                    "DELETE FROM threads WHERE session_id = ?",
                    (session_id,),
                )
                self._connection.execute(
                    "DELETE FROM plugin_state WHERE thread_id = ?",
                    (session_id,),
                )
                self._connection.execute(
                    "DELETE FROM sessions WHERE thread_id = ?",
                    (session_id,),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    @staticmethod
    def _encode_model(value: str | list[str]) -> str:
        return value if isinstance(value, str) else json.dumps(value)

    @staticmethod
    def _decode_model(value: str) -> str | list[str]:
        if value.startswith("["):
            decoded = json.loads(value)
            if isinstance(decoded, list) and all(
                isinstance(item, str) for item in decoded
            ):
                return decoded
        return value

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionInfo | None:
        if row is None:
            return None
        return SessionInfo(
            row["thread_id"],
            row["cwd"],
            SessionStore._decode_model(row["model"]),
            row["created"],
            row["title"],
            row["mode"],
            row["leaf_id"],
            row["current_thread"],
            row["parent_session"],
        )

    @staticmethod
    def _thread(row: sqlite3.Row | None) -> ThreadInfo | None:
        if row is None:
            return None
        return ThreadInfo(
            row["thread_id"],
            row["session_id"],
            row["seeded_from"],
            row["captured"],
        )

    def get(self, thread_id: str) -> SessionInfo | None:
        row = self._connection.execute(
            """
            SELECT thread_id, cwd, model, created, title, mode,
                   leaf_id, current_thread, parent_session
            FROM sessions
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        return self._session(row)

    def exists(self, thread_id: str) -> bool:
        return self.get(thread_id) is not None

    def resolve_session(self, prefix: str) -> SessionInfo:
        if exact := self.get(prefix):
            return exact
        rows = self._connection.execute(
            """
            SELECT thread_id, cwd, model, created, title, mode,
                   leaf_id, current_thread, parent_session
            FROM sessions
            WHERE substr(thread_id, 1, ?) = ?
            ORDER BY thread_id
            """,
            (len(prefix), prefix),
        ).fetchall()
        matches = [
            session
            for row in rows
            if (session := self._session(row)) is not None
        ]
        if not matches:
            raise LookupError(f"No session matches {prefix!r}")
        if len(matches) > 1:
            candidates = ", ".join(session.thread_id for session in matches)
            raise LookupError(
                f"Session prefix {prefix!r} is ambiguous: {candidates}"
            )
        return matches[0]

    def get_thread(self, thread_id: str) -> ThreadInfo | None:
        row = self._connection.execute(
            """
            SELECT thread_id, session_id, seeded_from, captured
            FROM threads
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        return self._thread(row)

    def next_thread_id(self, session_id: str) -> str:
        prefix = f"{session_id}."
        rows = self._connection.execute(
            "SELECT thread_id FROM threads WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        numbers = [
            int(suffix)
            for row in rows
            if row["thread_id"].startswith(prefix)
            and (suffix := row["thread_id"][len(prefix) :]).isdigit()
        ]
        return f"{prefix}{max(numbers, default=-1) + 1}"

    def create_thread(
        self,
        session_id: str,
        *,
        seeded_from: str | None = None,
        captured: int = 0,
        thread_id: str | None = None,
    ) -> ThreadInfo:
        with self.saver.lock:
            try:
                identifier = thread_id or self.next_thread_id(session_id)
                info = ThreadInfo(identifier, session_id, seeded_from, captured)
                self._connection.execute(
                    """
                    INSERT INTO threads(thread_id, session_id, seeded_from, captured)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        info.thread_id,
                        info.session_id,
                        info.seeded_from,
                        info.captured,
                    ),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return info

    def set_current_thread(self, session_id: str, thread_id: str) -> None:
        thread = self.get_thread(thread_id)
        if thread is None or thread.session_id != session_id:
            raise LookupError(
                f"Thread {thread_id!r} does not belong to session {session_id!r}"
            )
        self._write(
            "UPDATE sessions SET current_thread = ? WHERE thread_id = ?",
            (thread_id, session_id),
        )

    def checkpoint_exists(self, thread_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ''
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        return row is not None

    def checkpoint_values(self, thread_id: str) -> Mapping[str, Any] | None:
        checkpoint = self.saver.get_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        return values if isinstance(values, Mapping) else {}

    def checkpoint_has_pending_interrupt(self, thread_id: str) -> bool:
        checkpoint = self.saver.get_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        if checkpoint is None:
            return False
        return any(
            len(write) >= 2 and write[1] == "__interrupt__"
            for write in checkpoint.pending_writes or ()
        )

    def set_title(self, thread_id: str, title: str) -> None:
        self._write(
            "UPDATE sessions SET title = ? WHERE thread_id = ?",
            (title, thread_id),
        )

    def set_model(self, thread_id: str, model: str | list[str]) -> None:
        self._write(
            "UPDATE sessions SET model = ? WHERE thread_id = ?",
            (self._encode_model(model), thread_id),
        )

    def set_mode(self, thread_id: str, mode: str) -> None:
        self._write(
            "UPDATE sessions SET mode = ? WHERE thread_id = ?",
            (mode, thread_id),
        )

    def list(self) -> list[SessionInfo]:
        rows = self._connection.execute(
            """
            SELECT thread_id, cwd, model, created, title, mode,
                   leaf_id, current_thread, parent_session
            FROM sessions
            ORDER BY created DESC
            """
        ).fetchall()
        return [
            session
            for row in rows
            if (session := self._session(row)) is not None
        ]

    def set_plugin_state(
        self,
        thread_id: str,
        plugin: str,
        state: dict[str, Any],
    ) -> None:
        payload = json.dumps(state, separators=(",", ":"), sort_keys=True)
        self._write(
            """
            INSERT INTO plugin_state(thread_id, plugin, state) VALUES (?, ?, ?)
            ON CONFLICT(thread_id, plugin) DO UPDATE SET state = excluded.state
            """,
            (thread_id, plugin, payload),
        )

    def get_plugin_state(self, thread_id: str, plugin: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT state FROM plugin_state WHERE thread_id = ? AND plugin = ?",
            (thread_id, plugin),
        ).fetchone()
        return {} if row is None else json.loads(row["state"])

    def all_plugin_state(self, thread_id: str) -> dict[str, dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT plugin, state FROM plugin_state WHERE thread_id = ? ORDER BY plugin",
            (thread_id,),
        ).fetchall()
        return {row["plugin"]: json.loads(row["state"]) for row in rows}
