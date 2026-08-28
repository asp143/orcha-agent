"""SQLite-backed agent sessions and plugin state."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import stat
import threading
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
    captured_message_ids: tuple[str, ...] = ()


class _AsyncSqliteSaver(SqliteSaver):
    """Expose SqliteSaver's synchronized operations to async graph execution."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.lock = threading.RLock()

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
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path.parent.chmod(0o700)
        self._reject_database_symlinks()
        if self.db_path.exists():
            owner = self.db_path.stat().st_uid
            current_user = os.getuid()
            if owner != current_user:
                raise PermissionError(
                    f"Session database {self.db_path} is owned by uid {owner}, "
                    f"not the current uid {current_user}"
                )

        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            self._secure_database_files()
            self._connection.row_factory = sqlite3.Row
            self.saver = _AsyncSqliteSaver(self._connection)
            self.saver.setup()
            self._secure_database_files()
            self._migrate_schema()
            self._secure_database_files()
        except BaseException:
            self._connection.close()
            raise

    def _database_files(self) -> tuple[Path, Path, Path]:
        return (
            self.db_path,
            self.db_path.with_name(f"{self.db_path.name}-wal"),
            self.db_path.with_name(f"{self.db_path.name}-shm"),
        )

    def _reject_database_symlinks(self) -> None:
        for path in self._database_files():
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise PermissionError(
                    f"Session database file {path} must not be a symlink"
                )

    def _secure_database_files(self) -> None:
        self._reject_database_symlinks()
        for path in self._database_files():
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                pass

    def _table_exists(self, table: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            is not None
        )

    def _schema_version(self) -> int:
        if not self._table_exists("meta"):
            return 0
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        value = row["value"]
        try:
            version = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid schema_version value {value!r}") from error
        if version not in (0, 1):
            raise ValueError(f"Unsupported schema_version value {value!r}")
        return version

    def _legacy_checkpoints(
        self,
    ) -> dict[str, tuple[str | None, CheckpointTuple | None]]:
        if not self._table_exists("sessions"):
            return {}
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(sessions)")
        }
        created_expression = "created" if "created" in columns else "NULL"
        rows = self._connection.execute(
            f"SELECT thread_id, {created_expression} AS created "
            "FROM sessions ORDER BY thread_id"
        ).fetchall()
        checkpoints: dict[str, tuple[str | None, CheckpointTuple | None]] = {}
        for row in rows:
            session_id = row["thread_id"]
            checkpoint = self.saver.get_tuple(
                {
                    "configurable": {
                        "thread_id": session_id,
                        "checkpoint_ns": "",
                    }
                }
            )
            checkpoints[session_id] = (row["created"], checkpoint)
        return checkpoints

    def _migrate_schema(self) -> None:
        with self.saver.lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                version = self._schema_version()
                legacy_checkpoints = (
                    self._legacy_checkpoints() if version == 0 else {}
                )
                self._ensure_v1_schema()
                if version == 0:
                    self._migrate_v0_to_v1(legacy_checkpoints)
                self._connection.execute(
                    """
                    INSERT INTO meta(key, value) VALUES ('schema_version', '1')
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _ensure_v1_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT NOT NULL,
                created TEXT NOT NULL,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'ask',
                leaf_id TEXT,
                current_thread TEXT,
                parent_session TEXT
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

        session_columns = {
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
            if name not in session_columns:
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
                captured INTEGER NOT NULL DEFAULT 0,
                captured_message_ids TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        thread_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(threads)")
        }
        if "captured_message_ids" not in thread_columns:
            self._connection.execute(
                "ALTER TABLE threads ADD COLUMN captured_message_ids "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS entries_session_seq "
            "ON entries(session_id, seq)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS entries_session_parent "
            "ON entries(session_id, parent_id)"
        )

    def _migrate_v0_to_v1(
        self,
        legacy_checkpoints: Mapping[
            str, tuple[str | None, CheckpointTuple | None]
        ],
    ) -> None:
        for session_id, (created, checkpoint_tuple) in legacy_checkpoints.items():
            if checkpoint_tuple is None:
                continue

            checkpoint = checkpoint_tuple.checkpoint
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

            metadata = checkpoint_tuple.metadata
            timestamp = (
                metadata.get("ts") if isinstance(metadata, Mapping) else None
            )
            if not isinstance(timestamp, str) or not timestamp:
                timestamp = (
                    created
                    if isinstance(created, str) and created
                    else datetime.now(UTC).isoformat(timespec="microseconds")
                )

            parent_id: str | None = None
            captured_message_ids: list[str] = []
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
                    message if isinstance(message, dict) else message_to_dict(message)
                )
                message_id = getattr(message, "id", None)
                if not isinstance(message_id, str) and isinstance(serialized, Mapping):
                    data = serialized.get("data")
                    if isinstance(data, Mapping):
                        message_id = data.get("id")
                    if not isinstance(message_id, str):
                        message_id = serialized.get("id")
                if isinstance(message_id, str):
                    captured_message_ids.append(message_id)
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
                        timestamp,
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
                INSERT INTO threads(
                    thread_id, session_id, seeded_from, captured,
                    captured_message_ids
                ) VALUES (?, ?, NULL, ?, ?)
                """,
                (
                    session_id,
                    session_id,
                    captured,
                    self._encode_captured_message_ids(captured_message_ids),
                ),
            )
            self._connection.execute(
                """
                UPDATE sessions
                SET leaf_id = ?, current_thread = ?
                WHERE thread_id = ?
                """,
                (parent_id, session_id, session_id),
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
                    INSERT INTO threads(
                        thread_id, session_id, seeded_from, captured,
                        captured_message_ids
                    ) VALUES (?, ?, NULL, 0, '[]')
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
    def _encode_captured_message_ids(value: Sequence[str]) -> str:
        if isinstance(value, (str, bytes, bytearray)) or not all(
            isinstance(message_id, str) for message_id in value
        ):
            raise TypeError("captured_message_ids must contain only strings")
        return json.dumps(list(value), separators=(",", ":"))

    @staticmethod
    def _decode_captured_message_ids(value: str) -> tuple[str, ...]:
        decoded = json.loads(value)
        if not isinstance(decoded, list) or not all(
            isinstance(message_id, str) for message_id in decoded
        ):
            raise ValueError("Invalid captured_message_ids value")
        return tuple(decoded)

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
            SessionStore._decode_captured_message_ids(
                row["captured_message_ids"]
            ),
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
            SELECT thread_id, session_id, seeded_from, captured,
                   captured_message_ids
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
        captured_message_ids: Sequence[str] = (),
        thread_id: str | None = None,
    ) -> ThreadInfo:
        with self.saver.lock:
            try:
                identifier = thread_id or self.next_thread_id(session_id)
                message_ids = tuple(captured_message_ids)
                info = ThreadInfo(
                    identifier,
                    session_id,
                    seeded_from,
                    captured,
                    message_ids,
                )
                self._connection.execute(
                    """
                    INSERT INTO threads(
                        thread_id, session_id, seeded_from, captured,
                        captured_message_ids
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        info.thread_id,
                        info.session_id,
                        info.seeded_from,
                        info.captured,
                        self._encode_captured_message_ids(
                            info.captured_message_ids
                        ),
                    ),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return info

    def activate_thread(
        self,
        session_id: str,
        thread_id: str,
        *,
        seeded_from: str | None = None,
        captured: int = 0,
        captured_message_ids: Sequence[str] = (),
    ) -> ThreadInfo:
        message_ids = tuple(captured_message_ids)
        info = ThreadInfo(
            thread_id,
            session_id,
            seeded_from,
            captured,
            message_ids,
        )
        encoded_message_ids = self._encode_captured_message_ids(message_ids)
        with self.saver.lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._connection.execute(
                    "SELECT 1 FROM sessions WHERE thread_id = ?",
                    (session_id,),
                ).fetchone()
                if session is None:
                    raise LookupError(f"No session {session_id!r}")
                existing = self._connection.execute(
                    "SELECT session_id FROM threads WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                if existing is not None and existing["session_id"] != session_id:
                    raise LookupError(
                        f"Thread {thread_id!r} does not belong to session "
                        f"{session_id!r}"
                    )
                self._connection.execute(
                    """
                    INSERT INTO threads(
                        thread_id, session_id, seeded_from, captured,
                        captured_message_ids
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        seeded_from = excluded.seeded_from,
                        captured = excluded.captured,
                        captured_message_ids = excluded.captured_message_ids
                    """,
                    (
                        info.thread_id,
                        info.session_id,
                        info.seeded_from,
                        info.captured,
                        encoded_message_ids,
                    ),
                )
                self._connection.execute(
                    "UPDATE sessions SET current_thread = ? WHERE thread_id = ?",
                    (thread_id, session_id),
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

    def copy_plugin_state(
        self,
        source_thread_id: str,
        target_thread_id: str,
    ) -> None:
        self._write(
            """
            INSERT INTO plugin_state(thread_id, plugin, state)
            SELECT ?, plugin, state
            FROM plugin_state
            WHERE thread_id = ?
            ON CONFLICT(thread_id, plugin) DO UPDATE SET state = excluded.state
            """,
            (target_thread_id, source_thread_id),
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
