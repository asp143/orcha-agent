"""SQLite-backed agent sessions and plugin state."""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT NOT NULL,
                created TEXT NOT NULL,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'ask'
            );
            CREATE TABLE IF NOT EXISTS plugin_state (
                thread_id TEXT NOT NULL,
                plugin TEXT NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (thread_id, plugin),
                FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
            );
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(sessions)")
        }
        if "mode" not in columns:
            self._connection.execute(
                "ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'ask'"
            )
        self._connection.commit()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _write(self, sql: str, parameters: Sequence[Any]) -> None:
        with self.saver.lock:
            self._connection.execute(sql, parameters)
            self._connection.commit()

    def create(
        self,
        cwd: str | Path,
        model: str | list[str],
        mode: str = "ask",
        *,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> SessionInfo:
        created = datetime.now(UTC).isoformat(timespec="microseconds")
        identifier = thread_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        model_text = self._encode_model(model)
        stored_model = model if isinstance(model, str) else list(model)
        info = SessionInfo(identifier, str(cwd), stored_model, created, title, mode)
        self._write(
            "INSERT INTO sessions(thread_id, cwd, model, created, title, mode) VALUES (?, ?, ?, ?, ?, ?)",
            (info.thread_id, info.cwd, model_text, info.created, info.title, info.mode),
        )
        return info

    @staticmethod
    def _encode_model(value: str | list[str]) -> str:
        return value if isinstance(value, str) else json.dumps(value)

    @staticmethod
    def _decode_model(value: str) -> str | list[str]:
        if value.startswith("["):
            decoded = json.loads(value)
            if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
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
        )

    def get(self, thread_id: str) -> SessionInfo | None:
        row = self._connection.execute(
            "SELECT thread_id, cwd, model, created, title, mode FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return self._session(row)

    def exists(self, thread_id: str) -> bool:
        return self.get(thread_id) is not None

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
            "SELECT thread_id, cwd, model, created, title, mode FROM sessions ORDER BY created DESC"
        ).fetchall()
        return [session for row in rows if (session := self._session(row)) is not None]

    def set_plugin_state(self, thread_id: str, plugin: str, state: dict[str, Any]) -> None:
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
