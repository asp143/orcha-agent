"""SQLite-backed agent sessions and plugin state."""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


@dataclass(frozen=True, slots=True)
class SessionInfo:
    thread_id: str
    cwd: str
    model: str
    created: str
    title: str | None


class SessionStore:
    """Own one SQLite connection shared by checkpoints and session metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.saver = SqliteSaver(self._connection)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT NOT NULL,
                created TEXT NOT NULL,
                title TEXT
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
        self._connection.commit()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def create(
        self,
        cwd: str | Path,
        model: str | list[str],
        *,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> SessionInfo:
        created = datetime.now(UTC).isoformat(timespec="microseconds")
        identifier = thread_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        model_text = model if isinstance(model, str) else ",".join(model)
        info = SessionInfo(identifier, str(cwd), model_text, created, title)
        self._connection.execute(
            "INSERT INTO sessions(thread_id, cwd, model, created, title) VALUES (?, ?, ?, ?, ?)",
            (info.thread_id, info.cwd, info.model, info.created, info.title),
        )
        self._connection.commit()
        return info

    @staticmethod
    def _session(row: sqlite3.Row | None) -> SessionInfo | None:
        if row is None:
            return None
        return SessionInfo(row["thread_id"], row["cwd"], row["model"], row["created"], row["title"])

    def get(self, thread_id: str) -> SessionInfo | None:
        row = self._connection.execute(
            "SELECT thread_id, cwd, model, created, title FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return self._session(row)

    def exists(self, thread_id: str) -> bool:
        return self.get(thread_id) is not None

    def set_title(self, thread_id: str, title: str) -> None:
        self._connection.execute(
            "UPDATE sessions SET title = ? WHERE thread_id = ?",
            (title, thread_id),
        )
        self._connection.commit()

    def list(self) -> list[SessionInfo]:
        rows = self._connection.execute(
            "SELECT thread_id, cwd, model, created, title FROM sessions ORDER BY created DESC"
        ).fetchall()
        return [session for row in rows if (session := self._session(row)) is not None]

    def set_plugin_state(self, thread_id: str, plugin: str, state: dict[str, Any]) -> None:
        payload = json.dumps(state, separators=(",", ":"), sort_keys=True)
        self._connection.execute(
            """
            INSERT INTO plugin_state(thread_id, plugin, state) VALUES (?, ?, ?)
            ON CONFLICT(thread_id, plugin) DO UPDATE SET state = excluded.state
            """,
            (thread_id, plugin, payload),
        )
        self._connection.commit()

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
