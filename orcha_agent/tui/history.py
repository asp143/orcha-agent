"""Transactional SQLite prompt history with FTS search."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from prompt_toolkit.history import FileHistory, History


def history_path() -> Path:
    return Path.home() / ".local/share/orcha-agent/history.db"


def legacy_history_path() -> Path:
    return Path.home() / ".local/share/orcha-agent/history"


class SQLiteHistory(History):
    """Prompt-toolkit history backed by SQLite and an FTS5 index."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        cwd: str | Path | None = None,
        session_id: str | None = None,
        legacy_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else history_path()
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else ""
        self.session_id = session_id or ""
        self.legacy_path = (
            Path(legacy_path)
            if legacy_path is not None
            else self.path.with_suffix("")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        super().__init__()

    def rebind(
        self,
        *,
        cwd: str | Path,
        session_id: str,
    ) -> None:
        """Update prompt metadata and force prompt-toolkit to reload history."""

        self.cwd = str(Path(cwd).resolve())
        self.session_id = session_id
        self._loaded = False
        self._loaded_strings.clear()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    session_id TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                    prompt,
                    content='history',
                    content_rowid='id'
                );
                CREATE TABLE IF NOT EXISTS history_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS history_ai AFTER INSERT ON history BEGIN
                    INSERT INTO history_fts(rowid, prompt) VALUES (new.id, new.prompt);
                END;
                CREATE TRIGGER IF NOT EXISTS history_ad AFTER DELETE ON history BEGIN
                    INSERT INTO history_fts(history_fts, rowid, prompt)
                    VALUES ('delete', old.id, old.prompt);
                END;
                CREATE TRIGGER IF NOT EXISTS history_au AFTER UPDATE ON history BEGIN
                    INSERT INTO history_fts(history_fts, rowid, prompt)
                    VALUES ('delete', old.id, old.prompt);
                    INSERT INTO history_fts(rowid, prompt) VALUES (new.id, new.prompt);
                END;
                """
            )
            migrated = connection.execute(
                "SELECT value FROM history_meta WHERE key='legacy_migrated'"
            ).fetchone()
            if migrated is None:
                if self.legacy_path.is_file() and self.legacy_path != self.path:
                    legacy = FileHistory(str(self.legacy_path))
                    for prompt in reversed(list(legacy.load_history_strings())):
                        self._insert(connection, prompt, skip_duplicate=True)
                connection.execute(
                    "INSERT INTO history_meta(key, value) VALUES('legacy_migrated', '1')"
                )

    def _insert(
        self,
        connection: sqlite3.Connection,
        prompt: str,
        *,
        skip_duplicate: bool,
    ) -> bool:
        if not prompt:
            return False
        if skip_duplicate:
            latest = connection.execute(
                "SELECT prompt FROM history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest is not None and latest[0] == prompt:
                return False
        connection.execute(
            "INSERT INTO history(prompt, created_at, cwd, session_id) VALUES (?, ?, ?, ?)",
            (prompt, datetime.now(UTC).isoformat(), self.cwd, self.session_id),
        )
        return True

    def append_string(self, string: str) -> None:
        with self._connect() as connection:
            inserted = self._insert(connection, string, skip_duplicate=True)
        if inserted:
            self._loaded_strings.insert(0, string)

    def store_string(self, string: str) -> None:
        with self._connect() as connection:
            self._insert(connection, string, skip_duplicate=True)

    def load_history_strings(self) -> Iterable[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT prompt FROM history ORDER BY id DESC"
            ).fetchall()
        return (str(row[0]) for row in rows)

    def search(self, query: str, *, limit: int = 50) -> list[str]:
        terms = [term for term in query.split() if term]
        if not terms:
            return []
        expression = " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"*' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT history.prompt
                FROM history_fts
                JOIN history ON history.id = history_fts.rowid
                WHERE history_fts MATCH ?
                ORDER BY bm25(history_fts), history.id DESC
                LIMIT ?
                """,
                (expression, max(1, limit)),
            ).fetchall()
        return [str(row[0]) for row in rows]


__all__ = ["SQLiteHistory", "history_path", "legacy_history_path"]
