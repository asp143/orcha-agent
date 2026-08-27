import json
import re
import sqlite3
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver

from orcha_agent.core.ledger import Ledger, MessageEntry
from orcha_agent.core.session import SessionStore


def _create_v0_database(
    db_path: Path,
    cwd: Path,
) -> tuple[str, str, list[HumanMessage | AIMessage]]:
    """Create the exact pre-ledger schema, including one legacy checkpoint."""
    checkpoint_session = "legacy-with-checkpoint"
    empty_session = "legacy-without-checkpoint"
    messages = [
        HumanMessage("legacy question", id="legacy-human"),
        AIMessage("legacy answer", id="legacy-assistant"),
    ]

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                thread_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT NOT NULL,
                created TEXT NOT NULL,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'ask'
            );
            CREATE TABLE plugin_state (
                thread_id TEXT NOT NULL,
                plugin TEXT NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (thread_id, plugin),
                FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO sessions(thread_id, cwd, model, created, title, mode)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    checkpoint_session,
                    str(cwd),
                    "fake:legacy",
                    "2026-08-26T10:00:00.000000+00:00",
                    "Migrated session",
                    "ask",
                ),
                (
                    empty_session,
                    str(cwd),
                    "fake:empty",
                    "2026-08-26T11:00:00.000000+00:00",
                    "Empty session",
                    "plan",
                ),
            ],
        )

        saver = SqliteSaver(connection)
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"messages": messages}
        saver.put(
            {
                "configurable": {
                    "thread_id": checkpoint_session,
                    "checkpoint_ns": "",
                }
            },
            checkpoint,
            {},
            {},
        )

    return checkpoint_session, empty_session, messages


def _table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def _primary_key_columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return tuple(row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5])


def _index_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[tuple[str, ...]]:
    indexes = connection.execute(f'PRAGMA index_list("{table}")').fetchall()
    return {
        tuple(
            column[2]
            for column in connection.execute(
                f'PRAGMA index_info("{index[1]}")'
            )
        )
        for index in indexes
    }


def test_create_get_exists_and_list_newest_first(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        first = store.create(tmp_path / "first", "fake:first", title="First session")
        second = store.create(tmp_path / "second", "fake:second")

        assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", first.thread_id)
        assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", second.thread_id)
        assert first.thread_id != second.thread_id
        assert Path(first.cwd) == tmp_path / "first"
        assert first.model == "fake:first"
        assert first.title == "First session"
        assert store.exists(first.thread_id)
        assert store.get(first.thread_id) == first
        assert store.list() == [second, first]


def test_session_can_be_resumed_after_store_is_reopened(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        created = store.create(tmp_path, "fake:model", title="Persistent session")

    with SessionStore(db_path) as reopened:
        resumed = reopened.get(created.thread_id)

        assert reopened.exists(created.thread_id)
        assert resumed == created
        assert reopened.list() == [created]


def test_plugin_state_round_trips_as_json_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    state = {
        "nested": {"count": 3, "enabled": True},
        "items": ["one", 2, None],
        "message": "persist me",
    }

    with SessionStore(db_path) as store:
        session = store.create(tmp_path, "fake:model")
        store.set_plugin_state(session.thread_id, "planner", state)
        store.set_plugin_state(session.thread_id, "theme", {"name": "dark"})

        assert store.get_plugin_state(session.thread_id, "planner") == state
        assert store.all_plugin_state(session.thread_id) == {
            "planner": state,
            "theme": {"name": "dark"},
        }

    with SessionStore(db_path) as reopened:
        assert reopened.get_plugin_state(session.thread_id, "planner") == state
        assert reopened.all_plugin_state(session.thread_id) == {
            "planner": state,
            "theme": {"name": "dark"},
        }


def test_session_title_can_be_set_from_first_user_message(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "fake:model")

        store.set_title(session.thread_id, "Implement plugin loading")

        assert store.get(session.thread_id).title == "Implement plugin loading"


def test_fallback_model_chain_round_trips_without_losing_specs(tmp_path: Path) -> None:
    chain = ["anthropic:primary", "openai:fallback"]
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        created = store.create(tmp_path, chain)
        assert created.model == chain

    with SessionStore(db_path) as reopened:
        assert reopened.get(created.thread_id).model == chain


def test_set_model_persists_fallback_chain_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    replacement = ["anthropic:primary", "openai:fallback"]

    with SessionStore(db_path) as store:
        session = store.create(tmp_path, "fake:original", mode="ask")
        store.set_model(session.thread_id, replacement)

    with SessionStore(db_path) as reopened:
        resumed = reopened.get(session.thread_id)

        assert resumed is not None
        assert resumed.model == replacement
        assert resumed.mode == "ask"


def test_set_mode_persists_without_changing_model_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        session = store.create(tmp_path, "fake:model", mode="ask")
        store.set_mode(session.thread_id, "plan")

    with SessionStore(db_path) as reopened:
        resumed = reopened.get(session.thread_id)

        assert resumed is not None
        assert resumed.model == "fake:model"
        assert resumed.mode == "plan"


def test_metadata_and_plugin_writes_use_saver_lock(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "fake:model")

        class RecordingLock:
            def __init__(self) -> None:
                self.entries = 0
                self.exits = 0

            def __enter__(self) -> "RecordingLock":
                self.entries += 1
                return self

            def __exit__(self, *_: object) -> None:
                self.exits += 1

        lock = RecordingLock()
        store.saver.lock = lock

        store.set_title(session.thread_id, "Locked")
        store.set_model(session.thread_id, ["fake:primary", "fake:fallback"])
        store.set_mode(session.thread_id, "plan")
        store.set_plugin_state(
            session.thread_id,
            "planner",
            {"step": "locked"},
        )

        assert lock.entries == 4
        assert lock.exits == 4
        assert store.get(session.thread_id).title == "Locked"
        assert store.get(session.thread_id).model == [
            "fake:primary",
            "fake:fallback",
        ]
        assert store.get(session.thread_id).mode == "plan"
        assert store.get_plugin_state(session.thread_id, "planner") == {
            "step": "locked"
        }


def test_schema_version_one_has_ledger_tables_columns_keys_and_indexes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path):
        pass

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"sessions", "entries", "threads", "meta"} <= tables
        assert _table_columns(connection, "sessions") == {
            "thread_id": "TEXT",
            "cwd": "TEXT",
            "model": "TEXT",
            "created": "TEXT",
            "title": "TEXT",
            "mode": "TEXT",
            "leaf_id": "TEXT",
            "current_thread": "TEXT",
            "parent_session": "TEXT",
        }
        assert _table_columns(connection, "entries") == {
            "session_id": "TEXT",
            "id": "TEXT",
            "parent_id": "TEXT",
            "seq": "INTEGER",
            "type": "TEXT",
            "ts": "TEXT",
            "payload": "TEXT",
        }
        assert _table_columns(connection, "threads") == {
            "thread_id": "TEXT",
            "session_id": "TEXT",
            "seeded_from": "TEXT",
            "captured": "INTEGER",
        }
        assert _table_columns(connection, "meta") == {
            "key": "TEXT",
            "value": "TEXT",
        }
        assert _primary_key_columns(connection, "sessions") == ("thread_id",)
        assert _primary_key_columns(connection, "entries") == ("session_id", "id")
        assert _primary_key_columns(connection, "threads") == ("thread_id",)
        assert _primary_key_columns(connection, "meta") == ("key",)
        assert {
            ("session_id", "seq"),
            ("session_id", "parent_id"),
        } <= _index_columns(connection, "entries")
        assert connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone() == ("1",)


def test_new_session_starts_on_thread_zero_and_thread_metadata_advances(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(
            tmp_path,
            "fake:model",
            thread_id="thread-metadata-session",
        )

        assert session.leaf_id is None
        assert session.current_thread == "thread-metadata-session.0"
        assert session.parent_session is None

        initial = store.get_thread("thread-metadata-session.0")
        assert initial is not None
        assert initial.thread_id == "thread-metadata-session.0"
        assert initial.session_id == session.thread_id
        assert initial.seeded_from is None
        assert initial.captured == 0
        assert store.next_thread_id(session.thread_id) == "thread-metadata-session.1"

        branch = store.create_thread(
            session.thread_id,
            seeded_from="deadbeef",
            captured=3,
        )

        assert branch.thread_id == "thread-metadata-session.1"
        assert store.get_thread(branch.thread_id) == branch
        assert branch.seeded_from == "deadbeef"
        assert branch.captured == 3
        assert store.next_thread_id(session.thread_id) == "thread-metadata-session.2"

        store.set_current_thread(session.thread_id, branch.thread_id)

        updated = store.get(session.thread_id)
        assert updated is not None
        assert updated.current_thread == branch.thread_id


def test_session_prefix_resolution_prefers_exact_then_unique_match(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        exact = store.create(tmp_path, "fake:model", thread_id="alpha")
        store.create(tmp_path, "fake:model", thread_id="alpha-child")
        unique = store.create(tmp_path, "fake:model", thread_id="unique-session")

        assert store.resolve_session("alpha") == exact
        assert store.resolve_session("unique") == unique


def test_session_prefix_resolution_reports_sorted_ambiguous_matches(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        store.create(tmp_path, "fake:model", thread_id="project-zeta")
        store.create(tmp_path, "fake:model", thread_id="project-alpha")

        with pytest.raises(LookupError) as raised:
            store.resolve_session("project-")

        message = str(raised.value)
        assert "project-alpha" in message
        assert "project-zeta" in message
        assert message.index("project-alpha") < message.index("project-zeta")


def test_session_prefix_resolution_rejects_a_missing_prefix(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        store.create(tmp_path, "fake:model", thread_id="present-session")

        with pytest.raises(LookupError, match="missing"):
            store.resolve_session("missing")


def test_v0_database_migrates_checkpoints_and_empty_sessions_atomically(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    checkpoint_session, empty_session, messages = _create_v0_database(
        db_path,
        tmp_path,
    )
    expected_messages = [message_to_dict(message) for message in messages]

    with SessionStore(db_path) as store:
        ledger = Ledger(store)
        migrated = store.get(checkpoint_session)
        entries = ledger.all(checkpoint_session)

        assert migrated is not None
        assert migrated.current_thread == checkpoint_session
        assert migrated.parent_session is None
        assert migrated.cwd == str(tmp_path)
        assert migrated.model == "fake:legacy"
        assert migrated.title == "Migrated session"
        assert migrated.mode == "ask"
        assert migrated.leaf_id == entries[-1].id
        assert len(entries) == 2
        assert all(isinstance(entry, MessageEntry) for entry in entries)
        assert [entry.message for entry in entries] == expected_messages
        assert entries[0].parent_id is None
        assert entries[1].parent_id == entries[0].id
        assert all(re.fullmatch(r"[0-9a-f]{8}", entry.id) for entry in entries)

        legacy_thread = store.get_thread(checkpoint_session)
        assert legacy_thread is not None
        assert legacy_thread.thread_id == checkpoint_session
        assert legacy_thread.session_id == checkpoint_session
        assert legacy_thread.seeded_from is None
        assert legacy_thread.captured == len(messages)
        assert store.checkpoint_exists(checkpoint_session)

        empty = store.get(empty_session)
        assert empty is not None
        assert empty.leaf_id is None
        assert empty.current_thread is None
        assert ledger.all(empty_session) == []
        assert store.get_thread(empty_session) is None
        assert not store.checkpoint_exists(empty_session)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, parent_id, seq, type, payload
            FROM entries
            WHERE session_id = ?
            ORDER BY seq
            """,
            (checkpoint_session,),
        ).fetchall()
        assert [row["seq"] for row in rows] == [0, 1]
        assert [row["type"] for row in rows] == ["message", "message"]
        assert [json.loads(row["payload"]) for row in rows] == [
            {"message": message} for message in expected_messages
        ]
        assert connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "1"


def test_v0_migration_rolls_back_schema_and_backfill_on_checkpoint_failure(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    checkpoint_session, _, _ = _create_v0_database(db_path, tmp_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE checkpoints SET checkpoint = ? WHERE thread_id = ?",
            (b"corrupt checkpoint", checkpoint_session),
        )

    with pytest.raises(Exception):
        SessionStore(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"entries", "threads", "meta"}.isdisjoint(tables)
        assert set(_table_columns(connection, "sessions")) == {
            "thread_id",
            "cwd",
            "model",
            "created",
            "title",
            "mode",
        }


def test_explicit_schema_version_zero_is_migrated_to_version_one(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    checkpoint_session, _, _ = _create_v0_database(db_path, tmp_path)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO meta(key, value) VALUES ('schema_version', '0');
            """
        )

    with SessionStore(db_path) as store:
        assert len(Ledger(store).all(checkpoint_session)) == 2
        migrated = store.get(checkpoint_session)
        assert migrated is not None
        assert migrated.current_thread == checkpoint_session

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone() == ("1",)


def test_v0_migration_is_idempotent_when_store_is_reopened(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    checkpoint_session, empty_session, _ = _create_v0_database(db_path, tmp_path)

    with SessionStore(db_path) as store:
        first_entries = Ledger(store).all(checkpoint_session)
        first_thread = store.get_thread(checkpoint_session)

    with SessionStore(db_path) as reopened:
        assert Ledger(reopened).all(checkpoint_session) == first_entries
        assert Ledger(reopened).all(empty_session) == []
        assert reopened.get_thread(checkpoint_session) == first_thread

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM entries WHERE session_id = ?",
            (checkpoint_session,),
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT COUNT(*) FROM threads WHERE session_id = ?",
            (checkpoint_session,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone() == ("1",)


def test_fork_copies_only_the_selected_leaf_path(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        source = store.create(
            tmp_path,
            "fake:model",
            thread_id="source-session",
        )
        ledger = Ledger(store)
        root = ledger.append(
            source.thread_id,
            MessageEntry(message=message_to_dict(HumanMessage("root"))),
        )
        discarded = ledger.append(
            source.thread_id,
            MessageEntry(message=message_to_dict(AIMessage("discarded branch"))),
        )
        ledger.branch(source.thread_id, root.id)
        selected = ledger.append(
            source.thread_id,
            MessageEntry(message=message_to_dict(AIMessage("selected branch"))),
        )
        target = store.create(
            tmp_path,
            "fake:model",
            thread_id="forked-session",
            parent_session=source.thread_id,
        )

        ledger.fork(source.thread_id, target.thread_id)

        copied = ledger.all(target.thread_id)
        assert [entry.id for entry in copied] == [root.id, selected.id]
        assert [entry.parent_id for entry in copied] == [None, root.id]
        assert discarded.id not in {entry.id for entry in copied}
        forked = store.get(target.thread_id)
        assert forked is not None
        assert forked.parent_session == source.thread_id
        assert forked.leaf_id == selected.id
