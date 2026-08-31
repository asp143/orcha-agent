from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from orcha_agent.core import persistence
from orcha_agent.core.ledger import Ledger, MessageEntry
from orcha_agent.core.persistence import (
    TursoPersistenceError,
    TursoSessionStore,
    open_session_store,
)
from orcha_agent.core.session import SessionStore

_AUTH_TOKEN = "secret-token-that-must-not-leak"
_SYNC_URL = "libsql://sessions-example.turso.io"


class FakeReplicaConnection(sqlite3.Connection):
    sync_result: object = None
    sync_error: Exception | None = None
    close_error: Exception | None = None
    sync_calls: int = 0
    close_calls: int = 0
    events: list[str]

    def sync(self) -> object:
        self.events.append("sync")
        self.sync_calls += 1
        if self.sync_error is not None:
            raise self.sync_error
        return self.sync_result

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        super().close()


class FakeConnector:
    def __init__(self, *, sync_result: object = None) -> None:
        self.sync_result = sync_result
        self.calls: list[tuple[str, str, str]] = []
        self.events: list[str] = []
        self.connection: FakeReplicaConnection | None = None

    def __call__(
        self,
        database: str,
        *,
        sync_url: str,
        auth_token: str,
        _check_same_thread: bool,
    ) -> FakeReplicaConnection:
        assert _check_same_thread is False
        self.calls.append((database, sync_url, auth_token))
        connection = sqlite3.connect(
            database,
            check_same_thread=False,
            factory=FakeReplicaConnection,
        )
        connection.sync_result = self.sync_result
        connection.events = self.events
        connection.set_trace_callback(lambda _sql: self.events.append("sql"))
        self.connection = connection
        return connection


def _turso_config(db_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        db_path=db_path,
        persistence_backend="turso",
        turso_url=_SYNC_URL,
    )


def _assert_secret_redacted(error: BaseException) -> None:
    assert _AUTH_TOKEN not in str(error)
    assert _AUTH_TOKEN not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        for value in traceback.tb_frame.f_locals.values():
            assert _AUTH_TOKEN not in repr(value)
        traceback = traceback.tb_next


def test_sqlite_is_default_and_does_not_import_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(name: str) -> Any:
        raise AssertionError(f"must not import {name}")

    monkeypatch.setattr(persistence.importlib, "import_module", unexpected_import)

    with open_session_store(SimpleNamespace(db_path=tmp_path / "sessions.db")) as store:
        created = store.create(tmp_path, "fake:model", thread_id="sqlite-default")

        assert type(store) is SessionStore
        assert store.get(created.thread_id) == created


def test_turso_missing_dependency_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_libsql(name: str) -> Any:
        assert name == "libsql"
        raise ModuleNotFoundError("No module named 'libsql'", name="libsql")

    monkeypatch.setattr(persistence.importlib, "import_module", missing_libsql)

    with pytest.raises(TursoPersistenceError) as raised:
        open_session_store(
            _turso_config(tmp_path / "replica.db"),
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        )

    message = str(raised.value)
    assert "libsql" in message
    assert "install" in message.lower()
    _assert_secret_redacted(raised.value)


def test_turso_url_can_come_from_environment(tmp_path: Path) -> None:
    connector = FakeConnector()
    config = SimpleNamespace(
        db_path=tmp_path / "legacy.db",
        persistence=SimpleNamespace(
            backend="turso",
            replica_path=tmp_path / "replica.db",
            url=None,
            sync_on_start=False,
            sync_on_exit=False,
        ),
    )

    with open_session_store(
        config,
        environ={
            "TURSO_DATABASE_URL": _SYNC_URL,
            "TURSO_AUTH_TOKEN": _AUTH_TOKEN,
        },
        connector=connector,
    ):
        pass

    assert connector.calls == [(str(tmp_path / "replica.db"), _SYNC_URL, _AUTH_TOKEN)]


def test_fake_turso_connector_opens_syncs_and_closes_replica(tmp_path: Path) -> None:
    sync_result = {"frames_synced": 7}
    connector = FakeConnector(sync_result=sync_result)
    db_path = tmp_path / "replica.db"

    store = open_session_store(
        _turso_config(db_path),
        environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        connector=connector,
    )
    assert isinstance(store, TursoSessionStore)
    assert isinstance(store, SessionStore)
    assert store.supports_sync is True
    assert connector.calls == [(str(db_path), _SYNC_URL, _AUTH_TOKEN)]
    assert _AUTH_TOKEN not in repr(store)
    assert _AUTH_TOKEN not in repr(store.__dict__)
    assert not hasattr(store, "auth_token")
    assert not hasattr(store, "_auth_token")

    created = store.create(tmp_path, "fake:model", thread_id="turso-session")
    assert store.get(created.thread_id) == created
    assert store.last_sync_result is None
    assert store.sync() is sync_result
    assert store.last_sync_result is sync_result
    assert connector.connection is not None
    assert connector.connection.sync_calls == 1

    store.close()
    store.close()
    assert connector.connection.close_calls == 1


def test_structural_persistence_settings_control_sync_lifecycle(tmp_path: Path) -> None:
    sync_result = {"frames_synced": 3}
    connector = FakeConnector(sync_result=sync_result)
    settings = SimpleNamespace(
        backend="turso",
        replica_path=tmp_path / "configured-replica.db",
        url=_SYNC_URL,
        sync_on_start=True,
        sync_on_exit=True,
    )
    application_config = SimpleNamespace(
        db_path=tmp_path / "legacy-sessions.db",
        persistence=settings,
    )

    store = open_session_store(
        application_config,
        environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        connector=connector,
    )
    assert isinstance(store, TursoSessionStore)
    assert connector.connection is not None
    assert connector.calls == [(str(settings.replica_path), _SYNC_URL, _AUTH_TOKEN)]
    assert connector.connection.sync_calls == 1
    assert connector.events[0] == "sync"
    assert "sql" in connector.events[1:]
    assert store.last_sync_result is sync_result

    store.close()
    assert connector.connection.sync_calls == 2
    assert connector.connection.close_calls == 1


def test_structured_memory_setup_failure_closes_replica_and_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = FakeConnector()
    settings = SimpleNamespace(
        backend="turso",
        replica_path=tmp_path / "replica.db",
        url=_SYNC_URL,
        sync_on_start=False,
        sync_on_exit=True,
    )
    config = SimpleNamespace(
        db_path=tmp_path / "legacy.db",
        persistence=settings,
        memory_store=SimpleNamespace(backend="hybrid", workspace="orcha-agent"),
    )

    class FailedMemoryStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(f"backend leaked {_AUTH_TOKEN}")

    monkeypatch.setattr(persistence, "MemoryStore", FailedMemoryStore)

    with pytest.raises(TursoPersistenceError) as raised:
        open_session_store(
            config,
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
            connector=connector,
        )

    _assert_secret_redacted(raised.value)
    assert connector.connection is not None
    assert connector.connection.close_calls == 1
    assert connector.connection.sync_calls == 0


def test_turso_rejects_credentials_embedded_in_url_without_echoing_them(
    tmp_path: Path,
) -> None:
    credential_url = "libsql://user:embedded-secret@example.turso.io"
    config = SimpleNamespace(
        db_path=tmp_path / "replica.db",
        persistence_backend="turso",
        turso_url=credential_url,
    )

    with pytest.raises(TursoPersistenceError) as raised:
        open_session_store(
            config,
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
            connector=FakeConnector(),
        )

    assert credential_url not in str(raised.value)
    assert "embedded-secret" not in str(raised.value)
    assert "TURSO_AUTH_TOKEN" in str(raised.value)


def test_initial_sync_failure_is_redacted_and_closes_replica(tmp_path: Path) -> None:
    connector = FakeConnector()
    settings = SimpleNamespace(
        backend="turso",
        replica_path=tmp_path / "replica.db",
        url=_SYNC_URL,
        sync_on_start=True,
        sync_on_exit=True,
    )
    config = SimpleNamespace(db_path=tmp_path / "legacy.db", persistence=settings)
    original_call = connector.__call__

    def connect_with_failed_sync(*args: object, **kwargs: object) -> FakeReplicaConnection:
        connection = original_call(*args, **kwargs)
        connection.sync_error = RuntimeError(_AUTH_TOKEN)
        return connection

    with pytest.raises(TursoPersistenceError) as raised:
        open_session_store(
            config,
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
            connector=connect_with_failed_sync,
        )

    _assert_secret_redacted(raised.value)
    assert connector.connection is not None
    assert connector.connection.close_calls == 1


def test_preconnect_filesystem_failure_does_not_retain_token_in_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_before_connect(_self: object) -> None:
        raise PermissionError("replica path denied")

    monkeypatch.setattr(
        TursoSessionStore,
        "_reject_database_symlinks",
        fail_before_connect,
    )

    with pytest.raises(PermissionError) as raised:
        open_session_store(
            _turso_config(tmp_path / "replica.db"),
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
            connector=FakeConnector(),
        )

    _assert_secret_redacted(raised.value)


def test_turso_open_error_does_not_render_or_retain_token(tmp_path: Path) -> None:
    def failing_connector(*args: object, **kwargs: object) -> Any:
        raise RuntimeError(f"SDK rejected auth_token={_AUTH_TOKEN}")

    with pytest.raises(TursoPersistenceError) as raised:
        open_session_store(
            _turso_config(tmp_path / "replica.db"),
            environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
            connector=failing_connector,
        )

    assert "open" in str(raised.value).lower()
    _assert_secret_redacted(raised.value)


def test_turso_sync_and_close_errors_are_redacted(tmp_path: Path) -> None:
    connector = FakeConnector()
    store = open_session_store(
        _turso_config(tmp_path / "replica.db"),
        environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        connector=connector,
    )
    assert isinstance(store, TursoSessionStore)
    assert connector.connection is not None

    connector.connection.sync_error = RuntimeError(f"sync failed for auth_token={_AUTH_TOKEN}")
    with pytest.raises(TursoPersistenceError) as sync_raised:
        store.sync()
    _assert_secret_redacted(sync_raised.value)

    connector.connection.close_error = RuntimeError(f"close failed for auth_token={_AUTH_TOKEN}")
    with pytest.raises(TursoPersistenceError) as close_raised:
        store.close()
    _assert_secret_redacted(close_raised.value)

    connector.connection.close_error = None
    store.close()


def test_turso_context_exit_preserves_primary_runtime_error(tmp_path: Path) -> None:
    connector = FakeConnector()
    store = open_session_store(
        _turso_config(tmp_path / "replica.db"),
        environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        connector=connector,
        sync_on_close=True,
    )
    assert connector.connection is not None
    connector.connection.sync_error = RuntimeError("shutdown sync failed")

    with pytest.raises(ValueError, match="primary failure"):
        with store:
            raise ValueError("primary failure")

    assert connector.connection.close_calls == 1


def test_installed_libsql_driver_satisfies_session_and_checkpoint_contract(
    tmp_path: Path,
) -> None:
    libsql = pytest.importorskip("libsql")

    def local_connector(database: str, **kwargs: object) -> object:
        return libsql.connect(
            database,
            _check_same_thread=bool(kwargs.get("_check_same_thread", True)),
        )

    config = _turso_config(tmp_path / "native-libsql.db")
    with open_session_store(
        config,
        environ={"TURSO_AUTH_TOKEN": _AUTH_TOKEN},
        connector=local_connector,
    ) as store:
        session = store.create(tmp_path, "fake:model", thread_id="native-libsql")
        ledger = Ledger(store)
        ledger.append(
            session.thread_id,
            MessageEntry(message={"type": "human", "data": {"content": "hello"}}),
        )
        assert ledger.count(session.thread_id) == 1

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"messages": []}
        config_value = {
            "configurable": {
                "thread_id": session.current_thread,
                "checkpoint_ns": "",
            }
        }
        stored_config = store.saver.put(config_value, checkpoint, {}, {})
        restored = store.saver.get_tuple(stored_config)

        assert restored is not None
        assert restored.checkpoint["id"] == checkpoint["id"]
