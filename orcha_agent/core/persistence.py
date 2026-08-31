"""Session-store factory with optional Turso embedded-replica support.

SQLite remains the default and does not import the optional ``libsql`` module.
The Turso adapter uses libsql's DB-API-compatible connection for every session
operation and exposes explicit synchronization. Synchronization behavior and
availability are determined by libsql and the configured Turso service; this
module does not add or promise offline write semantics.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlsplit

from .memory_store import MemoryStore
from .session import SessionStore, _AsyncSqliteSaver


class PersistenceSettings(Protocol):
    """Structural persistence settings accepted by :func:`open_session_store`."""

    @property
    def backend(self) -> str: ...

    @property
    def replica_path(self) -> str | Path: ...

    @property
    def url(self) -> str | None: ...

    @property
    def sync_on_start(self) -> bool: ...

    @property
    def sync_on_exit(self) -> bool: ...


class ApplicationPersistenceConfig(Protocol):
    """Structural full-config shape accepted by :func:`open_session_store`."""

    @property
    def db_path(self) -> str | Path: ...

    @property
    def persistence(self) -> PersistenceSettings: ...


class TursoPersistenceError(RuntimeError):
    """A sanitized failure while opening or operating a Turso replica."""


ReplicaConnector = Callable[..., Any]
_MISSING = object()


class _SecretValue:
    """Short-lived secret holder whose representation is always redacted."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value: str | None = value

    def take(self) -> str:
        if self._value is None:
            raise RuntimeError("Secret value has already been consumed")
        value = self._value
        self._value = None
        return value

    def __repr__(self) -> str:
        return "<redacted secret>"


class _HybridRow:
    """Tuple-like row that also supports SQLite-style column-name lookup."""

    __slots__ = ("_values", "_names")

    def __init__(self, values: Any, names: tuple[str, ...]) -> None:
        self._values = tuple(values)
        self._names = names

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, str):
            try:
                key = self._names.index(key)
            except ValueError as exc:
                raise IndexError(f"No item with that key: {key}") from exc
        return self._values[key]

    def keys(self) -> list[str]:
        return list(self._names)


class _CursorAdapter:
    """Add dual index/name rows to a DB-API-compatible libsql cursor."""

    __slots__ = ("_cursor",)

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def _names(self) -> tuple[str, ...]:
        description = getattr(self._cursor, "description", None) or ()
        return tuple(str(column[0]) for column in description)

    def _row(self, row: Any) -> Any:
        if row is None or isinstance(row, sqlite3.Row):
            return row
        if isinstance(row, _HybridRow):
            return row
        return _HybridRow(row, self._names())

    def execute(self, sql: str, parameters: Any = ()) -> _CursorAdapter:
        self._cursor.execute(sql, parameters)
        return self

    def executemany(self, sql: str, parameters: Any) -> _CursorAdapter:
        self._cursor.executemany(sql, parameters)
        return self

    def fetchone(self) -> Any:
        return self._row(self._cursor.fetchone())

    def fetchall(self) -> list[Any]:
        return [self._row(row) for row in self._cursor.fetchall()]

    def __iter__(self):
        while True:
            row = self._cursor.fetchone()
            if row is None:
                return
            yield self._row(row)

    def close(self) -> None:
        self._cursor.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _LibsqlConnectionAdapter:
    """Normalize libsql's DB-API surface for SessionStore and SqliteSaver."""

    __slots__ = ("_connection", "row_factory")

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.row_factory: Any = sqlite3.Row

    def cursor(self) -> _CursorAdapter:
        return _CursorAdapter(self._connection.cursor())

    def execute(self, sql: str, parameters: Any = ()) -> _CursorAdapter:
        return _CursorAdapter(self._connection.execute(sql, parameters))

    def executemany(self, sql: str, parameters: Any) -> _CursorAdapter:
        return _CursorAdapter(self._connection.executemany(sql, parameters))

    def executescript(self, sql: str) -> Any:
        return self._connection.executescript(sql)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()

    def sync(self) -> Any:
        return self._connection.sync()

    @property
    def in_transaction(self) -> bool:
        return bool(getattr(self._connection, "in_transaction", False))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def _maybe_config_value(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        return config.get(name, _MISSING)
    return getattr(config, name, _MISSING)


def _config_value(config: object, name: str, default: object = _MISSING) -> object:
    value = _maybe_config_value(config, name)
    if value is _MISSING:
        if default is _MISSING:
            raise TypeError(f"Persistence configuration must define {name!r}")
        return default
    return value


def _validated_turso_url(value: str) -> str:
    """Validate a non-secret Turso endpoint without echoing it in failures."""

    parsed = urlsplit(value)
    if parsed.scheme not in {"libsql", "https"} or not parsed.hostname:
        raise TursoPersistenceError(
            "Turso persistence URL must be a libsql:// or https:// endpoint"
        )
    sensitive_names = {"token", "auth", "password", "secret", "credential", "key"}
    query_has_secret = any(
        any(marker in name.lower() for marker in sensitive_names)
        for name, _value in parse_qsl(parsed.query, keep_blank_values=True)
    )
    if parsed.username is not None or parsed.password is not None or query_has_secret:
        raise TursoPersistenceError(
            "Turso credentials must not be embedded in the database URL; use TURSO_AUTH_TOKEN"
        )
    return value


def _load_libsql_connector() -> ReplicaConnector:
    module: Any = None
    import_failed = False
    try:
        module = importlib.import_module("libsql")
    except ImportError:
        import_failed = True

    if import_failed:
        raise TursoPersistenceError(
            "Turso persistence requires the optional 'libsql' module; install "
            "the Turso libsql Python SDK or provide a compatible connector"
        )

    connector = getattr(module, "connect", None)
    if not callable(connector):
        raise TursoPersistenceError(
            "The installed 'libsql' module does not expose a callable connect()"
        )
    return connector


def _open_turso_replica(
    connector: ReplicaConnector,
    db_path: Path,
    sync_url: str,
    auth_token: str,
) -> Any:
    """Call the small portion of the libsql API on which the adapter depends."""

    connection: Any = _MISSING
    token = auth_token
    del auth_token
    try:
        connection = connector(
            str(db_path),
            sync_url=sync_url,
            auth_token=token,
            _check_same_thread=False,
        )
    except Exception:
        # Do not retain, chain, or render an SDK exception: it may echo credentials.
        pass
    finally:
        del token

    if connection is _MISSING or connection is None:
        raise TursoPersistenceError(
            "Could not open the Turso embedded replica; verify the replica URL, "
            "authentication token, and local database path"
        )
    return connection


class TursoSessionStore(SessionStore):
    """A :class:`SessionStore` backed by one libsql embedded-replica connection."""

    supports_sync = True

    def __init__(
        self,
        db_path: str | Path,
        *,
        sync_url: str,
        auth_token: str,
        connector: ReplicaConnector,
        sync_on_start: bool = False,
        sync_on_close: bool = False,
    ) -> None:
        secret = _SecretValue(auth_token)
        del auth_token
        self.db_path = Path(db_path)
        self._closed = False
        self._sync_on_close = sync_on_close
        self._last_sync_result: Any = None

        # Keep SessionStore's filesystem and ownership protections while replacing
        # only its sqlite3.connect call with the injected libsql connector.
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

        token = secret.take()
        del secret
        try:
            connection = _open_turso_replica(
                connector,
                self.db_path,
                sync_url,
                token,
            )
        finally:
            del token
        self._connection = _LibsqlConnectionAdapter(connection)

        # Pull the remote schema before SqliteSaver or SessionStore creates
        # local tables. Initializing first can conflict with an existing remote
        # database when this is a newly created replica.
        initial_sync_failed = False
        if sync_on_start:
            try:
                self._last_sync_result = self._connection.sync()
            except Exception:
                initial_sync_failed = True
        if initial_sync_failed:
            try:
                self._connection.close()
            except Exception:
                pass
            self._closed = True
            raise TursoPersistenceError(
                "Could not perform the configured initial Turso replica synchronization"
            )

        initialization_failed = False
        try:
            self._secure_database_files()
            self._connection.row_factory = sqlite3.Row
            self.saver = _AsyncSqliteSaver(cast(sqlite3.Connection, self._connection))
            self.saver.setup()
            self._secure_database_files()
            self._migrate_schema()
            self._secure_database_files()
        except Exception:
            initialization_failed = True

        if initialization_failed:
            try:
                self._connection.close()
            except Exception:
                pass
            self._closed = True
            raise TursoPersistenceError(
                "Could not initialize the Turso embedded replica as a session store; "
                "verify that the installed libsql SDK is DB-API compatible"
            )

    @property
    def last_sync_result(self) -> Any:
        """Return the exact result from the most recent successful SDK sync call."""

        return self._last_sync_result

    def sync(self) -> Any:
        """Ask libsql to synchronize the embedded replica and return its result."""

        if self._closed:
            raise TursoPersistenceError("Cannot synchronize a closed Turso session store")

        result: Any = _MISSING
        with self.saver.lock:
            try:
                result = self._connection.sync()
            except Exception:
                # The SDK error may include its URL or authentication token.
                pass

        if result is _MISSING:
            raise TursoPersistenceError(
                "Could not synchronize the Turso embedded replica; verify network "
                "connectivity and Turso credentials"
            )
        self._last_sync_result = result
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close without replacing an exception already raised by the runtime."""

        try:
            self.close()
        except TursoPersistenceError:
            if exc_type is None:
                raise

    def close(self) -> None:
        """Optionally synchronize, then close, sanitizing any SDK error."""

        if self._closed:
            return

        sync_failed = False
        if self._sync_on_close:
            try:
                self.sync()
            except TursoPersistenceError:
                sync_failed = True

        close_failed = False
        try:
            self._connection.close()
        except Exception:
            close_failed = True

        if close_failed:
            raise TursoPersistenceError("Could not close the Turso embedded replica")
        self._closed = True
        if sync_failed:
            raise TursoPersistenceError(
                "Could not synchronize the Turso embedded replica before closing it"
            )


def open_session_store(
    config: (
        PersistenceSettings | ApplicationPersistenceConfig | Mapping[str, object] | str | Path
    ),
    *,
    backend: str | None = None,
    turso_url: str | None = None,
    environ: Mapping[str, str] | None = None,
    connector: ReplicaConnector | None = None,
    sqlite_store_factory: Callable[[str | Path], SessionStore] = SessionStore,
    initial_sync: bool | None = None,
    sync_on_close: bool | None = None,
) -> SessionStore:
    """Open the selected session store without changing the SQLite default.

    The preferred structural settings expose ``backend``, ``replica_path``,
    ``url``, ``sync_on_start``, and ``sync_on_exit``. A full application config
    containing those settings under ``persistence`` is accepted too. The legacy
    ``db_path``/``persistence_backend``/``turso_url`` shape and direct paths are
    supported so existing SQLite callers retain their behavior. Keyword backend
    and URL arguments take precedence.
    """

    sync_on_start: object = False
    sync_on_exit: object = False
    if isinstance(config, (str, Path)):
        db_path = Path(config)
        selected_backend: object = "sqlite" if backend is None else backend
        selected_url: object = turso_url
    else:
        nested_settings = _maybe_config_value(config, "persistence")
        settings = config if nested_settings is _MISSING else nested_settings
        raw_replica_path = _maybe_config_value(settings, "replica_path")
        modern_settings = nested_settings is not _MISSING or raw_replica_path is not _MISSING

        if raw_replica_path is _MISSING:
            raw_db_path = _config_value(config, "db_path")
        else:
            raw_db_path = raw_replica_path
        if not isinstance(raw_db_path, (str, Path)):
            raise TypeError("Persistence database path must be a string or Path")
        db_path = Path(raw_db_path)

        selected_backend = backend
        if selected_backend is None:
            backend_name = "backend" if modern_settings else "persistence_backend"
            selected_backend = _config_value(settings, backend_name, "sqlite")
        selected_url = turso_url
        if selected_url is None:
            url_name = "url" if modern_settings else "turso_url"
            selected_url = _config_value(settings, url_name, None)
        if modern_settings:
            sync_on_start = _config_value(settings, "sync_on_start", False)
            sync_on_exit = _config_value(settings, "sync_on_exit", False)

    if not isinstance(selected_backend, str):
        raise TypeError("Persistence backend must be a string")
    normalized_backend = selected_backend.strip().lower()

    if normalized_backend == "sqlite":
        # A full Config has long exposed db_path, and many plugins/tests construct
        # Config directly. Preserve that authoritative SQLite path even though the
        # new persistence settings have their own replica path for Turso.
        if not isinstance(config, (str, Path)):
            legacy_path = _maybe_config_value(config, "db_path")
            if isinstance(legacy_path, (str, Path)):
                db_path = Path(legacy_path)
        return sqlite_store_factory(db_path)
    if normalized_backend != "turso":
        raise ValueError(
            f"Unsupported persistence backend {selected_backend!r}; expected 'sqlite' or 'turso'"
        )

    environment = os.environ if environ is None else environ
    if selected_url is None:
        selected_url = environment.get("TURSO_DATABASE_URL")
    if not isinstance(selected_url, str) or not selected_url.strip():
        raise TursoPersistenceError(
            "Turso persistence requires TURSO_DATABASE_URL or a non-empty configured URL"
        )
    selected_url = _validated_turso_url(selected_url.strip())
    if initial_sync is not None:
        sync_on_start = initial_sync
    if sync_on_close is not None:
        sync_on_exit = sync_on_close
    if not isinstance(sync_on_start, bool) or not isinstance(sync_on_exit, bool):
        raise TypeError("Turso synchronization settings must be booleans")

    auth_token = environment.get("TURSO_AUTH_TOKEN")
    if not auth_token:
        del environment
        del environ
        raise TursoPersistenceError(
            "Turso persistence requires TURSO_AUTH_TOKEN in the provided environment"
        )
    secret = _SecretValue(auth_token)
    del auth_token
    del environment
    del environ
    replica_connector = connector if connector is not None else _load_libsql_connector()

    token = secret.take()
    del secret
    open_failure: str | None = None
    store: TursoSessionStore | None = None
    try:
        store = TursoSessionStore(
            db_path,
            sync_url=selected_url,
            auth_token=token,
            connector=replica_connector,
            sync_on_start=sync_on_start,
            sync_on_close=sync_on_exit,
        )
    except TursoPersistenceError as exc:
        open_failure = str(exc)
    finally:
        del token
    if open_failure is not None:
        raise TursoPersistenceError(open_failure)
    if store is None:
        raise TursoPersistenceError("Could not open the Turso embedded replica")

    memory_settings = _maybe_config_value(config, "memory_store")
    memory_setup_failed = False
    if memory_settings is not _MISSING:
        memory_backend = _config_value(memory_settings, "backend", "files")
        if memory_backend in {"hybrid", "turso"}:
            try:
                store.structured_memory = MemoryStore(
                    store._connection,
                    store.saver.lock,
                )
            except Exception:
                memory_setup_failed = True
    if memory_setup_failed:
        store._sync_on_close = False
        try:
            store.close()
        except Exception:
            pass
        raise TursoPersistenceError(
            "Could not initialize structured memory in the Turso embedded replica"
        )
    return store


__all__ = [
    "ApplicationPersistenceConfig",
    "PersistenceSettings",
    "TursoPersistenceError",
    "TursoSessionStore",
    "open_session_store",
]
