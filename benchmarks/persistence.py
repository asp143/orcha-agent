from __future__ import annotations

import gc
import json
import sqlite3
import tempfile
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, message_to_dict

from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.context import AppContext
from orcha_agent.tui.history import SQLiteHistory
from orcha_agent.tui.overlays.history import HistoryOverlay
from orcha_agent.tui.overlays.session import SessionOverlay

from .common import RunConfig, database_file_bytes, measurement, result_document

ACTIVE_LEDGER_ENTRIES = 1_000
ABANDONED_LEDGER_ENTRIES = (0, 10_000, 100_000)
TURN_COUNTS = (100, 1_000)
TURN_STATE_BYTES = (0, 100 * 1024)
LOAD_ROWS = (10_000, 100_000)


class _CaptureGraph:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get_state(self, _: object) -> SimpleNamespace:
        return SimpleNamespace(values=self.values)


class _CaptureConsole:
    def error(self, message: str) -> None:
        raise RuntimeError(message)


def _populate_ledger(
    store: SessionStore,
    session_id: str,
    *,
    active_entries: int,
    abandoned_entries: int,
) -> None:
    message = message_to_dict(HumanMessage(content="ledger fixture", id="fixture-message"))
    active_payload = json.dumps(
        {"message": message}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    abandoned_payload = json.dumps(
        {"custom_type": "abandoned", "data": {"fixture": True}},
        separators=(",", ":"),
        sort_keys=True,
    )
    timestamp = "2026-01-01T00:00:00+00:00"

    def rows() -> Any:
        parent: str | None = None
        for index in range(active_entries):
            entry_id = f"a{index:07x}"
            yield (
                session_id,
                entry_id,
                parent,
                index,
                "message",
                timestamp,
                active_payload,
            )
            parent = entry_id
        parent = "a0000000"
        for index in range(abandoned_entries):
            entry_id = f"b{index:07x}"
            yield (
                session_id,
                entry_id,
                parent,
                active_entries + index,
                "custom",
                timestamp,
                abandoned_payload,
            )
            parent = entry_id

    with store.saver.lock:
        connection = store._connection
        connection.execute("BEGIN")
        try:
            connection.executemany(
                """
                INSERT INTO entries(session_id, id, parent_id, seq, type, ts, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows(),
            )
            connection.execute(
                "UPDATE sessions SET leaf_id = ? WHERE thread_id = ?",
                (f"a{active_entries - 1:07x}", session_id),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def run_ledger(config: RunConfig) -> dict[str, Any]:
    active_entries = 50 if config.quick else ACTIVE_LEDGER_ENTRIES
    abandoned_counts = (0, 100) if config.quick else ABANDONED_LEDGER_ENTRIES
    cases: list[dict[str, Any]] = []

    for abandoned_entries in abandoned_counts:
        with tempfile.TemporaryDirectory(prefix="orcha-ledger-benchmark-") as directory:
            root = Path(directory)
            database = root / "sessions.db"
            with SessionStore(database) as store:
                source = store.create(root, "fixture:model", thread_id="source")
                _populate_ledger(
                    store,
                    source.thread_id,
                    active_entries=active_entries,
                    abandoned_entries=abandoned_entries,
                )
                fixture_sizes = database_file_bytes(database)
                active_path = Ledger(store).path(source.thread_id)
                ledger = Ledger(store)

                path_samples: list[float] = []
                for _ in range(config.repetitions):
                    started = perf_counter_ns()
                    resolved = ledger.path(source.thread_id)
                    path_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                    if len(resolved) != active_entries:
                        raise AssertionError("ledger path did not resolve the active fixture chain")

                fork_samples: list[float] = []
                for repetition in range(config.repetitions):
                    target_id = f"fork-{repetition}"
                    store.create(root, "fixture:model", thread_id=target_id)
                    started = perf_counter_ns()
                    ledger.fork(source.thread_id, target_id)
                    fork_samples.append((perf_counter_ns() - started) / 1_000_000_000)

                context_samples: list[float] = []
                for _ in range(config.repetitions):
                    started = perf_counter_ns()
                    context = build_context(active_path)
                    context_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                    if len(context.messages) != active_entries:
                        raise AssertionError("build_context lost active fixture messages")

                cases.append(
                    {
                        "name": f"active_{active_entries}_abandoned_{abandoned_entries}",
                        "parameters": {
                            "active_entries": active_entries,
                            "abandoned_entries": abandoned_entries,
                            "fixture_population_timed": False,
                        },
                        "measurements": {
                            "path_wall": measurement(path_samples, "seconds"),
                            "fork_wall": measurement(fork_samples, "seconds"),
                            "build_context_wall": measurement(context_samples, "seconds"),
                            "fixture_database": measurement([fixture_sizes["database"]], "bytes"),
                            "fixture_wal": measurement([fixture_sizes["wal"]], "bytes"),
                            "fixture_shm": measurement([fixture_sizes["shm"]], "bytes"),
                            "fixture_total": measurement([sum(fixture_sizes.values())], "bytes"),
                        },
                    }
                )

    return result_document("ledger", cases)


def _capture_case(
    turns: int,
    state_bytes: int,
    repetitions: int,
) -> tuple[list[float], list[int], list[int], list[int]]:
    capture_samples: list[float] = []
    database_samples: list[int] = []
    wal_samples: list[int] = []
    final_database_samples: list[int] = []

    for repetition in range(repetitions):
        with tempfile.TemporaryDirectory(prefix="orcha-capture-benchmark-") as directory:
            root = Path(directory)
            database = root / "sessions.db"
            with SessionStore(database) as store:
                session = store.create(
                    root,
                    "fixture:model",
                    thread_id=f"capture-{repetition}",
                )
                if session.current_thread is None:
                    raise AssertionError("capture fixture requires a current graph thread")
                messages: list[HumanMessage] = []
                files = {"state.bin": "x" * state_bytes} if state_bytes else {}
                values: dict[str, Any] = {
                    "messages": messages,
                    "todos": [],
                    "files": files,
                }
                graph = _CaptureGraph(values)
                cfg: Any = SimpleNamespace()
                bus: Any = object()
                console: Any = _CaptureConsole()
                context = AppContext(
                    cfg=cfg,
                    registry=Registry(),
                    bus=bus,
                    session=store,
                    plugins=[],
                    plugin_states={},
                    console=console,
                    session_id=session.thread_id,
                    thread_id=session.current_thread,
                    agent=graph,
                )

                for turn in range(turns - 1):
                    messages.append(HumanMessage(content="turn", id=f"message-{turn}"))
                    context.capture_turn()

                messages.append(HumanMessage(content="turn", id=f"message-{turns - 1}"))
                started = perf_counter_ns()
                context.capture_turn()
                capture_samples.append((perf_counter_ns() - started) / 1_000_000_000)

                sizes = database_file_bytes(database)
                database_samples.append(sizes["database"])
                wal_samples.append(sizes["wal"])
            final_database_samples.append(database.stat().st_size)

    return capture_samples, database_samples, wal_samples, final_database_samples


def run_turn_capture(config: RunConfig) -> dict[str, Any]:
    turn_counts = (10, 20) if config.quick else TURN_COUNTS
    state_sizes = (0, 1024) if config.quick else TURN_STATE_BYTES
    cases: list[dict[str, Any]] = []

    for turns in turn_counts:
        for state_bytes in state_sizes:
            capture, database, wal, final_database = _capture_case(
                turns,
                state_bytes,
                config.repetitions,
            )
            state_name = "stable" if state_bytes == 0 else f"unchanged_{state_bytes}_bytes"
            cases.append(
                {
                    "name": f"{turns}_turns_{state_name}",
                    "parameters": {
                        "turns": turns,
                        "unchanged_state_bytes": state_bytes,
                        "fixture_creation_timed": False,
                        "setup_captures": turns - 1,
                        "timed_capture_depth": turns,
                    },
                    "measurements": {
                        "capture_turn_wall": measurement(capture, "seconds"),
                        "database_while_open": measurement(database, "bytes"),
                        "wal_while_open": measurement(wal, "bytes"),
                        "database_after_close": measurement(final_database, "bytes"),
                    },
                }
            )

    return result_document("turn_capture", cases)


def _populate_history(path: Path, rows: int) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO history(prompt, created_at, cwd, session_id)
            VALUES (?, ?, ?, ?)
            """,
            (
                (f"prompt {index:06d} deterministic fixture text", timestamp, "/tmp", "fixture")
                for index in range(rows)
            ),
        )


def run_history_load(config: RunConfig) -> dict[str, Any]:
    row_counts = (100, 1_000) if config.quick else LOAD_ROWS
    cases: list[dict[str, Any]] = []

    for rows in row_counts:
        with tempfile.TemporaryDirectory(prefix="orcha-history-benchmark-") as directory:
            root = Path(directory)
            database = root / "history.db"
            history = SQLiteHistory(
                database,
                cwd=root,
                session_id="fixture",
                legacy_path=root / "missing-legacy-history",
            )
            _populate_history(database, rows)
            fixture_sizes = database_file_bytes(database)

            load_samples: list[float] = []
            for _ in range(config.repetitions):
                started = perf_counter_ns()
                loaded = list(history.load_history_strings())
                load_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                if len(loaded) != rows:
                    raise AssertionError("history loader returned the wrong fixture size")
                del loaded
                gc.collect()

            overlay_samples: list[float] = []
            context = SimpleNamespace(ui=SimpleNamespace(history=history))
            for _ in range(config.repetitions):
                started = perf_counter_ns()
                overlay = HistoryOverlay(context)
                overlay_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                if len(overlay.items) != rows:
                    raise AssertionError("history overlay returned the wrong fixture size")
                del overlay
                gc.collect()

            cases.append(
                {
                    "name": f"{rows}_rows",
                    "parameters": {
                        "rows": rows,
                        "prompt_bytes": len("prompt 000000 deterministic fixture text"),
                        "fixture_population_timed": False,
                    },
                    "measurements": {
                        "load_history_strings_wall": measurement(load_samples, "seconds"),
                        "history_overlay_wall": measurement(overlay_samples, "seconds"),
                        "fixture_database": measurement([fixture_sizes["database"]], "bytes"),
                        "fixture_wal": measurement([fixture_sizes["wal"]], "bytes"),
                    },
                }
            )

    return result_document("history_load", cases)


def _populate_sessions(store: SessionStore, rows: int) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    with store.saver.lock:
        connection = store._connection
        connection.execute("BEGIN")
        try:
            connection.executemany(
                """
                INSERT INTO sessions(
                    thread_id, cwd, model, created, title, mode,
                    leaf_id, current_thread, parent_session
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                """,
                (
                    (
                        f"session-{index:06d}",
                        "/tmp",
                        "fixture:model",
                        timestamp,
                        f"Session {index:06d}",
                        "ask",
                        f"session-{index:06d}.0",
                    )
                    for index in range(rows)
                ),
            )
            connection.executemany(
                """
                INSERT INTO threads(
                    thread_id, session_id, seeded_from, captured, captured_message_ids
                ) VALUES (?, ?, NULL, 0, '[]')
                """,
                ((f"session-{index:06d}.0", f"session-{index:06d}") for index in range(rows)),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


async def _unused_resume(_: str) -> None:
    return None


def run_session_overlay_load(config: RunConfig) -> dict[str, Any]:
    row_counts = (100, 1_000) if config.quick else LOAD_ROWS
    cases: list[dict[str, Any]] = []

    for rows in row_counts:
        with tempfile.TemporaryDirectory(prefix="orcha-session-overlay-benchmark-") as directory:
            root = Path(directory)
            database = root / "sessions.db"
            with SessionStore(database) as store:
                _populate_sessions(store, rows)
                fixture_sizes = database_file_bytes(database)

                list_samples: list[float] = []
                for _ in range(config.repetitions):
                    started = perf_counter_ns()
                    sessions = store.list()
                    list_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                    if len(sessions) != rows:
                        raise AssertionError("session loader returned the wrong fixture size")
                    del sessions
                    gc.collect()

                overlay_samples: list[float] = []
                context = SimpleNamespace(
                    session=store,
                    ledger=Ledger(store),
                    resume=_unused_resume,
                )
                for _ in range(config.repetitions):
                    started = perf_counter_ns()
                    overlay = SessionOverlay(context)
                    overlay_samples.append((perf_counter_ns() - started) / 1_000_000_000)
                    if len(overlay.items) != rows:
                        raise AssertionError("session overlay returned the wrong fixture size")
                    del overlay
                    gc.collect()

                cases.append(
                    {
                        "name": f"{rows}_rows",
                        "parameters": {
                            "rows": rows,
                            "fixture_population_timed": False,
                            "overlay_scope": "constructor data load only",
                        },
                        "measurements": {
                            "session_store_list_wall": measurement(list_samples, "seconds"),
                            "session_overlay_wall": measurement(overlay_samples, "seconds"),
                            "fixture_database": measurement([fixture_sizes["database"]], "bytes"),
                            "fixture_wal": measurement([fixture_sizes["wal"]], "bytes"),
                            "fixture_shm": measurement([fixture_sizes["shm"]], "bytes"),
                            "fixture_total": measurement([sum(fixture_sizes.values())], "bytes"),
                        },
                    }
                )

    return result_document("session_overlay_load", cases)
