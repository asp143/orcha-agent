import re
import threading
from collections.abc import Callable
from pathlib import Path

from orcha_agent.core.session import SessionStore


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


def test_metadata_and_plugin_writes_wait_for_saver_operation_lock(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "fake:model")
        writer_names = {"metadata-write", "plugin-state-write"}
        observed_writers: set[str] = set()
        waiting_writers: set[str] = set()
        completed_writers: set[str] = set()
        state_lock = threading.Lock()
        all_writers_observed = threading.Event()
        saver_holds_lock = threading.Event()
        release_saver = threading.Event()
        errors: list[BaseException] = []

        def observe_writer(name: str, *, waiting: bool = False) -> None:
            if name not in writer_names:
                return
            with state_lock:
                observed_writers.add(name)
                if waiting:
                    waiting_writers.add(name)
                else:
                    completed_writers.add(name)
                if observed_writers == writer_names:
                    all_writers_observed.set()

        backing_lock = threading.Lock()

        class ObservedLock:
            def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
                if backing_lock.locked():
                    observe_writer(threading.current_thread().name, waiting=True)
                return backing_lock.acquire(blocking, timeout)

            def release(self) -> None:
                backing_lock.release()

            def __enter__(self) -> "ObservedLock":
                self.acquire()
                return self

            def __exit__(self, *_: object) -> None:
                self.release()

        store.saver.lock = ObservedLock()
        original_setup = store.saver.setup

        def blocking_setup() -> None:
            saver_holds_lock.set()
            if not release_saver.wait(2):
                raise TimeoutError("test did not release the saver operation")
            original_setup()

        store.saver.setup = blocking_setup

        def run_saver_operation() -> None:
            try:
                store.saver.get_tuple(
                    {
                        "configurable": {
                            "thread_id": session.thread_id,
                            "checkpoint_ns": "",
                        }
                    }
                )
            except BaseException as error:
                errors.append(error)

        def run_writer(name: str, write: Callable[[], None]) -> None:
            try:
                write()
            except BaseException as error:
                errors.append(error)
            finally:
                observe_writer(name)

        saver_thread = threading.Thread(
            target=run_saver_operation,
            name="checkpoint-saver",
        )
        writer_threads = [
            threading.Thread(
                target=run_writer,
                args=("metadata-write", lambda: store.set_title(session.thread_id, "Locked")),
                name="metadata-write",
            ),
            threading.Thread(
                target=run_writer,
                args=(
                    "plugin-state-write",
                    lambda: store.set_plugin_state(
                        session.thread_id,
                        "planner",
                        {"step": "locked"},
                    ),
                ),
                name="plugin-state-write",
            ),
        ]

        saver_thread.start()
        try:
            assert saver_holds_lock.wait(1)
            for thread in writer_threads:
                thread.start()

            assert all_writers_observed.wait(1)
            with state_lock:
                assert waiting_writers == writer_names
                assert completed_writers == set()
        finally:
            release_saver.set()
            saver_thread.join(1)
            for thread in writer_threads:
                if thread.ident is not None:
                    thread.join(1)

        assert not saver_thread.is_alive()
        assert all(not thread.is_alive() for thread in writer_threads)
        assert errors == []
        assert store.get(session.thread_id).title == "Locked"
        assert store.get_plugin_state(session.thread_id, "planner") == {
            "step": "locked"
        }
