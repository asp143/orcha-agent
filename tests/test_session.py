import re
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
