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
