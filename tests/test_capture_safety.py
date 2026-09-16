from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orcha_agent.core.capture import capture_graph_values
from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.core.session import SessionStore


@pytest.mark.parametrize("change", ["content", "tool", "duplicate", "reorder", "state"])
def test_capture_detects_checkpoint_changes(tmp_path: Path, change: str) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        assert session.current_thread is not None
        values: dict[str, Any] = {
            "messages": [HumanMessage(content="a", id="a"), AIMessage(content="b", id="b")]
        }

        def capture() -> bool:
            return capture_graph_values(
                store, session.thread_id, session.current_thread or "", values, only_if_new=True
            )

        capture()
        if change == "content":
            values["messages"][0].content = "changed"
        elif change == "tool":
            values["messages"][1].tool_calls = [
                {"id": "call", "name": "tool", "args": {}, "type": "tool_call"}
            ]
            values["messages"].append(ToolMessage(content="done", tool_call_id="call", id="tool"))
        elif change == "duplicate":
            values["messages"].append(HumanMessage(content="duplicate", id="a"))
        elif change == "reorder":
            values["messages"].reverse()
        else:
            values["files"] = {"file": "state-only"}
        assert capture()
        context = build_context(Ledger(store).path(session.thread_id))
        assert context.messages == values["messages"]
        assert context.files == values.get("files", {})
        assert not capture()


def test_capture_retries_after_partial_transaction(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        values = {"messages": [HumanMessage(content="a", id="a")]}
        store._connection.execute(
            "CREATE TRIGGER fail_capture BEFORE UPDATE ON threads BEGIN SELECT RAISE(FAIL, 'injected'); END"
        )
        with pytest.raises(RuntimeError, match="injected"):
            capture_graph_values(
                store, session.thread_id, session.current_thread or "", values, only_if_new=True
            )
        assert Ledger(store).path(session.thread_id) == []
        store._connection.execute("DROP TRIGGER fail_capture")
        assert capture_graph_values(
            store, session.thread_id, session.current_thread or "", values, only_if_new=True
        )
        assert len(build_context(Ledger(store).path(session.thread_id)).messages) == 1


def test_capture_writes_only_new_cursor_and_changed_state(tmp_path: Path) -> None:
    from orcha_agent.core.ledger import CustomEntry

    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        values: dict[str, Any] = {"messages": [], "files": {"large": "x" * 100_000}}
        for index in range(30):
            values["messages"].append(HumanMessage(content="next", id=str(index)))
            before = store._connection.total_changes
            capture_graph_values(
                store, session.thread_id, session.current_thread or "", values, only_if_new=False
            )
            if index > 0:
                # One message + leaf + count watermark + one digest row.
                assert store._connection.total_changes - before == 4
        path = Ledger(store).path(session.thread_id)
        assert (
            sum(
                isinstance(entry, CustomEntry) and entry.custom_type == "turn_state"
                for entry in path
            )
            == 1
        )
        assert (
            store._connection.execute("SELECT captured_message_ids FROM threads").fetchone()[0]
            == "[]"
        )
        assert store.get_thread(session.current_thread or "").captured_message_ids == tuple(
            str(i) for i in range(30)
        )


def test_capture_digest_survives_reopen_and_seeded_branch(tmp_path: Path) -> None:
    database = tmp_path / "capture.db"
    original = HumanMessage(content=[{"type": "text", "text": "original"}], id="same")
    with SessionStore(database) as store:
        session = store.create(tmp_path, "fake:model")
        session_id = session.thread_id
        capture_graph_values(
            store,
            session_id,
            session.current_thread or "",
            {"messages": [original]},
            only_if_new=False,
        )
    with SessionStore(database) as store:
        thread_id = store.get(session_id).current_thread or ""
        changed = HumanMessage(content="changed", id="same")
        assert capture_graph_values(
            store, session_id, thread_id, {"messages": [changed]}, only_if_new=True
        )
        assert build_context(Ledger(store).path(session_id)).messages == [changed]
        target = store.create(tmp_path, "fake:model")
        Ledger(store).fork(session_id, target.thread_id)
        branch_thread = store.activate_thread(
            target.thread_id,
            "branch.0",
            captured=1,
            captured_message_ids=("same",),
            captured_messages=[changed],
        )
        changed.content = "changed again after seed"
        assert capture_graph_values(
            store,
            target.thread_id,
            branch_thread.thread_id,
            {"messages": [changed]},
            only_if_new=True,
        )
        assert build_context(Ledger(store).path(target.thread_id)).messages == [changed]
        assert build_context(Ledger(store).path(session_id)).messages[0].content == "changed"


def test_capture_nested_mutation_invalidates_cached_digest(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        content: Any = [{"type": "text", "text": "original"}]
        message = HumanMessage(content=content, id="same")
        values = {"messages": [message], "files": {"file": {"content": ["first"]}}}
        capture_graph_values(
            store, session.thread_id, session.current_thread or "", values, only_if_new=False
        )
        content[0]["text"] = "nested mutation"
        values["files"]["file"]["content"].append("second")
        assert capture_graph_values(
            store, session.thread_id, session.current_thread or "", values, only_if_new=True
        )
        context = build_context(Ledger(store).path(session.thread_id))
        assert context.messages == [message]
        assert context.files == values["files"]
