from pathlib import Path

from langchain_core.messages import HumanMessage

from orcha_agent.core.capture import capture_graph_values
from orcha_agent.core.export import export_session
from orcha_agent.core.ledger import (
    CompactionEntry,
    CustomEntry,
    Ledger,
    MessageEntry,
    build_context,
)
from orcha_agent.core.session import SessionStore


def test_same_id_edit_has_constant_write_cost(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        messages = [HumanMessage(content=str(i), id=str(i)) for i in range(2000)]

        def capture() -> bool:
            return capture_graph_values(
                store,
                session.thread_id,
                session.current_thread or "",
                {"messages": messages},
                only_if_new=False,
            )

        capture()
        before = store._connection.total_changes
        messages[1000].content = "edited"
        assert capture()
        assert store._connection.total_changes - before <= 5
        path = Ledger(store).path(session.thread_id)
        assert len(path) == 2002
        assert build_context(path).messages == messages
        assert not capture()


def test_compaction_reuses_retained_messages(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        messages = [HumanMessage(content=str(i), id=str(i)) for i in range(20)]

        def capture(current: list[HumanMessage]) -> None:
            capture_graph_values(
                store,
                session.thread_id,
                session.current_thread or "",
                {"messages": current},
                only_if_new=True,
            )

        capture(messages)
        summary = HumanMessage(
            content="summary", id="summary", additional_kwargs={"lc_source": "summarization"}
        )
        capture([summary, *messages[-6:]])
        path = Ledger(store).path(session.thread_id)
        assert next(e for e in path if isinstance(e, CompactionEntry)).first_kept_id is not None
        assert len([e for e in path if isinstance(e, MessageEntry)]) == 20
        assert build_context(path).messages[1:] == messages[-6:]
        output = export_session(store, session.thread_id, tmp_path / "export.jsonl")
        assert output.read_text().count('"type":"message"') == 20


def test_reset_keeps_unchanged_live_state_without_duplicate_write(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        messages = [HumanMessage(content=str(i), id=str(i)) for i in range(2)]
        values = {"messages": messages, "files": {"file": "content"}}
        for _ in range(2):
            capture_graph_values(
                store, session.thread_id, session.current_thread or "", values, only_if_new=False
            )
            messages.reverse()
        path = Ledger(store).path(session.thread_id)
        assert sum(isinstance(e, CustomEntry) and e.custom_type == "turn_state" for e in path) == 1
        assert build_context(path).files == values["files"]


def test_untagged_summary_reset_is_marked(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        for messages in (
            [HumanMessage(content="old", id="old")],
            [
                HumanMessage(
                    content="Here is a summary of the conversation to date:\n\nsummary", id="new"
                )
            ],
        ):
            capture_graph_values(
                store,
                session.thread_id,
                session.current_thread or "",
                {"messages": messages},
                only_if_new=True,
            )
        assert any(
            isinstance(e, CustomEntry) and e.custom_type == "unrecognized_summary"
            for e in Ledger(store).path(session.thread_id)
        )


def test_compaction_full_retention_survives_reopen_and_fork(tmp_path: Path) -> None:
    database = tmp_path / "capture.db"
    messages = [HumanMessage(content=str(i), id=str(i)) for i in range(3)]
    with SessionStore(database) as store:
        session = store.create(tmp_path, "fake:model")
        session_id = session.thread_id
        for current in (
            messages,
            [
                HumanMessage(
                    content="summary",
                    id="summary",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                *messages,
            ],
        ):
            capture_graph_values(
                store,
                session_id,
                session.current_thread or "",
                {"messages": current},
                only_if_new=True,
            )
        assert (
            export_session(store, session_id, tmp_path / "export.jsonl")
            .read_text()
            .count('"type":"message"')
            == 3
        )
    with SessionStore(database) as store:
        target = store.create(tmp_path, "fake:model")
        Ledger(store).fork(session_id, target.thread_id)
        for selected in (session_id, target.thread_id):
            path = Ledger(store).path(selected)
            assert len([entry for entry in path if isinstance(entry, MessageEntry)]) == 3
            assert build_context(path).messages[1:] == messages


def test_compaction_retains_prior_replacement_and_updates_changed_tail(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        messages = [HumanMessage(content=str(i), id=str(i)) for i in range(20)]

        def capture(current: list[HumanMessage]) -> None:
            capture_graph_values(
                store,
                session.thread_id,
                session.current_thread or "",
                {"messages": current},
                only_if_new=True,
            )

        capture(messages)
        messages[-1] = HumanMessage(content="first edit", id="19")
        capture(messages)
        messages[-2] = HumanMessage(content="second edit", id="18")
        capture(
            [
                HumanMessage(
                    content="summary",
                    id="summary",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                *messages[-6:],
            ]
        )
        path = Ledger(store).path(session.thread_id)
        assert len([entry for entry in path if isinstance(entry, MessageEntry)]) == 20
        assert build_context(path).messages[1:] == messages[-6:]


def test_same_id_summary_edit_updates_compaction(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "capture.db") as store:
        session = store.create(tmp_path, "fake:model")
        messages = [HumanMessage(content=str(i), id=str(i)) for i in range(3)]
        summary = HumanMessage(
            content="first summary", id="summary", additional_kwargs={"lc_source": "summarization"}
        )
        for current in (messages, [summary, messages[-1]]):
            capture_graph_values(
                store,
                session.thread_id,
                session.current_thread or "",
                {"messages": current},
                only_if_new=True,
            )
        summary.content = "changed summary"
        capture_graph_values(
            store,
            session.thread_id,
            session.current_thread or "",
            {"messages": [summary, messages[-1]]},
            only_if_new=True,
        )
        path = Ledger(store).path(session.thread_id)
        assert build_context(path).messages[0].content == "[Conversation summary]\nchanged summary"
        assert len([entry for entry in path if isinstance(entry, MessageEntry)]) == 3
