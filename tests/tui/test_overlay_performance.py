from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from orcha_agent.core.ledger import Ledger, MessageEntry
from orcha_agent.core.session import SessionStore
from orcha_agent.core.session_picker import session_page
from orcha_agent.tui.history import SQLiteHistory
from orcha_agent.tui.overlays.session import SessionOverlay


def test_history_window_is_bounded_and_search_reaches_older_prompts(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "history.db", load_limit=10)
    for i in range(50):
        history.append_string(f"prompt-{i}")
    assert len(history._loaded_strings) == 10
    assert list(history.load_history_strings()) == [f"prompt-{i}" for i in range(49, 39, -1)]
    assert history.search("prompt-0") == ["prompt-0"]
    with sqlite3.connect(history.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 50


def test_session_page_precomputes_labels_without_redraw_queries(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        for i in range(205):
            session = store.create(
                cwd=str(tmp_path), model="test:model", thread_id=f"session-{i:03}"
            )
            store.set_title(session.thread_id, "Needle title" if i == 0 else f"Session {i:03}")
        ledger = Ledger(store)
        statements: list[str] = []
        store._connection.set_trace_callback(statements.append)
        ctx = SimpleNamespace(session=store, ledger=ledger, resume=lambda _: None)
        overlay = SessionOverlay(ctx)
        assert len(overlay.items) == 200
        assert len(statements) == 1
        statements.clear()
        for _ in range(30):
            overlay.render_text()
        assert statements == []
        overlay.index = 199
        overlay._move(1)
        assert len(overlay.items) == 5
        overlay._move(-1)
        assert len(overlay.items) == 200
        assert overlay.index == 199
        overlay.filter.text = "Needle title"
        assert len(overlay.items) == 1
        assert "Needle title" in overlay.render_text()
        overlay.filter.text = ""
        assert len(overlay.items) == 200


def test_session_snapshot_first_prompt_and_count(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(cwd=str(tmp_path), model="test:model")
        ledger = Ledger(store)
        ledger.append(
            session.thread_id,
            MessageEntry(
                message={
                    "type": "human",
                    "data": {"content": "first prompt"},
                }
            ),
        )
        page = session_page(store)
        assert len(page) == 1
        assert page[0].count == 1
        assert page[0].first_message == {"type": "human", "data": {"content": "first prompt"}}


def test_selection_only_formats_requested_rows() -> None:
    from orcha_agent.tui.overlays.select import SelectList

    labels: list[int] = []

    def label(item: int) -> str:
        labels.append(item)
        return str(item)

    picker = SelectList("Large list", list(range(10000)), label=label)
    # Empty filters need no labels for matching.
    content = picker.list_control.create_content(80, 8)
    for row in range(8):
        content.get_line(row)
    assert content.line_count == 10000
    assert labels == list(range(8))


def test_session_filter_matches_unicode_first_prompt(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(cwd=str(tmp_path), model="test:model")
        ledger = Ledger(store)
        ledger.append(
            session.thread_id,
            MessageEntry(
                message={
                    "type": "human",
                    "data": {"content": "Änderung planen"},
                }
            ),
        )
        overlay = SessionOverlay(
            SimpleNamespace(session=store, ledger=ledger, resume=lambda _: None)
        )
        overlay.filter.text = "änderung"
        assert len(overlay.items) == 1
        assert "Änderung planen" in overlay.render_text()


def test_bounded_session_search_reaches_hidden_id_and_full_path(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        target_id = "older-session-unique-id"
        full_path = "/repo/long-component-hidden-from-label/packages/target"
        target = store.create(cwd=full_path, model="test:model", thread_id=target_id)
        store.set_title(target.thread_id, "Parser repair")
        for i in range(205):
            store.create(cwd=str(tmp_path), model="test:model", thread_id=f"recent-{i:03}")
        overlay = SessionOverlay(SimpleNamespace(session=store, ledger=Ledger(store)))
        assert len(overlay.items) == 200
        assert target_id not in {item.thread_id for item in overlay.items}
        for query in (target_id, "long-component-hidden-from-label", full_path):
            overlay.filter.text = query
            assert [item.thread_id for item in overlay.items] == [target_id]
            label = overlay.label(overlay.items[0])
            assert target_id not in label
            assert "long-component-hidden-from-label" not in label
            statements: list[str] = []
            store._connection.set_trace_callback(statements.append)
            overlay.render_text()
            overlay.render_text()
            assert statements == []
            store._connection.set_trace_callback(None)
