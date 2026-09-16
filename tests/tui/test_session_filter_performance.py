from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.core.ledger import Ledger, MessageEntry
from orcha_agent.core.session import SessionStore
from orcha_agent.core.session_picker import session_page
from orcha_agent.tui.overlays.session import SessionOverlay


@pytest.mark.asyncio
async def test_filter_yields_and_discards_stale_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        first = store.create(tmp_path, "test:model")
        store.set_title(first.thread_id, "first")
        second = store.create(tmp_path, "test:model")
        store.set_title(second.thread_id, "second")
        overlay = SessionOverlay(SimpleNamespace(session=store, ledger=Ledger(store)))
        original = SessionOverlay._read_page
        entered = Event()
        release = Event()

        def slow(*args: Any) -> Any:
            if args[3] == "first":
                entered.set()
                assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(SessionOverlay, "_read_page", staticmethod(slow))
        overlay.filter.text = "first"
        old_task = overlay._search_task
        try:
            # A synchronous scan would block before this heartbeat can execute.
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
            heartbeat = False

            async def tick() -> None:
                nonlocal heartbeat
                await asyncio.sleep(0)
                heartbeat = True

            await asyncio.wait_for(tick(), 1)
            assert heartbeat
            assert overlay.items == ()
            overlay.filter.text = "second"
            assert overlay._search_task is not None
            await asyncio.wait_for(overlay._search_task, 2)
            assert [item.thread_id for item in overlay.items] == [second.thread_id]
        finally:
            release.set()
            if old_task is not None:
                await asyncio.wait_for(old_task, 2)
        assert [item.thread_id for item in overlay.items] == [second.thread_id]
        overlay.cancel()


@pytest.mark.asyncio
async def test_closed_filter_stops_before_querying_closed_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create(tmp_path, "test:model")
    overlay = SessionOverlay(SimpleNamespace(session=store, ledger=Ledger(store)))
    original = SessionOverlay._read_page
    entered = Event()
    release = Event()

    def slow(*args: Any) -> Any:
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(SessionOverlay, "_read_page", staticmethod(slow))
    overlay.filter.text = "missing"
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
        overlay.cancel()
        store.close()
    finally:
        release.set()
        if overlay._search_task is not None:
            await asyncio.wait_for(overlay._search_task, 2)
    assert overlay.done
    assert overlay.items == ()
    assert overlay._error is None


def test_first_prompt_skips_empty_and_image_only_human_messages(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "test:model")
        ledger = Ledger(store)
        for content in (
            " \n\t",
            [{"type": "image_url", "image_url": "https://example.org/a"}],
            [{"type": "text", "text": "meaningful prompt"}],
        ):
            ledger.append(
                session.thread_id,
                MessageEntry(
                    message={
                        "type": "human",
                        "data": {"content": content},
                    }
                ),
            )
        rows = session_page(store)
        assert rows[0].first_message is not None
        assert rows[0].first_message["data"]["content"] == [
            {"type": "text", "text": "meaningful prompt"},
        ]
