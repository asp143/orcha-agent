from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from orcha_agent.builtin.statusbar import register
from orcha_agent.core.events import EventBus, SessionSwitch, ThreadSwitch, TurnEnd
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.statusline import reset_session_state, session_segment


def test_session_paints_never_wait_for_database_and_drop_stale_refresh() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls: list[tuple[str, int]] = []
    state: dict = {}

    def get(session_id: str) -> SimpleNamespace:
        calls.append((session_id, threading.get_ident()))
        if session_id == "old":
            started.set()
            assert release.wait(2)
        return SimpleNamespace(title=session_id)

    ctx = SimpleNamespace(
        session_id="old",
        session=SimpleNamespace(get=get),
        plugin_states={"statusbar": state},
        ui=SimpleNamespace(invalidate=finished.set),
    )
    assert session_segment(ctx) is None
    assert started.wait(2)
    for _ in range(120):
        assert session_segment(ctx) is None
    assert len(calls) == 1
    assert calls[0][1] != threading.get_ident()
    reset_session_state(state)
    ctx.session_id = "new"
    session_segment(ctx)
    assert finished.wait(2)
    release.set()
    assert session_segment(ctx).text == "new"
    for _ in range(120):
        assert session_segment(ctx).text == "new"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_session_snapshot_invalidates_on_relevant_events() -> None:
    state: dict = {"_session_title": "persisted stale title", "_session_ready": True}
    bus = EventBus()
    register(
        PluginAPI(
            name="statusbar",
            config={},
            state=state,
            registry=Registry(),
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    assert "_session_title" not in state
    for event in (
        SessionSwitch(old="a", new="b"),
        ThreadSwitch(old="a.0", new="a.1", session_id="a", reason="branch"),
        TurnEnd(thread_id="a.1"),
    ):
        state["_session_title"] = "stale"
        state["_session_ready"] = True
        await bus.emit(event)
        assert "_session_title" not in state
        assert not state.get("_session_ready")
