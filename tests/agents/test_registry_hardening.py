from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from orcha_agent.core.agents import AgentRegistry
from orcha_agent.core.config import AgentsConfig, Config
from orcha_agent.core.events import (
    AgentDelivered,
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    EventBus,
)
from orcha_agent.core.ledger import CustomEntry, Ledger
from orcha_agent.core.plugin import ModeSpec
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


class _Graph:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.started = asyncio.Event()
        self.model = FakeListChatModel(responses=["ok"] * 16)

    async def astream(self, value: Any, **_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        text = value["messages"][0]["content"]
        response = await self.model.ainvoke([HumanMessage(content=text)])
        yield "messages", (response, {"langgraph_node": "agent"})

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": [], "todos": [], "files": {}})


def _config(tmp_path: Path, **overrides: Any) -> Config:
    return Config(
        model="fake:main",
        subagent_model="fake:legacy",
        summarizer_model="fake:summary",
        mode="yolo",
        backend="test",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
        model_roles={
            "task": "fake:task",
            "scout": "fake:scout",
            "reviewer": "fake:reviewer",
            "advisor": "fake:advisor",
        },
        agents=replace(AgentsConfig(), **overrides),
    )


def _registry() -> Registry:
    registry = Registry()
    registry.modes["yolo"] = ModeSpec(description="all", interrupt_on={}, allowed_tools=None)
    registry.modes["ask"] = ModeSpec(description="ask", interrupt_on={}, allowed_tools=frozenset())
    return registry


def _job(run_id: str, child: Any, *, status: str) -> CustomEntry:
    return CustomEntry(
        custom_type="agent_job",
        data={
            "run_id": run_id,
            "name": "Worker",
            "agent_type": "task",
            "parent_id": "main",
            "session_id": child.thread_id,
            "thread_id": child.current_thread,
            "depth": 0,
            "status": status,
            "visible": True,
            "delivered": False,
        },
    )


@pytest.mark.asyncio
async def test_terminal_status_wakes_wait_and_wait_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        child_info = store.get(run.session_id)
        assert child_info is not None
        assert child_info.kind == "agent"
        await run.wait_status("idle")
        first = asyncio.create_task(agents.wait([run.id], timeout_s=1))
        all_done = asyncio.create_task(agents.wait_all([run.id], timeout_s=1))
        await asyncio.sleep(0)

        await run.complete({"ok": True})

        assert await asyncio.wait_for(first, 0.2) == [run]
        assert await asyncio.wait_for(all_done, 0.2) == [run]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_inactive_view_keeps_events_children_messages_jobs_and_delivery_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_gate = asyncio.Event()
    graphs = iter((_Graph(old_gate), _Graph()))

    async def build(*_args: Any, **_kwargs: Any) -> _Graph:
        return next(graphs)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    bus = EventBus()
    events: list[Any] = []

    async def observe(event: Any) -> None:
        events.append(event)

    for event_type in (AgentSpawned, AgentStatus, AgentFinished, AgentDelivered):
        bus.on(event_type, observe, plugin="test")

    with SessionStore(tmp_path / "sessions.db") as store:
        first = store.create(tmp_path, "fake:main", thread_id="first")
        second = store.create(tmp_path, "fake:main", thread_id="second")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, bus, first.thread_id)
        old = await agents.spawn("task", "old", parent="main")
        await old.agent_ready.wait()
        await old.agent.started.wait()
        agents.retarget(second.thread_id)
        events.clear()

        child = await agents.spawn("task", "child", parent=old.id)
        await child.wait_status("idle")
        await child.complete({"view": "old"})
        assert await agents.wait_all([child.id], caller=old.id, timeout_s=0.2) == [child]
        assert await agents.wait([child.id], caller=old.id, timeout_s=0.2) == [child]
        assert [run.id for run in agents.list(caller=old.id)] == [old.id, child.id]
        assert agents.list() == []
        assert agents.resolve(child.id, caller=old.id) == child.id
        with pytest.raises(LookupError):
            agents.resolve(child.id)
        assert [run.id for run in agents.jobs(old.id)] == [child.id]
        assert await agents.deliver(old.id) == [child]

        assert await agents.post_message(old.id, "main", "old view") is False
        assert agents.drain_messages("main") == []
        assert events == []

        old_gate.set()
        await old.wait_status("idle")
        assert events == []
        agents.retarget(first.thread_id)
        assert [item["message"] for item in agents.drain_messages("main")] == ["old view"]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_post_message_reports_synchronous_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await run.wait_status("idle")
        waiting = asyncio.create_task(agents.wait_activity(run.id, timeout_s=1, peer="main"))
        await asyncio.sleep(0)

        assert await agents.post_message("main", run.id, "first") is True
        assert await asyncio.wait_for(waiting, 0.2) is True
        assert agents.drain_messages(run.id)[0]["message"] == "first"
        assert await agents.post_message("main", run.id, "second") is False
        assert agents.drain_messages(run.id)[0]["message"] == "second"
        reservation = agents.reserve_activity_waiter(run.id)
        try:
            assert await agents.post_message("main", run.id, "reserved") is True
            assert await agents.wait_activity(
                run.id,
                timeout_s=1,
                peer="main",
                reserved=True,
            )
            assert agents.drain_messages(run.id)[0]["message"] == "reserved"
        finally:
            agents.release_activity_waiter(reservation)
        assert await agents.post_message("main", run.id, "after") is False
        await agents.shutdown()


@pytest.mark.asyncio
async def test_hidden_run_requires_internal_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        advisor = await agents.spawn(
            "advisor", "advise", name="Advisor", parent="main", visible=False
        )
        await advisor.wait_status("idle")

        assert agents.list() == []
        with pytest.raises(LookupError):
            agents.resolve(advisor.id)
        with pytest.raises(LookupError):
            agents.resolve("Advisor")
        assert agents.resolve(advisor.id, visible_only=False) == advisor.id
        assert agents.get(advisor.id) is advisor
        assert agents.advisor_run(parent.thread_id) is advisor
        await agents.shutdown()


@pytest.mark.asyncio
async def test_cancel_aborts_every_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()

    async def build(*_args: Any, **_kwargs: Any) -> _Graph:
        return _Graph(gate)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent_session = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(
            _registry(), _config(tmp_path, max_depth=3), store, EventBus(), parent_session.thread_id
        )
        parent = await agents.spawn("task", "parent", parent="main")
        child = await agents.spawn("task", "child", parent=parent.id)
        grandchild = await agents.spawn("task", "grandchild", parent=child.id)
        await asyncio.gather(
            parent.agent_ready.wait(),
            child.agent_ready.wait(),
            grandchild.agent_ready.wait(),
        )

        await agents.cancel(parent.id)
        await asyncio.gather(
            parent.wait_status("aborted"),
            child.wait_status("aborted"),
            grandchild.wait_status("aborted"),
        )

        assert {run.abort_reason for run in (parent, child, grandchild)} == {"cancel"}
        await agents.shutdown()


@pytest.mark.asyncio
async def test_direct_parent_abort_cascades_budget_reason_to_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()

    async def build(*_args: Any, **_kwargs: Any) -> _Graph:
        return _Graph(gate)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        root = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(
            _registry(), _config(tmp_path, max_depth=3), store, EventBus(), root.thread_id
        )
        parent = await agents.spawn("task", "parent", parent="main")
        child = await agents.spawn("task", "child", parent=parent.id)
        grandchild = await agents.spawn("task", "grandchild", parent=child.id)
        await asyncio.gather(
            parent.agent_ready.wait(),
            child.agent_ready.wait(),
            grandchild.agent_ready.wait(),
        )

        await parent.request_abort("budget")
        await asyncio.gather(
            parent.wait_status("aborted"),
            child.wait_status("aborted"),
            grandchild.wait_status("aborted"),
        )

        assert {run.abort_reason for run in (parent, child, grandchild)} == {"budget"}
        await agents.shutdown()


@pytest.mark.asyncio
async def test_fork_rejects_live_foreign_child_but_restores_terminal_snapshot(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        source = store.create(tmp_path, "fake:main", thread_id="source")
        child = store.create(
            tmp_path,
            "fake:task",
            thread_id="source-child",
            parent_session=source.thread_id,
        )
        ledger = Ledger(store)
        ledger.append(source.thread_id, _job("live-run", child, status="running"))
        live_fork = store.create(tmp_path, "fake:main", thread_id="live-fork")
        ledger.fork(source.thread_id, live_fork.thread_id)

        live = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), live_fork.thread_id)
        assert live.list() == []

        ledger.append(source.thread_id, _job("done-run", child, status="done"))
        terminal_fork = store.create(tmp_path, "fake:main", thread_id="terminal-fork")
        ledger.fork(source.thread_id, terminal_fork.thread_id)
        terminal = AgentRegistry(
            _registry(),
            _config(tmp_path),
            store,
            EventBus(),
            terminal_fork.thread_id,
        )
        restored = terminal.get("done-run")
        assert restored is not None
        assert restored.status == "done"
        assert restored.session_id == child.thread_id
        await live.shutdown()
        await terminal.shutdown()


@pytest.mark.asyncio
async def test_agent_inboxes_and_mailboxes_drop_oldest_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    graph = _Graph(gate)
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "hold", parent="main")
        await graph.started.wait()

        for index in range(130):
            await agents.send(run.id, f"inbox {index}")
        assert run.inbox.maxsize == 128
        assert run.inbox.qsize() == 128
        queued = list(run.inbox._queue)
        assert any(isinstance(item, str) and "[dropped oldest" in item for item in queued)
        assert "inbox 129" in queued

        for index in range(130):
            await agents.post_message("main", run.id, f"mailbox {index}")
        messages = agents.drain_messages(run.id)
        assert len(messages) == 128
        assert any(
            item.get("warning") is True and "dropped oldest" in item["message"] for item in messages
        )
        assert any(item["message"] == "mailbox 129" for item in messages)

        gate.set()
        await agents.shutdown()


@pytest.mark.asyncio
async def test_active_run_detaches_when_parent_branches_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    graph = _Graph(gate)
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        ledger = Ledger(store)
        root = ledger.append(
            parent.thread_id,
            CustomEntry(custom_type="root", data={"active": True}),
        )
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await graph.started.wait()
        ledger.branch(parent.thread_id, root.id)
        agents.retarget(parent.thread_id)
        assert agents.list() == []
        gate.set()

        await run.wait_status("aborted")

        assert run.abort_reason == "cancel"
        assert run.id not in ledger.latest_custom(parent.thread_id, "agent_job", key="run_id")
        assert await agents.deliver("main", [run.id]) == []
        await agents.shutdown()


@pytest.mark.asyncio
async def test_detach_abort_task_is_retained_and_awaited_by_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_gate = asyncio.Event()
    graph = _Graph(graph_gate)
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        ledger = Ledger(store)
        root = ledger.append(
            parent.thread_id,
            CustomEntry(custom_type="root", data={"active": True}),
        )
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await graph.started.wait()
        original_abort = run.request_abort
        abort_started = asyncio.Event()
        release_abort = asyncio.Event()

        async def blocked_abort(reason: Any) -> None:
            abort_started.set()
            await release_abort.wait()
            await original_abort(reason)

        run.request_abort = blocked_abort
        ledger.branch(parent.thread_id, root.id)
        agents.retarget(parent.thread_id)
        await asyncio.wait_for(abort_started.wait(), 0.2)

        assert len(agents._detach_tasks) == 1
        shutdown = asyncio.create_task(agents.shutdown())
        await asyncio.sleep(0)
        assert not shutdown.done()
        release_abort.set()
        graph_gate.set()
        await asyncio.wait_for(shutdown, 0.5)
        assert agents._detach_tasks == set()


@pytest.mark.asyncio
async def test_direct_run_task_cancellation_settles_one_aborted_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    graph = _Graph(gate)

    async def build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(_registry(), _config(tmp_path), store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await graph.started.wait()
        assert run.task is not None

        run.task.cancel()
        await asyncio.wait_for(run.task, 0.2)

        exits = [
            entry
            for entry in Ledger(store).path(run.session_id)
            if isinstance(entry, CustomEntry) and entry.custom_type == "session_exit"
        ]
        assert run.status == "aborted"
        assert run.abort_reason == "shutdown"
        assert [entry.data for entry in exits] == [{"kind": "aborted", "reason": "shutdown"}]
        await agents.shutdown()


def test_hydration_restores_child_mode_cwd_and_recomputes_trust(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    child_cwd = trusted / "project"
    child_cwd.mkdir(parents=True)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path / "elsewhere", "fake:main", thread_id="main")
        child = store.create(
            child_cwd,
            "fake:task",
            "ask",
            thread_id="child",
            parent_session=parent.thread_id,
        )
        Ledger(store).append(parent.thread_id, _job("restored", child, status="idle"))
        cfg = replace(
            _config(tmp_path),
            cwd=tmp_path / "elsewhere",
            mode="yolo",
            trust_cwd=False,
            trusted_dirs=(trusted.resolve(),),
        )

        agents = AgentRegistry(_registry(), cfg, store, EventBus(), parent.thread_id)
        restored = agents.get("restored")

        assert restored is not None
        assert restored.cfg.cwd == child_cwd
        assert restored.cwd == child_cwd
        assert restored.cfg.mode == "yolo"
        assert restored.cfg.trust_cwd is True


@pytest.mark.asyncio
async def test_mode_refresh_discards_graph_built_for_stale_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _Graph()
    second = _Graph()
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    built_modes: list[str] = []

    async def build(_registry: Registry, cfg: Config, *_args: Any, **_kwargs: Any) -> _Graph:
        built_modes.append(cfg.mode)
        if len(built_modes) == 1:
            build_started.set()
            await release_build.wait()
            return first
        return second

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        yolo = _config(tmp_path)
        agents = AgentRegistry(_registry(), yolo, store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await build_started.wait()

        agents.retarget(parent.thread_id, replace(yolo, mode="ask"))
        release_build.set()

        await asyncio.wait_for(second.started.wait(), 0.2)
        await run.wait_status("idle")
        assert built_modes == ["yolo", "ask"]
        assert not first.started.is_set()
        assert run.agent is second
        await agents.shutdown()


@pytest.mark.asyncio
async def test_mode_refresh_defers_graph_rebuild_until_active_turn_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    first = _Graph(gate)
    second = _Graph()
    graphs = deque((first, second))
    built_modes: list[str] = []

    async def build(_registry: Registry, cfg: Config, *_args: Any, **_kwargs: Any) -> _Graph:
        built_modes.append(cfg.mode)
        return graphs.popleft()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        yolo = _config(tmp_path)
        agents = AgentRegistry(_registry(), yolo, store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await first.started.wait()

        agents.retarget(parent.thread_id, replace(yolo, mode="ask"))
        assert run.cfg.mode == "ask"
        assert run.agent is first

        gate.set()
        await run.wait_status("idle")
        assert run.agent is None
        await agents.send(run.id, "again")
        await second.started.wait()
        await run.wait_status("idle")
        assert built_modes == ["yolo", "ask"]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_retarget_and_revive_rebuild_child_with_current_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_modes: list[str] = []
    built_interrupts: list[dict[str, Any]] = []
    rebuilt = asyncio.Event()

    async def build(registry: Registry, cfg: Config, *_args: Any, **_kwargs: Any) -> _Graph:
        built_modes.append(cfg.mode)
        built_interrupts.append(dict(registry.modes[cfg.mode].interrupt_on))
        if cfg.mode == "ask":
            rebuilt.set()
        return _Graph()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        yolo = _config(tmp_path)
        registry = _registry()
        registry.modes["ask"] = ModeSpec(
            description="ask",
            interrupt_on={"shell": True},
            allowed_tools=frozenset(),
        )
        agents = AgentRegistry(registry, yolo, store, EventBus(), parent.thread_id)
        run = await agents.spawn("task", "work", parent="main")
        await run.wait_status("idle")
        assert built_modes == ["yolo"]
        await run._set_status("parked")

        agents.retarget(parent.thread_id, replace(yolo, mode="ask"))

        assert run.cfg.mode == "ask"
        assert run.agent is None
        await agents.revive(run.session_id)
        await agents.send(run.id, "again")
        await asyncio.wait_for(rebuilt.wait(), 0.2)
        await run.wait_status("idle")
        assert built_modes == ["yolo", "ask"]
        assert built_interrupts[-1] == {"shell": True}
        await agents.shutdown()
