from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from orcha_agent.core.agents import AgentRegistry
from orcha_agent.core.config import AgentsConfig, Config
from orcha_agent.core.events import (
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    EventBus,
)
from orcha_agent.core.ledger import CompactionEntry, CustomEntry, Ledger, MessageEntry
from orcha_agent.core.plugin import ModeSpec
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


class _Graph:
    def __init__(
        self,
        *responses: str,
        gate: asyncio.Event | None = None,
        active: list[int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.model = FakeListChatModel(responses=list(responses) or ["ok"] * 32)
        self.gate = gate
        self.active = active
        self.error = error
        self.started = asyncio.Event()
        self.inputs: list[str] = []
        self.messages: list[Any] = []

    async def astream(self, value: Any, **_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        text = value["messages"][0]["content"]
        self.inputs.append(text)
        self.messages.append(HumanMessage(content=text))
        if self.active is not None:
            self.active[0] += 1
            self.active[1] = max(self.active[1], self.active[0])
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.error is not None:
                raise self.error
            response = await self.model.ainvoke([HumanMessage(content=text)])
            self.messages.append(response)
            yield "messages", (response, {"langgraph_node": "agent"})
        finally:
            if self.active is not None:
                self.active[0] -= 1

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": self.messages, "todos": [], "files": {}})


async def _eventually(predicate: Any, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


def _config(tmp_path: Path, **agent_overrides: Any) -> Config:
    agents = replace(AgentsConfig(), **agent_overrides)
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
        agents=agents,
    )


def _plugin_registry() -> Registry:
    registry = Registry()
    registry.modes["yolo"] = ModeSpec(
        description="all", interrupt_on={}, allowed_tools=None
    )
    return registry


@pytest.mark.asyncio
async def test_registry_spawn_send_complete_and_tree_use_independent_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph("first", "second")
    builds: list[tuple[Config, dict[str, Any]]] = []

    async def fake_build(_registry: Registry, cfg: Config, _store: SessionStore, _bus: Any, **kwargs: Any) -> _Graph:
        builds.append((cfg, kwargs))
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)
    bus = EventBus()
    observed: list[object] = []

    async def record(event: object) -> None:
        observed.append(event)

    for event_type in (AgentSpawned, AgentStatus, AgentFinished):
        bus.on(event_type, record, plugin="test")

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(_plugin_registry(), _config(tmp_path), store, bus, parent.thread_id)

        run = await agents.spawn("scout", "inspect files", name="FileScout", parent="main")
        await run.wait_status("idle")

        assert graph.inputs == ["inspect files"]
        assert builds[0][0].model == "fake:scout"
        assert builds[0][1]["exclude_general_purpose"] is True
        assert builds[0][1]["tool_scope"] == {"ls", "read_file", "glob", "grep"}
        assert store.get(run.session_id).parent_session == parent.thread_id
        assert run.thread_config == {"configurable": {"thread_id": run.thread_id}}

        await agents.send(run.id, "inspect tests")
        await _eventually(lambda: graph.inputs == ["inspect files", "inspect tests"])
        await run.wait_status("idle")
        await run.complete({"summary": "done"})
        await run.wait_status("done")

        assert agents.get(run.id) is run
        assert agents.list("done") == [run]
        assert agents.tree() == [run]
        assert await agents.wait([run.id], timeout_s=0.1) == [run]
        assert any(isinstance(event, AgentSpawned) for event in observed)
        assert any(isinstance(event, AgentFinished) for event in observed)
        result_entry = next(
            entry
            for entry in Ledger(store).all(run.session_id)
            if isinstance(entry, CustomEntry) and entry.custom_type == "agent_result"
        )
        assert result_entry.data["result"] == {"summary": "done"}

        await agents.shutdown()


@pytest.mark.asyncio
async def test_registry_semaphore_gates_running_turns_not_idle_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    active = [0, 0]
    graphs = [_Graph("one", gate=gate, active=active), _Graph("two", gate=gate, active=active)]

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.pop(0)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_concurrency=1),
            store,
            EventBus(),
            parent.thread_id,
        )
        first = await agents.spawn("task", "one", name="One", parent="main")
        second = await agents.spawn("task", "two", name="Two", parent="main")
        await _eventually(lambda: active[0] == 1)
        await asyncio.sleep(0.01)
        assert active[1] == 1

        gate.set()
        await first.wait_status("idle")
        await second.wait_status("idle")
        assert active[1] == 1
        await agents.shutdown()


@pytest.mark.asyncio
async def test_idle_run_parks_revives_and_accepts_a_new_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph("one", "two")
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, idle_ttl_s=0.01),
            store,
            EventBus(),
            parent.thread_id,
        )
        run = await agents.spawn("task", "one", parent="main")
        await run.wait_status("parked")

        revived = await agents.revive(run.session_id)
        assert revived is run
        await agents.send(run.id, "two")
        await _eventually(lambda: graph.inputs == ["one", "two"])
        await run.wait_status("parked")
        await agents.shutdown()


@pytest.mark.asyncio
async def test_cancel_timeout_budget_and_shutdown_record_abort_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    graphs = [_Graph(gate=gate), _Graph(), _Graph(), _Graph()]

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.pop(0)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")

        cancelled_registry = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        cancelled = await cancelled_registry.spawn("task", "block", parent="main")
        await cancelled.agent_ready.wait()
        await cancelled_registry.cancel(cancelled.id)
        await cancelled.wait_status("aborted")
        assert cancelled.abort_reason == "cancel"

        timeout_registry = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_runtime_s=0.01, idle_ttl_s=10),
            store,
            EventBus(),
            parent.thread_id,
        )
        timed_out = await timeout_registry.spawn("task", "timeout", parent="main")
        await timed_out.wait_status("aborted")
        assert timed_out.abort_reason == "timeout"

        budget_registry = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, soft_request_budget=1, idle_ttl_s=10),
            store,
            EventBus(),
            parent.thread_id,
        )
        budgeted = await budget_registry.spawn("task", "budget", parent="main")
        await budgeted.wait_status("aborted")
        assert budgeted.abort_reason == "budget"
        assert budgeted.requests == 11
        assert budgeted.agent.inputs[1] == "Wrap up and yield now."

        idle = await cancelled_registry.spawn("task", "idle", parent="main")
        gate.set()
        await idle.wait_status("idle")
        await cancelled_registry.shutdown()
        assert idle.status == "aborted"
        assert idle.abort_reason == "shutdown"
        await timeout_registry.shutdown()
        await budget_registry.shutdown()


@pytest.mark.asyncio
async def test_failed_turn_settles_the_run_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph(error=RuntimeError("model failed"))
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "fail", parent="main")

        await run.wait_status("failed")

        assert run.result == {"error": "RuntimeError: model failed"}
        await agents.shutdown()


@pytest.mark.asyncio
async def test_cancelled_run_never_starts_after_waiting_for_the_semaphore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    first_graph = _Graph(gate=gate)
    queued_graph = _Graph()
    graphs = [first_graph, queued_graph]

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.pop(0)

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_concurrency=1),
            store,
            EventBus(),
            parent.thread_id,
        )
        first = await agents.spawn("task", "first", parent="main")
        queued = await agents.spawn("task", "queued", parent="main")
        await first_graph.started.wait()
        await queued.agent_ready.wait()
        assert queued.status == "queued"

        await agents.cancel(queued.id, "cancel")
        await queued.wait_status("aborted")
        await queued.request_abort("shutdown")

        assert queued_graph.inputs == []
        assert queued.abort_reason == "cancel"
        gate.set()
        await first.wait_status("idle")
        await agents.shutdown()


@pytest.mark.asyncio
async def test_agent_capture_records_summarization_compaction_and_new_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph("first")
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "first", parent="main")
        await run.wait_status("idle")
        graph.messages = [
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nsummary",
                id="summary",
                additional_kwargs={"lc_source": "summarization"},
            ),
            AIMessage(content="new answer", id="new-answer"),
        ]

        run.capture_turn()

        path = Ledger(store).path(run.session_id)
        assert any(isinstance(entry, CompactionEntry) for entry in path)
        assert any(
            isinstance(entry, MessageEntry)
            and entry.message["data"].get("id") == "new-answer"
            for entry in path
        )
        await agents.shutdown()


@pytest.mark.asyncio
async def test_runtime_budget_does_not_reset_when_a_parked_run_is_revived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph("first", "second")
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, idle_ttl_s=0.01, max_runtime_s=0.06),
            store,
            EventBus(),
            parent.thread_id,
        )
        run = await agents.spawn("task", "first", parent="main")
        await run.wait_status("parked")
        await asyncio.sleep(0.04)
        await agents.revive(run.session_id)

        await run.wait_status("aborted", timeout_s=0.04)

        assert run.abort_reason == "timeout"
        await agents.shutdown()


@pytest.mark.asyncio
async def test_depth_cap_removes_task_tool_from_descendant_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[dict[str, Any]] = []

    async def fake_build(*_args: Any, **kwargs: Any) -> _Graph:
        builds.append(kwargs)
        return _Graph()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent_session = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_depth=2),
            store,
            EventBus(),
            parent_session.thread_id,
        )
        parent = await agents.spawn("task", "parent", name="Parent", parent="main")
        await parent.wait_status("idle")
        child = await agents.spawn("task", "child", name="Child", parent=parent.id)
        await child.wait_status("idle")

        assert parent.depth == 1
        assert child.depth == 2
        assert "task" not in builds[1]["tool_scope"]
        assert store.get(child.session_id).parent_session == parent.session_id
        assert agents.tree() == [parent, child]
        with pytest.raises(ValueError, match="maximum agent depth"):
            await agents.spawn("task", "grandchild", parent=child.id)
        await agents.shutdown()


@pytest.mark.asyncio
async def test_registry_caps_live_runs_and_releases_capacity_after_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=_Graph()),
    )
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_live_runs=1),
            store,
            EventBus(),
            parent.thread_id,
        )
        first = await agents.spawn("task", "first", parent="main")
        await first.wait_status("idle")

        with pytest.raises(RuntimeError, match="live agent limit 1 reached"):
            await agents.spawn("task", "second", parent="main")

        await first.complete({"ok": True})
        await first.wait_status("done")
        replacement = await agents.spawn("task", "replacement", parent="main")
        assert replacement.id != first.id
        await agents.shutdown()
