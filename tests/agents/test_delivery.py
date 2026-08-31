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
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.builtin.tools_agents import agent_tools
from orcha_agent.core.agents import AgentRegistry, AgentRun
from orcha_agent.core.config import AgentsConfig, Config
from orcha_agent.core.events import AgentDelivered, AgentFinished, EventBus
from orcha_agent.core.ledger import CustomEntry, Ledger
from orcha_agent.core.plugin import ModeSpec
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade


class _Graph:
    def __init__(
        self,
        *responses: str,
        gate: asyncio.Event | None = None,
        error: Exception | None = None,
        active: list[int] | None = None,
    ) -> None:
        self.model = FakeListChatModel(responses=list(responses) or ["ok"] * 16)
        self.gate = gate
        self.error = error
        self.active = active
        self.started = asyncio.Event()
        self.inputs: list[str] = []

    async def astream(self, value: Any, **_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        text = value["messages"][0]["content"]
        self.inputs.append(text)
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
            yield "messages", (response, {"langgraph_node": "agent"})
        finally:
            if self.active is not None:
                self.active[0] -= 1

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": [], "todos": [], "files": {}})


def _config(tmp_path: Path, **agent_overrides: Any) -> Config:
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
        agents=replace(AgentsConfig(), **agent_overrides),
    )


def _plugin_registry() -> Registry:
    registry = Registry()
    registry.modes["yolo"] = ModeSpec(description="all", interrupt_on={}, allowed_tools=None)
    return registry


def _agent_jobs(store: SessionStore, parent_session: str) -> list[CustomEntry]:
    return [
        entry
        for entry in Ledger(store).path(parent_session)
        if isinstance(entry, CustomEntry) and entry.custom_type == "agent_job"
    ]


async def _spawn_done(
    agents: AgentRegistry,
    result: Any,
    *,
    name: str = "Worker",
) -> AgentRun:
    run = await agents.spawn("task", "work", name=name, parent="main")
    await run.wait_status("idle")
    await run.complete(result)
    await run.wait_status("done")
    return run


def _main_hub(agents: AgentRegistry) -> Any:
    host = SimpleNamespace(source_id="main", agents=agents)
    return next(tool for tool in agent_tools(host) if tool.name == "hub")


def _runtime_context(tmp_path: Path, agents: AgentRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=tmp_path,
            model="fake:main",
            models={},
            providers={},
            thinking="summary",
        ),
        registry=_plugin_registry(),
        agents=agents,
        plugin_states={},
        persist_plugin_states=lambda: None,
        ui=UIFacade(),
        bus=agents.bus,
        _bus=agents.bus,
        switch_model=lambda _model: asyncio.sleep(0),
        rebuild=lambda: asyncio.sleep(0),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["done", "failed", "aborted"])
async def test_settlement_mirrors_terminal_job_to_parent_before_finished_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    gate = asyncio.Event() if terminal_status == "aborted" else None
    error = RuntimeError("model failed") if terminal_status == "failed" else None
    graph = _Graph(gate=gate, error=error)
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        bus = EventBus()
        jobs_visible_at_finish: list[dict[str, Any]] = []
        finished_seen = asyncio.Event()

        async def observe_finished(_event: AgentFinished) -> None:
            jobs_visible_at_finish.append(dict(_agent_jobs(store, parent.thread_id)[-1].data))
            finished_seen.set()

        bus.on(AgentFinished, observe_finished, plugin="test")
        agents = AgentRegistry(_plugin_registry(), _config(tmp_path), store, bus, parent.thread_id)
        run = await agents.spawn("task", "work", name="Worker", parent="main")

        if terminal_status == "done":
            await run.wait_status("idle")
            await run.complete({"value": 1})
        elif terminal_status == "aborted":
            await graph.started.wait()
            await agents.cancel(run.id)
        await run.wait_status(terminal_status)
        await finished_seen.wait()

        expected_result = {
            "done": {"value": 1},
            "failed": {"error": "RuntimeError: model failed"},
            "aborted": {"error": "cancel"},
        }[terminal_status]
        parent_job = _agent_jobs(store, parent.thread_id)[-1].data
        assert parent_job["run_id"] == run.id
        assert parent_job["status"] == terminal_status
        assert parent_job["result"] == expected_result
        assert parent_job["schema_overridden"] is False
        assert parent_job["delivered"] is False
        assert jobs_visible_at_finish == [parent_job]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_jobs_wait_and_automatic_claim_compete_for_one_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        bus = EventBus()
        delivery_events: list[AgentDelivered] = []

        async def observe_delivery(event: AgentDelivered) -> None:
            delivery_events.append(event)

        bus.on(AgentDelivered, observe_delivery, plugin="test")
        agents = AgentRegistry(_plugin_registry(), _config(tmp_path), store, bus, parent.thread_id)
        run = await _spawn_done(agents, {"value": 1})
        hub = _main_hub(agents)
        original_deliver = agents.deliver
        all_consumers_ready = asyncio.Event()
        arrivals = 0

        async def gated_deliver(parent_id: str, ids: Any = None) -> list[AgentRun]:
            nonlocal arrivals
            arrivals += 1
            if arrivals == 3:
                all_consumers_ready.set()
            await asyncio.wait_for(all_consumers_ready.wait(), 1)
            return await original_deliver(parent_id, ids)

        monkeypatch.setattr(agents, "deliver", gated_deliver)

        with create_pipe_input() as pipe:
            runtime = ApplicationRuntime(
                lambda _text: asyncio.sleep(0),
                ctx=_runtime_context(tmp_path, agents),
                input=pipe,
                output=DummyOutput(),
            )
            jobs_result, wait_result, automatic_result = await asyncio.gather(
                hub.ainvoke({"op": "jobs"}),
                hub.ainvoke({"op": "wait", "ids": [run.id], "timeout_s": 1}),
                runtime._claim_agent_delivery(),
            )

        assert arrivals == 3
        assert delivery_events == [AgentDelivered(parent_id="main", run_ids=(run.id,))]
        assert jobs_result["jobs"][0]["id"] == run.id
        assert jobs_result["jobs"][0]["delivered"] is True
        fresh_wait = wait_result["jobs"]
        fresh_automatic = automatic_result is not None
        assert bool(fresh_wait) + fresh_automatic <= 1
        if fresh_wait:
            assert [job["id"] for job in fresh_wait] == [run.id]
        if automatic_result is not None:
            assert f"Job {run.id} (Worker) finished: done" in automatic_result
        assert await agents.deliver("main", [run.id]) == []
        await agents.shutdown()


@pytest.mark.asyncio
async def test_automatic_delivery_precedes_queued_user_prompt_without_entering_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "work", name="Worker", parent="main")
        await run.wait_status("idle")
        submitted: list[str] = []

        async def submit(text: str) -> None:
            submitted.append(text)
            if text == "active user prompt":
                await run.complete({"answer": 42})
                await run.wait_status("done")

        with create_pipe_input() as pipe:
            runtime = ApplicationRuntime(
                submit,
                ctx=_runtime_context(tmp_path, agents),
                input=pipe,
                output=DummyOutput(),
            )
            runtime.queue.append("queued user prompt")
            assert runtime.queue.items == ("queued user prompt",)

            await runtime._submit_serially("active user prompt")

            assert runtime.queue.items == ()

        assert submitted[0] == "active user prompt"
        assert submitted[2] == "queued user prompt"
        assert submitted[1].startswith("<system-notification>\n")
        assert f"Job {run.id} (Worker) finished: done" in submitted[1]
        assert '{"answer": 42}' in submitted[1]
        assert submitted[1].endswith("\n</system-notification>")
        await agents.shutdown()


@pytest.mark.asyncio
async def test_delivered_job_survives_store_reopen_without_redelivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await _spawn_done(agents, {"persisted": True})
        run_id = run.id
        assert [claimed.id for claimed in await agents.deliver("main", [run_id])] == [run_id]
        assert _agent_jobs(store, parent.thread_id)[-1].data["delivered"] is True
        await agents.shutdown()

    with SessionStore(db_path) as reopened_store:
        delivery_events: list[AgentDelivered] = []
        bus = EventBus()

        async def observe_delivery(event: AgentDelivered) -> None:
            delivery_events.append(event)

        bus.on(AgentDelivered, observe_delivery, plugin="test")
        reopened = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path),
            reopened_store,
            bus,
            "main-session",
        )
        restored = reopened.get(run_id)
        assert restored is not None
        assert restored.status == "done"
        assert restored.result == {"persisted": True}
        assert restored.delivered is True
        assert await reopened.deliver("main", [run_id]) == []
        assert delivery_events == []
        await reopened.shutdown()


@pytest.mark.asyncio
async def test_undelivered_completion_is_restored_once_and_claim_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await _spawn_done(agents, {"restored": "once"})
        run_id = run.id
        assert run.delivered is False
        await agents.shutdown()

    with SessionStore(db_path) as reopened_store:
        reopened = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path),
            reopened_store,
            EventBus(),
            "main-session",
        )
        restored = reopened.jobs("main")
        assert [run.id for run in restored] == [run_id]
        assert restored[0].status == "done"
        assert restored[0].result == {"restored": "once"}
        assert restored[0].delivered is False
        assert [run.id for run in await reopened.deliver("main")] == [run_id]
        assert await reopened.deliver("main") == []
        await reopened.shutdown()

    with SessionStore(db_path) as final_store:
        final = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path),
            final_store,
            EventBus(),
            "main-session",
        )
        restored = final.get(run_id)
        assert restored is not None
        assert restored.delivered is True
        assert await final.deliver("main") == []
        await final.shutdown()


@pytest.mark.asyncio
async def test_hydration_ignores_agent_job_on_abandoned_parent_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    db_path = tmp_path / "sessions.db"

    with SessionStore(db_path) as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        ledger = Ledger(store)
        root = ledger.append(
            parent.thread_id,
            CustomEntry(custom_type="test_root", data={"active": True}),
        )
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await _spawn_done(agents, {"abandoned": True})
        run_id = run.id
        assert _agent_jobs(store, parent.thread_id)[-1].data["run_id"] == run_id

        ledger.branch(parent.thread_id, root.id)
        ledger.append(
            parent.thread_id,
            CustomEntry(custom_type="active_branch", data={"active": True}),
        )
        assert _agent_jobs(store, parent.thread_id) == []
        await agents.shutdown()

    with SessionStore(db_path) as reopened_store:
        delivery_events: list[AgentDelivered] = []
        bus = EventBus()

        async def observe_delivery(event: AgentDelivered) -> None:
            delivery_events.append(event)

        bus.on(AgentDelivered, observe_delivery, plugin="test")
        reopened = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path),
            reopened_store,
            bus,
            "main-session",
        )
        assert reopened.jobs("main") == []
        assert await reopened.deliver("main", [run_id]) == []
        assert delivery_events == []
        await reopened.shutdown()


@pytest.mark.asyncio
async def test_retarget_preserves_live_runs_per_parent_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    graph = _Graph(gate=gate)
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )

    with SessionStore(tmp_path / "sessions.db") as store:
        first = store.create(tmp_path, "fake:main", thread_id="first")
        second = store.create(tmp_path, "fake:main", thread_id="second")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), first.thread_id
        )
        run = await agents.spawn("task", "work", parent="main")
        await graph.started.wait()

        agents.retarget(second.thread_id)
        assert agents.list() == []
        agents.retarget(first.thread_id)
        assert agents.get(run.id) is run
        assert run.status == "running"

        gate.set()
        await run.wait_status("idle")
        await agents.shutdown()


@pytest.mark.asyncio
async def test_retargeted_sessions_share_application_concurrency_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    active = [0, 0]
    old_graph = _Graph(gate=gate, active=active)
    new_graph = _Graph(active=active)
    graphs = deque([old_graph, new_graph])

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.popleft()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        first = store.create(tmp_path, "fake:main", thread_id="first")
        second = store.create(tmp_path, "fake:main", thread_id="second")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, max_concurrency=1),
            store,
            EventBus(),
            first.thread_id,
        )
        await agents.spawn("task", "old", parent="main")
        await old_graph.started.wait()
        assert active[0] == 1

        agents.retarget(second.thread_id)
        new_run = await agents.spawn("task", "new", parent="main")
        await new_run.agent_ready.wait()
        await asyncio.sleep(0.01)
        assert new_graph.started.is_set() is False
        assert active == [1, 1]

        gate.set()
        await new_run.wait_status("idle")
        assert active[1] == 1
        await agents.shutdown()


@pytest.mark.asyncio
async def test_same_session_retarget_preserves_unread_mailbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph()
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
        run = await agents.spawn("task", "work", parent="main")
        await run.wait_status("parked")
        await agents.post_message(run.id, "main", "saved message")

        agents.retarget(parent.thread_id)

        assert [item["message"] for item in agents.drain_messages("main")] == ["saved message"]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_hydration_prefers_terminal_child_result_over_stale_parent_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _Graph()
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    database = tmp_path / "sessions.db"

    with SessionStore(database) as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path, idle_ttl_s=0.01),
            store,
            EventBus(),
            parent.thread_id,
        )
        run = await agents.spawn("task", "work", parent="main")
        await run.wait_status("parked")
        Ledger(store).append(
            run.session_id,
            CustomEntry(
                custom_type="agent_result",
                data={
                    "run_id": run.id,
                    "status": "done",
                    "result": {"durable": True},
                    "schema_overridden": False,
                },
            ),
        )

    with SessionStore(database) as reopened:
        restored = AgentRegistry(
            _plugin_registry(),
            _config(tmp_path),
            reopened,
            EventBus(),
            "main-session",
        ).get(run.id)

        assert restored is not None
        assert restored.status == "done"
        assert restored.result == {"durable": True}
        assert restored.task is None
