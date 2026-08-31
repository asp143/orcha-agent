from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from orcha_agent.builtin.tools_agents import _timeout, agent_tools
from orcha_agent.core.agents import AgentRegistry
from orcha_agent.core.config import AgentsConfig, Config, MemoryStoreConfig
from orcha_agent.core.events import EventBus
from orcha_agent.core.memory_store import MemoryStore
from orcha_agent.core.plugin import ModeSpec
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.core.ledger import CustomEntry, Ledger


class _Graph:
    def __init__(
        self,
        *responses: str,
        gate: asyncio.Event | None = None,
        active: list[int] | None = None,
    ) -> None:
        self.model = FakeListChatModel(responses=list(responses) or ["ok"] * 16)
        self.gate = gate
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
            response = await self.model.ainvoke([HumanMessage(content=text)])
            yield "messages", (response, {"langgraph_node": "agent"})
        finally:
            if self.active is not None:
                self.active[0] -= 1

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": [], "todos": [], "files": {}})


async def _eventually(predicate: Any, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


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


def _tool_map(host: Any) -> dict[str, Any]:
    return {tool.name: tool for tool in agent_tools(host)}


def _main_host(agents: AgentRegistry) -> SimpleNamespace:
    return SimpleNamespace(source_id="main", agents=agents)


@pytest.mark.asyncio
async def test_task_spawns_all_items_concurrently_with_shared_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    active = [0, 0]
    graphs = deque([_Graph(gate=gate, active=active), _Graph(gate=gate, active=active)])

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.popleft()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        main_tools = _tool_map(_main_host(agents))
        assert set(main_tools) == {"task", "hub"}
        task = main_tools["task"]

        response = await task.ainvoke(
            {
                "context": "Shared constraints apply to every worker.",
                "tasks": [
                    {"name": "First", "task": "Inspect the API."},
                    {"name": "Second", "agent": "scout", "task": "Inspect tests."},
                ],
            }
        )
        runs = agents.list()
        await asyncio.gather(*(run.agent_ready.wait() for run in runs))
        await asyncio.gather(*(run.agent.started.wait() for run in runs))

        assert [run.name for run in runs] == ["First", "Second"]
        assert [run.agent_type.name for run in runs] == ["task", "scout"]
        assert set(_tool_map(runs[1])) == {"yield", "hub"}
        assert active[1] == 2
        assert all(
            "Shared constraints apply to every worker." in run.agent.inputs[0] for run in runs
        )
        assert "Inspect the API." in runs[0].agent.inputs[0]
        assert "Inspect tests." in runs[1].agent.inputs[0]
        assert response == {
            "spawned": [
                {
                    "id": run.id,
                    "name": run.name,
                    "type": run.agent_type.name,
                    "status": "running",
                    "blocking": False,
                }
                for run in runs
            ],
            "results": [],
            "timed_out": [],
            "errors": [],
        }

        gate.set()
        await asyncio.gather(*(run.wait_status("idle") for run in runs))
        await agents.shutdown()


@pytest.mark.asyncio
async def test_yield_accumulates_findings_then_terminal_result_settles_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph()

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "Review this", name="Worker", parent="main")
        await run.wait_status("idle")
        tools = _tool_map(run)

        assert set(tools) == {"task", "yield", "hub"}
        finding = {"title": "First finding", "priority": "P1"}
        response = await tools["yield"].ainvoke({"type": "findings", "data": finding})

        assert response == {"accepted": True, "terminal": False, "findings": 1}
        assert run.status == "idle"
        assert run.partial_findings == [finding]

        result = {"summary": "review complete"}
        response = await tools["yield"].ainvoke({"type": "result", "data": result})
        await run.wait_status("done")

        assert response == {
            "accepted": True,
            "terminal": True,
            "status": "done",
            "schema_overridden": False,
        }
        assert run.result == result
        assert run.partial_findings == [finding]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_payloads_are_bounded_and_event_snapshots_omit_growing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "orcha_agent.core.agents.build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=_Graph()),
    )
    oversized = "x" * (300 * 1024)
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "Bound payloads", parent="main")
        await run.wait_status("idle")
        yield_tool = _tool_map(run)["yield"]

        await yield_tool.ainvoke({"type": "findings", "data": oversized})
        finding = run.partial_findings[0]
        finding_json = json.dumps(finding, ensure_ascii=False)
        assert "[truncated" in finding_json
        assert len(finding_json.encode()) <= 256 * 1024

        await agents.post_message("main", run.id, oversized)
        message = agents.drain_messages(run.id)[0]["message"]
        assert "[truncated" in message
        assert len(message.encode()) <= 256 * 1024

        before = len(
            [
                entry
                for entry in Ledger(store).path(parent.thread_id)
                if isinstance(entry, CustomEntry)
                and entry.custom_type == "agent_job"
                and entry.data.get("run_id") == run.id
            ]
        )
        for count in range(3):
            run.tool_calls = count + 1
            agents._persist_job(run)
        snapshots = [
            entry.data
            for entry in Ledger(store).path(parent.thread_id)
            if isinstance(entry, CustomEntry)
            and entry.custom_type == "agent_job"
            and entry.data.get("run_id") == run.id
        ]
        assert len(snapshots) == before + 3
        for snapshot in snapshots[-3:]:
            assert "partial_findings" not in snapshot
            assert "last_yield" not in snapshot
            assert "result" not in snapshot

        await yield_tool.ainvoke({"type": "result", "data": oversized})
        await run.wait_status("done")
        child_results = [
            entry.data
            for entry in Ledger(store).path(run.session_id)
            if isinstance(entry, CustomEntry) and entry.custom_type == "agent_result"
        ]
        assert "[truncated" in child_results[-1]["result"]
        assert (
            len(json.dumps(child_results[-1]["result"], ensure_ascii=False).encode()) <= 256 * 1024
        )
        terminal = (
            Ledger(store).latest_custom(parent.thread_id, "agent_job", key="run_id")[run.id].data
        )
        assert "partial_findings" in terminal
        assert "last_yield" in terminal
        assert "result" in terminal
        await agents.shutdown()


_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["priority"],
                "properties": {"priority": {"type": "string", "enum": ["P0", "P1"]}},
            },
        }
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema_mode", "expected_status", "schema_overridden"),
    [("permissive", "done", True), ("strict", "failed", False)],
)
async def test_yield_schema_failure_on_third_attempt_obeys_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_mode: str,
    expected_status: str,
    schema_overridden: bool,
) -> None:
    graph = _Graph()

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn(
            "task",
            "Return findings",
            parent="main",
            output_schema=_SCHEMA,
            schema_mode=schema_mode,
        )
        await run.wait_status("idle")
        yield_tool = _tool_map(run)["yield"]
        invalid = {"findings": [{"priority": "P9"}]}

        first = await yield_tool.ainvoke({"type": "result", "data": invalid})
        second = await yield_tool.ainvoke({"type": "result", "data": invalid})

        assert run.status == "idle"
        assert first["accepted"] is False
        assert first["terminal"] is False
        assert first["attempt"] == 1
        assert "priority" in first["error"]
        assert second["accepted"] is False
        assert second["terminal"] is False
        assert second["attempt"] == 2
        assert "priority" in second["error"]

        third = await yield_tool.ainvoke({"type": "result", "data": invalid})
        await run.wait_status(expected_status)

        assert run.schema_overridden is schema_overridden
        assert third["terminal"] is True
        assert third["status"] == expected_status
        if schema_mode == "permissive":
            assert third["accepted"] is True
            assert third["schema_overridden"] is True
            assert run.result == invalid
        else:
            assert third["accepted"] is False
            assert "priority" in third["error"]
            assert "error" in run.result
        await agents.shutdown()


@pytest.mark.asyncio
async def test_hub_lists_sends_and_drains_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _Graph("first", "second")

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "first", name="Worker", parent="main")
        await run.wait_status("idle")
        main_hub = _tool_map(_main_host(agents))["hub"]
        child_hub = _tool_map(run)["hub"]

        roster = await main_hub.ainvoke({"op": "list"})
        assert len(roster["agents"]) == 1
        assert roster["agents"][0]["id"] == run.id
        assert roster["agents"][0]["name"] == "Worker"
        assert roster["agents"][0]["status"] == "idle"

        sent = await child_hub.ainvoke({"op": "send", "to": "main", "message": "progress report"})
        inbox = await main_hub.ainvoke({"op": "inbox"})
        drained = await main_hub.ainvoke({"op": "inbox"})
        assert sent["sent"]["to"] == "main"
        assert inbox["messages"][0]["from"] == run.id
        assert inbox["messages"][0]["to"] == "main"
        assert inbox["messages"][0]["message"] == "progress report"
        assert drained == {"messages": []}

        sent = await main_hub.ainvoke(
            {"op": "send", "to": run.id, "message": "second", "await": False}
        )
        assert sent["sent"]["to"] == run.id
        await _eventually(
            lambda: (
                graph.inputs
                == [
                    "first",
                    '<agent-message from="main" role="parent">second</agent-message>',
                ]
            )
        )
        await run.wait_status("idle")
        await agents.shutdown()


@pytest.mark.asyncio
async def test_interrupt_send_steers_running_agent_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    graph = _Graph(gate=gate)

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graph

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        run = await agents.spawn("task", "first", name="Worker", parent="main")
        await graph.started.wait()

        sent = await _tool_map(_main_host(agents))["hub"].ainvoke(
            {
                "op": "send",
                "to": run.id,
                "message": "steer now",
                "interrupt": True,
            }
        )
        await _eventually(
            lambda: (
                graph.inputs
                == [
                    "first",
                    ('<agent-message from="main" role="parent">steer now</agent-message>'),
                ]
            )
        )
        gate.set()
        await run.wait_status("idle")

        assert sent["sent"]["to"] == run.id
        assert run.abort_reason is None
        assert run.status == "idle"
        await agents.shutdown()


@pytest.mark.asyncio
async def test_awaited_send_ignores_unrelated_mailbox_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = deque([_Graph(), _Graph()])

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.popleft()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        target = await agents.spawn("task", "target", parent="main")
        unrelated = await agents.spawn("task", "other", parent="main")
        await target.wait_status("idle")
        await unrelated.wait_status("idle")
        await agents.post_message(unrelated.id, "main", "unrelated")

        waiting = asyncio.create_task(
            _tool_map(_main_host(agents))["hub"].ainvoke(
                {
                    "op": "send",
                    "to": target.id,
                    "message": "please reply",
                    "await": True,
                }
            )
        )
        await asyncio.sleep(0.01)
        assert waiting.done() is False

        await agents.record_yield(target, {"type": "findings", "data": "reply"})
        response = await waiting

        assert response["event"]["kind"] == "yield"
        inbox = await _tool_map(_main_host(agents))["hub"].ainvoke({"op": "inbox"})
        assert [message["from"] for message in inbox["messages"]] == [unrelated.id]
        await agents.shutdown()


@pytest.mark.asyncio
async def test_hub_jobs_wait_and_cancel_report_observable_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = asyncio.Event()
    graphs = deque([_Graph(), _Graph(), _Graph(gate=blocked)])

    async def fake_build(*_args: Any, **_kwargs: Any) -> _Graph:
        return graphs.popleft()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", fake_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        agents = AgentRegistry(
            _plugin_registry(), _config(tmp_path), store, EventBus(), parent.thread_id
        )
        main_hub = _tool_map(_main_host(agents))["hub"]

        job_run = await agents.spawn("task", "job", name="Job", parent="main")
        await job_run.wait_status("idle")
        await _tool_map(job_run)["yield"].ainvoke({"type": "result", "data": {"value": 1}})
        await job_run.wait_status("done")

        jobs = await main_hub.ainvoke({"op": "jobs"})
        assert jobs == {
            "jobs": [
                {
                    "id": job_run.id,
                    "name": "Job",
                    "type": "task",
                    "status": "done",
                    "result": {"value": 1},
                    "schema_overridden": False,
                    "findings": [],
                    "delivered": True,
                }
            ]
        }

        wait_run = await agents.spawn("task", "wait", name="Wait", parent="main")
        await wait_run.wait_status("idle")
        waiter = asyncio.create_task(
            main_hub.ainvoke({"op": "wait", "ids": [wait_run.id], "timeout_s": 1})
        )
        await asyncio.sleep(0)
        await _tool_map(wait_run)["yield"].ainvoke({"type": "result", "data": {"value": 2}})
        waited = await waiter
        assert waited["messages"] == []
        assert waited["timed_out"] is False
        assert len(waited["jobs"]) == 1
        assert waited["jobs"][0]["id"] == wait_run.id
        assert waited["jobs"][0]["status"] == "done"
        assert waited["jobs"][0]["result"] == {"value": 2}

        cancel_run = await agents.spawn("task", "block", name="Cancel", parent="main")
        await cancel_run.agent_ready.wait()
        cancelled = await main_hub.ainvoke(
            {"op": "cancel", "ids": [cancel_run.id], "reason": "cancel"}
        )
        await cancel_run.wait_status("aborted")
        assert cancelled["errors"] == []
        assert [item["id"] for item in cancelled["cancelled"]] == [cancel_run.id]
        assert cancel_run.abort_reason == "cancel"
        await agents.shutdown()


@pytest.mark.asyncio
async def test_main_agent_gets_structured_memory_tools_when_turso_memory_is_enabled(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        parent = store.create(tmp_path, "fake:main", thread_id="main-session")
        cfg = replace(
            _config(tmp_path),
            memory_store=MemoryStoreConfig(
                backend="hybrid",
                workspace="orcha-agent",
            ),
        )
        agents = AgentRegistry(_plugin_registry(), cfg, store, EventBus(), parent.thread_id)
        store.structured_memory = MemoryStore(store._connection, store.saver.lock)
        rebuilds: list[str] = []
        host = SimpleNamespace(
            source_id="main",
            agents=agents,
            session=store,
            cfg=cfg,
            request_rebuild=lambda: rebuilds.append("rebuild"),
        )
        tools = _tool_map(host)

        assert {"list_memories", "read_memory", "save_memory"} <= set(tools)
        saved = await tools["save_memory"].ainvoke(
            {
                "name": "test-command",
                "content": "Run uv run pytest.",
                "scope": "workspace",
            }
        )
        assert saved == {
            "id": "test-command",
            "scope": "workspace",
            "revision": 1,
        }
        assert rebuilds == ["rebuild"]
        assert await tools["read_memory"].ainvoke({"name": "test-command"}) == [
            {
                "id": "test-command",
                "scope": "workspace",
                "path": None,
                "revision": 1,
                "content": "Run uv run pytest.",
            }
        ]
        await agents.shutdown()


def test_blocking_wait_timeout_uses_runtime_or_safety_bound() -> None:
    agents = SimpleNamespace(max_runtime_s=17.5)
    registry = SimpleNamespace(cfg=SimpleNamespace(agents=agents))

    assert _timeout(registry) == 17.5
    agents.max_runtime_s = 0
    assert _timeout(registry) == 300.0
    agents.max_runtime_s = -1
    assert _timeout(registry) == 300.0
