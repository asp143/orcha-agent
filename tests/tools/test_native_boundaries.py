"""Exercise native-tool mode, backend, subagent, and overflow boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from test_native_integration import FakeToolsModel, config, kernel

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import ToolsConfig
from orcha_agent.core.events import AgentBuildBefore, AppExit
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.session import SessionStore


def plugin(registry, bus) -> PluginAPI:
    return PluginAPI(
        name="boundary_test",
        config={},
        state={},
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )


def capture_kwargs(bus) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def capture(event: AgentBuildBefore) -> None:
        captured.update(event.kwargs)

    bus.on(AgentBuildBefore, capture, plugin="boundary_test")
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disable", "backend"])
async def test_loaded_native_registry_cannot_leak_into_fallback(
    tmp_path: Path, change: str, monkeypatch
) -> None:
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(cfg, [])
    assert {"read", "bash", "write"} <= registry.tools.keys()
    if change == "disable":
        cfg = replace(cfg, tools=ToolsConfig(native=False))
    else:
        plugin(registry, bus).add_backend(
            "alternate",
            lambda cfg: FilesystemBackend(root_dir=cfg.cwd, virtual_mode=True),
        )
        cfg = replace(cfg, backend="alternate")
    captured = capture_kwargs(bus)
    bound: list[str] = []

    def bind(model, tools, **kwargs):
        bound.extend(item.name for item in tools)
        return model

    monkeypatch.setattr(FakeToolsModel, "bind_tools", bind)
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Inspect the available tools."}]},
                {"configurable": {"thread_id": "fallback"}},
            )
        assert not captured["tools"]
        assert {"read_file", "write_file", "edit_file", "grep", "glob", "ls"} <= set(bound)
        assert not {"read", "write", "edit", "bash", "bash_jobs"} & set(bound)
        assert len(bound) == len(set(bound))
        assert "Use read for file contents" not in captured["system_prompt"]
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, {"read_file", "glob"}])
async def test_plan_restricts_native_tools_and_maps_legacy_scope(tmp_path: Path, scope) -> None:
    cfg = config(tmp_path, "plan")
    registry, bus = kernel(cfg, [])
    captured = capture_kwargs(bus)
    try:
        with SessionStore(cfg.db_path) as session:
            await build_agent(
                registry, cfg, session, bus, exclude_general_purpose=True, tool_scope=scope
            )
        available = {item.name for item in captured["tools"]}
        expected = {"read", "glob"} if scope else {"read", "grep", "glob", "ls"}
        assert available == expected
        filesystem = next(
            item for item in captured["middleware"] if isinstance(item, FilesystemMiddleware)
        )
        assert filesystem.tools == []
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_explicit_subagent_approvals_and_readonly_filesystem_scope(tmp_path: Path) -> None:
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(cfg, [])
    api = plugin(registry, bus)
    api.add_subagent(
        {
            "name": "executor",
            "description": "Run a command after approval.",
            "system_prompt": "Run commands carefully.",
            "tools": ["execute", "bash_jobs"],
            "interrupt_on": {"execute": True},
        }
    )
    api.add_subagent(
        {
            "name": "reader",
            "description": "Read files only.",
            "system_prompt": "Read files without mutation.",
            "tools": ["read_file", "grep"],
        }
    )
    captured = capture_kwargs(bus)
    try:
        with SessionStore(cfg.db_path) as session:
            await build_agent(registry, cfg, session, bus)
        agents = {spec["name"]: spec for spec in captured["subagents"]}
        executor = agents["executor"]
        assert executor["interrupt_on"] == {"bash": True, "bash_jobs": True}
        assert {item.name for item in executor["tools"]} == {"bash", "bash_jobs"}
        reader = agents["reader"]
        assert {item.name for item in reader["tools"]} == {"read", "grep"}
        filesystem = next(
            item for item in reader["middleware"] if isinstance(item, FilesystemMiddleware)
        )
        assert filesystem.tools == []
        assert "delete" not in {item.name for item in reader["tools"]}
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_custom_large_result_offloads_and_native_read_recovers_it(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    destination = tmp_path / ".orcha" / "artifacts" / "large_tool_results" / "call-0"
    payload = "".join(f"entry {index:05d}: bounded recovery evidence\n" for index in range(6000))
    registry, bus = kernel(
        cfg,
        [
            ("large_result", {}),
            ("read", {"path": f"{destination}:5000-5002"}),
        ],
    )

    @tool
    def large_result() -> str:
        """Return a large custom plugin response to test overflow storage."""
        return payload

    plugin(registry, bus).add_tool(large_result)
    captured = capture_kwargs(bus)
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Produce and recover a large result."}]},
                {"configurable": {"thread_id": "offload"}},
            )
        messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
        assert len(messages) == 2
        assert all(item.status == "success" for item in messages)
        assert destination.read_text() == payload
        assert str(destination) in messages[0].content
        assert len(messages[0].content) < len(payload)
        assert "entry 04999" in messages[1].content
        assert "entry 05001" in messages[1].content
        filesystem = next(
            item for item in captured["middleware"] if isinstance(item, FilesystemMiddleware)
        )
        assert {item.name for item in filesystem.tools} == {"delete"}
    finally:
        await bus.emit(AppExit())
