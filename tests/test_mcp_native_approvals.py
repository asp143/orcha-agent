"""MCP discovery must retain execution approvals after native tool remapping."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.builtin.mcp import register
from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart, EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.mcp import Connection, MCPManager, ServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_name", ["execute", "bash"])
@pytest.mark.parametrize("approval_required", [True, False])
async def test_discovered_mcp_tools_follow_execution_policy(
    tmp_path, monkeypatch, execution_name, approval_required
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    async def discover(manager):
        connection = Connection(ServerConfig("local", {}, tmp_path / "mcp.json"))
        connection.tools = [{"name": "change", "inputSchema": {"type": "object", "properties": {}}}]
        manager.publish(connection)
        manager.loaded.set()

    monkeypatch.setattr(MCPManager, "start", discover)
    registry, bus = Registry(), EventBus()
    api = PluginAPI(
        name="mcp", config={}, state={}, registry=registry, bus=bus, request_rebuild=lambda: None
    )
    register(api)
    # The registry retains legacy names; the build event carries the effective
    # policy after the native tool layer has mapped execute to bash.
    registry.modes["ask"] = ModeSpec("ask", {"execute": True}, None)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False, mode="ask"), registry=registry
    )
    policy = {execution_name: approval_required}
    name = "mcp__local__change"
    await bus.emit(AppStart(ctx))
    try:
        event = AgentBuildBefore({"tools": [], "interrupt_on": {}}, mode_interrupt_on=policy)
        await bus.emit(event)
        assert [tool.name for tool in event.kwargs["tools"]] == [name]
        assert event.kwargs["interrupt_on"] == ({name: True} if approval_required else {})

        approved = AgentBuildBefore(
            {"tools": [], "interrupt_on": {}},
            mode_interrupt_on=policy,
            always_allowed=frozenset({name}),
        )
        await bus.emit(approved)
        assert approved.kwargs["interrupt_on"] == {}

        scoped = AgentBuildBefore(
            {"tools": [], "interrupt_on": {}}, mode_interrupt_on=policy, tool_scope={"read"}
        )
        await bus.emit(scoped)
        assert scoped.kwargs == {"tools": [], "interrupt_on": {}}
    finally:
        await bus.emit(AppExit())
