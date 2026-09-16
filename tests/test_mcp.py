"""Integration tests using the official SDK server in a stdio subprocess."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from mcp import types

from orcha_agent.extensibility.mcp import (
    Connection,
    MCPManager,
    ServerConfig,
    flatten_result,
    load_servers,
    tool_name,
)


class API:
    def __init__(self):
        self.tools = {}
        self.rebuilds = 0

    def add_tool(self, tool, **_kwargs):
        self.tools[tool.name] = tool

    def remove_tool(self, name):
        return self.tools.pop(name, None)

    def request_rebuild(self):
        self.rebuilds += 1


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def test_config_precedence_and_trust(tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "project"
    write(home / ".config/orcha-agent/mcp.json", {"mcpServers": {"a": {"command": "user"}}})
    write(cwd / ".mcp.json", {"mcpServers": {"a": {"command": "claude"}}})
    write(cwd / ".orcha-agent/mcp.json", {"mcpServers": {"a": {"command": "native"}}})
    assert load_servers(cwd, home, False, {})["a"].values["command"] == "user"
    assert load_servers(cwd, home, True, {})["a"].values["command"] == "native"
    (cwd / ".orcha-agent/mcp.json").unlink()
    assert load_servers(cwd, home, True, {})["a"].values["command"] == "claude"
    assert load_servers(cwd, home, True, {"import_claude": False})["a"].values["command"] == "user"


def test_codex_import_and_toggle(tmp_path):
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(
        '[mcp_servers.example]\nurl="https://example.com/mcp"\n[mcp_servers.example.http_headers]\nX-Test="ok"\n'
    )
    values = load_servers(tmp_path / "cwd", tmp_path, False, {})["example"].values
    assert values["type"] == "http"
    assert values["headers"] == {"X-Test": "ok"}
    assert load_servers(tmp_path, tmp_path, False, {"import_codex": False}) == {}


def test_names_are_bounded_distinct_and_safe():
    assert tool_name("server", "echo") == "mcp__server__echo"
    assert len(tool_name("s" * 70, "t")) == 64
    assert tool_name("s" * 70, "t") != tool_name("s" * 70, "u")
    assert tool_name("a.b", "c") != tool_name("a_b", "c")


def test_flatten_and_truncate():
    result = types.CallToolResult(
        is_error=True,
        content=[
            types.TextContent(type="text", text="failed"),
            types.ImageContent(type="image", data="secret-image", mime_type="image/png"),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(uri="file:///test", text="resource body"),
            ),
        ],
    )
    text = flatten_result(result)
    assert text.startswith("MCP tool error:")
    assert "resource body" in text and "Image: image/png" in text
    assert "secret-image" not in text
    long = flatten_result(
        types.CallToolResult(content=[types.TextContent(type="text", text="x" * 200000)])
    )
    assert len(long) < 200000


@pytest_asyncio.fixture
async def live_manager(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("""import asyncio
from mcp.server.mcpserver import MCPServer
server = MCPServer("test")
@server.tool()
async def echo(value: str) -> str:
    return "echo:" + value
@server.tool()
async def fail() -> str:
    raise ValueError("intentional failure")
@server.tool()
async def slow() -> str:
    await asyncio.sleep(5)
    return "late"
server.run(transport="stdio")
""")
    config = {"type": "stdio", "command": sys.executable, "args": [str(script)], "timeout": 3}
    write(tmp_path / ".config/orcha-agent/mcp.json", {"mcpServers": {"test": config}})
    api = API()
    manager = MCPManager(api, tmp_path, tmp_path, False, {})
    await manager.start()
    try:
        await asyncio.wait_for(manager.connections["test"].ready.wait(), 8)
        yield manager
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_stdio_connect_list_call_error_timeout_and_shutdown(live_manager):
    manager = live_manager
    assert manager.connections["test"].status == "connected"
    assert "mcp__test__echo" in manager.api.tools
    tool = manager.api.tools["mcp__test__echo"]
    assert tool.args_schema["properties"]["value"]["type"] == "string"
    assert await tool.ainvoke({"value": "hello"}) == "echo:hello"
    assert "MCP tool error:" in await manager.call("test", "fail", {})
    manager.connections["test"].config.values["timeout"] = 0.05
    assert "timed out" in await manager.call("test", "slow", {})
    await manager.close()
    assert manager.connections["test"].task.done()


@pytest.mark.asyncio
async def test_cache_fallback_and_fingerprint(tmp_path):
    config = {"type": "stdio", "command": "/does-not-exist", "timeout": 0.02}
    path = tmp_path / ".config/orcha-agent/mcp.json"
    write(path, {"mcpServers": {"offline": config}})
    api = API()
    manager = MCPManager(api, tmp_path, tmp_path, False, {})
    connection = Connection(ServerConfig("offline", config, path))
    connection.tools = [
        {
            "name": "cached",
            "description": "cached tool",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]
    manager.write_cache(connection)
    await manager.start()
    try:
        assert "mcp__offline__cached" in api.tools
        assert "timed out" in await api.tools["mcp__offline__cached"].ainvoke({})
        changed = ServerConfig("offline", {**config, "command": "other"}, path)
        assert manager.read_cache(changed) == []
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
async def test_network_transports(tmp_path, transport):
    import socket
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    # Reserve a socket and hand it directly to uvicorn, avoiding port allocation races.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    server = MCPServer("network")

    @server.tool()
    def echo(value: str) -> str:
        return value

    app = server.streamable_http_app() if transport == "http" else server.sse_app()
    runner = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    serving = asyncio.create_task(runner.serve(sockets=[listener]))
    port = listener.getsockname()[1]
    suffix = "/mcp" if transport == "http" else "/sse"
    write(
        tmp_path / ".config/orcha-agent/mcp.json",
        {
            "mcpServers": {
                "network": {
                    "type": transport,
                    "url": f"http://127.0.0.1:{port}{suffix}",
                    "timeout": 3,
                }
            }
        },
    )
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    try:
        await manager.start()
        await asyncio.wait_for(manager.connections["network"].ready.wait(), 8)
        assert await manager.call("network", "echo", {"value": "network-ok"}) == "network-ok"
    finally:
        await manager.close()
        runner.should_exit = True
        await serving
        listener.close()


@pytest.mark.asyncio
async def test_management_and_reload_remove_tools(live_manager):
    from io import StringIO
    from rich.console import Console
    from orcha_agent.builtin.mcp import command

    manager = live_manager
    output = StringIO()
    ctx = SimpleNamespace(console=Console(file=output))
    for action in ["list", "test test", "resources test", "prompts test", "reconnect test"]:
        await command(manager, ctx, action)
    assert "connected" in output.getvalue()
    await command(manager, ctx, "disable test")
    assert manager.connections["test"].status == "disabled"
    assert "mcp__test__echo" not in manager.api.tools
    await command(manager, ctx, "enable test")
    await asyncio.wait_for(manager.connections["test"].ready.wait(), 8)
    assert "mcp__test__echo" in manager.api.tools
    previous_rebuilds = manager.api.rebuilds
    await command(manager, ctx, "remove test")
    assert manager.connections == {} and manager.api.tools == {}
    assert manager.api.rebuilds > previous_rebuilds
    await command(manager, ctx, "add second --url http://127.0.0.1:1/mcp")
    assert manager.connections["second"].config.values["type"] == "http"
    await command(manager, ctx, "remove second")
    await command(manager, ctx, "add third /does-not-exist --arg")
    assert manager.connections["third"].config.values["args"] == ["--arg"]


@pytest.mark.asyncio
async def test_plugin_startup_gate_and_approval(tmp_path, monkeypatch):
    from orcha_agent.builtin.mcp import register
    from orcha_agent.core.events import AppStart, AppExit, AgentBuildBefore, EventBus
    from orcha_agent.core.registry import Registry
    from orcha_agent.core.plugin import PluginAPI, ModeSpec

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {"type": "stdio", "command": "/does-not-exist", "timeout": 0.01}
    path = tmp_path / ".config/orcha-agent/mcp.json"
    write(path, {"mcpServers": {"offline": config}})
    cache_manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    cached = Connection(ServerConfig("offline", config, path))
    cached.tools = [{"name": "cached", "inputSchema": {"type": "object", "properties": {}}}]
    cache_manager.write_cache(cached)
    registry, bus = Registry(), EventBus()
    api = PluginAPI(
        name="mcp", config={}, state={}, registry=registry, bus=bus, request_rebuild=lambda: None
    )
    register(api)
    registry.modes["ask"] = ModeSpec("ask", {"execute": True}, None)
    registry.modes["yolo"] = ModeSpec("yolo", {}, None)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False, mode="ask"), registry=registry
    )
    started = asyncio.get_running_loop().time()
    await bus.emit(AppStart(ctx))
    assert asyncio.get_running_loop().time() - started < 0.1
    try:
        event = AgentBuildBefore({"tools": [], "interrupt_on": {}})
        await bus.emit(event)
        assert len(event.kwargs["tools"]) == 1
        assert event.kwargs["interrupt_on"]["mcp__offline__cached"] is True
        assert registry.status_segments[0].render(ctx) == "MCP 0/1"
        approved = AgentBuildBefore(
            {"tools": [], "interrupt_on": {}},
            always_allowed=frozenset({"mcp__offline__cached"}),
            mode_interrupt_on={"execute": True},
        )
        await bus.emit(approved)
        assert approved.kwargs["interrupt_on"] == {}
        readonly = AgentBuildBefore(
            {"tools": [], "interrupt_on": {}},
            tool_scope={"read_file"},
            mode_interrupt_on={},
        )
        await bus.emit(readonly)
        assert readonly.kwargs["tools"] == []
        scoped = AgentBuildBefore({"tools": [], "interrupt_on": {}}, tool_scope=set())
        await bus.emit(scoped)
        assert scoped.kwargs["tools"] == []
        ctx.cfg.mode = "yolo"
        yolo = AgentBuildBefore({"tools": [], "interrupt_on": {}})
        await bus.emit(yolo)
        assert yolo.kwargs["interrupt_on"] == {}
        assert api.state == {}  # Runtime objects must never leak into persisted plugin state.
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_remove_imported_and_shadowed_servers(tmp_path):
    from io import StringIO
    from rich.console import Console
    from orcha_agent.builtin.mcp import command

    codex = tmp_path / ".codex/config.toml"
    codex.parent.mkdir()
    original = '[mcp_servers.hidden]\ncommand="/does-not-exist"\n'
    codex.write_text(original)
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    ctx = SimpleNamespace(console=Console(file=StringIO()))
    await manager.start()
    try:
        await command(manager, ctx, "remove hidden")
        assert "hidden" not in manager.connections
        assert codex.read_text() == original
        await command(manager, ctx, "add hidden /does-not-exist")
        assert "hidden" in manager.connections
        await command(manager, ctx, "remove hidden")
        assert "hidden" not in manager.connections
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_bridge_preserves_reserved_argument_names(tmp_path):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    seen = []

    async def record(server, tool, arguments):
        seen.append((server, tool, arguments))
        return "ok"

    manager.call = record
    config = ServerConfig("original", {"command": "unused"}, tmp_path)
    connection = Connection(
        config,
        tools=[
            {
                "name": "tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"_server": {"type": "string"}, "_tool": {"type": "string"}},
                },
            }
        ],
    )
    manager.publish(connection)
    await manager.api.tools["mcp__original__tool"].ainvoke(
        {"_server": "other", "_tool": "different"}
    )
    assert seen == [("original", "tool", {"_server": "other", "_tool": "different"})]
    connection.tools = []
    manager.publish(connection)
    assert manager.api.tools == {}


@pytest.mark.asyncio
async def test_plugin_tolerates_minimal_start_context():
    from orcha_agent.builtin.mcp import register
    from orcha_agent.core.events import AppStart, AppExit, EventBus
    from orcha_agent.core.plugin import PluginAPI
    from orcha_agent.core.registry import Registry

    bus = EventBus()
    register(
        PluginAPI(
            name="mcp",
            config={},
            state={},
            registry=Registry(),
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    await bus.emit(AppStart(object()))
    await bus.emit(AppExit())
