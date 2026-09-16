"""Regression coverage for MCP trust, lifecycle, and diagnostics review findings."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from contextlib import asynccontextmanager
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp import types
from mcp.shared.exceptions import MCPError
from rich.console import Console

from orcha_agent.builtin.mcp import command
from orcha_agent.extensibility import mcp
from orcha_agent.extensibility.mcp import Connection, MCPManager, ServerConfig, load_servers

from test_mcp import API, write


def test_invalid_entries_do_not_hide_valid_servers(tmp_path, caplog):
    path = tmp_path / ".config/orcha-agent/mcp.json"
    invalid = {
        "reserved__name": {"command": "unused"},
        "list": [],
        "transport": {"type": "invalid"},
        "timeout": {"command": "unused", "timeout": "secret-value"},
        "headers": {"url": "https://example.com", "headers": ["secret-value"]},
        "args": {"command": "unused", "args": "bad"},
        "boolean": {"command": "unused", "enabled": "false"},
    }
    write(path, {"mcpServers": {**invalid, "valid": {"command": "unused"}}})
    assert set(load_servers(tmp_path, tmp_path, False, {})) == {"valid"}
    for name in invalid:
        assert name in caplog.text
    assert str(path) in caplog.text
    assert "secret-value" not in caplog.text
    with pytest.raises(ValueError, match="__"):
        mcp.tool_name("a__b", "c")


@pytest.mark.parametrize("host", ["example.com", "127.0.0.1.example.com", "192.168.1.1"])
@pytest.mark.parametrize("key", ["headers", "env_http_headers", "bearer_token_env_var"])
def test_cleartext_headers_rejected(tmp_path, host, key):
    value = "TOKEN" if key == "bearer_token_env_var" else {"X-Key": "TOKEN"}
    write(
        tmp_path / ".config/orcha-agent/mcp.json",
        {"mcpServers": {"unsafe": {"url": f"http://{host}/mcp", key: value}}},
    )
    assert load_servers(tmp_path, tmp_path, False, {}) == {}


@pytest.mark.parametrize(
    "url",
    ["https://example.com/mcp", "http://localhost/mcp", "http://127.0.0.1/mcp", "http://[::1]/mcp"],
)
def test_header_urls_preserve_legitimate_controls(url):
    assert mcp.validate_server("a", {"url": url, "headers": {"X": "value"}})["url"] == url


def test_bridge_refuses_other_server_and_original_tool_collisions(tmp_path, monkeypatch, caplog):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    monkeypatch.setattr(mcp, "tool_name", lambda *_: "mcp__collision")
    first = Connection(ServerConfig("one", {}, tmp_path), tools=[{"name": "first"}])
    second = Connection(ServerConfig("two", {}, tmp_path), tools=[{"name": "second"}])
    manager.publish(first)
    original = manager.api.tools["mcp__collision"]
    manager.publish(second)
    assert manager.api.tools["mcp__collision"] is original
    second.tools = []
    manager.publish(second)
    assert manager.api.tools["mcp__collision"] is original
    first.tools = [{"name": "imposter"}]
    manager.publish(first)
    assert "Refusing duplicate" in caplog.text
    assert manager.api.tools == {}  # The removed original is no longer published.


def test_cache_permissions_and_atomic_creation(tmp_path):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    conn = Connection(ServerConfig("private", {}, tmp_path), tools=[{"name": "secret"}])
    old_umask = os.umask(0)
    try:
        manager.write_cache(conn)
    finally:
        os.umask(old_umask)
    path = manager.cache_path("private")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert manager.read_cache(conn.config) == conn.tools
    assert sorted(p.name for p in path.parent.iterdir()) == ["private.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["remove", "disable", "enable"])
async def test_claude_management_keeps_foreign_content_and_mode(tmp_path, action):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    source = cwd / ".mcp.json"
    write(source, {"mcpServers": {"foreign": {"command": "/missing", "enabled": False}}})
    source.chmod(0o644)
    original = source.read_bytes()
    manager = MCPManager(API(), cwd, home, True, {})
    await manager.start()
    try:
        await command(
            manager, SimpleNamespace(console=Console(file=StringIO())), f"{action} foreign"
        )
        assert source.read_bytes() == original
        assert source.stat().st_mode & 0o777 == 0o644
        override = json.loads((cwd / ".orcha-agent/mcp.json").read_text())["mcpServers"]["foreign"]
        if action == "remove":
            assert override == {"_orcha_removed": True}
            assert "foreign" not in manager.connections
        else:
            assert override["enabled"] is (action == "enable")
            assert manager.connections["foreign"].config.enabled is (action == "enable")
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,reconnect",
    [
        (MCPError(-32602, "Invalid argument"), False),
        (ValueError("bad local argument"), False),
        (MCPError(types.CONNECTION_CLOSED, "Connection closed"), True),
        (MCPError(types.REQUEST_TIMEOUT, "Request timed out"), True),
        (anyio.ClosedResourceError(), True),
        (anyio.BrokenResourceError(), True),
        (ConnectionError(), True),
        (OSError(), True),
        (TimeoutError(), True),
    ],
)
async def test_only_transport_errors_reconnect(tmp_path, error, reconnect):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    session = SimpleNamespace(call_tool=AsyncMock(side_effect=error))
    conn = Connection(ServerConfig("test", {}, tmp_path), session=session)
    conn.ready.set()
    manager.connections["test"] = conn
    result = await manager.call("test", "tool", {})
    assert conn.reconnect.is_set() is reconnect
    assert conn.session is session
    if isinstance(error, MCPError):
        assert result == str(error)


@pytest.mark.asyncio
async def test_list_and_test_show_last_connection_error(tmp_path):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    conn = Connection(ServerConfig("bad", {"timeout": 0.01}, tmp_path), error="FileNotFoundError")
    manager.connections["bad"] = conn
    manager.loaded.set()
    for action in ["list", "test bad"]:
        output = StringIO()
        await command(manager, SimpleNamespace(console=Console(file=output)), action)
        assert "FileNotFoundError" in output.getvalue()


def test_codex_timeouts_and_late_environment_headers(tmp_path, monkeypatch, caplog):
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text("""[mcp_servers.remote]
url = "https://example.com/mcp"
startup_timeout_sec = 12
tool_timeout_sec = 34
bearer_token_env_var = "TEST_MCP_BEARER"
[mcp_servers.remote.env_http_headers]
X-Token = "TEST_MCP_HEADER"
""")
    monkeypatch.delenv("TEST_MCP_BEARER", raising=False)
    monkeypatch.delenv("TEST_MCP_HEADER", raising=False)
    cfg = load_servers(tmp_path, tmp_path, False, {})["remote"]
    assert cfg.timeout == 34 and cfg.startup_timeout == 12
    monkeypatch.setenv("TEST_MCP_BEARER", "late-secret")
    monkeypatch.setenv("TEST_MCP_HEADER", "late-header")
    headers = mcp.connection_headers(cfg.values)
    assert headers == {"Authorization": "Bearer late-secret", "X-Token": "late-header"}
    assert "late-secret" not in json.dumps(cfg.values) + caplog.text
    assert "late-header" not in json.dumps(cfg.values) + caplog.text
    with pytest.raises(ValueError, match="HTTPS"):
        mcp.connection_headers({**cfg.values, "url": "http://example.com"})


@pytest.mark.asyncio
async def test_management_rejects_invalid_names_before_write(tmp_path):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    manager.loaded.set()
    output = StringIO()
    await command(manager, SimpleNamespace(console=Console(file=output)), "add evil__server unused")
    assert not (tmp_path / ".config/orcha-agent/mcp.json").exists()
    assert "must not contain __" in output.getvalue()


@pytest.mark.asyncio
async def test_pagination_and_reconnect_flag_during_initialization(tmp_path, monkeypatch):
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    conn = Connection(ServerConfig("paged", {"type": "stdio", "command": "unused"}, tmp_path))
    entered = 0
    cursors = []

    @asynccontextmanager
    async def transport(*args, **kwargs):
        yield (None, None)

    @asynccontextmanager
    async def client(*args, **kwargs):
        nonlocal entered
        entered += 1

        async def initialize():
            conn.reconnect.set()  # A request made while the existing attempt was connecting.

        async def list_tools(params=None):
            cursor = params.cursor if params else None
            cursors.append(cursor)
            return types.ListToolsResult(
                tools=[types.Tool(name="first" if cursor is None else "second", input_schema={})],
                next_cursor="page-2" if cursor is None else None,
            )

        yield SimpleNamespace(initialize=initialize, list_tools=list_tools)

    monkeypatch.setattr(mcp, "stdio_client", transport)
    monkeypatch.setattr(mcp, "ClientSession", client)
    conn.task = asyncio.create_task(manager.run(conn))
    manager.connections["paged"] = conn
    try:
        await asyncio.wait_for(conn.ready.wait(), 2)
        await asyncio.sleep(0.05)
        assert cursors == [None, "page-2"]
        assert entered == 1 and not conn.reconnect.is_set()
        assert set(manager.api.tools) == {"mcp__paged__first", "mcp__paged__second"}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_real_stdio_deferred_live_reconnect_and_private_stderr_tail(tmp_path):
    script, pidfile = tmp_path / "server.py", tmp_path / "pid"
    script.write_text("""import os, sys, time
from pathlib import Path
from mcp.server.mcpserver import MCPServer
Path(sys.argv[1]).write_text(str(os.getpid()))
for i in range(75):
    print(f"diagnostic-{i}", file=sys.stderr)
print(os.environ["TEST_SERVER_SECRET"], file=sys.stderr, flush=True)
time.sleep(0.2)
server = MCPServer("restartable")
@server.tool()
def echo(value: str) -> str:
    return value
server.run(transport="stdio")
""")
    values = {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(script), str(pidfile)],
        "env": {"TEST_SERVER_SECRET": "redaction-control"},
        "timeout": 4,
    }
    source = tmp_path / ".config/orcha-agent/mcp.json"
    write(source, {"mcpServers": {"live": values}})
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    cached = Connection(
        ServerConfig("live", values, source),
        tools=[
            {
                "name": "echo",
                "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        ],
    )
    manager.write_cache(cached)
    await manager.start()
    conn = manager.connections["live"]
    deferred = manager.api.tools["mcp__live__echo"]
    assert not conn.ready.is_set()
    try:
        # Invocation starts while the tool is deferred and completes on the live session.
        assert await deferred.ainvoke({"value": "mid-wait"}) == "mid-wait"
        pid = int(pidfile.read_text())
        os.kill(pid, signal.SIGKILL)
        # EOF reaches the MCP session before this call, or while it is waiting for a reply.
        await manager.call("live", "echo", {"value": "detect-death"})
        async with asyncio.timeout(10):
            while (
                not pidfile.exists()
                or int(pidfile.read_text() or pid) == pid
                or not conn.ready.is_set()
            ):
                await asyncio.sleep(0.02)
        assert await deferred.ainvoke({"value": "after-restart"}) == "after-restart"
    finally:
        await manager.close()
    logfile = manager.cache_path("live").with_suffix(".log")
    log = logfile.read_text()
    assert len(log.splitlines()) <= 50
    assert "diagnostic-74" in log and "diagnostic-0\n" not in log
    assert "redaction-control" not in log and "[redacted]" in log
    assert logfile.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_stdio_retry_backoff_before_eventual_connection(tmp_path):
    script, attempts = tmp_path / "retry_server.py", tmp_path / "attempts"
    script.write_text("""import sys, time
from pathlib import Path
attempts = Path(sys.argv[1])
previous = attempts.read_text().splitlines() if attempts.exists() else []
with attempts.open("a") as output:
    output.write(str(time.monotonic()) + "\\n")
if len(previous) < 2:
    sys.exit(1)
from mcp.server.mcpserver import MCPServer
server = MCPServer("recovered")
@server.tool()
def echo() -> str:
    return "recovered"
server.run(transport="stdio")
""")
    write(
        tmp_path / ".config/orcha-agent/mcp.json",
        {
            "mcpServers": {
                "retry": {
                    "command": sys.executable,
                    "args": [str(script), str(attempts)],
                    "timeout": 3,
                }
            }
        },
    )
    manager = MCPManager(API(), tmp_path, tmp_path, False, {})
    await manager.start()
    try:
        await asyncio.wait_for(manager.connections["retry"].ready.wait(), 8)
        starts = [float(value) for value in attempts.read_text().splitlines()]
        assert len(starts) == 3
        assert starts[1] - starts[0] >= 0.25
        assert starts[2] - starts[1] >= 0.5
        assert await manager.call("retry", "echo", {}) == "recovered"
    finally:
        await manager.close()


def test_native_config_write_private_at_creation(tmp_path, monkeypatch):
    from orcha_agent.builtin.mcp import _write

    path = tmp_path / ".orcha-agent/mcp.json"
    real_replace = type(path).replace
    seen_modes = []

    def observe_replace(temporary, destination):
        seen_modes.append(temporary.stat().st_mode & 0o777)
        assert temporary.read_text().find("private-token") >= 0
        return real_replace(temporary, destination)

    monkeypatch.setattr(type(path), "replace", observe_replace)
    old_umask = os.umask(0)
    try:
        _write(path, "secret", {"headers": {"X-Key": "private-token"}})
    finally:
        os.umask(old_umask)
    assert seen_modes == [0o600]
    assert path.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in path.parent.iterdir()) == ["mcp.json"]


@pytest.mark.asyncio
async def test_http_url_userinfo_is_an_implicit_authorization_header():
    import httpx2

    observed = []

    async def receive(request):
        observed.append((request.url.scheme, "authorization" in request.headers))
        return httpx2.Response(200, json={})

    url = "http://user:fixture-password@example.com/mcp"
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(receive)) as client:
        await client.post(url)
    assert observed == [("http", True)]
    with pytest.raises(ValueError, match="HTTPS"):
        mcp.validate_server("remote", {"url": url})
    with pytest.raises(ValueError, match="HTTPS"):
        mcp.connection_headers({"url": url})
    assert mcp.validate_server("local", {"url": "http://user:fixture@127.0.0.1/mcp"})


@pytest.mark.asyncio
async def test_https_url_credentials_remain_supported_as_basic_auth():
    import httpx2

    observed = []

    async def receive(request):
        observed.append((request.url.scheme, request.headers["authorization"].startswith("Basic ")))
        return httpx2.Response(200)

    values = mcp.validate_server(
        "remote", {"url": "https://fixture:fixture-password@example.com/mcp"}
    )
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(receive), headers=mcp.connection_headers(values)
    ) as client:
        await client.post(values["url"])
    assert observed == [("https", True)]
