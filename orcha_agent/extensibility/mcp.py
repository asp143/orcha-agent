"""MCP configuration, task-owned connections, cache, and LangChain tool bridge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tomllib
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from deepagents.backends.utils import truncate_if_too_long
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


def tool_name(server: str, name: str) -> str:
    raw = f"mcp__{server}__{name}"
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if clean != raw or len(clean) > 64:
        clean = clean[:53] + "_" + hashlib.sha256(raw.encode()).hexdigest()[:10]
    return clean


@dataclass
class ServerConfig:
    name: str
    values: dict[str, Any]
    source: Path

    @property
    def timeout(self) -> float:
        return max(0.01, float(self.values.get("timeout", 30)))

    @property
    def enabled(self) -> bool:
        return self.values.get("enabled", True) is not False

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.values, sort_keys=True).encode()).hexdigest()


def load_servers(cwd: Path, home: Path, trusted: bool, settings: Any) -> dict[str, ServerConfig]:
    paths: list[Path] = []
    if settings.get("import_codex", True):
        paths.append(home / ".codex/config.toml")
    paths.append(home / ".config/orcha-agent/mcp.json")
    if trusted:
        if settings.get("import_claude", True):
            paths.append(cwd / ".mcp.json")
        paths.append(cwd / ".orcha-agent/mcp.json")
    servers: dict[str, ServerConfig] = {}
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        document = tomllib.loads(raw.decode()) if path.suffix == ".toml" else json.loads(raw)
        entries = document.get("mcp_servers" if path.suffix == ".toml" else "mcpServers", {})
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid MCP server table in {path}")
        for name, values in entries.items():
            if not isinstance(values, dict):
                raise ValueError(f"Invalid MCP server {name}")
            values = dict(values)
            if values.get("_orcha_removed") is True:
                servers.pop(name, None)
                continue
            values.setdefault("type", "http" if values.get("url") else "stdio")
            if values["type"] not in {"stdio", "http", "sse"}:
                raise ValueError(f"Unsupported MCP transport for {name}")
            if values["type"] == "stdio" and not isinstance(values.get("command"), str):
                raise ValueError(f"Missing MCP command for {name}")
            if values["type"] != "stdio" and not str(values.get("url", "")).startswith(
                ("http://", "https://")
            ):
                raise ValueError(f"Invalid MCP URL for {name}")
            if "http_headers" in values and "headers" not in values:
                values["headers"] = values["http_headers"]
            servers[name] = ServerConfig(name, values, path)
    return servers


def flatten_result(result: Any) -> str:
    lines: list[str] = []
    if getattr(result, "is_error", False):
        lines.append("MCP tool error:")
    for block in result.content:
        kind = getattr(block, "type", "")
        if kind == "text":
            lines.append(block.text)
        elif kind == "image":
            lines.append(f"[Image: {block.mime_type}; image data omitted]")
        elif kind == "resource":
            resource = block.resource
            lines.append(getattr(resource, "text", f"[Resource: {resource.uri}]"))
        elif kind == "resource_link":
            lines.append(f"[Resource: {block.name}] {block.uri}")
        else:
            lines.append(f"[{kind or 'MCP content'} omitted]")
    return str(truncate_if_too_long("\n".join(lines)))


@dataclass
class Connection:
    config: ServerConfig
    session: Any = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    reconnect: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    published: set[str] = field(default_factory=set)
    status: str = "connecting"
    error: str | None = None


class MCPManager:
    def __init__(self, api: Any, cwd: Path, home: Path, trusted: bool, settings: Any):
        self.api, self.cwd, self.home, self.trusted = api, cwd, home, trusted
        self.settings = settings
        self.connections: dict[str, Connection] = {}
        self.names: set[str] = set()
        self.loaded = asyncio.Event()
        self.startup: asyncio.Task[None] | None = None
        self.closed = False
        self.error: str | None = None
        self.gated = False

    def cache_path(self, name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        if safe != name:
            safe += "_" + hashlib.sha256(name.encode()).hexdigest()[:10]
        return self.home / ".cache/orcha-agent/mcp" / f"{safe}.json"

    def read_cache(self, config: ServerConfig) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.cache_path(config.name).read_text())
            if isinstance(value, dict) and value.get("fingerprint") == config.fingerprint:
                definitions = value.get("tools")
                if isinstance(definitions, list) and all(
                    isinstance(d, dict)
                    and isinstance(d.get("name"), str)
                    and isinstance(d.get("inputSchema", {}), dict)
                    for d in definitions
                ):
                    return definitions
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return []

    def write_cache(self, connection: Connection) -> None:
        path = self.cache_path(connection.config.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"fingerprint": connection.config.fingerprint, "tools": connection.tools})
        )
        temporary.replace(path)

    def _caller(self, server: str, original: str):
        async def call(**arguments: Any) -> str:
            return await self.call(server, original, arguments)

        return call

    def publish(self, connection: Connection) -> None:
        current = {
            tool_name(connection.config.name, definition["name"]) for definition in connection.tools
        }
        for obsolete in connection.published - current:
            self.api.remove_tool(obsolete)
            self.names.discard(obsolete)
        connection.published = current
        for definition in connection.tools:
            name = tool_name(connection.config.name, definition["name"])
            original = definition["name"]
            server = connection.config.name

            self.api.add_tool(
                StructuredTool.from_function(
                    coroutine=self._caller(server, original),
                    name=name,
                    description=definition.get("description") or f"MCP {server}: {original}",
                    args_schema=definition.get("inputSchema", {"type": "object", "properties": {}}),
                ),
                replace=name in self.names,
            )
            self.names.add(name)
        self.api.request_rebuild()

    async def start(self) -> None:
        try:
            configs = await asyncio.to_thread(
                load_servers, self.cwd, self.home, self.trusted, self.settings
            )
            for config in configs.values():
                connection = Connection(config)
                self.connections[config.name] = connection
                if not config.enabled:
                    connection.status = "disabled"
                    continue
                connection.tools = await asyncio.to_thread(self.read_cache, config)
                self.publish(connection)
                connection.task = asyncio.create_task(self.run(connection))
        except Exception as exc:
            self.error = f"MCP configuration failed ({type(exc).__name__})"
        finally:
            self.loaded.set()

    async def run(self, connection: Connection) -> None:
        delay = 0.25
        config = connection.config
        while not self.closed:
            try:
                connection.status = "connecting"
                connection.reconnect.clear()
                async with AsyncExitStack() as stack:
                    values = config.values
                    if values["type"] == "stdio":
                        # Keep server diagnostics out of the terminal: they may contain credentials.
                        errlog = stack.enter_context(open(os.devnull, "w"))
                        transport = stdio_client(
                            StdioServerParameters(
                                command=values["command"],
                                args=values.get("args", []),
                                env=values.get("env"),
                                cwd=str(self.cwd),
                            ),
                            errlog=errlog,
                        )
                    elif values["type"] == "sse":
                        transport = sse_client(
                            values["url"], headers=values.get("headers"), timeout=config.timeout
                        )
                    else:
                        client = await stack.enter_async_context(
                            httpx2.AsyncClient(
                                headers=values.get("headers"), timeout=config.timeout
                            )
                        )
                        transport = streamable_http_client(values["url"], http_client=client)
                    streams = await stack.enter_async_context(transport)
                    session = await stack.enter_async_context(
                        ClientSession(streams[0], streams[1], read_timeout_seconds=config.timeout)
                    )
                    async with asyncio.timeout(config.timeout):
                        await session.initialize()
                        definitions = []
                        cursor = None
                        from mcp import types

                        while True:
                            result = await session.list_tools(
                                params=types.PaginatedRequestParams(cursor=cursor)
                                if cursor
                                else None
                            )
                            definitions.extend(
                                tool.model_dump(by_alias=True) for tool in result.tools
                            )
                            cursor = result.next_cursor
                            if not cursor:
                                break
                    connection.tools = definitions
                    connection.session = session
                    connection.status, connection.error = "connected", None
                    connection.ready.set()
                    self.publish(connection)
                    try:
                        await asyncio.to_thread(self.write_cache, connection)
                    except OSError:
                        pass
                    delay = 0.25
                    while not connection.reconnect.is_set():
                        try:
                            await asyncio.wait_for(connection.reconnect.wait(), 15)
                        except TimeoutError:
                            await session.send_ping()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                connection.error = type(exc).__name__
                connection.status = "disconnected"
            finally:
                connection.session = None
                connection.ready.clear()
            if not self.closed:
                try:
                    await asyncio.wait_for(connection.reconnect.wait(), delay)
                except TimeoutError:
                    pass
                delay = min(delay * 2, 30)

    async def call(self, server: str, name: str, arguments: dict[str, Any]) -> str:
        connection = self.connections.get(server)
        if connection is None or not connection.config.enabled:
            return "MCP server is disabled or removed."
        try:
            async with asyncio.timeout(connection.config.timeout):
                await connection.ready.wait()
                result = await connection.session.call_tool(name, arguments)
            return flatten_result(result)
        except TimeoutError:
            return f"MCP tool timed out after {connection.config.timeout:g}s."
        except Exception as exc:
            connection.reconnect.set()
            return f"MCP tool failed ({type(exc).__name__}); reconnecting."

    async def close(self) -> None:
        self.closed = True
        tasks = [c.task for c in self.connections.values() if c.task]
        if self.startup and not self.startup.done():
            tasks.append(self.startup)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def reload(self) -> None:
        await self.close()
        for name in self.names:
            self.api.remove_tool(name)
        self.names.clear()
        self.connections.clear()
        self.closed = False
        self.loaded.clear()
        self.gated = False
        self.error = None
        await self.start()
        self.api.request_rebuild()
