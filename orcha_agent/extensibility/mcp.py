"""MCP configuration, task-owned connections, cache, and LangChain tool bridge."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
import threading
import tomllib
from collections import deque
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx2
from deepagents.backends.utils import truncate_if_too_long
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

logger = logging.getLogger(__name__)


def tool_name(server: str, name: str) -> str:
    if "__" in server:
        raise ValueError("MCP server names must not contain __")
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
    def startup_timeout(self) -> float:
        return float(self.values.get("startup_timeout_sec", self.timeout))

    @property
    def enabled(self) -> bool:
        return self.values.get("enabled", True) is not False

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.values, sort_keys=True).encode()).hexdigest()


def validate_server(name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate one entry without exposing configuration secrets."""
    if not isinstance(name, str) or not name or "__" in name:
        raise ValueError("Server names must be nonempty and must not contain __")
    if not isinstance(values, dict):
        raise ValueError("Expected a server object")
    values = dict(values)
    if values.get("_orcha_removed") is True:
        return values
    values.setdefault("type", "http" if values.get("url") else "stdio")
    if values["type"] not in {"stdio", "http", "sse"}:
        raise ValueError("Unsupported transport")
    if values["type"] == "stdio" and (
        not isinstance(values.get("command"), str) or not values["command"]
    ):
        raise ValueError("Missing command")
    if "args" in values and (
        not isinstance(values["args"], list)
        or not all(isinstance(arg, str) for arg in values["args"])
    ):
        raise ValueError("Arguments must be strings")
    if "http_headers" in values and "headers" not in values:
        values["headers"] = values["http_headers"]
    for key in ("env", "headers", "env_http_headers"):
        if key in values and (
            not isinstance(values[key], dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in values[key].items())
        ):
            raise ValueError(f"{key} must map strings to strings")
    if "bearer_token_env_var" in values and not isinstance(values["bearer_token_env_var"], str):
        raise ValueError("bearer_token_env_var must be a string")
    for key in ("timeout", "startup_timeout_sec", "tool_timeout_sec"):
        if key in values:
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a positive number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value <= 0:
                raise ValueError(f"{key} must be a positive number")
    if "timeout" not in values:
        if "tool_timeout_sec" in values:
            values["timeout"] = values["tool_timeout_sec"]
        elif "startup_timeout_sec" in values:
            values["timeout"] = values["startup_timeout_sec"]
    if "enabled" in values and not isinstance(values["enabled"], bool):
        raise ValueError("enabled must be a boolean")
    if values["type"] != "stdio":
        validate_http(values)
    return values


def validate_http(values: dict[str, Any]) -> None:
    url = values.get("url")
    if not isinstance(url, str):
        raise ValueError("Invalid MCP URL")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        raise ValueError("Invalid MCP URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Invalid MCP URL")
    credentials = parsed.username is not None or any(
        values.get(key) for key in ("headers", "env_http_headers", "bearer_token_env_var")
    )
    if parsed.scheme == "http" and credentials:
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback:
            raise ValueError("MCP headers require HTTPS except on loopback")


def connection_headers(values: dict[str, Any]) -> dict[str, str]:
    """Resolve environment-backed credentials only when opening a transport."""
    headers = dict(values.get("headers", {}))
    for header, variable in values.get("env_http_headers", {}).items():
        if variable not in os.environ:
            raise ValueError("Missing MCP header environment variable")
        headers[header] = os.environ[variable]
    if variable := values.get("bearer_token_env_var"):
        if variable not in os.environ:
            raise ValueError("Missing MCP bearer environment variable")
        headers["Authorization"] = "Bearer " + os.environ[variable]
    validate_http({**values, "headers": headers})
    return headers


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
        try:
            raw = path.read_bytes()
            document = tomllib.loads(raw.decode()) if path.suffix == ".toml" else json.loads(raw)
            entries = document.get("mcp_servers" if path.suffix == ".toml" else "mcpServers", {})
            if not isinstance(entries, dict):
                raise ValueError("Invalid server table")
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Invalid MCP configuration in %s (%s)", path, type(exc).__name__)
            continue
        for name, entry in entries.items():
            try:
                values = validate_server(name, entry)
            except (ValueError, TypeError) as exc:
                logger.warning("Invalid MCP server %r in %s: %s", name, path, exc)
                continue
            if values.get("_orcha_removed") is True:
                servers.pop(name, None)
            else:
                servers[name] = ServerConfig(name, values, path)
    return servers


def private_write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def stderr_log(path: Path, secrets: list[str]):
    """Drain subprocess stderr into a private, bounded tail without terminal output."""
    private_write(path, "")
    read_fd, write_fd = os.pipe()
    output = os.fdopen(write_fd, "w")

    def capture() -> None:
        lines: deque[str] = deque(maxlen=50)
        pending = b""

        def redact(text: str) -> str:
            for secret in secrets:
                if secret:
                    text = text.replace(secret, "[redacted]")
            return text

        try:
            while chunk := os.read(read_fd, 4096):
                parts = (pending + chunk).split(b"\n")
                pending = parts.pop()[-8192:]
                lines.extend(redact(part.decode(errors="replace")[-8192:]) for part in parts)
                private_write(path, "\n".join(lines) + ("\n" if lines else ""))
            if pending:
                lines.append(redact(pending.decode(errors="replace")))
            private_write(path, "\n".join(lines) + ("\n" if lines else ""))
        except OSError:
            logger.warning("Unable to capture MCP stderr in %s", path)
        finally:
            os.close(read_fd)

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    try:
        yield output
    finally:
        output.close()
        # The transport closes its subprocess before this outer context exits.
        thread.join(timeout=1)


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
        self.owners: dict[str, tuple[str, str]] = {}
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
        private_write(
            path,
            json.dumps({"fingerprint": connection.config.fingerprint, "tools": connection.tools}),
        )

    def _caller(self, server: str, original: str):
        async def call(**arguments: Any) -> str:
            return await self.call(server, original, arguments)

        return call

    def publish(self, connection: Connection) -> None:
        current: set[str] = set()
        server = connection.config.name
        for definition in connection.tools:
            original = definition["name"]
            name = tool_name(server, original)
            owner = (server, original)
            if name in current or (name in self.owners and self.owners[name] != owner):
                logger.warning("Refusing duplicate MCP tool %r from server %r", name, server)
                continue
            self.api.add_tool(
                StructuredTool.from_function(
                    coroutine=self._caller(server, original),
                    name=name,
                    description=definition.get("description") or f"MCP {server}: {original}",
                    args_schema=definition.get("inputSchema", {"type": "object", "properties": {}}),
                ),
                replace=name in self.owners,
            )
            self.owners[name] = owner
            self.names.add(name)
            current.add(name)
        for obsolete in connection.published - current:
            if self.owners.get(obsolete, (None,))[0] == server:
                self.api.remove_tool(obsolete)
                self.names.discard(obsolete)
                self.owners.pop(obsolete, None)
        connection.published = current
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
                        errlog = stack.enter_context(
                            stderr_log(
                                self.cache_path(config.name).with_suffix(".log"),
                                list(values.get("env", {}).values()),
                            )
                        )
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
                            values["url"],
                            headers=connection_headers(values),
                            timeout=config.timeout,
                        )
                    else:
                        client = await stack.enter_async_context(
                            httpx2.AsyncClient(
                                headers=connection_headers(values), timeout=config.timeout
                            )
                        )
                        transport = streamable_http_client(values["url"], http_client=client)
                    streams = await stack.enter_async_context(transport)
                    session = await stack.enter_async_context(
                        ClientSession(
                            streams[0],
                            streams[1],
                            read_timeout_seconds=max(config.timeout, config.startup_timeout),
                        )
                    )
                    async with asyncio.timeout(config.startup_timeout):
                        await session.initialize()
                        definitions = []
                        cursor = None
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
                    connection.reconnect.clear()
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
            connection.reconnect.set()
            return f"MCP tool timed out after {connection.config.timeout:g}s."
        except MCPError as exc:
            if exc.code in {types.CONNECTION_CLOSED, types.REQUEST_TIMEOUT}:
                connection.reconnect.set()
            return str(exc)
        except (
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            ConnectionError,
            OSError,
        ) as exc:
            connection.reconnect.set()
            return f"MCP tool failed ({type(exc).__name__}); reconnecting."
        except Exception as exc:
            return f"MCP tool failed ({type(exc).__name__})."

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
        self.owners.clear()
        self.connections.clear()
        self.closed = False
        self.loaded.clear()
        self.gated = False
        self.error = None
        await self.start()
        self.api.request_rebuild()
