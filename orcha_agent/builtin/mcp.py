"""MCP tools and runtime management, registered solely through PluginAPI."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any

from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.mcp import MCPManager, load_servers, private_write, validate_server

PLUGIN = PluginSpec(name="mcp", version="1.0.0")


def _write(path: Path, name: str, value: dict[str, Any] | None) -> None:
    document = json.loads(path.read_text()) if path.exists() else {}
    entries = document.setdefault("mcpServers", {})
    if value is None:
        entries.pop(name, None)
    else:
        entries[name] = value
    private_write(path, json.dumps(document, indent=2) + "\n")


async def command(manager: MCPManager, ctx: Any, args: str) -> None:
    try:
        words = shlex.split(args)
        action = words[0] if words else "list"
        await manager.loaded.wait()
        if action == "list":
            from orcha_agent.tui.panels import summary_panel

            rows = [
                (name, connection.status, str(len(connection.tools)), connection.error or "")
                for name, connection in manager.connections.items()
            ]
            if manager.error:
                rows.append(("Discovery", "error", "", manager.error))
            ctx.console.print(
                summary_panel(
                    "MCP servers",
                    ("Name", "Status", "Tools", "Details"),
                    rows or [("No MCP servers configured.", "", "", "")],
                )
            )
            return
        if action == "reload":
            await manager.reload()
            ctx.console.print("MCP configuration reloaded.")
            return
        if action == "reconnect" and len(words) == 1:
            for connection in manager.connections.values():
                connection.reconnect.set()
            ctx.console.print("MCP reconnect requested.")
            return
        if len(words) < 2:
            raise ValueError(
                "Usage: /mcp list|add|remove|enable|disable|reload|reconnect|test|resources|prompts [name]"
            )
        name = words[1]
        default_path = (
            manager.cwd / ".orcha-agent/mcp.json"
            if manager.trusted
            else manager.home / ".config/orcha-agent/mcp.json"
        )
        if action == "add":
            rest = words[2:]
            if name == "--url":
                if len(rest) != 2:
                    raise ValueError("Usage: /mcp add --url <url> <name>")
                url, name = rest
                values = {"type": "http", "url": url}
            elif rest and rest[0] == "--url" and len(rest) == 2:
                values = {"type": "http", "url": rest[1]}
            elif rest:
                values = {"type": "stdio", "command": rest[0], "args": rest[1:]}
            else:
                raise ValueError("Usage: /mcp add <name> <command...> or <name> --url <url>")
            if values["type"] == "http" and not str(values["url"]).startswith(
                ("http://", "https://")
            ):
                raise ValueError("MCP URL must use http:// or https://")
            values = validate_server(name, values)
            await asyncio.to_thread(_write, default_path, name, values)
            await manager.reload()
            ctx.console.print(f"Added MCP server {name}.", markup=False)
            return
        if name not in manager.connections:
            raise ValueError(f"Unknown MCP server: {name}")
        connection = manager.connections[name]
        if action in {"remove", "enable", "disable"}:
            path = connection.config.source
            values = dict(connection.config.values)
            native_paths = {
                manager.home / ".config/orcha-agent/mcp.json",
                manager.cwd / ".orcha-agent/mcp.json",
            }
            if path not in native_paths:
                path = default_path
                # Native overrides mask imports without ever rewriting foreign files.
                if action == "remove":
                    values = {"_orcha_removed": True}
                else:
                    values["enabled"] = action == "enable"
            elif action == "remove":
                values = None
            else:
                values["enabled"] = action == "enable"
            await asyncio.to_thread(_write, path, name, values)
            if action == "remove":
                remaining = await asyncio.to_thread(
                    load_servers, manager.cwd, manager.home, manager.trusted, manager.settings
                )
                if name in remaining:
                    await asyncio.to_thread(_write, default_path, name, {"_orcha_removed": True})
            await manager.reload()
            ctx.console.print(f"MCP server {name}: {action} complete.", markup=False)
            return
        if action == "reconnect":
            connection.reconnect.set()
            ctx.console.print(f"Reconnecting {name}.", markup=False)
            return
        if action not in {"test", "resources", "prompts"}:
            raise ValueError(f"Unknown MCP action: {action}")
        if not connection.config.enabled:
            raise ValueError(f"MCP server {name} is disabled")
        if action == "test" and connection.error:
            ctx.console.print(f"{name}: {connection.error}", markup=False)
        async with asyncio.timeout(connection.config.timeout):
            await connection.ready.wait()
            if action == "test":
                await connection.session.send_ping()
                ctx.console.print(
                    f"{name}: connected, {len(connection.tools)} tools.", markup=False
                )
            else:
                from mcp import types

                cursor = None
                while True:
                    params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                    result = await (
                        connection.session.list_resources(params=params)
                        if action == "resources"
                        else connection.session.list_prompts(params=params)
                    )
                    for item in getattr(result, action):
                        ctx.console.print(item.model_dump_json(), markup=False)
                    cursor = result.next_cursor
                    if not cursor:
                        break
    except ValueError as exc:
        ctx.console.print(str(exc), markup=False)
    except Exception as exc:
        # Transport errors can include headers/URLs; surface only the error class.
        ctx.console.print(f"MCP operation failed ({type(exc).__name__}).", markup=False)


def register(api: PluginAPI) -> None:
    manager: MCPManager | None = None
    context: Any = None

    async def start(event: AppStart) -> None:
        nonlocal manager
        cfg = getattr(event.ctx, "cfg", None)
        if cfg is None or getattr(cfg, "cwd", None) is None:
            return
        manager = MCPManager(
            api, Path(cfg.cwd), Path.home(), bool(getattr(cfg, "trust_cwd", False)), api.config
        )
        manager.startup = asyncio.create_task(manager.start())

    async def before_build(event: AgentBuildBefore) -> None:
        if manager is None:
            return

        active_manager = manager

        async def gate() -> None:
            await active_manager.loaded.wait()
            await asyncio.gather(
                *(c.ready.wait() for c in active_manager.connections.values() if c.config.enabled)
            )

        if not manager.gated:
            manager.gated = True
            try:
                await asyncio.wait_for(gate(), 0.25)
            except TimeoutError:
                pass
        # Tools discovered during the gate were not in the pre-hook registry snapshot.
        tools = event.kwargs.setdefault("tools", [])
        existing = {getattr(tool, "name", "") for tool in tools}
        cfg = getattr(context, "cfg", None)
        # Never broaden plan/custom allowlists; registry filtering is owned by the kernel.
        if cfg is not None:
            for name in manager.names - existing:
                scope = getattr(event, "tool_scope", None)
                if scope is None or name in scope:
                    tools.append(context.registry.tools[name])
        interrupts = event.kwargs.setdefault("interrupt_on", {})
        policy = event.mode_interrupt_on
        if policy is None and cfg is not None:
            policy = context.registry.modes[cfg.mode].interrupt_on
        if policy and (policy.get("execute") or policy.get("bash")):
            for tool in tools:
                if getattr(tool, "name", "") in manager.names and tool.name not in getattr(
                    event, "always_allowed", ()
                ):
                    interrupts[tool.name] = True

    async def remember(event: AppStart) -> None:
        nonlocal context
        context = event.ctx
        await start(event)

    async def stop(_event: AppExit) -> None:
        if manager:
            await manager.close()

    async def manage(ctx: Any, args: str) -> None:
        if manager:
            await command(manager, ctx, args)

    def status(_ctx: Any) -> str | None:
        if manager is None or not manager.connections:
            return None
        active = [c for c in manager.connections.values() if c.config.enabled]
        connected = sum(c.status == "connected" for c in active)
        return f"MCP {connected}/{len(active)}"

    api.on(AppStart, remember)
    api.on(AgentBuildBefore, before_build)
    api.on(AppExit, stop)
    api.add_command(
        "mcp",
        manage,
        help="Manage MCP servers (list/add/remove/enable/disable/reload/reconnect/test/resources/prompts)",
    )
    api.add_status_segment("mcp", status, priority=80)
