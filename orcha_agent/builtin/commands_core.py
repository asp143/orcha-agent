"""Core slash commands."""

from __future__ import annotations

import os
from typing import Any

from rich.table import Table

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="commands_core", version="1.0.0")


def _plugin_value(record: object, name: str, default: str = "") -> str:
    value = getattr(record, name, default)
    return default if value is None else str(value)


async def _help(ctx: Any, _args: str) -> None:
    table = Table(title="Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Help")
    for name, command in sorted(ctx.registry.commands.items()):
        table.add_row(f"/{name}", command.help)
    ctx.console.print(table)


async def _clear(ctx: Any, _args: str) -> None:
    await ctx.clear()


async def _exit(ctx: Any, _args: str) -> None:
    ctx.exit_requested = True


async def _plugins(ctx: Any, _args: str) -> None:
    table = Table(title="Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Status")
    for record in ctx.plugins:
        status = _plugin_value(record, "status", "loaded")
        error = _plugin_value(record, "error")
        if error:
            status = f"{status}: {error}"
        table.add_row(
            _plugin_value(record, "name"),
            _plugin_value(record, "version", "0"),
            _plugin_value(record, "source"),
            status,
        )
    ctx.console.print(table)


def _capabilities(value: object) -> str:
    fields = ("tool_calling", "streaming", "thinking", "structured_output", "max_context")
    return ", ".join(
        f"{field}={getattr(value, field)}" for field in fields if hasattr(value, field)
    )


async def _providers(ctx: Any, _args: str) -> None:
    table = Table(title="Providers")
    table.add_column("Prefix", style="cyan")
    table.add_column("Available")
    table.add_column("Environment")
    table.add_column("Capabilities")
    for prefix, provider in sorted(ctx.registry.providers.items()):
        unavailable = provider.available()
        availability = "yes" if unavailable is None else f"no ({unavailable})"
        environment = ", ".join(
            f"{key}: {'yes' if key in os.environ else 'no'}" for key in provider.env_keys
        ) or "n/a"
        table.add_row(
            prefix,
            availability,
            environment,
            _capabilities(provider.capabilities),
        )
    ctx.console.print(table)


def register(api: PluginAPI) -> None:
    api.add_command("help", _help, help="List available slash commands")
    api.add_command("clear", _clear, help="Clear the current conversation")
    api.add_command("exit", _exit, help="Exit orcha-agent")
    api.add_command("plugins", _plugins, help="List loaded plugins")
    api.add_command("providers", _providers, help="List model providers and availability")
