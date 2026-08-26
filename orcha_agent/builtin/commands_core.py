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


async def _auth_action(ctx: Any, args: str, *, action: str) -> None:
    parts = args.split()
    if len(parts) != 1:
        ctx.console.error(f"Usage: /{action} <prefix>")
        return
    prefix = parts[0]
    registration = ctx.registry.auth.get(prefix)
    if registration is None:
        ctx.console.error(f"Unknown auth prefix: {prefix}")
        return
    callback = (
        registration.flow.login
        if action == "login"
        else registration.flow.logout
    )
    await callback(ctx)


async def _login(ctx: Any, args: str) -> None:
    await _auth_action(ctx, args, action="login")


async def _logout(ctx: Any, args: str) -> None:
    await _auth_action(ctx, args, action="logout")


def _provider_flags(capabilities: Any) -> str:
    return " ".join(
        symbol if enabled else "-"
        for symbol, enabled in (
            ("T", capabilities.tool_calling),
            ("S", capabilities.streaming),
            ("R", capabilities.thinking),
            ("O", capabilities.structured_output),
        )
    )


def _provider_auth_or_keys(ctx: Any, prefix: str, provider: Any) -> str:
    auth = ctx.registry.auth.get(prefix)
    if auth is not None:
        return auth.flow.status()
    return ", ".join(
        f"{key}: {'yes' if key in os.environ else 'no'}"
        for key in provider.env_keys
    ) or "n/a"


async def _providers(ctx: Any, args: str) -> None:
    selected = args.strip()
    providers = ctx.registry.providers
    if selected:
        provider = providers.get(selected)
        if provider is None:
            ctx.console.error(f"Unknown provider prefix: {selected}")
            return
        unavailable = provider.available()
        detail = Table(title=f"Provider: {selected}")
        detail.add_column("Field", style="cyan", no_wrap=True)
        detail.add_column("Value", overflow="fold")
        detail.add_row("Available", "yes" if unavailable is None else "no")
        detail.add_row(
            "Auth / Keys",
            _provider_auth_or_keys(ctx, selected, provider),
        )
        detail.add_row("T/S/R/O", _provider_flags(provider.capabilities))
        detail.add_row("Status", unavailable or "ready")
        detail.add_row("Models", ", ".join(provider.models) or "provider-defined")
        ctx.console.print(detail)
        return

    width = int(getattr(ctx.console.console, "width", 80))
    table = Table(title="Providers", padding=(0, 0))
    table.add_column("Prefix", style="cyan", min_width=20, no_wrap=True)
    table.add_column("Available", no_wrap=True)
    table.add_column("Auth / Keys", overflow="fold")
    table.add_column("T/S/R/O", no_wrap=True)
    table.add_column("Status", overflow="fold")
    for prefix, provider in sorted(providers.items()):
        unavailable = provider.available()
        status = unavailable or "ready"
        if width >= 120 and provider.models:
            status = f"{status}\nmodels: {', '.join(provider.models)}"
        table.add_row(
            prefix,
            "yes" if unavailable is None else "no",
            _provider_auth_or_keys(ctx, prefix, provider),
            _provider_flags(provider.capabilities),
            status,
        )
    ctx.console.print(table)


def register(api: PluginAPI) -> None:
    api.add_command("help", _help, help="List available slash commands")
    api.add_command("clear", _clear, help="Clear the current conversation")
    api.add_command("exit", _exit, help="Exit orcha-agent")
    api.add_command("plugins", _plugins, help="List loaded plugins")
    api.add_command("providers", _providers, help="List model providers and availability")
    api.add_command("login", _login, help="Log in to a provider: /login <prefix>")
    api.add_command("logout", _logout, help="Log out of a provider: /logout <prefix>")
