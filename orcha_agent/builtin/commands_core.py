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


async def _help(ctx: Any, args: str) -> None:
    if args.strip():
        ctx.console.error("Usage: /help")
        return
    ui = getattr(ctx, "ui", None)
    if ui is not None and hasattr(ui, "show"):
        try:
            await ui.show("help")
            return
        except RuntimeError:
            pass
    table = Table(title="Commands")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Help")
    for name, command in sorted(ctx.registry.commands.items()):
        table.add_row(f"/{name}", command.help)
    ctx.console.print(table)




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


def _auth_registration(ctx: Any, prefix: str) -> Any | None:
    registration = ctx.registry.auth.get(prefix)
    if registration is None:
        ctx.console.error(f"Unknown auth prefix: {prefix}")
    return registration


def _model_prefix(model: Any) -> str | None:
    if isinstance(model, list):
        model = model[0] if model else None
    if not isinstance(model, str) or ":" not in model:
        return None
    return model.split(":", 1)[0]


def _provider_usable(ctx: Any, prefix: str) -> bool:
    provider = ctx.registry.providers.get(prefix)
    if provider is None or provider.available() is not None:
        return False
    if provider.env_keys and not any(os.environ.get(key) for key in provider.env_keys):
        return False
    auth = ctx.registry.auth.get(prefix)
    return auth is None or auth.flow.status() != "not logged in"


async def _after_login(ctx: Any, prefix: str) -> None:
    cfg = getattr(ctx, "cfg", None)
    provider = ctx.registry.providers.get(prefix)
    default_model = None if provider is None else provider.default_model
    if cfg is None or default_model is None:
        return
    model_spec = f"{prefix}:{default_model}"
    current_prefix = _model_prefix(getattr(cfg, "model", None))
    if current_prefix != prefix or not _provider_usable(ctx, current_prefix):
        await ctx.switch_model(model_spec)
        ctx.console.print(
            f"Switched model to {model_spec} (use /model to change)"
        )
    else:
        ctx.console.print(f"use /model {model_spec} to switch")


async def _login(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) not in {1, 2}:
        ctx.console.error("Usage: /login <prefix> [browser|device|paste]")
        return
    prefix = parts[0]
    mode_argument = "auto" if len(parts) == 1 else parts[1]
    mode = mode_argument[2:] if mode_argument.startswith("--") else mode_argument
    if mode not in {"auto", "browser", "device", "paste"} or (
        len(parts) == 2 and mode == "auto"
    ):
        ctx.console.error("Usage: /login <prefix> [browser|device|paste]")
        return
    registration = _auth_registration(ctx, prefix)
    if registration is not None:
        await registration.flow.login(ctx, mode)
        await _after_login(ctx, prefix)


async def _logout(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) != 1:
        ctx.console.error("Usage: /logout <prefix>")
        return
    registration = _auth_registration(ctx, parts[0])
    if registration is not None:
        await registration.flow.logout(ctx)


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


async def _theme(ctx: Any, args: str) -> None:
    name = args.strip()
    if not name:
        ui = getattr(ctx, "ui", None)
        if ui is None or not hasattr(ui, "show"):
            ctx.console.error("Theme picker is unavailable.")
            return
        try:
            await ui.show("theme")
        except RuntimeError:
            ctx.console.error("Theme picker is unavailable.")
        return
    if any(character.isspace() for character in name):
        ctx.console.error("Usage: /theme <name>")
        return
    ui = getattr(ctx, "ui", None)
    if ui is None or not hasattr(ui, "set_theme"):
        ctx.console.error("Theme selection is unavailable.")
        return
    try:
        selected = ui.set_theme(name)
    except (KeyError, ValueError):
        ctx.console.error(f"Unknown theme: {name}")
        return
    except RuntimeError as exc:
        ctx.console.error(str(exc))
        return
    states = getattr(ctx, "plugin_states", None)
    if isinstance(states, dict):
        states.setdefault("commands_core", {})["theme"] = name
        persist = getattr(ctx, "persist_plugin_states", None)
        if persist is not None:
            persist()
    selected_name = getattr(selected, "id", name)
    ctx.console.print(f"Theme: {selected_name}")


async def _keys(ctx: Any, _args: str) -> None:
    effective = getattr(getattr(ctx, "ui", None), "effective_keys", {})
    if not effective:
        ctx.console.error("Keybindings are unavailable.")
        return
    lines = ["Keybindings"]
    for action, bindings in sorted(effective.items()):
        lines.append(f"{action}: {', '.join(bindings) or 'unbound'}")
    ctx.console.print("\n".join(lines))


def register(api: PluginAPI) -> None:
    api.add_command("help", _help, help="List available slash commands")
    api.add_command("exit", _exit, help="Exit orcha-agent")
    api.add_command("plugins", _plugins, help="List loaded plugins")
    api.add_command("providers", _providers, help="List model providers and availability")
    api.add_command("login", _login, help="Log in to a provider: /login <prefix>")
    api.add_command("logout", _logout, help="Log out of a provider: /logout <prefix>")
    api.add_command("theme", _theme, help="Switch themes: /theme <name>")
    api.add_command("keys", _keys, help="List effective keybindings")
