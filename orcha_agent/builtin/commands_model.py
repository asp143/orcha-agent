"""Model and operating-mode slash commands."""

from __future__ import annotations

from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="commands_model", version="1.0.0")


async def _model(ctx: Any, args: str) -> None:
    spec = args.strip()
    if not spec or any(character.isspace() for character in spec):
        ctx.console.error("Usage: /model <provider:model>")
        return
    await ctx.switch_model(spec)


async def _mode(ctx: Any, args: str) -> None:
    name = args.strip()
    if not name or any(character.isspace() for character in name):
        ctx.console.error("Usage: /mode <name>")
        return
    await ctx.switch_mode(name)


def register(api: PluginAPI) -> None:
    api.add_command("model", _model, help="Switch models: /model <provider:model>")
    api.add_command("mode", _mode, help="Switch operating modes: /mode <name>")
