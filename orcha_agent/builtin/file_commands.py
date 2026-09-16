"""Register file-backed slash commands without blocking initial rendering."""

from __future__ import annotations

import asyncio
from typing import Any

from orcha_agent.core.events import AppExit, AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.commands import FileCommand, discover_commands, render_command

PLUGIN = PluginSpec(name="file_commands", version="1.0.0")


def register(api: PluginAPI) -> None:
    task: asyncio.Task[None] | None = None

    def handler(command: FileCommand):
        async def run(ctx: Any, args: str) -> None:
            try:
                text = await render_command(command, args, cwd=ctx.cfg.cwd)
            except (OSError, ValueError, TimeoutError) as exc:
                ctx.console.error(str(exc) or "Command interpolation timed out")
                return
            await ctx.submit_prompt(text, model=command.model)

        return run

    async def discover(ctx: Any) -> None:
        commands = await asyncio.to_thread(
            discover_commands,
            ctx.cfg.cwd,
            trust_cwd=ctx.cfg.trust_cwd,
            import_claude=api.config.get("import_claude", True) is not False,
        )
        for name, command in commands.items():
            help_text = command.description
            if command.argument_hint:
                help_text += f" ({command.argument_hint})"
            try:
                api.add_command(name, handler(command), help=help_text)
            except ValueError:
                # A file must never replace a built-in or another plugin's command.
                continue

    async def start(event: AppStart) -> None:
        nonlocal task
        if getattr(getattr(event.ctx, "cfg", None), "cwd", None) is None:
            return
        task = asyncio.create_task(discover(event.ctx))
        track = getattr(event.ctx, "add_command_discovery_task", None)
        if callable(track):
            track(task)

    async def stop(_event: AppExit) -> None:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    api.on(AppStart, start)
    api.on(AppExit, stop)
