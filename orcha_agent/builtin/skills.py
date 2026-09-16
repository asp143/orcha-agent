"""Discover skills in the background and expose tools and slash commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from prompt_toolkit.completion import Completion

from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.skills import Skill, discover_skills, render_skills

PLUGIN = PluginSpec(name="skills", version="1.0.0")


def register(api: PluginAPI) -> None:
    skills: dict[str, Skill] = {}
    task: asyncio.Task[None] | None = None
    prompt = ""

    async def ready() -> None:
        if task is not None:
            await asyncio.shield(task)

    async def invoke(ctx: Any, name: str, args: str) -> None:
        await ready()
        skill = skills.get(name)
        if skill is None:
            ctx.console.error(f"Unknown skill: {name}. Use /skills to list available skills.")
            return
        try:
            text = await asyncio.to_thread(skill.invocation, args)
        except (OSError, UnicodeError, ValueError) as exc:
            ctx.console.error(f"Could not read skill {name}: {exc}")
            return
        await ctx.submit_prompt(text)

    async def command(ctx: Any, args: str) -> None:
        parts = args.strip().split(maxsplit=1)
        if not parts:
            ctx.console.print("Usage: /skill <name> [args]")
            return
        await invoke(ctx, parts[0], parts[1] if len(parts) > 1 else "")

    async def listing(ctx: Any, _args: str) -> None:
        await ready()
        if not skills:
            ctx.console.print("No skills found.")
        for skill in sorted(skills.values(), key=lambda item: item.name):
            ctx.console.print(f"/skill:{skill.name} — {skill.description}")

    async def read_skill(name: str) -> str:
        """Read a skill's instructions by name when relevant to the user's task."""
        await ready()
        skill = skills.get(name.removeprefix("skill://"))
        if skill is None:
            return f"Error: unknown skill {name}"
        if skill.disable_model_invocation:
            return f"Error: skill {skill.name} allows user invocation only"
        try:
            return await asyncio.to_thread(skill.invocation)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error reading skill {skill.name}: {exc}"

    async def load(ctx: Any) -> None:
        nonlocal prompt
        found, warnings = await asyncio.to_thread(
            discover_skills, Path(ctx.cfg.cwd), Path.home(), api.config
        )
        skills.update(found)
        for warning in warnings:
            ctx.console.warning(warning)
        for name, skill in skills.items():

            async def shortcut(ctx: Any, args: str, name: str = name) -> None:
                await invoke(ctx, name, args)

            try:
                api.add_command(
                    f"skill:{name}", shortcut, help=skill.description or f"Run skill {name}"
                )
            except ValueError as exc:
                ctx.console.warning(f"Could not register /skill:{name}: {exc}; use /skill {name}")
        render_warnings: list[str] = []
        prompt = await asyncio.to_thread(render_skills, skills, warnings=render_warnings)
        for warning in render_warnings:
            ctx.console.warning(warning)
        if prompt:
            api.system_prompt_fragment(prompt, priority=60)
            api.request_rebuild()

    async def start(event: AppStart) -> None:
        nonlocal task
        if getattr(getattr(event.ctx, "cfg", None), "cwd", None) is None:
            return
        if task is None:
            task = asyncio.create_task(load(event.ctx), name="skills-discovery")
            if hasattr(event.ctx, "add_command_discovery_task"):
                event.ctx.add_command_discovery_task(task)

    async def before_build(event: AgentBuildBefore) -> None:
        await ready()
        current = event.kwargs.get("system_prompt", "")
        if prompt and prompt not in current:
            event.kwargs["system_prompt"] = current + "\n\n" + prompt

    async def stop(_event: AppExit) -> None:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def complete(document: Any) -> list[Completion]:
        prefix = document.text_before_cursor.removeprefix("/skill ")
        return [
            Completion(name, start_position=-len(prefix), display_meta=skills[name].description)
            for name in sorted(skills)
            if name.startswith(prefix)
        ]

    api.add_tool(StructuredTool.from_function(coroutine=read_skill, name="skill"))
    api.add_command("skill", command, help="Run a skill: /skill <name> [args]")
    api.add_command("skills", listing, help="List discovered skills")
    api.add_completer("/skill ", complete)
    api.on(AppStart, start)
    api.on(AgentBuildBefore, before_build)
    api.on(AppExit, stop)
