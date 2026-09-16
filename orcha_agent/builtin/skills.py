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
from orcha_agent.extensibility.skill_globs import SkillGlobsMiddleware

PLUGIN = PluginSpec(name="skills", version="1.0.0")


def register(api: PluginAPI) -> None:
    skills: dict[str, Skill] = {}
    task: asyncio.Task[None] | None = None
    prompt = ""
    glob_middleware = SkillGlobsMiddleware(
        skills, enabled=api.config.get("auto_attach_globs", True)
    )

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
        from orcha_agent.tui.panels import summary_panel

        rows = [
            (f"/skill:{skill.name}", skill.description)
            for skill in sorted(skills.values(), key=lambda item: item.name)
        ]
        ctx.console.print(
            summary_panel("Skills", ("Command", "Description"), rows or [("No skills found.", "")])
        )

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
            discover_skills,
            Path(ctx.cfg.cwd),
            Path.home(),
            api.config,
            trust_cwd=bool(getattr(ctx.cfg, "trust_cwd", False)),
        )
        skills.update(found)
        if warnings:
            transcript = getattr(ctx.console, "transcript", None)
            if transcript is not None:
                skipped = sum(warning.startswith("Skipping skill") for warning in warnings)
                summary = (
                    f"{skipped} skills skipped"
                    if skipped == len(warnings)
                    else f"{len(warnings)} skill warnings"
                )
                block = transcript.append_banner(summary + " · Ctrl+O for details", level="warning")
                block.data["details"] = "\n".join(warnings)
                block.revision += 1
            else:
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
            glob_middleware.cwd = Path(event.ctx.cfg.cwd)

            async def guarded_load() -> None:
                try:
                    await load(event.ctx)
                except Exception as exc:
                    event.ctx.console.warning(
                        f"Skill discovery failed: {type(exc).__name__}: {exc}"
                    )

            task = asyncio.create_task(guarded_load(), name="skills-discovery")
            if hasattr(event.ctx, "add_command_discovery_task"):
                event.ctx.add_command_discovery_task(task)

    async def before_build(event: AgentBuildBefore) -> None:
        try:
            await asyncio.wait_for(ready(), timeout=0.25)
        except TimeoutError:
            return
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
    api.add_middleware(glob_middleware, priority=60)
    api.add_command("skill", command, help="Run a skill: /skill <name> [args]")
    api.add_command("skills", listing, help="List discovered skills")
    api.add_completer("/skill ", complete)
    api.on(AppStart, start)
    api.on(AgentBuildBefore, before_build)
    api.on(AppExit, stop)
