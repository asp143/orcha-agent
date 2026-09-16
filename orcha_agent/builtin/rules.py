"""Rulebook discovery, model reminders and time-traveling stream rules plugin."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from rich.panel import Panel
from rich.text import Text

from orcha_agent.core.events import AppExit, AppStart, TurnEnd, TurnStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.rules import (
    MARKER,
    Rule,
    RulesMiddleware,
    discover_rules,
    rulebook,
)
from orcha_agent.extensibility.stream_rules import StreamInspect, StreamRetry, StreamRules

PLUGIN = PluginSpec(name="rules", version="1.0.0")


def register(api: PluginAPI) -> None:
    settings = dict(api.config)
    invalid: list[str] = []
    for key, allowed, default in (
        ("contextMode", {"keep", "discard"}, "discard"),
        ("interruptMode", {"always", "never", "prose-only", "tool-only"}, "always"),
        ("repeatMode", {"once", "after-gap"}, "once"),
    ):
        value = settings.get(key, default)
        if not isinstance(value, str) or value not in allowed:
            invalid.append(key)
            settings[key] = default
    gap = settings.get("repeatGap", 10)
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 0:
        invalid.append("repeatGap")
        settings["repeatGap"] = 10
    rules: dict[str, Rule] = {}
    sessions: dict[str, StreamRules] = {}
    context: Any = None
    discovery: asyncio.Task[None] | None = None
    middleware = RulesMiddleware(rules, Path.cwd())
    pending = middleware.pending

    def manager(session_id: str) -> StreamRules:
        if session_id not in sessions:
            sessions[session_id] = StreamRules(rules, settings, cwd=middleware.paths.cwd)
        return sessions[session_id]

    async def start(event: AppStart) -> None:
        nonlocal context
        context = event.ctx
        for key in invalid:
            context.console.warning(f"Invalid rules setting {key}; using its default.")
        cwd = getattr(getattr(context, "cfg", None), "cwd", None)
        if cwd is None:
            return
        middleware.paths.cwd = Path(cwd)

        async def load() -> None:
            found, warnings = await asyncio.to_thread(
                discover_rules,
                Path(cwd),
                Path.home(),
                trust_cwd=getattr(context.cfg, "trust_cwd", False),
            )
            rules.update(found)
            sessions.clear()
            for warning in warnings:
                context.console.warning(warning)
            if rules:
                api.system_prompt_fragment(rulebook(rules), priority=65)
                api.request_rebuild()

        nonlocal discovery
        discovery = asyncio.create_task(load(), name="rules-discovery")
        await ready()

    async def ready() -> None:
        if discovery is not None:
            try:
                await asyncio.wait_for(asyncio.shield(discovery), timeout=0.25)
            except TimeoutError:
                pass

    async def stop(_event: AppExit) -> None:
        if discovery is not None:
            discovery.cancel()
            await asyncio.gather(discovery, return_exceptions=True)

    async def read_rule(name: str) -> str:
        """Read a rule body by name or rule://name URL."""
        if discovery is not None:
            await asyncio.shield(discovery)
        rule = rules.get(name.removeprefix("rule://"))
        return rule.reminder().text if rule is not None else f"Error: unknown rule {name}"

    async def listing(ctx: Any, _args: str) -> None:
        ctx.console.print(rulebook(rules) or "No rules found.", markup=False)

    async def turn_start(event: TurnStart) -> None:
        if context is not None and event.source_id == "main":
            state = manager(context.session_id)
            state.start()
            agent = getattr(context, "agent", None)
            if agent is not None and hasattr(agent, "aget_state"):
                snapshot = await agent.aget_state(context.thread_config)
                restored = {
                    name
                    for message in snapshot.values.get("messages", [])
                    if isinstance(message, SystemMessage)
                    for name in message.additional_kwargs.get(MARKER, [])
                }
                restored.update(snapshot.values.get("rule_reminders", {}))
                state.fired = {name: state.fired.get(name, state.turn) for name in restored}

    async def turn_end(event: TurnEnd) -> None:
        if context is not None and event.source_id == "main":
            manager(context.session_id).finish()

    async def inspect(event: StreamInspect) -> StreamRetry | None:
        session_id = event.host.session_id
        state = manager(session_id)
        pending_key = str(event.host.thread_config.get("configurable", {}).get("thread_id", ""))
        if event.item is None:
            reminders = pending.pop(pending_key, [])
            if reminders:
                await event.host.agent.aupdate_state(
                    event.host.thread_config, {"messages": reminders}
                )
            return None
        matched, text = state.inspect(event.item)
        if not matched:
            return None
        reminders = [rule.reminder() for rule in matched]
        for rule in matched:
            event.host.console.print(
                Panel(Text(f"⚠ Injecting rule: {rule.name}"), border_style="yellow")
            )

        def interrupts(rule: Rule) -> bool:
            mode = rule.interrupt_mode or settings.get("interruptMode", "always")
            source = state.match_sources[rule.name]
            return (
                mode == "always"
                or (mode == "prose-only" and source != "tool")
                or (mode == "tool-only" and source == "tool")
            )

        interrupt = any(interrupts(rule) for rule in matched)
        if not interrupt:
            pending.setdefault(pending_key, []).extend(reminders)
            return None
        messages: list[Any] = pending.pop(pending_key, []) + reminders
        if settings.get("contextMode", "discard") == "keep" and text:
            messages.insert(0, AIMessage(content=text))
        state.buffers.clear()
        state.tool_buffers.clear()
        return StreamRetry(messages=messages)

    api.add_tool(StructuredTool.from_function(coroutine=read_rule, name="rule"))
    api.add_command("rules", listing, help="List the rulebook")
    api.add_middleware(middleware, priority=65)
    api.on(AppStart, start)
    api.on(AppExit, stop)
    api.on(TurnStart, turn_start)
    api.on(TurnEnd, turn_end)
    api.on(StreamInspect, inspect)
