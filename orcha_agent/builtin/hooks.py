"""Expose declarative hooks through the plugin event and middleware surfaces."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from orcha_agent.core.events import (
    AgentFinished,
    AgentSpawned,
    AppExit,
    AppStart,
    Compaction,
    Event,
    ModelSwitch,
    SessionSwitch,
    ToolCallAfter,
    ToolCallBefore,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.hooks import HooksMiddleware, WRITE_TOOLS, matches, run_hook

PLUGIN = PluginSpec(name="hooks", version="1.0.0")
_EVENTS = {
    TurnStart: "turn_start",
    TurnEnd: "turn_end",
    ToolCallBefore: "tool_call_before",
    ToolCallAfter: "tool_call_after",
    ModelSwitch: "model_switch",
    Compaction: "compaction",
    AgentSpawned: "agent_spawned",
    AgentFinished: "agent_finished",
}


def register(api: PluginAPI) -> None:
    ctx: Any = None
    tasks: set[asyncio.Task[None]] = set()

    async def dispatch(name: str, payload: dict[str, Any], event: Event | None = None) -> None:
        if ctx is None:
            return
        payload = {"event": name, **payload}
        for hook in getattr(getattr(ctx, "cfg", None), "hooks", ()):
            if hook.event != name or not matches(hook, payload):
                continue

            async def execute(hook: Any = hook) -> None:
                try:
                    result = await run_hook(hook, payload, Path(ctx.cfg.cwd))
                    if result.code == 2 and hook.blocking and isinstance(event, ToolCallBefore):
                        event.block_message = result.error.strip() or "Blocked by hook"
                    elif result.code != 0:
                        ctx.console.warning(
                            result.error.strip() or f"Hook exited with code {result.code}"
                        )
                    elif (
                        hook.blocking
                        and isinstance(event, ToolCallBefore)
                        and result.output.strip()
                    ):
                        data = json.loads(result.output)
                        if not isinstance(data, dict):
                            raise ValueError("hook output must be a JSON object")
                        if data.get("block"):
                            event.block_message = str(
                                data.get("message", data.get("reason", "Blocked by hook"))
                            )
                        args = data.get("args", data.get("updatedInput"))
                        if args is not None:
                            if not isinstance(args, dict):
                                raise ValueError("hook rewritten args must be an object")
                            if event.name in WRITE_TOOLS:
                                event.args = args
                                payload["args"] = args
                except Exception as exc:
                    ctx.console.warning(f"Hook failed: {type(exc).__name__}: {exc}")
                    if hook.blocking and isinstance(event, ToolCallBefore):
                        event.block_message = "Hook failed; tool execution blocked"

            if hook.blocking:
                await execute()
                if isinstance(event, ToolCallBefore) and event.block_message:
                    break
            else:
                task = asyncio.create_task(execute(), name=f"hook:{name}")
                tasks.add(task)
                task.add_done_callback(tasks.discard)

    async def start(event: AppStart) -> None:
        nonlocal ctx
        ctx = event.ctx
        await dispatch("session_start", {"session_id": getattr(ctx, "session_id", None)})

    async def stop(_event: AppExit) -> None:
        await dispatch("session_end", {"session_id": getattr(ctx, "session_id", None)})
        if tasks:
            await asyncio.gather(*tuple(tasks), return_exceptions=True)

    async def switch(event: SessionSwitch) -> None:
        await dispatch("session_end", {"session_id": event.old})
        await dispatch("session_start", {"session_id": event.new})

    async def handle(event: Event) -> None:
        await dispatch(
            _EVENTS[type(event)],
            {field.name: getattr(event, field.name) for field in fields(event)},
            event,
        )

    async def listing(context: Any, _args: str) -> None:
        hooks = getattr(context.cfg, "hooks", ())
        if not hooks:
            context.console.print("No declarative hooks configured.")
        for hook in hooks:
            context.console.print(
                f"{hook.event} [{hook.scope}] {hook.matcher} → {hook.command or hook.python} ({'blocking' if hook.blocking else 'background'}, {hook.timeout:g}s)",
                markup=False,
            )

    api.add_middleware(HooksMiddleware(api.emit), priority=10)
    api.add_command("hooks", listing, help="List configured declarative hooks")
    api.on(AppStart, start)
    api.on(AppExit, stop, priority=10)
    api.on(SessionSwitch, switch)
    for event_type in _EVENTS:
        api.on(event_type, handle)
