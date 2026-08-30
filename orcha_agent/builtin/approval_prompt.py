"""Fail-closed adapter from graph interrupts to the shared TUI approval overlay."""

from __future__ import annotations
import asyncio

from typing import Any, cast

from langchain.agents.middleware.human_in_the_loop import HITLRequest

from orcha_agent.core.events import AppStart, InterruptRaised
from orcha_agent.core.plugin import PluginAPI, PluginSpec, Resolved

PLUGIN = PluginSpec(name="approval_prompt", version="1.0.0", priority=1000)

_ctx: Any = None


async def _prompt_action(
    name: str,
    args: dict[str, Any],
    description: str | None,
) -> Any:
    """Open the registered approval overlay without creating a nested prompt."""

    if _ctx is None:
        return "reject"
    ui = getattr(_ctx, "ui", None)
    if ui is None or not hasattr(ui, "show"):
        return "reject"
    action = {"name": name, "args": args}
    if description is not None:
        action["description"] = description
    try:
        return await ui.show("approval", action=action)
    except (Exception, KeyboardInterrupt, asyncio.CancelledError):
        return "reject"


async def _choice(name: str, args: dict[str, Any], description: str | None) -> str:
    try:
        answer = await _prompt_action(name, args, description)
    except (Exception, KeyboardInterrupt, asyncio.CancelledError):
        return "reject"
    if isinstance(answer, str):
        normalized = answer.strip().casefold()
        if normalized in {"y", "yes", "approve"}:
            return "approve"
        if normalized in {"a", "always"}:
            return "always"
    return "reject"


def register(api: PluginAPI) -> None:
    async def capture_context(event: AppStart) -> None:
        global _ctx
        _ctx = event.ctx

    async def handle_approval(event: InterruptRaised) -> Resolved | None:
        request = cast(HITLRequest, event.payload)
        actions = request.get("action_requests")
        if not isinstance(actions, list):
            return None

        always_allowed = {
            value for value in api.state.get("always_allowed", []) if isinstance(value, str)
        }
        changed = False
        decisions: list[dict[str, str]] = []
        for action in actions:
            if not isinstance(action, dict):
                decisions.append({"type": "reject"})
                continue
            name = action.get("name")
            args = action.get("args")
            description = action.get("description")
            if not isinstance(name, str) or not isinstance(args, dict):
                decisions.append({"type": "reject"})
                continue

            if name in always_allowed:
                choice = "approve"
            else:
                choice = await _choice(
                    name,
                    args,
                    description if isinstance(description, str) else None,
                )
            if choice == "always":
                always_allowed.add(name)
                changed = True
                choice = "approve"
            decisions.append({"type": choice})

        if changed:
            api.state["always_allowed"] = sorted(always_allowed)
            api.request_rebuild()
        return Resolved(resume_value={"decisions": decisions})

    api.on(AppStart, capture_context, priority=1000)
    api.on(InterruptRaised, handle_approval, priority=1000)
