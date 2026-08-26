"""Interactive approval for human-in-the-loop tool requests."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain.agents.middleware.human_in_the_loop import HITLRequest
from prompt_toolkit import PromptSession

from orcha_agent.core.events import InterruptRaised
from orcha_agent.core.plugin import PluginAPI, PluginSpec, Resolved

PLUGIN = PluginSpec(name="approval_prompt", version="1.0.0", priority=1000)


async def _prompt_action(
    name: str,
    args: dict[str, Any],
    description: str | None,
) -> str:
    detail = description or json.dumps(args, ensure_ascii=False, default=str)
    session: PromptSession[str] = PromptSession()
    return await session.prompt_async(f"Allow {name}: {detail}? [y/n/always] ")


async def _choice(name: str, args: dict[str, Any], description: str | None) -> str:
    while True:
        try:
            answer = (await _prompt_action(name, args, description)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "reject"
        if answer in {"y", "yes"}:
            return "approve"
        if answer in {"n", "no"}:
            return "reject"
        if answer in {"a", "always"}:
            return "always"


def register(api: PluginAPI) -> None:
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

    api.on(InterruptRaised, handle_approval, priority=1000)
