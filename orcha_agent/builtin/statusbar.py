"""Default status-line segment adapter and accounting hooks."""

from __future__ import annotations

from typing import Any

from orcha_agent.core.events import ModelChunk, ThreadSwitch, TurnEnd, TurnStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.tui.statusline import (
    BUILTIN_SEGMENTS,
    DEFAULT_PRICING,
    WINDOWS,
    Segment,
    _configured_thinking,
    _display_model,
    _parse_git,
    _quantity,
    _spec,
    _thinking_level,
    _window,
    cache_segment,
    context_segment,
    cost_segment,
    git_segment,
    mode_segment,
    model_segment,
    path_segment,
    record_turn_end,
    record_turn_start,
    record_usage,
    reset_accounting,
    session_segment,
    subagents_segment,
    time_segment,
    tokens_segment,
    visible_segments,
)

PLUGIN = PluginSpec(name="statusbar", version="2.0.0")
SEGMENTS = BUILTIN_SEGMENTS


def cwd_segment(ctx: Any) -> Segment:
    """Compatibility name for the renamed path segment."""

    return path_segment(ctx)


def effort_segment(ctx: Any) -> Segment | None:
    """Compatibility helper; thinking now appears in the model segment."""

    spec, _ = _spec(ctx)
    effort = _configured_thinking(ctx, spec)
    return Segment(effort, "thinkingMedium", "icon.thinking") if effort else None


def _usage(event: ModelChunk, state: dict[str, Any]) -> None:
    """Compatibility wrapper for plugins/tests using the prior helper."""

    record_usage(event, state)


def register(api: PluginAPI) -> None:
    if not bool(api.config.get("statusbar", True)):
        return

    api.state["_git_refreshing"] = False
    api.state["_usage_seen_ids"] = []
    for priority, (name, render) in enumerate(BUILTIN_SEGMENTS, start=1):
        api.add_status_segment(name, render, priority=priority * 10)

    async def track(event: ModelChunk) -> None:
        record_usage(event, api.state)

    async def reset_usage(_event: ThreadSwitch) -> None:
        reset_accounting(api.state)

    async def turn_started(_event: TurnStart) -> None:
        record_turn_start(api.state)

    async def turn_finished(_event: TurnEnd) -> None:
        record_turn_end(api.state)

    async def show(ctx: Any, _args: str) -> None:
        for name, segment in visible_segments(ctx):
            ctx.console.print(f"{name}: {segment.text}")

    api.on(ModelChunk, track, priority=10)
    api.on(ThreadSwitch, reset_usage, priority=10)
    api.on(TurnStart, turn_started, priority=10)
    api.on(TurnEnd, turn_finished, priority=10)
    api.add_command("status", show, help="Show effective status-line segments")


__all__ = [
    "DEFAULT_PRICING",
    "PLUGIN",
    "SEGMENTS",
    "WINDOWS",
    "cache_segment",
    "context_segment",
    "cost_segment",
    "cwd_segment",
    "effort_segment",
    "git_segment",
    "mode_segment",
    "model_segment",
    "path_segment",
    "register",
    "session_segment",
    "subagents_segment",
    "time_segment",
    "tokens_segment",
]
