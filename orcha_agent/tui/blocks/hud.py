"""Compact todo and running-subagent HUD renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_symbol, theme_value
from .task import render as render_task
from .tool import render as render_tool

_MAX_ROWS = 8
_HUD_FIELDS = (
    "id",
    "run_id",
    "name",
    "agent_type",
    "description",
    "status",
    "requests",
    "tokens_in",
    "tokens_out",
    "cost",
    "last_tool",
    "last_tool_args",
    "current_tool",
    "current_tool_args",
    "created_at",
    "updated_at",
)


def _items(block: Block, key: str) -> list[Any]:
    value = block.data.get(key, [])
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _run_snapshot(run: Any) -> dict[str, Any]:
    snapshot = getattr(run, "snapshot", None)
    if callable(snapshot):
        source = snapshot()
        row = {key: source.get(key) for key in _HUD_FIELDS if key in source}
    elif isinstance(run, Mapping):
        row = {key: run.get(key) for key in _HUD_FIELDS if key in run}
    else:
        row = {key: getattr(run, key, None) for key in _HUD_FIELDS}
    name = str(row.get("name") or "agent")
    row.setdefault("run_id", row.get("id") or name)
    row["name"] = name
    row["description"] = str(row.get("description") or name)
    if row.get("current_tool"):
        row["last_tool"] = row["current_tool"]
        row["last_tool_args"] = row.get("current_tool_args")
    return row


def subagent_hud_data(ctx: Any, *, spinner_frame: int = 0) -> dict[str, Any] | None:
    """Build immutable HUD data directly from the application agent registry."""

    registry = getattr(ctx, "agents", None)
    list_runs = getattr(registry, "list", None)
    if not callable(list_runs):
        return None

    rows: list[dict[str, Any]] = []
    queued = running = idle = 0
    for run in list_runs():
        row = _run_snapshot(run)
        status = str(row.get("status") or "parked").casefold()
        row["status"] = status
        queued += status == "queued"
        running += status == "running"
        idle += status == "idle"
        rows.append(row)
    if not rows:
        return None
    return {
        "agents": rows,
        "queued": queued,
        "running": running,
        "idle": idle,
        "compact": True,
        "spinner_frame": spinner_frame,
    }


def render_todo(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    items = _items(block, "items")
    if not items or budget_rows <= 0:
        return None
    tool = Block(
        id=block.id,
        kind="tool",
        state=block.state,
        revision=block.revision,
        data={"name": "todo", "args": {"items": items}, "result": "ok"},
    )
    return render_tool(tool, theme, width, min(_MAX_ROWS, budget_rows), expanded)


def render_subagents(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    agents = [_run_snapshot(agent) for agent in _items(block, "agents")]
    if not agents or budget_rows <= 0:
        return None
    task = Block(
        id=block.id,
        kind="task",
        state=block.state,
        revision=block.revision,
        data={
            "agents": agents,
            "running": int(block.data.get("running", 0) or 0),
            "idle": int(block.data.get("idle", 0) or 0),
            "repeat_description": True,
            "spinner_frame": block.data.get("spinner_frame", 0),
        },
    )
    return render_task(task, theme, width, min(_MAX_ROWS, budget_rows), expanded)


def render_queue(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    del width, expanded
    prompts = _items(block, "prompts")
    if not prompts or budget_rows <= 0:
        return None
    rows = min(_MAX_ROWS, budget_rows)
    rendered = Text("Queue", style=f"bold {theme_value(theme, 'accent')}")
    glyph = theme_symbol(theme, "status.pending", "○")
    for prompt in prompts[: max(0, rows - 1)]:
        text = " ".join(str(prompt).split())
        rendered.append(f"\n{glyph} {text}")
    return rendered


__all__ = [
    "render_queue",
    "render_subagents",
    "render_todo",
    "subagent_hud_data",
]
