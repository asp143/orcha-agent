"""Compact todo and running-subagent HUD renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_spinner, theme_symbol, theme_value
from .tool import render as render_tool

_MAX_ROWS = 8


def _items(block: Block, key: str) -> list[Any]:
    value = block.data.get(key, [])
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


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
        data={
            "name": "todo",
            "args": {"items": items},
            "result": "ok",
            "leading_spacer": False,
        },
    )
    return render_tool(tool, theme, width, min(_MAX_ROWS, budget_rows), expanded)


def render_subagents(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    agents = _items(block, "agents")
    if not agents or budget_rows <= 0:
        return None
    normalized = []
    for agent in agents:
        if isinstance(agent, Mapping):
            normalized.append(
                {
                    **agent,
                    "description": agent.get("description", agent.get("name", "")),
                }
            )
        else:
            normalized.append({"id": "agent", "description": str(agent), "status": "running"})
    tool = Block(
        id=block.id,
        kind="tool",
        state=block.state,
        revision=block.revision,
        data={
            "name": "task",
            "result": {"agents": normalized},
            "spinner_frame": block.data.get("spinner_frame", 0),
            "leading_spacer": False,
        },
    )
    return render_tool(tool, theme, width, min(_MAX_ROWS, budget_rows), expanded)


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


__all__ = ["render_queue", "render_subagents", "render_todo"]
