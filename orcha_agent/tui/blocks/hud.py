"""Compact todo and running-subagent HUD renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

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
    del width, expanded
    items = _items(block, "items")
    if not items or budget_rows <= 0:
        return None
    rows = min(_MAX_ROWS, budget_rows)
    rendered = Text("Todo", style=f"bold {theme_value(theme, 'accent')}")
    for item in items[: max(0, rows - 1)]:
        if isinstance(item, Mapping):
            label = str(item.get("text", item.get("title", "")))
            done = bool(item.get("done") or item.get("status") == "done")
        else:
            label, done = str(item), False
        rendered.append(f"\n{'✔' if done else '○'} {label}", style="dim" if done else "")
    return rendered


def render_subagents(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    del width, expanded
    agents = _items(block, "agents")
    if not agents or budget_rows <= 0:
        return None
    rows = min(_MAX_ROWS, budget_rows)
    rendered = Text("Subagents", style=f"bold {theme_value(theme, 'accent')}")
    for agent in agents[: max(0, rows - 1)]:
        if isinstance(agent, Mapping):
            name = str(agent.get("name", agent.get("id", "agent")))
            status = str(agent.get("status", "running"))
        else:
            name, status = str(agent), "running"
        rendered.append(f"\n✻ {name} · {status}")
    return rendered


__all__ = ["render_subagents", "render_todo"]
