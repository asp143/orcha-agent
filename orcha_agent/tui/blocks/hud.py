"""Compact todo and running-subagent HUD renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_spinner, theme_symbol, theme_value

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
            label = str(item.get("content", item.get("text", item.get("title", ""))))
            done = bool(item.get("done") or item.get("status") in {"done", "completed"})
        else:
            label, done = str(item), False
        glyph = theme_symbol(
            theme,
            "status.success" if done else "status.pending",
            "✔" if done else "○",
        )
        rendered.append(f"\n{glyph} {label}", style="dim" if done else "")
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
    spinner_frame = int(block.data.get("spinner_frame", 0))
    spinner = theme_spinner(theme, "spinner.status", spinner_frame, ("✻",))
    separator = theme_symbol(theme, "sep.thin", "·")
    for agent in agents[: max(0, rows - 1)]:
        if isinstance(agent, Mapping):
            name = str(agent.get("name", agent.get("id", "agent")))
            status = str(agent.get("status", "running"))
        else:
            name, status = str(agent), "running"
        rendered.append(f"\n{spinner} {name} {separator} {status}")
    return rendered


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
