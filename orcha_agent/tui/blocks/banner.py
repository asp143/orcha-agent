"""Banner panel renderer."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.panel import Panel
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_symbol, theme_value


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel:
    del width, budget_rows, expanded
    level = str(block.data.get("level", "error")).lower()
    lines = str(block.data.get("message", block.data.get("text", ""))).splitlines()
    if level == "error" and len(lines) > 8:
        lines = [*lines[:7], "…"]
    titles = {"error": "Error", "warning": "Warning", "info": "Info"}
    colors = {"error": "error", "warning": "warning", "info": "accent"}
    return Panel(
        Text("\n".join(lines)),
        title=titles.get(level, level.title()),
        title_align="left",
        border_style=str(theme_value(theme, colors.get(level, "accent"))),
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        padding=(0, 1),
    )
