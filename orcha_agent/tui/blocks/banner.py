"""Banner panel renderer."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_symbol, theme_value

_MAX_CONTENT_ROWS = 6
_PANEL_HORIZONTAL_CHROME = 4
_WRAP_CONSOLE = Console(force_terminal=False, color_system=None)


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel:
    del expanded
    level = str(block.data.get("level", "error")).lower()
    message = str(block.data.get("message", block.data.get("text", "")))
    content_limit = max(1, min(_MAX_CONTENT_ROWS, budget_rows - 2))
    content_width = max(1, width - _PANEL_HORIZONTAL_CHROME)
    lines = list(
        Text(message).wrap(_WRAP_CONSOLE, content_width, overflow="fold")
    )
    if len(lines) > content_limit:
        lines = [*lines[: content_limit - 1], Text("…")]
    titles = {"error": "Error", "warning": "Warning", "info": "Info"}
    colors = {"error": "error", "warning": "warning", "info": "accent"}
    return Panel(
        Text("\n").join(lines),
        title=titles.get(level, level.title()),
        title_align="left",
        border_style=str(theme_value(theme, colors.get(level, "accent"))),
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        padding=(0, 1),
    )
