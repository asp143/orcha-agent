"""Themed, literal-text compaction transcript card."""

from __future__ import annotations
from typing import Any
from rich import box
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from orcha_agent.tui.frame import Block
from . import theme_symbol, theme_value


def render(block: Block, theme: Any, width: int, budget_rows: int, expanded: bool) -> Panel:
    del width, budget_rows, expanded
    color = str(theme_value(theme, "warning"))
    title = Text(
        f"Compacted · {block.data.get('method', 'summary')} · {block.data.get('tokens_before', 0):,} tokens before",
        style=Style(color=color),
    )
    return Panel(
        Text(str(block.data.get("summary", "")), style=str(theme_value(theme, "text"))),
        title=title,
        title_align="left",
        border_style=color,
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        padding=(0, 1),
    )
