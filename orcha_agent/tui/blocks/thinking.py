"""Visible and collapsed reasoning renderer."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_spinner, theme_symbol, theme_value, with_leading_spacer

SPINNER_FRAMES = ("✻", "✼", "❉", "❊", "✺", "✹", "✸", "✶")


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Group:
    del width, budget_rows, expanded
    visible = bool(block.data.get("visible", theme_value(theme, "thinking_visible", True)))
    if visible:
        content: Markdown | Text = Markdown(
            str(block.data.get("text", "")),
            style=f"italic {theme_value(theme, 'thinkingText')}",
            code_theme="monokai",
        )
    else:
        frame = int(block.data.get("spinner_frame", 0))
        tokens = int(block.data.get("reasoning_tokens", 0))
        rate = float(block.data.get("tokens_per_second", 0.0))
        separator = theme_symbol(theme, "sep.thin", "·")
        content = Text(
            f"{theme_spinner(theme, 'spinner.activity', frame, SPINNER_FRAMES)} "
            f"Thinking… {separator} {tokens} {separator} {rate:.1f} tok/s",
            style=str(theme_value(theme, "thinkingOff")),
        )
    return with_leading_spacer(content)


__all__ = ["SPINNER_FRAMES", "render"]
