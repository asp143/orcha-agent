"""Transient turn and retry indicator renderer."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_spinner, theme_value


_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text:
    del budget_rows, expanded
    level = str(block.data.get("level", "accent"))
    color = str(theme_value(theme, "warning" if level == "warning" else "accent"))
    frame = int(block.data.get("spinner_frame", 0))
    rendered = Text(theme_spinner(theme, "spinner.activity", frame, _FRAMES), style=color)
    rendered.append(" ")
    rendered.append(
        str(block.data.get("message", "Working… (Esc to interrupt)")),
        style=str(theme_value(theme, "muted", "dim")),
    )
    rendered.truncate(max(1, width), overflow="ellipsis")
    return rendered


__all__ = ["render"]
