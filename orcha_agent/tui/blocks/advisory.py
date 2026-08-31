"""Advisor watchdog transcript card renderer."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_symbol, theme_value

_SEVERITY_TOKEN = {
    "nit": "dim",
    "concern": "warning",
    "blocker": "error",
}


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel | None:
    """Render a settled advisor response, omitting no-op responses."""

    del width, budget_rows, expanded
    note = block.data.get("note")
    if note is None:
        return None

    severity = str(block.data.get("severity", "nit")).casefold()
    token = _SEVERITY_TOKEN.get(severity, "dim")
    color = str(theme_value(theme, token))
    advisor_id = str(block.data.get("advisor_id") or "advisor")
    title = Text(
        f"Advisor · {advisor_id} · {severity.title()}",
        style=Style(color=color, bold=severity != "nit", dim=severity == "nit"),
    )
    content = Text(
        str(note),
        style=Style(
            color=str(theme_value(theme, "text")),
            dim=severity == "nit",
        ),
    )
    return Panel(
        content,
        title=title,
        title_align="left",
        border_style=Style(color=color, dim=severity == "nit"),
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        padding=(0, 1),
    )
