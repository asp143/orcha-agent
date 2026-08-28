"""Assistant markdown renderer."""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown

from orcha_agent.tui.frame import Block

from . import theme_value


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Markdown:
    del width, budget_rows, expanded
    return Markdown(
        str(block.data.get("text", "")),
        style=str(theme_value(theme, "text")),
        code_theme="monokai",
    )
