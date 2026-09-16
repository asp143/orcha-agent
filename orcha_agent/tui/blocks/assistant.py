"""Assistant markdown renderer."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.padding import Padding

from orcha_agent.tui.frame import Block

from .syntax import syntax_style
from .markdown import StreamingMarkdown as Markdown

from . import theme_value, with_leading_spacer


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Group:
    del width, budget_rows, expanded
    subagent = bool(block.data.get("subagent"))
    aborted = bool(block.data.get("aborted"))
    markdown = Markdown(
        str(block.data.get("text", "")),
        style=(
            f"dim strike {theme_value(theme, 'text')}"
            if aborted
            else f"dim {theme_value(theme, 'text')}"
            if subagent
            else str(theme_value(theme, "text"))
        ),
        code_theme=syntax_style(theme),
        heading_color=str(theme_value(theme, "accent")),
    )
    content = Padding(markdown, (0, 2), expand=True) if subagent else markdown
    return with_leading_spacer(content)
