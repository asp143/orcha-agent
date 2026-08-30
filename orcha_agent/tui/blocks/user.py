"""User-message bubble renderer."""

from __future__ import annotations

import re
from typing import Any

from rich.padding import Padding
from rich.style import Style
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

_MARKUP = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`", re.DOTALL)


def _inline(text: str, *, dim: bool, color: str) -> Text:
    rendered = Text(style=Style(color=color, dim=dim))
    cursor = 0
    for match in _MARKUP.finditer(text):
        rendered.append(text[cursor : match.start()])
        if match.group(1) is not None:
            rendered.append(match.group(1), style=Style(bold=True, dim=dim))
        else:
            rendered.append(match.group(2) or "", style=Style(reverse=True, dim=dim))
        cursor = match.end()
    rendered.append(text[cursor:])
    return rendered


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Padding:
    del width, budget_rows, expanded
    dim = bool(block.data.get("synthetic") or block.data.get("queued"))
    text = _inline(
        str(block.data.get("text", "")),
        dim=dim,
        color=str(theme_value(theme, "userMessageText")),
    )
    return Padding(
        text,
        (0, 1),
        style=f"on {theme_value(theme, 'userMessageBg')}",
        expand=True,
    )
