"""User-message bubble renderer."""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.padding import Padding
from rich.style import Style
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

_MARKUP = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`", re.DOTALL)


class _WidthConstrainedPadding(Padding):
    """Padding that honors the transcript width even when Rich reports a wider TTY."""

    def __init__(self, renderable: Text, width: int, *, style: str) -> None:
        super().__init__(renderable, (1, 1), style=style, expand=True)
        self._render_width = width

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        constrained = options.update_width(min(self._render_width, options.max_width))
        yield from super().__rich_console__(console, constrained)


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
    del budget_rows, expanded
    dim = bool(block.data.get("synthetic") or block.data.get("queued"))
    text = _inline(
        str(block.data.get("text", "")),
        dim=dim,
        color=str(theme_value(theme, "userMessageText")),
    )
    return _WidthConstrainedPadding(
        text,
        width,
        style=f"on {theme_value(theme, 'userMessageBg')}",
    )
