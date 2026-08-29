"""Fixed-height startup welcome renderer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

_MAX_WIDTH = 100
_PREFERRED_LEFT = 26
_MIN_LEFT = 13
_MIN_RIGHT = 20
_SESSION_SLOTS = 4
_HINT_SLOTS = 4


def _slots(value: Any, count: int) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found = [str(item) for item in value[:count]]
    else:
        found = []
    return [*found, *([""] * (count - len(found)))]


def _logo(block: Block, *, compact: bool) -> Text:
    if compact:
        return Text("ORCHA", style="bold")
    lines = _slots(block.data.get("logo"), 5)
    styles = block.data.get("logo_styles", ())
    rendered = Text()
    for row, line in enumerate(lines):
        row_styles = styles[row] if isinstance(styles, Sequence) and row < len(styles) else ()
        for column, character in enumerate(line):
            style = row_styles[column] if isinstance(row_styles, Sequence) and column < len(row_styles) else None
            rendered.append(character, style=style)
        if row < len(lines) - 1:
            rendered.append("\n")
    return rendered


def _line(label: str, value: Any, width: int) -> str:
    text = f"{label}: {value}"
    return text if len(text) <= width else f"{text[: max(0, width - 3)]}..."


def _fit(value: Text | str, width: int, *, center: bool = False) -> Text:
    rendered = value.copy() if isinstance(value, Text) else Text(value)
    rendered.truncate(max(0, width), overflow="ellipsis")
    remaining = max(0, width - rendered.cell_len)
    if center:
        left = remaining // 2
        rendered.pad_left(left)
        rendered.pad_right(remaining - left)
    else:
        rendered.pad_right(remaining)
    return rendered


def _right(block: Block, width: int, *, ascii_only: bool) -> Text:
    rule = "----" if ascii_only else "────"
    lines = [
        f"{rule} Recent sessions",
        *_slots(block.data.get("sessions"), _SESSION_SLOTS),
        f"{rule} Hints",
        *_slots(block.data.get("hints"), _HINT_SLOTS),
        _line("Tip", block.data.get("tip", ""), width),
        "",
    ]
    rendered = Text()
    for index, line in enumerate(lines):
        fitted = _fit(f" {line}" if line else "", width)
        if index in {0, 5}:
            fitted.stylize("dim")
        rendered.append(fitted)
        if index < len(lines) - 1:
            rendered.append("\n")
    return rendered


def _left(block: Block, width: int) -> Text:
    logo = _logo(block, compact=False).split("\n")
    lines = [
        Text(),
        Text("Welcome back!", style="bold"),
        Text(),
        *logo,
        Text(),
        Text(str(block.data.get("model", "")), style="dim"),
        Text(str(block.data.get("mode", "")), style="dim"),
        Text(str(block.data.get("cwd", "")), style="dim"),
    ]
    rendered = Text()
    for index, line in enumerate(lines):
        rendered.append(_fit(line, width, center=True))
        if index < len(lines) - 1:
            rendered.append("\n")
    return rendered


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel:
    del budget_rows, expanded
    target = max(12, min(_MAX_WIDTH, width - 2))
    ascii_only = bool(block.data.get("ascii"))
    border = str(theme_value(theme, "border", "cyan"))
    inner = max(1, target - 2)
    dual_content = max(1, target - 3)
    desired_left = max(
        _MIN_LEFT,
        min(_PREFERRED_LEFT, max(_MIN_LEFT, int(dual_content * 0.35))),
    )
    left_width = min(desired_left, max(1, dual_content - _MIN_RIGHT))
    right_width = max(1, dual_content - left_width)
    show_right = left_width >= _MIN_LEFT and right_width >= _MIN_RIGHT
    if show_right:
        separator = "|" if ascii_only else "│"
        table = Table.grid(expand=False, padding=0)
        table.add_column(width=left_width, no_wrap=True)
        table.add_column(width=1, no_wrap=True)
        table.add_column(width=right_width, no_wrap=True)
        table.add_row(
            _left(block, left_width),
            Text("\n".join([separator] * 12), style="dim"),
            _right(block, right_width, ascii_only=ascii_only),
        )
        content: Any = table
    else:
        content = Text()
        content.append(_logo(block, compact=True))
        content.append("\n" + _line("Model", block.data.get("model", ""), inner))
        content.append("\n" + _line("Mode", block.data.get("mode", ""), inner))
        content.append("\n" + _line("Cwd", block.data.get("cwd", ""), inner))
        content.append("\n")
        content.append(_right(block, inner, ascii_only=ascii_only))
    return Panel(
        content,
        box=box.ASCII if ascii_only else box.ROUNDED,
        border_style=border,
        padding=0,
        width=target,
    )


__all__ = ["render"]
