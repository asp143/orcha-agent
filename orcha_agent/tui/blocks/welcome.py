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
_WIDE_MINIMUM = 96


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


def _right(block: Block, width: int, *, ascii_only: bool) -> Text:
    rule = "----" if ascii_only else "────"
    lines = [
        f"{rule} Recent sessions",
        *_slots(block.data.get("sessions"), 4),
        f"{rule} Hints",
        *_slots(block.data.get("hints"), 4),
        _line("Tip", block.data.get("tip", ""), width),
    ]
    rendered = Text()
    for index, line in enumerate(lines):
        rendered.append(line[:width], style="dim" if index in {0, 5} else "")
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
    target = max(12, min(_MAX_WIDTH, width))
    ascii_only = bool(block.data.get("ascii"))
    border = str(theme_value(theme, "border", "cyan"))
    if width >= _WIDE_MINIMUM:
        inner = max(1, target - 4)
        left_width = min(48, inner // 2)
        right_width = max(1, inner - left_width - 1)
        left = _logo(block, compact=False)
        left.append("\n\n")
        left.append(_line("Model", block.data.get("model", ""), left_width))
        left.append("\n" + _line("Mode", block.data.get("mode", ""), left_width))
        left.append("\n" + _line("Cwd", block.data.get("cwd", ""), left_width))
        table = Table.grid(expand=False, padding=(0, 1))
        table.add_column(width=left_width, no_wrap=True)
        table.add_column(width=right_width, no_wrap=True)
        table.add_row(left, _right(block, right_width, ascii_only=ascii_only))
        content: Any = table
    else:
        inner = max(1, target - 4)
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
        padding=(0, 1),
        width=target,
    )


__all__ = ["render"]
