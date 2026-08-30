"""Fixed-height startup welcome renderer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

_MAX_WIDTH = 100
_MIN_LEFT = 13
_MIN_RIGHT = 20
_LEFT_PADDING = 1
_SESSION_SLOTS = 4
_HINT_SLOTS = 4
_TIP_ROWS = 2
_WRAP_CONSOLE = Console(width=_MAX_WIDTH, force_terminal=False, color_system=None)


def _slots(value: Any, count: int) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found = [str(item) for item in value[:count]]
    else:
        found = []
    return [*found, *([""] * (count - len(found)))]


def _logo_width(block: Block) -> int:
    return max(
        (Text(line).cell_len for line in _slots(block.data.get("logo"), 5)),
        default=0,
    )


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


def _tip_lines(value: Any, width: int) -> list[str]:
    """Wrap the tip like omp: label on the first row, indented continuation."""

    label = "Tip: "
    words = str(value).split()
    body_width = max(1, width - 1 - len(label))
    rows: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > body_width:
            rows.append(current)
            current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    rows = rows[:_TIP_ROWS]
    if len(rows) == _TIP_ROWS and len(words) > sum(len(r.split()) for r in rows):
        rows[-1] = rows[-1][: max(0, body_width - 1)] + "…"
    rendered = [f"{label}{rows[0]}" if rows else ""]
    rendered.extend(" " * len(label) + row for row in rows[1:])
    return [*rendered, *([""] * (_TIP_ROWS - len(rendered)))]


def _display_bindings(value: object) -> str:
    if isinstance(value, str):
        bindings = (value,)
    elif isinstance(value, Sequence):
        bindings = tuple(str(binding) for binding in value)
    else:
        bindings = (str(value),)
    try:
        from orcha_agent.tui.keys import format_key_bindings
    except ImportError:
        return ", ".join(bindings)
    return format_key_bindings(bindings)


def _hint_text(value: object, theme: Any) -> Text:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        bindings, description = value
        key = _display_bindings(bindings)
        rendered = Text(key, style=str(theme_value(theme, "dim", "bright_black")))
        if key and description:
            rendered.append(" ")
        rendered.append(str(description), style=str(theme_value(theme, "muted", "bright_black")))
        return rendered
    return Text(str(value), style=str(theme_value(theme, "muted", "bright_black")))


def _hint_lines(value: object, width: int, theme: Any) -> list[Text]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        hints = value
    else:
        hints = ()
    rendered: list[Text] = []
    for hint in hints:
        if len(rendered) >= _HINT_SLOTS:
            break
        text = _hint_text(hint, theme)
        wrapped = text.wrap(_WRAP_CONSOLE, max(1, width), overflow="fold") if text else [Text()]
        rendered.extend(wrapped[: _HINT_SLOTS - len(rendered)])
    return [*rendered, *([Text()] * (_HINT_SLOTS - len(rendered)))]


def _right(block: Block, width: int, *, ascii_only: bool, theme: Any) -> Text:
    rule = "----" if ascii_only else "────"
    lines = [
        Text(f"{rule} Recent sessions", style="dim"),
        *(Text(session) for session in _slots(block.data.get("sessions"), _SESSION_SLOTS)),
        Text(f"{rule} Hints", style="dim"),
        *_hint_lines(block.data.get("hints"), max(1, width - 1), theme),
        *(Text(line) for line in _tip_lines(block.data.get("tip", ""), width)),
    ]
    rendered = Text()
    for index, line in enumerate(lines):
        indented = Text(" ") if line else Text()
        indented.append_text(line)
        rendered.append(_fit(indented, width))
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
    # The left column is sized by the logo itself, not a fixed ratio: omp's
    # constants assume a 12-column logo and would ellipsize wider ones.
    left_width = max(_MIN_LEFT, _logo_width(block) + 2 * _LEFT_PADDING)
    right_width = max(1, dual_content - left_width)
    show_right = right_width >= _MIN_RIGHT
    if show_right:
        separator = "|" if ascii_only else "│"
        table = Table.grid(expand=False, padding=0)
        table.add_column(width=left_width, no_wrap=True)
        table.add_column(width=1, no_wrap=True)
        table.add_column(width=right_width, no_wrap=True)
        table.add_row(
            _left(block, left_width),
            Text("\n".join([separator] * 12), style="dim"),
            _right(block, right_width, ascii_only=ascii_only, theme=theme),
        )
        content: Any = table
    else:
        content = Text()
        content.append(
            _logo(block, compact=_logo_width(block) > inner)
        )
        content.append("\n" + _line("Model", block.data.get("model", ""), inner))
        content.append("\n" + _line("Mode", block.data.get("mode", ""), inner))
        content.append("\n" + _line("Cwd", block.data.get("cwd", ""), inner))
        content.append("\n")
        content.append(_right(block, inner, ascii_only=ascii_only, theme=theme))
    return Panel(
        content,
        box=box.ASCII if ascii_only else box.ROUNDED,
        border_style=border,
        padding=0,
        width=target,
    )


__all__ = ["render"]
