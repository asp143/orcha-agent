"""Bounded headless terminal replay for shell output, including cursor rewrites."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from time import monotonic
from typing import Any

import pyte
from rich.style import Style
from rich.text import Text


@dataclass
class _Replay:
    width: int
    screen: Any = field(init=False)
    stream: Any = field(init=False)
    consumed: int = 0
    tail: str = ""
    updated: float = 0.0
    rows: list[Text] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.screen = pyte.HistoryScreen(self.width, 64, history=4096)
        self.stream = pyte.Stream(self.screen)


_REPLAYS: OrderedDict[str, _Replay] = OrderedDict()


def _color(value: str) -> str | None:
    if value == "default":
        return None
    if len(value) == 6 and all(char in "0123456789abcdef" for char in value.lower()):
        return f"#{value}"
    if value.startswith("bright"):
        suffix = value.removeprefix("bright")
        return "bright_" + ("yellow" if suffix == "brown" else suffix)
    return "yellow" if value == "brown" else value


def clear_terminal_cache() -> None:
    """Drop session-local replay state on clear or session switch."""
    _REPLAYS.clear()


@lru_cache(maxsize=512)
def _cell_style(attributes: tuple[Any, ...]) -> Style:
    fg, bg, bold, italics, underscore, reverse, strike = attributes
    return Style(
        color=_color(fg),
        bgcolor=_color(bg),
        bold=bold or None,
        italic=italics or None,
        underline=underscore or None,
        reverse=reverse or None,
        strike=strike or None,
    )


def terminal_rows(key: str, source: str, width: int, *, streaming: bool = False) -> list[Text]:
    """Feed only appended bytes; repaint at most every 50 ms while streaming."""
    width = max(1, width)
    replay = _REPLAYS.get(key)
    if (
        replay is None
        or replay.width != width
        or len(source) < replay.consumed
        or source[max(0, replay.consumed - 128) : replay.consumed] != replay.tail
    ):
        replay = _Replay(width)
        _REPLAYS[key] = replay
    _REPLAYS.move_to_end(key)
    while len(_REPLAYS) > 16:
        _REPLAYS.popitem(last=False)
    now = monotonic()
    if streaming and replay.updated and now - replay.updated < 0.05:
        return replay.rows
    if len(source) == replay.consumed and replay.updated:
        return replay.rows
    # PTYs translate LF to CRLF. Tool pipes do not, so model that translation here.
    replay.stream.feed(source[replay.consumed :].replace("\n", "\r\n"))
    replay.consumed = len(source)
    replay.tail = source[-128:]
    replay.updated = now
    rows: list[Text] = []
    last_content_row = max(
        (
            index
            for index, line in replay.screen.buffer.items()
            if any(char.data.strip() or char.bg != "default" for char in line.values())
        ),
        default=0,
    )
    last_row = max(replay.screen.cursor.y, last_content_row)
    cells = [
        *replay.screen.history.top,
        *(replay.screen.buffer[index] for index in range(last_row + 1)),
    ]
    for line in cells:
        row = Text()
        # Sparse pyte rows contain only touched cells. Preserve interior blanks,
        # but never allocate spans or cells for the untouched right margin.
        last_column = max(line, default=-1)
        run: list[str] = []
        previous: tuple[Any, ...] | None = None
        for column in range(min(width, last_column + 1)):
            char = line.get(column, replay.screen.default_char)
            attributes = (
                char.fg,
                char.bg,
                char.bold,
                char.italics,
                char.underscore,
                char.reverse,
                char.strikethrough,
            )
            if previous is not None and attributes != previous:
                row.append("".join(run), _cell_style(previous))
                run = []
            run.append(char.data)
            previous = attributes
        if previous is not None:
            row.append("".join(run), _cell_style(previous))
        row.rstrip()
        rows.append(row)
    while rows and not rows[-1].plain:
        rows.pop()
    replay.rows = rows
    return rows
