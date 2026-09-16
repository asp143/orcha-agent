"""Bounded headless terminal replay for shell output, including cursor rewrites."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
    cells = [*replay.screen.history.top, *(replay.screen.buffer[index] for index in range(64))]
    for line in cells:
        row = Text()
        for column in range(width):
            char = line[column]
            row.append(
                char.data,
                Style(
                    color=_color(char.fg),
                    bgcolor=_color(char.bg),
                    bold=char.bold,
                    italic=char.italics,
                    underline=char.underscore,
                    reverse=char.reverse,
                    strike=char.strikethrough,
                ),
            )
        row.rstrip()
        rows.append(row)
    while rows and not rows[-1].plain:
        rows.pop()
    replay.rows = rows
    return rows
