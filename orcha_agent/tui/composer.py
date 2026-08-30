"""Themed multi-line prompt composer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import History
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import AnyContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import Margin, ScrollbarMargin
from prompt_toolkit.utils import get_cwidth

_SHAPES = frozenset({"box", "claude", "borderless"})
_PADDING_X = 2


def _width(value: str) -> int:
    return sum(get_cwidth(character) for character in value)


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if _width(value) <= width:
        return value
    if width == 1:
        return "…"
    output: list[str] = []
    used = 0
    for character in value:
        cells = get_cwidth(character)
        if used + cells > width - 1:
            break
        output.append(character)
        used += cells
    return f"{''.join(output)}…"


class _BoxMargin(Margin):
    """Render omp's side chrome, merging the final row into the bottom edge."""

    def __init__(self, composer: "Composer", *, left: bool) -> None:
        self.composer = composer
        self.left = left

    def get_width(self, get_ui_content: Callable[[], Any]) -> int:
        del get_ui_content
        return _PADDING_X + 1

    @staticmethod
    def _thumb_rows(render_info: Any, height: int) -> set[int]:
        content_height = int(render_info.content_height)
        if content_height <= height or height <= 1:
            return set()
        usable = height - 1
        visible = min(content_height, len(render_info.displayed_lines))
        thumb_height = max(1, int(usable * visible / content_height))
        thumb_top = int(usable * render_info.vertical_scroll / content_height)
        return set(range(thumb_top, min(usable, thumb_top + thumb_height)))

    def create_margin(
        self,
        window_render_info: Any,
        width: int,
        height: int,
    ) -> StyleAndTextTuples:
        del width
        thumb_rows = self._thumb_rows(window_render_info, height)
        fragments: StyleAndTextTuples = []
        for row in range(height):
            last = row == height - 1
            if self.left:
                value = "╰─ " if last else "│  "
            elif last:
                value = " ─╯"
            else:
                value = f"  {'█' if row in thumb_rows else '│'}"
            fragments.append((self.composer.border_style, value))
            if not last:
                fragments.append(("", "\n"))
        return fragments


class Composer:
    """Buffer plus shape-specific chrome used by the inline application."""

    def __init__(
        self,
        *,
        shape: str = "box",
        theme: Any = None,
        model: Callable[[], str] = lambda: "model",
        thinking: Callable[[], str] = lambda: "off",
        history: History | None = None,
        completer: Completer | None = None,
        accept_handler: Callable[[Buffer], bool] | None = None,
    ) -> None:
        if shape not in _SHAPES:
            raise ValueError(f"unknown composer shape {shape!r}")
        self.shape = shape
        self.theme = theme
        self._model = model
        self._thinking = thinking
        self.buffer = Buffer(
            history=history,
            completer=completer,
            complete_while_typing=Condition(lambda: completer is not None),
            multiline=True,
            accept_handler=accept_handler,
        )
        self.control = BufferControl(buffer=self.buffer)
        self.input_window = Window(
            self.control,
            height=lambda: Dimension.exact(
                self.text_rows(self._content_width(self._current_width()))
            ),
            dont_extend_height=True,
            wrap_lines=True,
            left_margins=[_BoxMargin(self, left=True)] if shape == "box" else [],
            right_margins=(
                [_BoxMargin(self, left=False)]
                if shape == "box"
                else [ScrollbarMargin(display_arrows=False)]
            ),
        )
        self.container = self._build_container()

    @property
    def chrome_lines(self) -> int:
        if self.shape == "borderless":
            return 0
        return 1 if self.shape == "box" else 2

    @property
    def border_style(self) -> str:
        if self.buffer.text.lstrip().startswith("!"):
            return "class:bashmode"
        level = self._thinking().lower()
        if level not in {"off", "low", "medium", "high", "max"}:
            level = "off"
        return f"class:thinking{level}"

    def _chip(self) -> str:
        return f" {self._model()} · {self._thinking()} "

    def _top_line(self, width: int) -> str:
        width = max(1, width)
        chip = self._chip()
        if self.shape == "box":
            available = max(0, width - 6)
            if _width(chip) > available:
                chip = _truncate(chip, max(0, available - 1))
            fill = max(0, available - _width(chip))
            return _truncate(f"╭──{chip}{'─' * fill}──╮", width)
        if self.shape == "claude":
            chip = _truncate(chip, width)
            return f"{'─' * max(0, width - _width(chip))}{chip}"
        return ""

    def _top_fragments(self) -> StyleAndTextTuples:
        line = self._top_line(self._current_width())
        chip = self._chip()
        offset = line.find(chip)
        if offset < 0:
            return [(self.border_style, line)]
        return [
            (self.border_style, line[:offset]),
            ("class:statuslinemodel", chip),
            (self.border_style, line[offset + len(chip) :]),
        ]

    def _bottom_fragments(self) -> StyleAndTextTuples:
        return [(self.border_style, "─" * self._current_width())]

    def _build_container(self) -> AnyContainer:
        if self.shape == "borderless":
            return self.input_window
        top = Window(
            FormattedTextControl(self._top_fragments),
            height=1,
            style=lambda: self.border_style,
        )
        bottom = Window(
            FormattedTextControl(self._bottom_fragments),
            height=1,
            style=lambda: self.border_style,
        )
        if self.shape == "claude":
            middle = VSplit(
                [
                    Window(width=2, char="❯ ", style=lambda: self.border_style),
                    self.input_window,
                ]
            )
        else:
            middle = self.input_window
        rows = [top, middle] if self.shape == "box" else [top, middle, bottom]
        return HSplit(
            rows,
            height=lambda: Dimension.exact(
                self.height_for_width(self._current_width())
            ),
        )

    def _current_width(self) -> int:
        app = get_app_or_none()
        if app is None:
            return 80
        return max(1, app.output.get_size().columns)

    def _content_width(self, width: int) -> int:
        if self.shape == "box":
            return max(1, width - (_PADDING_X + 1) * 2)
        return max(1, width - (2 if self.shape == "claude" else 0) - 1)

    def render_lines(
        self,
        content: list[str],
        width: int,
        *,
        scrollbar_rows: set[int] | None = None,
    ) -> list[str]:
        """Render plain composer rows for goldens and terminal diagnostics."""

        width = max(1, width)
        scrollbar_rows = scrollbar_rows or set()
        if self.shape == "borderless":
            return [_truncate(line, width).ljust(width) for line in content]
        if self.shape == "claude":
            body = [f"❯ {_truncate(line, max(1, width - 2))}".ljust(width) for line in content]
            return [self._top_line(width), *body, "─" * width]
        inner = max(1, width - 6)
        rows = [self._top_line(width)]
        for index, line in enumerate(content):
            text = _truncate(line, inner)
            last = index == len(content) - 1
            if last:
                rows.append(f"╰─ {text.ljust(inner)} ─╯")
            else:
                border = "█" if index in scrollbar_rows else "│"
                rows.append(f"│  {text.ljust(inner)}  {border}")
        return rows

    def text_rows(self, width: int) -> int:
        width = max(1, width)
        rows = 0
        for line in self.buffer.text.split("\n"):
            columns = sum(get_cwidth(character) for character in line)
            rows += max(1, (columns + width - 1) // width)
        return min(8, max(1, rows))

    def height_for_width(self, width: int) -> int:
        return self.text_rows(self._content_width(width)) + self.chrome_lines


__all__ = ["Composer"]
