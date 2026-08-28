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
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.utils import get_cwidth

_SHAPES = frozenset({"box", "claude", "borderless"})


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
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        self.container = self._build_container()

    @property
    def chrome_lines(self) -> int:
        return 0 if self.shape == "borderless" else 2

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

    def _top_fragments(self) -> StyleAndTextTuples:
        prefix = "╭─" if self.shape == "box" else "──"
        return [
            (self.border_style, prefix),
            ("class:statuslinemodel", self._chip()),
            (self.border_style, "─"),
        ]

    def _bottom_fragments(self) -> StyleAndTextTuples:
        return [(self.border_style, "╰" if self.shape == "box" else ""), (self.border_style, "─")]

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
            middle = VSplit(
                [
                    Window(width=1, char="│", style=lambda: self.border_style),
                    self.input_window,
                    Window(width=1, char="│", style=lambda: self.border_style),
                ]
            )
        return HSplit(
            [top, middle, bottom],
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
        return max(
            1,
            width - (2 if self.shape in {"box", "claude"} else 0) - 1,
        )

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
