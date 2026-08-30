"""Shared prompt-toolkit overlay chrome and result lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import Float, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

Anchor = Literal["center", "bottom"]


def _text(value: str, *, width: int | None = None, style: str = "") -> Window:
    return Window(
        FormattedTextControl(FormattedText([(style, value)])),
        width=width,
        height=1,
        dont_extend_width=width is not None,
    )


def _bordered(title: str, body: Any) -> Any:
    title_text = f" {title} "
    top = VSplit(
        [
            _text("╭", width=1, style="class:overlay.border"),
            _text("─", width=1, style="class:overlay.border"),
            _text(title_text, width=len(title_text), style="class:overlay.title"),
            Window(char="─", height=1, style="class:overlay.border"),
            _text("╮", width=1, style="class:overlay.border"),
        ],
        height=1,
    )
    middle = VSplit(
        [
            Window(char="│", width=1, style="class:overlay.border"),
            Window(char=" ", width=1),
            body,
            Window(char=" ", width=1),
            Window(char="│", width=1, style="class:overlay.border"),
        ]
    )
    bottom = VSplit(
        [
            _text("╰", width=1, style="class:overlay.border"),
            Window(char="─", height=1, style="class:overlay.border"),
            _text("╯", width=1, style="class:overlay.border"),
        ],
        height=1,
    )
    return HSplit([top, middle, bottom])


class Overlay(Float):
    """A rounded floating container with awaitable result/cancel semantics."""

    def __init__(
        self,
        title: str,
        body: Any,
        *,
        anchor: Anchor = "center",
        width: float = 0.72,
        height: float = 0.62,
        margin: int = 2,
        min_height: int = 3,
        on_cancel: Callable[[], Any] | None = None,
        bindings: KeyBindings | None = None,
    ) -> None:
        if anchor not in {"center", "bottom"}:
            raise ValueError(f"unsupported overlay anchor: {anchor}")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError("overlay width and height must be percentages in (0, 1]")
        self.title = title
        self.anchor = anchor
        self.width_percent = width
        self.height_percent = height
        self.margin = max(0, margin)
        self.min_height = max(3, min_height)
        self._body = body
        self._on_cancel = on_cancel
        self._finished = False
        self._value: Any = None
        self._waiter: asyncio.Future[Any] | None = None
        self.bindings = bindings or KeyBindings()

        @self.bindings.add("escape")
        def _escape(_event: Any) -> None:
            self.cancel()

        self.container = _bordered(title, body)
        super().__init__(
            content=self.container,
            width=self._width,
            height=self._height,
            bottom=1 if anchor == "bottom" else None,
            z_index=10,
        )

    def _terminal_size(self) -> tuple[int, int]:
        size = get_app().output.get_size()
        return size.columns, size.rows

    def _width(self) -> int:
        columns, _ = self._terminal_size()
        available = max(1, columns - self.margin * 2)
        return max(4, min(80, available))

    @property
    def inner_width(self) -> int:
        return max(0, self._width() - 4)

    def _needed_height(self, available: int) -> int:
        preferred = self.container.preferred_height(self._width(), available)
        return max(3, int(preferred.preferred))

    def _height(self) -> int:
        _, rows = self._terminal_size()
        available = max(1, rows - self.margin * 2)
        maximum = max(self.min_height, int(rows * self.height_percent))
        needed = max(self.min_height, self._needed_height(available))
        return max(3, min(available, maximum, needed))

    def body_rows(self, terminal_rows: int | None = None) -> int:
        if terminal_rows is None:
            return max(0, self._height() - 2)
        _, current_rows = self._terminal_size()
        if terminal_rows == current_rows:
            return max(0, self._height() - 2)
        available = max(1, terminal_rows - self.margin * 2)
        maximum = max(self.min_height, int(terminal_rows * self.height_percent))
        needed = max(self.min_height, self._needed_height(available))
        return max(0, min(available, maximum, needed) - 2)

    @staticmethod
    def render_lines(
        title: str,
        lines: list[str],
        *,
        width: int,
        height: int,
        anchor: Anchor = "center",
    ) -> list[str]:
        """Plain overlay rendering used by goldens and terminal diagnostics."""

        width = max(4, width)
        height = max(3, height)
        inner = width - 4
        title_text = f" {title} ".replace("\n", " ")
        title_text = title_text[: max(0, width - 5)]
        fill = max(0, width - 3 - len(title_text))
        top = f"╭─{title_text}{'─' * fill}╮"
        capacity = height - 2
        visible = lines[-capacity:] if anchor == "bottom" else lines[:capacity]
        rows = [top]
        rows.extend(f"│ {line[:inner].ljust(inner)} │" for line in visible)
        rows.extend(f"│ {'':{inner}} │" for _ in range(capacity - len(visible)))
        rows.append(f"╰{'─' * (width - 2)}╯")
        return rows

    @property
    def focus_target(self) -> Any:
        return self.container

    @property
    def done(self) -> bool:
        return self._finished

    def resolve(self, value: Any) -> None:
        if self._finished:
            return
        self._finished = True
        self._value = value
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(value)

    def cancel(self) -> None:
        if self._finished:
            return
        if self._on_cancel is not None:
            self._on_cancel()
        self.resolve(None)

    async def wait(self) -> Any:
        if self._finished:
            return self._value
        if self._waiter is None:
            self._waiter = asyncio.get_running_loop().create_future()
        return await self._waiter

    def __await__(self):
        return self.wait().__await__()


__all__ = ["Anchor", "Overlay"]
