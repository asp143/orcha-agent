"""Shared prompt-toolkit overlay chrome and result lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import Float, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from orcha_agent.tui.keys import format_key_bindings

from .hints import key_hint

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


class ScrollableContent:
    """Shared clamped selection state, navigation, and footer rendering."""

    index: int
    page_size: int

    def _init_scrolling(self, page_size: int) -> None:
        self.index = 0
        self.page_size = max(1, page_size)

    def _scroll_count(self) -> int:
        raise NotImplementedError

    def _scroll_changed(self) -> None:
        return None

    def _move(self, delta: int) -> None:
        count = self._scroll_count()
        self.index = 0 if count == 0 else min(count - 1, max(0, self.index + delta))
        self._scroll_changed()

    def _bind_navigation(self, bindings: KeyBindings) -> None:
        movements = (
            ("up", -1),
            ("k", -1),
            ("down", 1),
            ("j", 1),
            ("pageup", -self.page_size),
            ("pagedown", self.page_size),
        )
        for key, delta in movements:

            @bindings.add(key)
            def _navigate(event: Any, delta: int = delta) -> None:
                self._move(delta)
                event.app.invalidate()

    def _scroll_footer(self, action: str) -> StyleAndTextTuples:
        count = self._scroll_count()
        current = self.index + 1 if count else 0
        fragments: StyleAndTextTuples = [
            ("class:muted", f" ({current}/{count})  "),
        ]
        hints = (
            ((format_key_bindings(("escape",)), "close"),)
            if count == 0
            else (
                (format_key_bindings(("j", "k")), "navigate"),
                (format_key_bindings(("pageup", "pagedown")), "page"),
                (format_key_bindings(("enter",)), action),
                (format_key_bindings(("escape",)), "close"),
            )
        )
        for offset, (key, description) in enumerate(hints):
            if offset:
                fragments.append(("class:muted", " · "))
            fragments.extend(key_hint(key, description, formatted=True))
        return fragments




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
        columns = max(1, columns)
        if columns < 4:
            return columns
        available = max(4, columns - self.margin * 2)
        return min(80, available, columns)

    @property
    def inner_width(self) -> int:
        return max(0, self._width() - 4)

    def _needed_height(self, available: int) -> int:
        preferred = self.container.preferred_height(self._width(), available)
        return max(3, int(preferred.preferred))

    def _height(self) -> int:
        _, rows = self._terminal_size()
        rows = max(1, rows)
        if rows < 3:
            return rows
        available = max(3, rows - self.margin * 2)
        maximum = max(self.min_height, int(rows * self.height_percent))
        needed = max(self.min_height, self._needed_height(available))
        return min(rows, max(3, min(available, maximum, needed)))

    def body_rows(self, terminal_rows: int | None = None) -> int:
        if terminal_rows is None:
            return max(0, self._height() - 2)
        _, current_rows = self._terminal_size()
        if terminal_rows == current_rows:
            return max(0, self._height() - 2)
        if terminal_rows < 3:
            return 0
        available = max(3, terminal_rows - self.margin * 2)
        maximum = max(self.min_height, int(terminal_rows * self.height_percent))
        needed = max(self.min_height, self._needed_height(available))
        return max(0, min(terminal_rows, available, maximum, needed) - 2)

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
class ScrollableOverlay(ScrollableContent, Overlay):
    """Scrollable static content using the same navigation grammar as pickers."""

    def __init__(
        self,
        title: str,
        rows: Sequence[StyleAndTextTuples],
        *,
        page_size: int = 8,
        width: float = 0.72,
        height: float = 0.62,
    ) -> None:
        self.rows = tuple(tuple(row) for row in rows)
        self._init_scrolling(page_size)
        self.content_control = FormattedTextControl(self._content_fragments, focusable=True)
        self.content_window = Window(self.content_control, always_hide_cursor=True)
        self.footer_control = FormattedTextControl(lambda: self._scroll_footer("close"))
        bindings = KeyBindings()
        self._bind_navigation(bindings)
        body = HSplit(
            [
                self.content_window,
                Window(self.footer_control, height=1, wrap_lines=False),
            ]
        )
        Overlay.__init__(
            self,
            title,
            body,
            width=width,
            height=height,
            bindings=bindings,
        )

    def _scroll_count(self) -> int:
        return len(self.rows)

    def _content_fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for offset, row in enumerate(self.rows):
            if offset == self.index:
                fragments.append(("[SetCursorPosition]", ""))
            fragments.extend(row)
            if not row or not row[-1][1].endswith("\n"):
                fragments.append(("", "\n"))
        return fragments

    def render_text(self) -> str:
        fragments = [*self._content_fragments(), *self._scroll_footer("close")]
        return "".join(text for _style, text in fragments)

    @property
    def focus_target(self) -> Any:
        return self.content_control


__all__ = ["Anchor", "Overlay", "ScrollableContent", "ScrollableOverlay"]
