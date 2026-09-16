"""Filterable single- and multi-selection overlays."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent

from .base import Anchor, Overlay, ScrollableContent

T = TypeVar("T")


def _fuzzy(query: str, text: str) -> bool:
    needle = iter(query.casefold())
    current = next(needle, None)
    if current is None:
        return True
    for character in text.casefold():
        if character == current:
            current = next(needle, None)
            if current is None:
                return True
    return False


class _SelectControl(FormattedTextControl):
    """Let the window request visible rows without formatting the whole list."""

    def __init__(self, picker: Any, *, focusable: bool) -> None:
        super().__init__(focusable=focusable)
        self.picker = picker

    def create_content(self, width: int, height: int | None) -> UIContent:
        picker = self.picker
        pairs = picker._filtered_pairs()
        error_rows = int(picker._error is not None)

        def line(index: int) -> StyleAndTextTuples:
            if error_rows and index == 0:
                return [("class:error", f"  {picker._error}")]
            index -= error_rows
            if not pairs:
                return [("class:overlay.empty", f"  {picker.empty_text}")]
            original, item = pairs[index]
            fragments = picker._item_fragments(index, original, item)
            if index == picker.index:
                used = sum(len(part[1]) for part in fragments)
                fragments.append(("class:overlay.selection", " " * max(0, width - used)))
            return fragments

        return UIContent(
            get_line=line,
            line_count=max(1, len(pairs)) + error_rows,
            cursor_position=Point(x=0, y=picker.index + error_rows),
            show_cursor=False,
        )


class SelectList(ScrollableContent, Overlay, Generic[T]):
    """A deterministic fuzzy-filtered picker driven entirely by key events."""

    def __init__(
        self,
        title: str,
        items: Sequence[T],
        *,
        label: Callable[[T], str] = str,
        multi: bool = False,
        page_size: int = 8,
        empty_text: str = "No matches",
        on_accept: Callable[[T | list[T]], Any] | None = None,
        on_change: Callable[[T | None], Any] | None = None,
        on_cancel: Callable[[], Any] | None = None,
        anchor: Anchor = "center",
        prefix: Any | None = None,
        show_filter: bool = True,
    ) -> None:
        self.items = tuple(items)
        self._filter_cache: tuple[tuple[T, ...], str, list[tuple[int, T]]] | None = None
        self.label = label
        self.multi = multi
        self._init_scrolling(page_size)
        self.empty_text = empty_text
        self._selected: set[int] = set()
        self._on_accept = on_accept
        self._on_change = on_change
        self._accepting = False
        self._error: str | None = None
        self.filter = Buffer(multiline=False)
        self.filter.on_text_changed += self._filter_changed
        self._show_filter = show_filter
        self.list_control = _SelectControl(
            self,
            focusable=not show_filter,
        )
        self.filter_control = BufferControl(buffer=self.filter)
        body_parts: list[Any] = []
        if show_filter:
            body_parts.extend(
                [
                    Window(
                        self.filter_control,
                        height=lambda: 1 if self.filter.text else 0,
                        dont_extend_height=True,
                        style="class:overlay.filter",
                    ),
                    Window(
                        char="─",
                        height=lambda: 1 if self.filter.text else 0,
                        dont_extend_height=True,
                        style="class:overlay.divider",
                    ),
                ]
            )
        if prefix is not None:
            body_parts.extend([prefix, Window(char="─", height=1, style="class:overlay.divider")])
        self.list_window = Window(
            self.list_control, always_hide_cursor=True,
            get_vertical_scroll=lambda window: max(
                self.index // self.page_size * self.page_size,
                self.index - (window.render_info.window_height if window.render_info else self.page_size) + 1,
            ),
        )
        self.footer_control = FormattedTextControl(lambda: self._scroll_footer("select"))
        body_parts.extend(
            [
                self.list_window,
                Window(self.footer_control, wrap_lines=True, dont_extend_height=True),
            ]
        )
        bindings = KeyBindings()
        self._bind_navigation(bindings)
        filter_focused = has_focus(self.filter_control)
        for key in ("j", "k"):
            binding = bindings.get_bindings_for_keys((key,))[0]
            bindings.remove(key)
            bindings.add(key, filter=~filter_focused)(binding)

        @bindings.add(" ")
        def _toggle(event: Any) -> None:
            if self.multi:
                filtered = self._filtered_pairs()
                if filtered:
                    original = filtered[self.index][0]
                    if original in self._selected:
                        self._selected.remove(original)
                    else:
                        self._selected.add(original)
                    event.app.invalidate()
            else:
                self.filter.insert_text(" ")

        @bindings.add("enter")
        def _enter(event: Any) -> None:
            filtered = self._filtered_pairs()
            if not filtered:
                return
            if self.multi:
                value: T | list[T] = [
                    item for offset, item in enumerate(self.items) if offset in self._selected
                ]
            else:
                value = filtered[self.index][1]
            self._accept(value, event)

        super().__init__(
            title,
            HSplit(body_parts),
            anchor=anchor,
            on_cancel=on_cancel,
            bindings=bindings,
        )

    @property
    def focus_target(self) -> Any:
        return self.filter_control if self._show_filter else self.list_control

    @property
    def accepting(self) -> bool:
        return self._accepting

    def _filtered_pairs(self) -> list[tuple[int, T]]:
        query = self.filter.text
        cached = self._filter_cache
        if cached is not None and cached[0] is self.items and cached[1] == query:
            return cached[2]
        pairs = [
            (offset, item)
            for offset, item in enumerate(self.items)
            if not query or _fuzzy(query, self.label(item))
        ]
        self._filter_cache = (self.items, query, pairs)
        return pairs

    @property
    def filtered_items(self) -> tuple[T, ...]:
        return tuple(item for _, item in self._filtered_pairs())

    def _filter_changed(self, _buffer: Buffer) -> None:
        self.index = 0
        self._changed()

    def _scroll_count(self) -> int:
        return len(self._filtered_pairs())

    def _scroll_changed(self) -> None:
        self._changed()

    def _changed(self) -> None:
        if self._on_change is None:
            return
        filtered = self._filtered_pairs()
        self._on_change(filtered[self.index][1] if filtered else None)

    def _accept(self, value: T | list[T], event: Any) -> None:
        if self._accepting:
            return
        if self._on_accept is None:
            self.resolve(value)
            return
        self._accepting = True
        self._error = None
        event.app.invalidate()
        try:
            result = self._on_accept(value)
        except Exception as exc:
            self._accept_failed(exc, event)
            return
        if not inspect.isawaitable(result):
            self._accepting = False
            self.resolve(value if result is None else result)
            return

        async def complete() -> None:
            try:
                accepted = await result
            except Exception as exc:
                self._accept_failed(exc, event)
                return
            self._accepting = False
            self.resolve(value if accepted is None else accepted)

        event.app.create_background_task(complete())

    def _accept_failed(self, exc: Exception, event: Any) -> None:
        self._accepting = False
        self._error = f"{type(exc).__name__}: {exc}"
        event.app.invalidate()

    def _item_fragments(self, visible: int, original: int, item: T) -> StyleAndTextTuples:
        current = visible == self.index
        marker = (
            ("◉" if original in self._selected else "○")
            if self.multi
            else ("›" if current else " ")
        )
        style = "class:overlay.selection" if current else "class:overlay.item"
        return [(style, f" {marker} {self.label(item)}")]

    def _fragments(self) -> StyleAndTextTuples:
        filtered = self._filtered_pairs()
        fragments: StyleAndTextTuples = []
        if self._error is not None:
            fragments.append(("class:error", f"  {self._error}\n"))
        if not filtered:
            fragments.append(("class:overlay.empty", f"  {self.empty_text}\n"))
            return fragments
        for visible, (original, item) in enumerate(filtered):
            current = visible == self.index
            if current:
                fragments.append(("[SetCursorPosition]", ""))
            fragments.extend(self._item_fragments(visible, original, item))
            fragments.append(("", "\n"))
        return fragments

    def render_text(self) -> str:
        fragments = [*self._fragments(), *self._scroll_footer("select")]
        return "".join(fragment[1] for fragment in fragments)


__all__ = ["SelectList"]
