"""Filterable single- and multi-selection overlays."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

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
        self.list_control = FormattedTextControl(
            self._fragments,
            focusable=not show_filter,
        )
        self.filter_control = BufferControl(buffer=self.filter)
        body_parts: list[Any] = []
        if show_filter:
            body_parts.extend(
                [
                    Window(self.filter_control, height=1, style="class:overlay.filter"),
                    Window(char="─", height=1, style="class:overlay.divider"),
                ]
            )
        if prefix is not None:
            body_parts.extend([prefix, Window(char="─", height=1, style="class:overlay.divider")])
        self.list_window = Window(self.list_control, always_hide_cursor=True)
        self.footer_control = FormattedTextControl(
            lambda: self._scroll_footer("select")
        )
        body_parts.extend(
            [
                self.list_window,
                Window(self.footer_control, height=1, wrap_lines=False),
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
        return [
            (offset, item)
            for offset, item in enumerate(self.items)
            if _fuzzy(query, self.label(item))
        ]

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
            if self.multi:
                marker = "◉" if original in self._selected else "○"
            else:
                marker = "›" if current else " "
            style = "class:overlay.selection" if current else "class:overlay.item"
            fragments.append((style, f" {marker} {self.label(item)}\n"))
        return fragments

    def render_text(self) -> str:
        fragments = [*self._fragments(), *self._scroll_footer("select")]
        return "".join(fragment[1] for fragment in fragments)


__all__ = ["SelectList"]
