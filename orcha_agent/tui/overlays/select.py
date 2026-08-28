"""Filterable single- and multi-selection overlays."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from .base import Anchor, Overlay

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


class SelectList(Overlay, Generic[T]):
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
    ) -> None:
        self.items = tuple(items)
        self.label = label
        self.multi = multi
        self.page_size = max(1, page_size)
        self.empty_text = empty_text
        self.index = 0
        self._selected: set[int] = set()
        self._on_accept = on_accept
        self._on_change = on_change
        self.filter = Buffer(multiline=False)
        self.filter.on_text_changed += self._filter_changed
        self.list_control = FormattedTextControl(self._fragments, focusable=False)
        self.filter_control = BufferControl(buffer=self.filter)
        body_parts: list[Any] = [
            Window(self.filter_control, height=1, style="class:overlay.filter"),
            Window(char="─", height=1, style="class:overlay.divider"),
        ]
        if prefix is not None:
            body_parts.extend([prefix, Window(char="─", height=1, style="class:overlay.divider")])
        self.list_window = Window(self.list_control, always_hide_cursor=True)
        body_parts.append(self.list_window)
        bindings = KeyBindings()

        @bindings.add("up")
        def _up(event: Any) -> None:
            self._move(-1)
            event.app.invalidate()

        @bindings.add("down")
        def _down(event: Any) -> None:
            self._move(1)
            event.app.invalidate()

        @bindings.add("pageup")
        def _page_up(event: Any) -> None:
            self._move(-self.page_size)
            event.app.invalidate()

        @bindings.add("pagedown")
        def _page_down(event: Any) -> None:
            self._move(self.page_size)
            event.app.invalidate()

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
    def focus_target(self) -> BufferControl:
        return self.filter_control

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

    def _move(self, delta: int) -> None:
        filtered = self._filtered_pairs()
        if not filtered:
            self.index = 0
            self._changed()
            return
        self.index = min(len(filtered) - 1, max(0, self.index + delta))
        self._changed()

    def _changed(self) -> None:
        if self._on_change is None:
            return
        filtered = self._filtered_pairs()
        self._on_change(filtered[self.index][1] if filtered else None)

    def _accept(self, value: T | list[T], event: Any) -> None:
        if self._on_accept is None:
            self.resolve(value)
            return
        try:
            result = self._on_accept(value)
        except Exception:
            return
        if not inspect.isawaitable(result):
            self.resolve(value if result is None else result)
            return

        async def complete() -> None:
            try:
                accepted = await result
            except Exception:
                return
            self.resolve(value if accepted is None else accepted)

        event.app.create_background_task(complete())

    def _fragments(self) -> StyleAndTextTuples:
        filtered = self._filtered_pairs()
        if not filtered:
            return [("class:overlay.empty", f"  {self.empty_text}\n")]
        fragments: StyleAndTextTuples = []
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
        return "".join(fragment[1] for fragment in self._fragments())


__all__ = ["SelectList"]
