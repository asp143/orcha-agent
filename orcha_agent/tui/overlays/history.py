"""Full-text prompt history picker."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.buffer import Buffer

from .select import SelectList


class HistoryOverlay(SelectList[str]):
    def __init__(self, ctx: Any) -> None:
        self._history = getattr(ctx.ui, "history", None)
        initial = (
            tuple(self._history.load_history_strings())
            if self._history is not None
            else ()
        )
        super().__init__(
            "Prompt history",
            initial,
            empty_text="No matching prompts",
        )

    def _filter_changed(self, buffer: Buffer) -> None:
        if self._history is not None:
            query = buffer.text.strip()
            self.items = tuple(
                self._history.search(query)
                if query
                else self._history.load_history_strings()
            )
        super()._filter_changed(buffer)


__all__ = ["HistoryOverlay"]
