"""Read-only inspection of an opaque composer paste chip."""

from .base import ScrollableOverlay


class PasteOverlay(ScrollableOverlay):
    def __init__(self, text: str) -> None:
        super().__init__(
            "Pasted text",
            [[("class:muted", line)] for line in text.split("\n")],
            width=0.9,
            height=0.7,
        )
        self.bindings.add("enter")(lambda _event: self.resolve(None))
