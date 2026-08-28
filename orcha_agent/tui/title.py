"""Deduplicated terminal-title state transitions."""

from __future__ import annotations

import unicodedata
from typing import Any


def _safe(value: Any, *, ascii_only: bool) -> str:
    text = "".join(character for character in str(value) if ord(character) >= 32 and ord(character) != 127)
    text = " ".join(text.split())
    if ascii_only:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


class TerminalTitle:
    """Own the base, turn, and approval title states for one application."""

    def __init__(self, output: Any, *, unicode: bool = True) -> None:
        self.output = output
        self.unicode = unicode
        self.session = "new session"
        self.turn_active = False
        self.approval_pending = False
        self.spinner = "✻" if unicode else "*"
        self._last: str | None = None

    @property
    def value(self) -> str:
        separator = " · " if self.unicode else " - "
        base = f"orcha{separator}{_safe(self.session, ascii_only=not self.unicode) or 'new session'}"
        if self.approval_pending:
            return f"{'⏳' if self.unicode else '[wait]'} {base}"
        if self.turn_active:
            glyph = _safe(self.spinner, ascii_only=not self.unicode) or "*"
            return f"{glyph} {base}"
        return base

    def _emit(self) -> bool:
        value = self.value
        if value == self._last:
            return False
        try:
            self.output.set_title(value)
        except Exception:
            return False
        self._last = value
        return True

    def set_session(self, title: Any) -> bool:
        self.session = _safe(title, ascii_only=not self.unicode) or "new session"
        return self._emit()

    def set_turn(self, active: bool, *, spinner: str | None = None) -> bool:
        self.turn_active = bool(active)
        if spinner is not None:
            self.spinner = spinner if self.unicode else "*"
        return self._emit()

    def set_approval(self, pending: bool) -> bool:
        self.approval_pending = bool(pending)
        return self._emit()

    def set_spinner(self, spinner: str) -> bool:
        self.spinner = spinner if self.unicode else "*"
        return self._emit()


__all__ = ["TerminalTitle"]
