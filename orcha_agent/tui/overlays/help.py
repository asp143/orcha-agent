"""Effective command and keybinding reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples

from orcha_agent.tui.keys import format_key_bindings

from .base import ScrollableOverlay
from .hints import key_hint


def _action_description(action: str) -> str:
    return action.replace("_", " ").replace(".", " ")


class KeyBindingsCard:
    """Reusable renderable containing the effective keybinding reference."""

    def __init__(self, effective_keys: Mapping[str, Sequence[str]]) -> None:
        self.effective_keys = effective_keys

    def rows(self) -> tuple[StyleAndTextTuples, ...]:
        return tuple(
            key_hint(
                format_key_bindings(bindings) or "Unbound",
                _action_description(action),
                formatted=True,
            )
            for action, bindings in sorted(self.effective_keys.items())
        )

    def fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for row in self.rows():
            fragments.extend(row)
            fragments.append(("", "\n"))
        return fragments

    @property
    def text(self) -> str:
        return "".join(text for _style, text in self.fragments())


class KeyBindingsOverlay(ScrollableOverlay):
    """Effective keybindings as a standalone surface for ``/keys``."""

    def __init__(self, ctx: Any) -> None:
        self.card = KeyBindingsCard(getattr(ctx.ui, "effective_keys", {}))
        self.text = self.card.text
        super().__init__(
            "Keybindings",
            self.card.rows(),
            width=0.84,
            height=0.78,
        )
        self.bindings.add("enter")(lambda _event: self.resolve(None))


class HelpOverlay(ScrollableOverlay):
    def __init__(self, ctx: Any) -> None:
        rows: list[StyleAndTextTuples] = [
            [("class:overlay.section", "Commands")],
        ]
        plain_lines = ["Commands"]
        for name, command in sorted(ctx.registry.commands.items()):
            line = f"/{name:<14} {command.help}"
            rows.append([("class:muted", line)])
            plain_lines.append(line)
        rows.extend(
            [
                [("", "")],
                [("class:overlay.section", "Keybindings")],
            ]
        )
        plain_lines.extend(["", "Keybindings"])

        card = KeyBindingsCard(getattr(ctx.ui, "effective_keys", {}))
        rows.extend(card.rows())
        plain_lines.extend(card.text.rstrip("\n").splitlines())
        self.text = "\n".join(plain_lines)
        super().__init__("Help", rows, width=0.84, height=0.78)
        self.bindings.add("enter")(lambda _event: self.resolve(None))


__all__ = ["HelpOverlay", "KeyBindingsCard", "KeyBindingsOverlay"]

