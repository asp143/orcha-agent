"""Effective command and keybinding reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from orcha_agent.tui.keys import format_key_bindings

from .base import Overlay
from .hints import key_hint


def _action_description(action: str) -> str:
    return action.replace("_", " ").replace(".", " ")


class KeyBindingsCard:
    """Reusable renderable containing the effective keybinding reference."""

    def __init__(self, effective_keys: Mapping[str, Sequence[str]]) -> None:
        self.effective_keys = effective_keys
        self.control = FormattedTextControl(self.fragments)
        self.container = Window(self.control, wrap_lines=False)

    def fragments(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        for action, bindings in sorted(self.effective_keys.items()):
            label = format_key_bindings(bindings) or "Unbound"
            fragments.extend(
                key_hint(label, _action_description(action), formatted=True)
            )
            fragments.append(("", "\n"))
        return fragments

    @property
    def text(self) -> str:
        return "".join(text for _style, text in self.fragments())


class KeyBindingsOverlay(Overlay):
    """Effective keybindings as a standalone surface for ``/keys``."""

    def __init__(self, ctx: Any) -> None:
        self.card = KeyBindingsCard(getattr(ctx.ui, "effective_keys", {}))
        self.text = self.card.text
        super().__init__("Keybindings", self.card.container, width=0.84, height=0.78)
        self.bindings.add("enter")(lambda _event: self.resolve(None))


class HelpOverlay(Overlay):
    def __init__(self, ctx: Any) -> None:
        fragments: StyleAndTextTuples = [("class:overlay.section", "Commands\n")]
        plain_lines = ["Commands"]
        for name, command in sorted(ctx.registry.commands.items()):
            line = f"/{name:<14} {command.help}"
            fragments.append(("class:muted", f"{line}\n"))
            plain_lines.append(line)
        fragments.extend([("", "\n"), ("class:overlay.section", "Keybindings\n")])
        plain_lines.extend(["", "Keybindings"])

        card = KeyBindingsCard(getattr(ctx.ui, "effective_keys", {}))
        fragments.extend(card.fragments())
        plain_lines.extend(card.text.rstrip("\n").splitlines())
        self.text = "\n".join(plain_lines)
        body = Window(
            FormattedTextControl(FormattedText(fragments)),
            wrap_lines=False,
        )
        super().__init__("Help", body, width=0.84, height=0.78)
        self.bindings.add("enter")(lambda _event: self.resolve(None))


__all__ = ["HelpOverlay", "KeyBindingsCard", "KeyBindingsOverlay"]

