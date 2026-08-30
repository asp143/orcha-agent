"""Effective command and keybinding reference."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from .base import Overlay


class HelpOverlay(Overlay):
    def __init__(self, ctx: Any) -> None:
        lines = ["Commands"]
        for name, command in sorted(ctx.registry.commands.items()):
            lines.append(f"/{name:<14} {command.help}")
        lines.append("")
        lines.append("Keybindings")
        for action, bindings in sorted(getattr(ctx.ui, "effective_keys", {}).items()):
            lines.append(f"{action:<22} {', '.join(bindings) or 'unbound'}")
        self.text = "\n".join(lines)
        body = Window(
            FormattedTextControl(FormattedText([("class:overlay.help", self.text)])),
            wrap_lines=False,
        )
        super().__init__("Help", body, width=0.84, height=0.78)
        self.bindings.add("enter")(lambda _event: self.resolve(None))


__all__ = ["HelpOverlay"]
