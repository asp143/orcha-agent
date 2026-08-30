"""Transcript boundary marker renderer."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text:
    del width, budget_rows, expanded
    supplied = block.data.get("text")
    if supplied:
        label = str(supplied)
    else:
        reason = str(block.data.get("reason", "compact"))
        labels = {
            "compact": "⊟ compacted",
            "clear": "⊠ cleared",
            "branch": f"⎇ branched to {block.data.get('new', '')}".rstrip(),
        }
        label = labels.get(reason, reason)
    return Text(label, style=f"dim {theme_value(theme, 'muted')}")
