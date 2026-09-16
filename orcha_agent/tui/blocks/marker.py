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
    del budget_rows, expanded
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
    caption = Text(f" {label} ")
    caption.truncate(max(1, width - 2), overflow="ellipsis")
    remaining = max(0, width - caption.cell_len)
    return Text(
        "─" * (remaining // 2) + caption.plain + "─" * (remaining - remaining // 2),
        style=f"dim {theme_value(theme, 'muted')}",
    )
