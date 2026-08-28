"""Ledger branch picker."""

from __future__ import annotations

from typing import Any

from .select import SelectList


class TreeOverlay(SelectList[Any]):
    def __init__(self, ctx: Any) -> None:
        entries = tuple(ctx.ledger.all(ctx.session_id))
        by_id = {entry.id: entry for entry in entries}
        leaf = ctx.ledger.leaf(ctx.session_id)

        def depth(entry: Any) -> int:
            value = 0
            parent = getattr(entry, "parent_id", None)
            seen: set[str] = set()
            while isinstance(parent, str) and parent in by_id and parent not in seen:
                seen.add(parent)
                value += 1
                parent = getattr(by_id[parent], "parent_id", None)
            return value

        def label(entry: Any) -> str:
            marker = " *" if entry.id == leaf else ""
            summary = getattr(entry, "content", None) or getattr(entry, "summary", None)
            suffix = "" if summary is None else f"  {str(summary).replace(chr(10), ' ')[:64]}"
            return f"{'  ' * depth(entry)}{entry.id}{marker}{suffix}"

        async def branch(entry: Any) -> str:
            await ctx.branch(entry.id)
            return str(entry.id)

        super().__init__(
            "Conversation tree",
            entries,
            label=label,
            empty_text="No ledger entries",
            on_accept=branch,
        )


__all__ = ["TreeOverlay"]
