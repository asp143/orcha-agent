"""Consistent keybinding hint fragments for overlay content and footers."""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.formatted_text import StyleAndTextTuples

from orcha_agent.tui.keys import format_key_name


def key_hint(
    key: str,
    description: str,
    *,
    formatted: bool = False,
) -> StyleAndTextTuples:
    """Render one dim key followed by its muted description."""

    label = key if formatted else format_key_name(key)
    return [("class:dim", label), ("class:muted", f" {description}")]


def key_hints(hints: Iterable[tuple[str, str]], *, width: int | None = None) -> StyleAndTextTuples:
    """Render multiple key hints separated by a muted middle dot."""

    fragments: StyleAndTextTuples = []
    column = 0
    for key, description in hints:
        hint = key_hint(key, description)
        size = sum(len(part[1]) for part in hint)
        if fragments:
            separator = "\n" if width is not None and column + 3 + size > width else " · "
            fragments.append(("class:muted", separator))
            column = 0 if separator == "\n" else column + 3
        fragments.extend(hint)
        column += size
    return fragments


__all__ = ["key_hint", "key_hints"]
