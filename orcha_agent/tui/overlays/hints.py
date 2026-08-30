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


def key_hints(hints: Iterable[tuple[str, str]]) -> StyleAndTextTuples:
    """Render multiple key hints separated by a muted middle dot."""

    fragments: StyleAndTextTuples = []
    for key, description in hints:
        if fragments:
            fragments.append(("class:muted", " · "))
        fragments.extend(key_hint(key, description))
    return fragments


__all__ = ["key_hint", "key_hints"]
