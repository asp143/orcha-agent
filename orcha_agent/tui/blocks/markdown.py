"""Markdown layout cache at top-level block boundaries, independent of transcript state."""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown

# Settled paragraphs/lists/fences survive streaming tail updates. Retain a bounded
# number of blocks, not every historical version of an assistant response.
_LAYOUTS: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()


class StreamingMarkdown(Markdown):
    """Preserve Rich's GFM parser and reuse layouts of unchanged root blocks."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        groups: list[list[Any]] = []
        current: list[Any] = []
        depth = 0
        for token in self.parsed:
            current.append(token)
            depth += token.nesting
            if depth == 0:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        newline = False
        styles = tuple(
            str(console.get_style(f"markdown.{name}", default="none"))
            for name in (
                "text",
                "paragraph",
                "h1",
                "h2",
                "h3",
                "strong",
                "em",
                "link",
                "link_url",
                "code",
                "item",
                "blockquote",
            )
        )
        for group in groups:
            key = (
                repr(group),
                newline,
                self.style,
                self.code_theme,
                self.hyperlinks,
                options.max_width,
                console.color_system,
                console.legacy_windows,
                styles,
            )
            segments = _LAYOUTS.get(key)
            if segments is None:
                piece = copy(self)
                prefix = Markdown("x") if newline else None
                piece.parsed = [*prefix.parsed, *group] if prefix is not None else group
                segments = tuple(Markdown.__rich_console__(piece, console, options))
                if prefix is not None:
                    prefix.style = self.style
                    consumed = len(tuple(Markdown.__rich_console__(prefix, console, options)))
                    segments = segments[consumed:]
                # Very large blocks should not displace the useful small-block cache.
                if len(segments) <= 4096:
                    _LAYOUTS[key] = segments
                    while len(_LAYOUTS) > 128:
                        _LAYOUTS.popitem(last=False)
            else:
                _LAYOUTS.move_to_end(key)
            yield from segments
            element = self.elements.get(group[0].type)
            newline = bool(element and element.new_line)
