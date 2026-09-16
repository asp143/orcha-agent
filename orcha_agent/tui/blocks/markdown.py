"""Markdown layout cache at top-level block boundaries, independent of transcript state."""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
import re
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
from rich.theme import Theme

# Settled paragraphs/lists/fences survive streaming tail updates. Retain a bounded
# number of blocks, not every historical version of an assistant response.
_REFERENCE = re.compile(r"(?m)^ {0,3}\[[^]\n]+\]:")

_PARSED: OrderedDict[str, tuple[int, int, list[Any]]] = OrderedDict()

_LAYOUTS: OrderedDict[tuple[Any, ...], tuple[Any, ...]] = OrderedDict()


class StreamingMarkdown(Markdown):
    """Preserve Rich's GFM parser and reuse layouts of unchanged root blocks."""

    def __init__(self, markup: str, *args: Any, **kwargs: Any) -> None:
        self.heading_color = kwargs.pop("heading_color", None)
        boundary = line_offset = 0
        prefix: list[Any] = []
        # Reference definitions can retroactively change any earlier inline link.
        # Keep those uncommon documents on the full parser for correctness.
        if not _REFERENCE.search(markup):
            for source, cached in reversed(_PARSED.items()):
                if markup.startswith(source):
                    boundary, line_offset, prefix = cached
                    break
        super().__init__(markup[boundary:], *args, **kwargs)
        for token in self.parsed:
            if token.map:
                token.map = [line + line_offset for line in token.map]
        self.parsed = [*prefix, *self.parsed]
        self.markup = markup
        self._source_lines = markup.splitlines(keepends=True)
        depth = 0
        starts: list[tuple[int, int]] = []
        for index, token in enumerate(self.parsed):
            if depth == 0 and token.map:
                starts.append((index, token.map[0]))
            depth += token.nesting
        if starts and not _REFERENCE.search(markup):
            index, line = starts[-1]
            _PARSED[markup] = (sum(map(len, self._source_lines[:line])), line, self.parsed[:index])
            while len(_PARSED) > 8:
                _PARSED.popitem(last=False)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        styles = (
            {f"markdown.h{level}": f"bold {self.heading_color}" for level in range(1, 7)}
            if self.heading_color
            else {}
        )
        with console.use_theme(Theme(styles)):
            yield from self._render(console, options)

    def _render(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if _REFERENCE.search(self.markup):
            yield from super().__rich_console__(console, options)
            return
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
                "".join(self._source_lines[group[0].map[0] : group[0].map[1]])
                if group[0].map
                else self.markup,
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
