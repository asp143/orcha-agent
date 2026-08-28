"""Unified-diff renderer with compact gutters and token highlights."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from orcha_agent.tui.frame import Block, BlockState

from . import theme_value

_HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_TOKENS = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


def _visible_lines(block: Block) -> list[str]:
    lines = str(block.data.get("text", block.data.get("diff", ""))).splitlines()
    if block.state is BlockState.ACTIVE:
        while lines and lines[-1].startswith("-") and not lines[-1].startswith("---"):
            lines.pop()
    return lines


def _changed_indexes(before: str, after: str) -> tuple[set[int], set[int]]:
    old = _TOKENS.findall(_visualize_indent(before))
    new = _TOKENS.findall(_visualize_indent(after))
    matcher = SequenceMatcher(a=old, b=new, autojunk=False)
    old_changed: set[int] = set()
    new_changed: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            old_changed.update(range(i1, i2))
            new_changed.update(range(j1, j2))
    return old_changed, new_changed


def _visualize_indent(value: str) -> str:
    match = re.match(r"[ \t]*", value)
    indent = match.group(0) if match else ""
    return indent.replace(" ", "·").replace("\t", "→") + value[len(indent) :]


def _append_content(target: Text, content: str, color: str, changed: set[int]) -> None:
    for index, token in enumerate(_TOKENS.findall(_visualize_indent(content))):
        style = Style(color=color, reverse=index in changed)
        target.append(token, style=style)


def _highlight_context(content: str, lexer: str) -> Text:
    visible = _visualize_indent(content)
    syntax = Syntax(
        visible,
        lexer,
        theme="ansi_dark",
        background_color="default",
        word_wrap=False,
    )
    return syntax.highlight(visible)


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text:
    del width, budget_rows, expanded
    lines = _visible_lines(block)
    rendered = Text()
    old_line = new_line = 0
    paired: dict[int, tuple[set[int], set[int]]] = {}
    lexer = "text"
    for index in range(len(lines) - 1):
        if lines[index].startswith("-") and not lines[index].startswith("---") and lines[index + 1].startswith("+") and not lines[index + 1].startswith("+++"):
            paired[index] = _changed_indexes(lines[index][1:], lines[index + 1][1:])

    for index, line in enumerate(lines):
        if rendered:
            rendered.append("\n")
        hunk = _HUNK.search(line)
        if hunk:
            old_line, new_line = int(hunk.group(1)), int(hunk.group(2))
            rendered.append(line, style=str(theme_value(theme, "accent")))
            continue
        if line.startswith(("---", "+++")):
            rendered.append(line, style=str(theme_value(theme, "muted")))
            if line.startswith("+++"):
                path = line[4:].strip()
                try:
                    lexer = Syntax.guess_lexer(path)
                except Exception:
                    lexer = "text"
            continue
        if line.startswith("-"):
            changed = paired.get(index, (set(), set()))[0]
            rendered.append(f"{old_line:03d} - ", style=str(theme_value(theme, "toolDiffRemoved")))
            _append_content(rendered, line[1:], str(theme_value(theme, "toolDiffRemoved")), changed)
            old_line += 1
            continue
        if line.startswith("+"):
            changed = paired.get(index - 1, (set(), set()))[1]
            rendered.append(f"{new_line:03d} + ", style=str(theme_value(theme, "toolDiffAdded")))
            _append_content(rendered, line[1:], str(theme_value(theme, "toolDiffAdded")), changed)
            new_line += 1
            continue
        content = line[1:] if line.startswith(" ") else line
        rendered.append(f"{new_line:03d}   ", style=str(theme_value(theme, "toolDiffContext")))
        rendered.append(_highlight_context(content, lexer))
        old_line += 1
        new_line += 1
    return rendered
