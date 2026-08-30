"""omp-style edit diff renderer with fixed gutters and inverse word changes."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from rich.style import Style
from rich.text import Text

from orcha_agent.tui.frame import Block, BlockState

from . import theme_value

_HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_TOKENS = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


def _visualize_indent(value: str) -> str:
    match = re.match(r"[ \t]*", value)
    indent = match.group(0) if match else ""
    return indent.replace(" ", "·").replace("\t", "→") + value[len(indent) :]


def _changes(before: str, after: str) -> tuple[set[int], set[int]]:
    old = _TOKENS.findall(_visualize_indent(before))
    new = _TOKENS.findall(_visualize_indent(after))
    left: set[int] = set()
    right: set[int] = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag != "equal":
            left.update(range(i1, i2))
            right.update(range(j1, j2))
    return left, right


def _append(target: Text, value: str, color: str, changed: set[int]) -> None:
    for index, token in enumerate(_TOKENS.findall(_visualize_indent(value))):
        target.append(token, style=Style(color=color, reverse=index in changed))


def render(block: Block, theme: Any, width: int, budget_rows: int, expanded: bool) -> Text:
    del width, budget_rows
    lines = str(block.data.get("text", block.data.get("diff", ""))).splitlines()
    if block.state is BlockState.ACTIVE:
        while lines and lines[-1].startswith("-") and not lines[-1].startswith("---"):
            lines.pop()
    hunks = sum(1 for line in lines if line.startswith("@@ "))
    maximum = len(lines) if expanded else 40
    output = Text()
    old_line = new_line = 0
    digits = 3
    paired: dict[int, tuple[set[int], set[int]]] = {}
    for index in range(len(lines) - 1):
        if (
            lines[index].startswith("-")
            and not lines[index].startswith("---")
            and lines[index + 1].startswith("+")
            and not lines[index + 1].startswith("+++")
        ):
            paired[index] = _changes(lines[index][1:], lines[index + 1][1:])
    visible_hunks = rendered_lines = 0
    for index, line in enumerate(lines):
        hunk = _HUNK.search(line)
        if hunk:
            visible_hunks += 1
            if not expanded and visible_hunks > 8:
                continue
            old_line, new_line = int(hunk.group(1)), int(hunk.group(2))
            digits = max(3, len(str(old_line)), len(str(new_line)))
            if output:
                output.append("\n")
            output.append("…", style="dim")
            rendered_lines += 1
            continue
        if line.startswith(("---", "+++")):
            continue
        if rendered_lines >= maximum:
            continue
        if output:
            output.append("\n")
        if line.startswith("-"):
            color = str(theme_value(theme, "toolDiffRemoved"))
            output.append(f"-{old_line:>{digits}}│", style=color)
            _append(output, line[1:], color, paired.get(index, (set(), set()))[0])
            old_line += 1
        elif line.startswith("+"):
            color = str(theme_value(theme, "toolDiffAdded"))
            output.append(f"+{new_line:>{digits}}│", style=color)
            _append(output, line[1:], color, paired.get(index - 1, (set(), set()))[1])
            new_line += 1
        else:
            content = line[1:] if line.startswith(" ") else line
            color = str(theme_value(theme, "toolDiffContext"))
            output.append(f" {new_line:>{digits}}│", style=color)
            output.append(_visualize_indent(content), style=f"dim {color}")
            old_line += 1
            new_line += 1
        rendered_lines += 1
    content_lines = [line for line in lines if not line.startswith(("@@ ", "---", "+++"))]
    hidden_lines = max(0, len(content_lines) - maximum)
    hidden_hunks = max(0, hunks - 8)
    if not expanded and (hidden_lines or hidden_hunks):
        if output:
            output.append("\n")
        output.append(
            f"… ({hidden_hunks} more hunks, {hidden_lines} more lines) ⟦Ctrl+O: Expand⟧",
            style="dim",
        )
    return output


__all__ = ["render"]
