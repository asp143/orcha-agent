"""Visual contracts for tool card chrome, glyphs, and transcript emphasis."""

from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from orcha_agent.tui.blocks import DEFAULT_RENDERERS, DEFAULT_THEME
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.symbols import resolve_symbols


def _capture(block: Block, *, width: int = 80, expanded: bool = False) -> tuple[Text, Console]:
    output = StringIO()
    theme = {**DEFAULT_THEME, "symbols": resolve_symbols("unicode")}
    console = Console(
        file=output, width=width, force_terminal=True, color_system="truecolor", no_color=False
    )
    console.print(DEFAULT_RENDERERS[block.kind](block, theme, width, 40, expanded))
    return Text.from_ansi(output.getvalue()), console


@pytest.mark.parametrize("width", (80, 120))
def test_long_tool_header_retains_title_path_styles_and_right_aligned_time(width: int) -> None:
    value, console = _capture(
        Block(
            id="read",
            kind="tool",
            data={
                "name": "read",
                "args": {"path": "nested/" * 20 + "source.py"},
                "result": "hello",
                "duration": 1.2,
            },
        ),
        width=width,
    )
    top = value.plain.splitlines()[1]
    assert len(top) == width and top.endswith("Took 1.2s ─╮")
    assert "source.py" in top and "…" in top
    title = value.get_style_at_offset(console, value.plain.index("Read"))
    glyph = value.get_style_at_offset(console, value.plain.index("≡"))
    path = value.get_style_at_offset(console, value.plain.index("source.py"))
    timing = value.get_style_at_offset(console, value.plain.index("Took"))
    assert title.bold and title.color != path.color
    assert glyph.color != title.color and timing.dim


def test_write_expand_reveals_hidden_lines_without_changing_preview_anchor() -> None:
    block = Block(
        id="write",
        kind="tool",
        state=BlockState.ACTIVE,
        data={
            "name": "write",
            "args": {"path": "a.py", "content": "\n".join(f"line {i}" for i in range(15))},
        },
    )
    collapsed, _ = _capture(block)
    expanded, _ = _capture(block, expanded=True)
    assert "line 0" in collapsed.plain and "line 14" not in collapsed.plain
    assert "line 0" in expanded.plain and "line 14" in expanded.plain
    assert "(streaming)" not in collapsed.plain
    assert collapsed.plain.count("⠋") == 1


def test_markdown_headings_have_accent_and_bold_styles() -> None:
    value, console = _capture(
        Block(id="markdown", kind="assistant", data={"text": "## Heading\n\nBody"})
    )
    heading = value.get_style_at_offset(console, value.plain.index("Heading"))
    body = value.get_style_at_offset(console, value.plain.index("Body"))
    assert heading.bold and heading.color != body.color


def test_todo_distinguishes_pending_active_and_complete_without_color() -> None:
    value, _ = _capture(
        Block(
            id="todos",
            kind="todo",
            data={
                "items": [
                    {"text": "next"},
                    {"text": "working", "status": "in_progress"},
                    {"text": "finished", "done": True},
                ]
            },
        )
    )
    assert "○ next" in value.plain
    assert "● working" in value.plain
    assert "✓ finished" in value.plain


def test_marker_rule_is_centered_and_dim() -> None:
    value, console = _capture(Block(id="marker", kind="marker", data={"reason": "compact"}))
    row = value.plain.rstrip("\n")
    left, _, right = row.partition(" ⊟ compacted ")
    assert len(row) == 80 and abs(len(left) - len(right)) <= 1
    assert value.get_style_at_offset(console, 0).dim
