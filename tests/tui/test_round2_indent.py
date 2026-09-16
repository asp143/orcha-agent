"""Display-only indentation must not inherit lexer error colors."""

import pytest
from rich.color import Color
from rich.console import Console

from orcha_agent.tui.blocks.diff import render
from orcha_agent.tui.blocks.tool import _read_display_rows
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.theme import load_themes


@pytest.mark.parametrize("indent", ["  ", "\t", " \t "])
@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_diff_indent_has_context_color_and_preserves_keyword_offsets(tmp_path, indent, theme_name):
    theme = load_themes(home=tmp_path)[theme_name]
    block = Block(
        id="indent",
        kind="diff",
        state=BlockState.SETTLED,
        data={
            "path": "example.py",
            "text": (
                "@@ -1,2 +1,2 @@\n"
                f"-{indent}return 'old'\n+{indent}return 'new'\n {indent}return 'context'"
            ),
        },
    )
    console = Console()
    for line in render(block, theme, 80, 20, False).split("\n")[1:]:
        start = line.plain.index("│") + 1
        for offset in range(start, start + len(indent)):
            assert line.get_style_at_offset(console, offset).color == Color.parse(
                theme.colors["toolDiffContext"]
            )
        keyword_start = start + len(indent)
        for offset in range(keyword_start, keyword_start + len("return")):
            assert line.get_style_at_offset(console, offset).color == Color.parse(
                theme.colors["syntaxKeyword"]
            )


@pytest.mark.parametrize("indent", ["  ", "\t", " \t "])
def test_read_preview_keeps_real_indentation_without_error_colors(tmp_path, indent):
    theme = load_themes(home=tmp_path)["dark"]
    rows = _read_display_rows(
        [("1", f"{indent}return 1")], expanded=False, theme=theme, path="example.py"
    )
    line = rows[0]
    console = Console()
    for offset in range(line.plain.index("│") + 1, line.plain.index("return")):
        assert line.get_style_at_offset(console, offset).color != Color.parse("#ed007e")
