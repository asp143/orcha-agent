"""R7/R11 lifecycle, inset, glyph and narrow-header regression fixtures."""

from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from orcha_agent.tui.blocks.tool import render
from orcha_agent.tui.frame import Block
from orcha_agent.tui.gallery_fixtures.blocks import TOOL_GALLERY_FIXTURES
from orcha_agent.tui.symbols import resolve_symbols
from orcha_agent.tui.theme import load_themes


def capture(name, state, theme, width):
    fixture = TOOL_GALLERY_FIXTURES[name][state]
    block = Block(id="round2-tool", kind="tool", state=fixture.state, data=fixture.data)
    output = StringIO()
    console = Console(
        file=output, width=width, force_terminal=True, color_system="truecolor", no_color=False
    )
    console.print(render(block, theme, width, 40, False))
    return output.getvalue()


@pytest.mark.parametrize(
    "preset,glyph", [("unicode", "×"), ("nerd", "󰆴"), ("ascii", "D"), ("colorblind", "×")]
)
def test_delete_has_explicit_tool_glyph(preset, glyph, tmp_path):
    symbols = resolve_symbols(preset)
    assert symbols["tool.delete"] == glyph
    theme = load_themes(home=tmp_path, symbols=preset)["dark"]
    output = Text.from_ansi(capture("delete", "success", theme, 80)).plain
    assert f"{glyph} Delete" in output


@pytest.mark.parametrize("width", [80, 120])
def test_read_lifecycle_and_long_path_header_golden(width, tmp_path, update_goldens):
    theme = replace(load_themes(home=tmp_path, symbols="nerd")["dark"], hyperlinks=False)
    states = ("streaming", "success")
    captures = {state: capture("long_read_file", state, theme, width) for state in states}
    actual = "\n".join(f"{state}\n{value}" for state, value in captures.items())
    actual += "\nDelete\n" + capture("delete", "success", theme, width)
    golden = Path(__file__).with_name("golden") / f"round2-tool-polish.{width}.ansi"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
    for state, value in captures.items():
        lines = Text.from_ansi(value).plain.splitlines()
        header = lines[1]
        assert len(header) == width
        assert header.startswith("╭─── ") and header.endswith(" ─╮")
        assert "Read " in header and "Read:" not in header
        assert "…" in header and "renderer_output.py" in header
        assert all(len(line) == width for line in lines[1:])
        if state == "success":
            hint = next(line for line in lines if "more lines" in line)
            assert hint.startswith("│ … 3 more lines")
    write = Text.from_ansi(capture("write_file", "success", theme, width)).plain
    write_hint = next(line for line in write.splitlines() if "more lines" in line)
    assert write_hint.startswith("│ … ")
