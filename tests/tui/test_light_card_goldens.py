"""Light-theme card snapshots keep diff backgrounds legible."""

from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from orcha_agent.tui.blocks import DEFAULT_RENDERERS
from orcha_agent.tui.frame import Block
from orcha_agent.tui.gallery_fixtures.blocks import GALLERY_FIXTURES, TOOL_GALLERY_FIXTURES
from orcha_agent.tui.theme import load_themes


@pytest.mark.parametrize("width", (80, 120))
@pytest.mark.parametrize("name", ("read_file", "edit_file", "execute", "diff"))
def test_light_card_golden(name: str, width: int, tmp_path: Path, update_goldens: bool) -> None:
    theme = replace(load_themes(home=tmp_path)["light"], hyperlinks=False)
    kind = "diff" if name == "diff" else "tool"
    fixture = (GALLERY_FIXTURES if kind == "diff" else TOOL_GALLERY_FIXTURES)[name]["success"]
    block = Block(id=name, kind=kind, state=fixture.state, data=fixture.data)
    stream = StringIO()
    console = Console(
        file=stream, width=width, force_terminal=True, color_system="truecolor", no_color=False
    )
    console.print(DEFAULT_RENDERERS[kind](block, theme, width, 30, False))
    actual = stream.getvalue().replace("\x1b", "<ESC>")
    golden = Path(__file__).with_name("golden") / f"polish-{name}.light.{width}.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
