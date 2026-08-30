from __future__ import annotations

from io import StringIO

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from orcha_agent.tui.blocks.banner import render
from orcha_agent.tui.frame import Block


THEME = {"colors": {"error": "red"}}


def _render(message: str, width: int) -> tuple[str, list[str]]:
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        height=24,
        force_terminal=True,
        color_system="standard",
        no_color=False,
    )
    block = Block(
        id="provider-error",
        kind="banner",
        data={"level": "error", "message": message},
    )

    console.print(render(block, THEME, width, 20, False))

    raw = output.getvalue()
    return raw, Text.from_ansi(raw).plain.splitlines()


@pytest.mark.parametrize(
    ("width", "message"),
    [
        pytest.param(
            40,
            "provider unavailable: " + "x" * 400,
            id="40-single-long-line",
        ),
        pytest.param(
            80,
            "first provider line\n" + "界" * 260 + "\nlast provider line",
            id="80-mixed-wide-and-newlines",
        ),
    ],
)
def test_provider_error_banner_caps_wrapped_content_at_six_rows(
    width: int,
    message: str,
) -> None:
    raw, rows = _render(message, width)

    assert len(rows) <= 8
    assert rows[0].startswith("╭─ Error ")
    assert rows[-1].startswith("╰")
    assert all(row.startswith("│") and row.endswith("│") for row in rows[1:-1])
    assert all(cell_len(row) == width for row in rows)
    assert "\x1b[31m╭─" in raw
    assert "\x1b[31m│" in raw
    assert "\x1b[31m╰" in raw

    assert sum(row.count("…") for row in rows) == 1
    assert rows[-2][1:-1].rstrip().endswith("…")
    assert raw == _render(message, width)[0]
