from __future__ import annotations

from io import StringIO

from rich.color import Color
from rich.console import Console
from rich.text import Text

from orcha_agent.tui.blocks.user import render
from orcha_agent.tui.frame import Block


WIDTH = 40
THEME = {
    "colors": {
        "userMessageBg": "grey19",
        "userMessageText": "white",
    }
}


def test_user_bubble_paints_three_full_width_terminal_rows() -> None:
    bubble = render(
        Block(id="user-1", kind="user", data={"text": "Use **bold** and `code`"}),
        THEME,
        WIDTH,
        20,
        False,
    )
    stream = StringIO()
    console = Console(
        file=stream,
        width=WIDTH,
        force_terminal=True,
        color_system="256",
        no_color=False,
        _environ={"TERM": "dumb"},
    )

    console.print(bubble)

    raw = stream.getvalue()
    parsed = Text.from_ansi(raw)
    assert parsed.plain == (f"{' ' * WIDTH}\n Use bold and code{' ' * 22}\n{' ' * WIDTH}\n")
    assert [len(row) for row in parsed.plain.splitlines()] == [WIDTH, WIDTH, WIDTH]

    background = Color.from_ansi(236)
    for offset, character in enumerate(parsed.plain):
        if character != "\n":
            assert parsed.get_style_at_offset(console, offset).bgcolor == background

    bold_offset = parsed.plain.index("bold")
    assert all(
        parsed.get_style_at_offset(console, offset).bold
        for offset in range(bold_offset, bold_offset + len("bold"))
    )

    code_offset = parsed.plain.index("code")
    assert all(
        parsed.get_style_at_offset(console, offset).reverse
        for offset in range(code_offset, code_offset + len("code"))
    )
