from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from orcha_agent.core.events import Advisory
from orcha_agent.tui.blocks.advisory import render
from orcha_agent.tui.frame import Block, BlockState, Frame
from orcha_agent.tui.transcript import Transcript


THEME = {
    "id": "advisory-test",
    "colors": {
        "text": "white",
        "dim": "bright_black",
        "warning": "yellow",
        "error": "red",
    },
}


def _capture(renderable: object, width: int = 80) -> str:
    output = StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    ).print(renderable)
    return output.getvalue()


def _block(*, note: str | None, severity: str = "nit") -> Block:
    return Block(
        id="advisory-1",
        kind="advisory",
        data={
            "note": note,
            "severity": severity,
            "advisor_id": "watchdog",
            "interrupt": severity == "blocker",
        },
    )


@pytest.mark.asyncio
async def test_advisory_event_adds_one_settled_transcript_block() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(
        Advisory(
            note="Check the retry boundary.",
            severity="concern",
            advisor_id="watchdog",
            interrupt=False,
        )
    )

    assert len(frame.blocks) == 1
    block = frame.blocks[0]
    assert block.kind == "advisory"
    assert block.state is BlockState.COMMITTED
    assert block.source_id == "watchdog"
    assert block.data == {
        "note": "Check the retry boundary.",
        "severity": "concern",
        "advisor_id": "watchdog",
        "interrupt": False,
    }


@pytest.mark.asyncio
async def test_none_advisory_produces_no_card() -> None:
    frame = Frame()
    transcript = Transcript(frame)

    await transcript.handle(
        Advisory(
            note=None,
            severity="nit",
            advisor_id="watchdog",
            interrupt=False,
        )
    )

    assert frame.blocks == []
    assert render(_block(note=None), THEME, 80, 20, False) is None


@pytest.mark.parametrize(
    ("severity", "semantic_color", "bold", "dim"),
    [
        ("nit", "bright_black", False, True),
        ("concern", "yellow", True, False),
        ("blocker", "red", True, False),
    ],
)
def test_advisory_cards_use_severity_theme_styles(
    severity: str,
    semantic_color: str,
    bold: bool,
    dim: bool,
) -> None:
    note = f"{severity} guidance"

    rendered = render(_block(note=note, severity=severity), THEME, 80, 20, False)

    assert isinstance(rendered, Panel)
    output = _capture(rendered)
    assert f"Advisor · watchdog · {severity.title()}" in output
    assert note in output

    assert isinstance(rendered.title, Text)
    assert rendered.title.style == Style(
        color=semantic_color,
        bold=bold,
        dim=dim,
    )
    assert rendered.border_style == Style(color=semantic_color, dim=dim)
    assert isinstance(rendered.renderable, Text)
    assert rendered.renderable.style == Style(color="white", dim=dim)
