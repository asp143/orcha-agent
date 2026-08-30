from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from orcha_agent.tui.blocks.welcome import render
from orcha_agent.tui.frame import Block


GOLDEN_DIR = Path(__file__).with_name("golden")
SAMPLE = {
    "logo": [
        "████████████",
        "   ██  ██   ",
        "   ██  ██   ",
        "   ▒▒  ██   ",
        "       ██   ",
    ],
    "model": "claude-sonnet-4",
    "mode": "normal",
    "cwd": "~/project",
    "sessions": ["• first (2h ago)", "• second (1d ago)"],
    "hints": ["✓ Trusted folder", "3 plugins loaded", "provider ready"],
    "tip": "Use Ctrl+O to expand tool output.",
}


def _capture(width: int, data: dict[str, object] = SAMPLE) -> str:
    stream = StringIO()
    console = Console(file=stream, width=width, force_terminal=False, color_system=None)
    console.print(render(Block("welcome", "welcome", data=data), None, width, 100, False))
    return stream.getvalue()


@pytest.mark.parametrize("width", [80, 120])
def test_welcome_golden(width: int, update_goldens: bool) -> None:
    actual = _capture(width)
    golden = GOLDEN_DIR / f"welcome.{width}.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual


def test_welcome_uses_omp_columns_and_fixed_slots_at_eighty_columns() -> None:
    empty = _capture(80, {**SAMPLE, "sessions": [], "hints": []})
    full = _capture(
        80,
        {
            **SAMPLE,
            "sessions": ["one", "two", "three", "four", "ignored"],
            "hints": ["one", "two", "three", "four", "ignored"],
        },
    )
    lines = full.splitlines()

    assert len(empty.splitlines()) == len(lines)
    assert all(len(line) == 78 for line in lines)
    # 12-col sample logo → left column is the 13-col minimum + padding
    assert all(line[15] == "│" for line in lines[1:-1])
    assert "ignored" not in full
    assert "Tip: Use Ctrl+O to expand tool output." in full


def test_welcome_tip_wraps_onto_an_indented_continuation_row() -> None:
    rendered = _capture(
        80,
        {**SAMPLE, "logo": WIDE_LOGO, "tip": "Queue another prompt with Ctrl+Q while running."},
    )
    lines = rendered.splitlines()
    assert any("Tip: Queue another prompt with" in line for line in lines)
    assert any("│      Ctrl+Q while running." in line for line in lines)
    assert "…" not in rendered


WIDE_LOGO = [
    " ██████╗ ██████╗  ██████╗██╗  ██╗ █████╗ ",
    "██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗",
    "██║   ██║██████╔╝██║     ███████║███████║",
    "██║   ██║██╔══██╗██║     ██╔══██║██╔══██║",
    "╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║",
]


@pytest.mark.parametrize("width", [80, 100, 120])
def test_welcome_never_truncates_the_real_logo(width: int) -> None:
    rendered = _capture(width, {**SAMPLE, "logo": WIDE_LOGO})
    assert "…" not in rendered
    for row in WIDE_LOGO:
        assert row.strip() in rendered
    # two-column layout survives: the right column keeps its section rules
    assert "──── Recent sessions" in rendered


def test_welcome_left_column_tracks_logo_width() -> None:
    rendered = _capture(100, {**SAMPLE, "logo": WIDE_LOGO})
    lines = rendered.splitlines()
    # frame + 1 pad + 42 logo columns, then the separator
    assert all(line[44] == "│" for line in lines[1:-1])


def test_welcome_falls_back_to_single_column_when_too_narrow() -> None:
    rendered = _capture(60, {**SAMPLE, "logo": WIDE_LOGO})
    assert "…" not in rendered
    assert WIDE_LOGO[0].strip() in rendered
    assert "──── Recent sessions" in rendered
