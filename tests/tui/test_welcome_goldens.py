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
    assert all(line[27] == "│" for line in lines[1:-1])
    assert "ignored" not in full
    assert "Tip: Use Ctrl+O to expand tool output." in full
