from __future__ import annotations

from pathlib import Path

import pytest

from orcha_agent.tui.overlays import Overlay


GOLDEN_DIR = Path(__file__).with_name("golden")


@pytest.mark.parametrize("anchor", ["center", "bottom"])
def test_overlay_chrome_and_clipping_golden(
    anchor: str,
    update_goldens: bool,
) -> None:
    lines = [f"row {index}" for index in range(1, 11)]
    actual = "\n".join(
        Overlay.render_lines(
            "Approval" if anchor == "bottom" else "Picker",
            lines,
            width=32,
            height=7,
            anchor=anchor,
        )
    ) + "\n"
    golden = GOLDEN_DIR / f"overlay-{anchor}.32x7.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual

def test_overlay_backdrop_rows_are_full_width_golden(
    update_goldens: bool,
) -> None:
    actual = "\n".join(
        row.replace(" ", "·")
        for row in Overlay.render_lines(
            "Picker",
            [f"row {index}" for index in range(1, 6)],
            width=32,
            height=7,
            terminal_width=50,
        )
    ) + "\n"
    golden = GOLDEN_DIR / "overlay-backdrop.50x7.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
