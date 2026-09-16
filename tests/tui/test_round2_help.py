from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.core.config import TuiConfig
from orcha_agent.tui.gallery_fixtures.surfaces import surface_fixtures
from orcha_agent.tui.overlays.base import Overlay
from orcha_agent.tui.overlays.help import HelpOverlay


@pytest.mark.parametrize("vim", [True, False])
def test_help_explains_insert_mode_escape_ladder(vim: bool) -> None:
    overlay = HelpOverlay(
        SimpleNamespace(
            cfg=SimpleNamespace(tui=TuiConfig(vim=vim)),
            ui=SimpleNamespace(effective_keys={}),
            registry=SimpleNamespace(commands={}),
        )
    )
    hint = "Vim insert: Esc leaves insert mode; Esc Esc aborts; Esc×3 opens tree."
    assert (hint in overlay.text) is vim
    assert (hint in overlay.render_text()) is vim


def test_vim_help_gallery_golden(update_goldens: bool) -> None:
    rows = surface_fixtures()["Help"]
    actual = "\n".join(Overlay.render_lines("Help", rows, width=76, height=len(rows) + 2)) + "\n"
    golden = Path(__file__).with_name("golden") / "omp-help.76.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
    assert "Esc×3 opens tree." in actual
