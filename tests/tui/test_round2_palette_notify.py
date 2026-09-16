from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.tui.notify import DesktopNotifier
from orcha_agent.tui.theme import apply_colorblind, load_themes


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, False])
async def test_windows_terminal_notification_detection(monkeypatch, override):
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.setenv("WT_SESSION", "test-session")
    writes = []
    notifier = DesktopNotifier(
        enabled=True,
        output=SimpleNamespace(write_raw=writes.append, flush=lambda: None),
        which=lambda _: None,
        run_terminal=lambda fn: fn(),
        osc9_supported=override,
    )
    notifier.set_focused(False)
    assert await notifier.notify("Finished", "Done")
    assert writes == (["\x1b]9;Done\x1b\\"] if override is None else ["\x07"])


@pytest.mark.parametrize("background", ["#ffffff", "#102030"])
def test_colorblind_empty_tool_background_uses_theme_surface(tmp_path: Path, background: str):
    theme = load_themes(home=tmp_path)["light"]
    theme = replace(
        theme, colors={**theme.colors, "toolSuccessBg": "", "userMessageBg": background}
    )
    accessible = apply_colorblind(theme)
    for token, accent in [("toolDiffAddedBg", "#56b4e9"), ("toolDiffRemovedBg", "#e69f00")]:
        expected = "#" + "".join(
            f"{round(int(background[i : i + 2], 16) * 0.9 + int(accent[i : i + 2], 16) * 0.1):02x}"
            for i in (1, 3, 5)
        )
        assert accessible.color(token) == expected


def test_light_colorblind_gallery_golden(monkeypatch, update_goldens):
    from io import StringIO

    from rich.console import Console
    from rich.style import Style

    from orcha_agent.tui.gallery_fixtures.palette import light_colorblind_fixture

    for variable in ("NO_COLOR", "TERM", "COLORTERM"):
        monkeypatch.delenv(variable, raising=False)
    Style.parse.cache_clear()
    Style._add.cache_clear()
    stream = StringIO()
    Console(file=stream, width=40, force_terminal=True, color_system="truecolor").print(
        light_colorblind_fixture(), end=""
    )
    actual = stream.getvalue().replace("\x1b", "<ESC>")
    golden = Path(__file__).with_name("golden") / "colorblind-light.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
