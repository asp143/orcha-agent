from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.tui.gallery_fixtures.surfaces import surface_fixtures
from orcha_agent.tui.overlays.base import Overlay
from orcha_agent.tui.overlays.help import HelpOverlay
from orcha_agent.tui.overlays.select import SelectList
from orcha_agent.tui.theme import load_themes, select_theme


def luminance(color: str) -> float:
    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in rgb
    ]
    return sum(
        value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )


def test_all_light_background_tokens_stay_light(tmp_path: Path) -> None:
    themes = load_themes(home=tmp_path)
    for name, theme in themes.items():
        if not any(part in name for part in ("light", "latte", "day", "dawn")):
            continue
        for token, color in theme.colors.items():
            if token.endswith("Bg") and color.startswith("#"):
                assert luminance(color) >= 0.4, (name, token, color)


def test_picker_selected_row_has_full_background_and_accent(tmp_path: Path) -> None:
    theme = load_themes(home=tmp_path)["light"]
    picker = SelectList("Pick", ["first", "second"])
    picker._move(1)
    row = picker.list_control.create_content(40, 2).get_line(1)
    assert sum(len(part[1]) for part in row) == 40
    style = theme.pt.get_attrs_for_style_str(row[0][0])
    assert style.bgcolor == theme.color("selectedBg").lstrip("#")
    assert style.color == theme.color("accent").lstrip("#")
    assert style.bold


def test_ported_theme_aliases_resolve_canonical_ids(tmp_path: Path) -> None:
    themes = load_themes(home=tmp_path)
    assert select_theme(themes, "dracula-omp").id == "dracula-purple"
    assert select_theme(themes, "nord-omp").id == "nord-muted"


def test_help_groups_commands_and_preserves_long_descriptions() -> None:
    description = "A long command description that must wrap rather than disappear at the edge."
    commands = {
        "help": SimpleNamespace(plugin="commands_core", help=description),
        "resume": SimpleNamespace(plugin="commands_session", help="Resume"),
        "model": SimpleNamespace(plugin="commands_model", help="Choose model"),
        "mcp": SimpleNamespace(plugin="commands_core", help="Servers"),
        "skills": SimpleNamespace(plugin="commands_core", help="Skills"),
        "custom": SimpleNamespace(plugin="my_plugin", help="Custom"),
    }
    overlay = HelpOverlay(
        SimpleNamespace(
            ui=SimpleNamespace(effective_keys={"submit": ("enter",)}),
            registry=SimpleNamespace(commands=commands),
        )
    )
    for group in ("Core", "Session", "Models", "Plugins/MCP", "Skills", "Custom", "Keybindings"):
        assert group in overlay.text
    assert description in overlay.text
    assert overlay.content_window.wrap_lines()
    overlay._move(8)
    assert overlay.content_window.get_vertical_scroll(overlay.content_window) == 8


@pytest.mark.parametrize("surface", ["Behaviour", "Themes", "Agent Hub"])
@pytest.mark.parametrize("width", [76, 48])
def test_polished_surface_goldens(surface: str, width: int, update_goldens: bool) -> None:
    rows = surface_fixtures()[surface]
    actual = (
        "\n".join(Overlay.render_lines(surface, rows, width=width, height=len(rows) + 2)) + "\n"
    )
    path = (
        Path(__file__).with_name("golden")
        / f"polish-{surface.lower().replace(' ', '-')}.{width}.txt"
    )
    if update_goldens:
        path.write_text(actual)
    assert path.read_text() == actual


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_selection_palette_ansi_golden(
    theme_name: str, tmp_path: Path, update_goldens: bool
) -> None:
    from io import StringIO
    from rich.console import Console
    from orcha_agent.tui.gallery_fixtures.overlay_palette import selection_fixture

    output = StringIO()
    theme = load_themes(home=tmp_path)[theme_name]
    Console(
        file=output, force_terminal=True, color_system="truecolor", no_color=False, width=40
    ).print(selection_fixture(theme), end="")
    actual = output.getvalue()
    path = Path(__file__).with_name("golden") / f"polish-selection-{theme_name}.ansi"
    if update_goldens:
        path.write_text(actual)
    assert path.read_text() == actual


@pytest.mark.parametrize("enabled", [True, False])
def test_auto_compact_setting_survives_config_reload(tmp_path: Path, enabled: bool) -> None:
    from orcha_agent.core.config import load_config
    from orcha_agent.tui.overlays.settings import SettingsOverlay

    path = tmp_path / "config.toml"
    path.write_text(f"[ui]\nauto_compact = {str(not enabled).lower()}\n")
    cfg = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=path)
    ctx = SimpleNamespace(cfg=cfg, ui=SimpleNamespace(apply_settings=lambda _: None))
    picker = SettingsOverlay(ctx)
    picker.category = picker.categories.index("Behaviour")
    picker._load()
    picker._accept("auto_compact", SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None)))
    assert picker._error is None
    assert ctx.cfg.auto_compact is enabled
    assert ctx.rebuild_requested
    restored = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=path)
    assert restored.auto_compact is enabled
