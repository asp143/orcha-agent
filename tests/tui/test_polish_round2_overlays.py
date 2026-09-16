from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from orcha_agent.tui.gallery_fixtures.overlay_palette import overlay_frame_fixture
from orcha_agent.tui.keys import format_key_name
from orcha_agent.tui.overlays.help import HelpOverlay
from orcha_agent.tui.overlays.settings import CATEGORIES, SettingsOverlay
from orcha_agent.tui.theme import load_themes, select_theme


def test_help_cursor_has_full_row_highlight(tmp_path: Path) -> None:
    overlay = HelpOverlay(
        SimpleNamespace(
            ui=SimpleNamespace(effective_keys={}),
            registry=SimpleNamespace(commands={"help": SimpleNamespace(help="Show help")}),
        )
    )
    overlay._move(2)
    content = overlay.content_control.create_content(72, 10)
    selected = content.get_line(content.cursor_position.y)
    theme = load_themes(home=tmp_path)["dark"]
    assert sum(len(part[1]) for part in selected) == 72
    for part in selected:
        if part[1].strip():
            attrs = theme.pt.get_attrs_for_style_str(part[0])
            assert attrs.bgcolor == theme.color("selectedBg").lstrip("#")
            assert attrs.color == theme.color("accent").lstrip("#")
    for _ in range(1100):
        repeated = overlay.content_control.create_content(72, 10)
        assert repeated.get_line(repeated.cursor_position.y) == selected
    assert "Enter close" not in overlay.render_text()
    assert "Esc close" in overlay.render_text()


def test_paging_key_names_preserve_capitalization() -> None:
    assert format_key_name("PgUp/PgDn") == "PgUp/PgDn"


def test_each_setting_has_exactly_one_tab() -> None:
    names = [name for options in CATEGORIES.values() for name, _ in options]
    assert len(names) == len(set(names))


def test_moved_terminal_settings_still_persist_to_tui(tmp_path: Path) -> None:
    from orcha_agent.core.config import load_config

    path = tmp_path / "config.toml"
    cfg = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=path)
    ctx = SimpleNamespace(cfg=cfg, ui=SimpleNamespace(apply_settings=lambda _: None))
    picker = SettingsOverlay(ctx)
    picker.category = picker.categories.index("Behaviour")
    picker._load()
    event = SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None))
    picker._accept("vim", event)
    assert picker._error is None
    assert ctx.cfg.tui.vim
    restored = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=path)
    assert restored.tui.vim


@pytest.mark.parametrize(
    "base,old,new",
    [
        ("dracula", "dracula-dark", "dracula-purple"),
        ("nord", "nord-dark", "nord-muted"),
        ("rose-pine", "rose-pine-classic", "rose-pine-raised"),
        ("catppuccin-latte", "catppuccin-latte-classic", "catppuccin-latte-mauve"),
    ],
)
def test_distinct_theme_variants_have_descriptive_names(tmp_path, base, old, new) -> None:
    themes = load_themes(home=tmp_path)
    assert themes[base].colors != themes[new].colors
    assert select_theme(themes, old) is themes[new]
    assert old not in themes


@pytest.mark.parametrize("surface", ["help", "hub", "models", "tree", "settings"])
@pytest.mark.parametrize("width", [80, 120])
def test_live_overlay_frame_golden(surface, width, tmp_path, update_goldens) -> None:
    theme = load_themes(home=tmp_path)["dark"]
    frame = overlay_frame_fixture(theme, width, surface)
    if surface == "hub":
        assert "R revive" in frame.plain and "Y copy" in frame.plain
        assert "PgUp/PgDn" in frame.plain
    if surface in {"models", "tree"}:
        assert frame.plain.splitlines()[1].strip("│ ")
        assert not frame.plain.splitlines()[1].strip("│ ─") == ""
    output = StringIO()
    Console(
        file=output, force_terminal=True, color_system="truecolor", no_color=False, width=width
    ).print(frame, end="")
    actual = output.getvalue()
    path = Path(__file__).with_name("golden") / f"polish-round2-{surface}.{width}.ansi"
    if update_goldens:
        path.write_text(actual)
    assert path.read_text() == actual
