from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.notify import DesktopNotifier
from orcha_agent.tui.overlays.session import SessionOverlay
from orcha_agent.tui.overlays.settings import CATEGORIES, persist_setting
from orcha_agent.tui.statusline import _hostname, brand_segment, hostname_segment
from orcha_agent.tui.theme import apply_colorblind, load_themes, select_theme


def test_symbol_preset_does_not_change_palette(tmp_path: Path) -> None:
    original = load_themes(home=tmp_path, symbols="unicode")["dark"]
    symbols_only = load_themes(home=tmp_path, symbols="colorblind")["dark"]
    assert original.colors == symbols_only.colors
    accessible = apply_colorblind(original)
    assert accessible.color("success") == "#56b4e9"
    assert accessible.color("error") == "#e69f00"
    assert accessible.color("toolDiffAddedBg") != original.color("toolDiffAddedBg")
    assert accessible.color("toolDiffRemovedBg") != original.color("toolDiffRemovedBg")
    assert accessible.symbols == original.symbols
    assert apply_colorblind(original, False) is original
    assert original.color("success") != accessible.color("success")


def test_canonical_theme_aliases_keep_palette_and_hide_duplicates(tmp_path: Path) -> None:
    themes = load_themes(home=tmp_path)
    assert "dark-catppuccin" not in themes
    assert select_theme(themes, "dark-catppuccin") is themes["catppuccin-mocha"]
    assert select_theme(themes, "dark-dracula") is themes["dracula-omp"]
    assert select_theme(themes, "dark-nord") is themes["nord-omp"]
    assert select_theme(themes, "dark-gruvbox") is themes["gruvbox-dark"]
    assert themes["dracula"].colors != themes["dracula-omp"].colors


def test_persist_setting_inserts_before_next_section_comments(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[ui]\ntheme = "dark"\n\n# Plugins follow\n[plugins]\nx = true\n')
    persist_setting(path, "ui", "composer", "rail")
    assert path.read_text() == (
        '[ui]\ntheme = "dark"\ncomposer = "rail"\n\n# Plugins follow\n[plugins]\nx = true\n'
    )


def test_hostname_cached_only_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "orcha_agent.tui.statusline.socket.gethostname", lambda: calls.append(1) or "host.example"
    )
    _hostname.cache_clear()
    ctx = SimpleNamespace(plugin_states={"statusbar": {}})
    try:
        assert hostname_segment(ctx).text == hostname_segment(ctx).text == "host"
        assert calls == [1]
        assert ctx.plugin_states == {"statusbar": {}}
    finally:
        _hostname.cache_clear()


@pytest.mark.parametrize("supported,expected", [(True, "\x1b]9;Done\x1b\\"), (False, "\x07")])
@pytest.mark.asyncio
async def test_notification_uses_exactly_one_fallback(supported: bool, expected: str) -> None:
    writes = []
    notifier = DesktopNotifier(
        enabled=True,
        output=SimpleNamespace(write_raw=writes.append, flush=lambda: None),
        which=lambda _: None,
        osc9_supported=supported,
        run_terminal=lambda fn: fn(),
    )
    notifier.set_focused(False)
    assert await notifier.notify("Finished", "Done")
    assert writes == [expected]


def test_sessions_search_cached_labels_without_reopening_ledger() -> None:
    sessions = [
        SimpleNamespace(thread_id="a", title="Parser repair", cwd="/repo", created=None),
        SimpleNamespace(thread_id="b", title="UI polish", cwd="/other", created=None),
    ]
    calls = []
    picker = SessionOverlay(
        SimpleNamespace(
            session=SimpleNamespace(list=lambda: sessions),
            ledger=SimpleNamespace(count=lambda key: calls.append(key) or 4),
        )
    )
    picker.filter.text = "parser"
    assert picker.filtered_items == (sessions[0],)
    picker.render_text()
    picker.render_text()
    assert calls == ["a", "b"]
    assert "4 entries" in picker.label(sessions[0])


@pytest.mark.parametrize(
    "kind,data,activity", [("thinking", {}, "thinking"), ("tool", {"name": "bash"}, "bash")]
)
def test_brand_reports_current_activity(monkeypatch, kind, data, activity) -> None:
    monkeypatch.setattr("orcha_agent.tui.statusline.monotonic", lambda: 12)
    frame = SimpleNamespace(
        blocks=[Block(id="active", kind=kind, state=BlockState.ACTIVE, data=data)]
    )
    ctx = SimpleNamespace(
        plugin_states={"statusbar": {"_turn_started": 10}}, ui=SimpleNamespace(frame=frame)
    )
    assert brand_segment(ctx).text == f"* orcha 2s · {activity}"


def test_settings_expose_explicit_colorblind_and_mouse_modes() -> None:
    options = dict(CATEGORIES["Behaviour"])
    assert options["colorblind"] == (False, True)
    assert options["mouse"] == ("scroll", "full", "off")
