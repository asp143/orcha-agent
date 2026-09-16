from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from orcha_agent.core.config import StatusLineConfig
from orcha_agent.tui.notify import DesktopNotifier
from orcha_agent.tui.overlays.settings import SettingsOverlay, persist_setting
from orcha_agent.tui.statusline import brand_segment, cache_hit_segment, token_rate_segment


@dataclass(frozen=True)
class SettingsConfig:
    user_config_path: Path
    composer: str = "box"
    theme: str = "dark"
    notify: bool = False
    statusline: StatusLineConfig = field(default_factory=StatusLineConfig)


def test_settings_preserves_unrelated_toml_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '# retained\nmodel = "demo:tiny"\n[ui]\n# layout\ncomposer = "box"\n[plugins]\nx = true'
    )
    persist_setting(path, "ui", "composer", "rail")
    persist_setting(path, "ui.statusline", "preset", "powerline")
    assert "# retained" in path.read_text() and "# layout" in path.read_text()
    assert tomllib.loads(path.read_text()) == {
        "model": "demo:tiny",
        "ui": {"composer": "rail", "statusline": {"preset": "powerline"}},
        "plugins": {"x": True},
    }


def test_settings_overlay_persists_without_closing(tmp_path: Path) -> None:
    applied = []
    ctx = SimpleNamespace(
        cfg=SettingsConfig(tmp_path / "config.toml"),
        ui=SimpleNamespace(apply_settings=applied.append),
    )
    picker = SettingsOverlay(ctx)
    picker._accept("composer", SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None)))
    assert ctx.cfg.composer == "claude"
    assert applied == [ctx.cfg]
    assert not picker.done
    assert tomllib.loads(ctx.cfg.user_config_path.read_text())["ui"]["composer"] == "claude"


def test_status_rate_is_per_turn_and_cache_ratio_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr("orcha_agent.tui.statusline.monotonic", lambda: 20.0)
    ctx = SimpleNamespace(
        plugin_states={
            "statusbar": {
                "_turn_started": 10.0,
                "_turn_output_start": 900,
                "output_tokens": 1000,
                "input_tokens": 100,
                "cache_read_tokens": 75,
                "cache_known": True,
            }
        },
        ui=SimpleNamespace(),
    )
    assert token_rate_segment(ctx).text == "10.0 tok/s"
    assert cache_hit_segment(ctx).text == "cache 75%"
    assert brand_segment(ctx).text == "* orcha 10s"


@pytest.mark.asyncio
async def test_notifications_obey_focus_with_bell_fallback() -> None:
    writes = []
    notifier = DesktopNotifier(
        enabled=True,
        output=SimpleNamespace(write_raw=writes.append, flush=lambda: None),
        clock=lambda: 0.0,
        which=lambda _: None,
        osc9_supported=False,
        run_terminal=lambda fn: fn(),
    )
    notifier.set_focused(True)
    assert not await notifier.notify("Done", "Completed")
    notifier.set_focused(False)
    assert await notifier.notify("Done", "Completed")
    assert writes == ["\x07"]


@pytest.mark.parametrize("surface", ["Settings", "Models", "Status"])
def test_surface_gallery_goldens(surface: str, update_goldens: bool) -> None:
    from orcha_agent.tui.gallery_fixtures.surfaces import surface_fixtures
    from orcha_agent.tui.overlays.base import Overlay

    rows = surface_fixtures()[surface]
    actual = "\n".join(Overlay.render_lines(surface, rows, width=76, height=len(rows) + 2)) + "\n"
    golden = Path(__file__).with_name("golden") / f"omp-{surface.lower()}.76.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual


@pytest.mark.asyncio
async def test_settings_keyboard_tabs_apply_and_escape(tmp_path: Path) -> None:
    import asyncio
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from orcha_agent.tui.runtime import ApplicationRuntime

    ctx = SimpleNamespace(cfg=SettingsConfig(tmp_path / "config.toml"), ui=SimpleNamespace())
    overlay = SettingsOverlay(ctx)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show(overlay))
        async with asyncio.timeout(2):
            while runtime.active_overlay is not overlay:
                await asyncio.sleep(0)
        pipe.send_bytes(b"\r\t\r\x1b")
        await asyncio.wait_for(shown, 2)
        assert ctx.cfg.composer == "claude"
        assert ctx.cfg.statusline.preset == "full"
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 2)


def test_settings_updates_existing_tui_alias_instead_of_shadowed_ui(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[ui]\ncomposer = "borderless"\n[tui]\ncomposer = "box"\n[tui.statusline]\npreset = "default"\nseparator = "pipe"\n'
    )
    ctx = SimpleNamespace(cfg=SettingsConfig(path), ui=SimpleNamespace())
    picker = SettingsOverlay(ctx)
    event = SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None))
    picker._accept("composer", event)
    picker.category = 1
    picker._load()
    picker._accept("preset", event)
    saved = tomllib.loads(path.read_text())
    assert saved["tui"]["composer"] == "claude"
    assert saved["ui"]["composer"] == "borderless"
    assert saved["tui"]["statusline"] == {"preset": "full", "separator": "pipe"}


def test_vim_segment_reads_actual_application_mode_without_status_callback() -> None:
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.key_binding.vi_state import InputMode
    from orcha_agent.tui.statusline import vim_segment

    application = SimpleNamespace(
        editing_mode=EditingMode.VI, vi_state=SimpleNamespace(input_mode=InputMode.NAVIGATION)
    )
    ctx = SimpleNamespace(ui=SimpleNamespace(application=application))
    assert vim_segment(ctx).text == "VI-NAVIGATION"
    application.editing_mode = EditingMode.EMACS
    assert vim_segment(ctx) is None


def test_settings_accepts_commented_table_header(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[ui] # appearance\ncomposer = "box"\n')
    persist_setting(path, "ui", "composer", "rail")
    assert path.read_text().startswith("[ui] # appearance\n")
    assert tomllib.loads(path.read_text())["ui"]["composer"] == "rail"
