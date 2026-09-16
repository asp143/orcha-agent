"""Deterministic picker and status surfaces for gallery and golden review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from time import monotonic
from typing import Any

from orcha_agent.core.config import StatusLineConfig, TuiConfig
from orcha_agent.tui.frame import Block
from orcha_agent.tui.overlays.help import HelpOverlay
from orcha_agent.tui.overlays.hub import HubOverlay
from orcha_agent.tui.overlays.theme import ThemeOverlay
from orcha_agent.tui.theme import _BUILTIN_NAMES, _PORTED_NAMES, load_theme_file
from orcha_agent.tui.overlays.model import ModelOverlay
from orcha_agent.tui.overlays.settings import SettingsOverlay
from orcha_agent.tui.statusline import brand_segment, cache_hit_segment, token_rate_segment


@dataclass(frozen=True)
class _Config:
    user_config_path: Path = Path("/tmp/orcha-gallery-config.toml")
    composer: str = "box"
    theme: str = "dark"
    notify: bool = False
    mode: str = "ask"
    auto_compact: bool = True
    model: str = "demo:small"
    tui: TuiConfig = field(default_factory=TuiConfig)
    statusline: StatusLineConfig = field(default_factory=StatusLineConfig)


def surface_fixtures() -> dict[str, list[str]]:
    """Use production surfaces; no providers, clocks or config writes are invoked."""
    config = _Config()
    ctx: Any = SimpleNamespace(
        cfg=config,
        ui=SimpleNamespace(),
        registry=SimpleNamespace(
            providers={
                "demo": SimpleNamespace(
                    models=("small", "large"),
                    available=lambda: None,
                    capabilities=SimpleNamespace(max_context=128_000),
                )
            }
        ),
        switch_model=lambda _: None,
        plugin_states={
            "statusbar": {
                "_last_turn_elapsed": 10.0,
                "output_tokens": 100,
                "input_tokens": 1000,
                "cache_read_tokens": 800,
                "cache_known": True,
            }
        },
    )
    settings = SettingsOverlay(ctx)
    models = ModelOverlay(ctx)
    segments = [brand_segment(ctx), token_rate_segment(ctx), cache_hit_segment(ctx)]
    settings_rows = settings.render_text().splitlines()
    settings.category = 3
    settings._load()
    terminal_rows = settings.render_text().splitlines()
    settings.category = 2
    settings._load()
    behaviour_rows = settings.render_text().splitlines()
    # This non-interactive picker fixture only reads names and the current ID.
    # The gallery already loads/validates the palettes for its active theme.
    ctx.ui.themes = {name: None for name in (*_BUILTIN_NAMES, *_PORTED_NAMES.values())}
    ctx.ui.theme = load_theme_file(Path(__file__).parents[1] / "themes" / "dark.json")
    themes = ThemeOverlay(ctx)
    hub = HubOverlay(ctx)
    activity_rows = []
    ctx.plugin_states["statusbar"]["_turn_started"] = monotonic()
    for kind, data in (("thinking", {}), ("tool", {"name": "bash"})):
        ctx.ui.frame = SimpleNamespace(blocks=[Block(id=kind, kind=kind, data=data)])
        activity_rows.append(brand_segment(ctx).text)
    help_overlay = HelpOverlay(
        SimpleNamespace(
            cfg=SimpleNamespace(tui=TuiConfig(vim=True)),
            ui=SimpleNamespace(effective_keys={"newline": ("escape enter",)}),
            registry=SimpleNamespace(commands={"help": SimpleNamespace(help="Show help")}),
        )
    )
    return {
        "Settings": [
            "Appearance | Status | Behaviour | Terminal",
            *settings_rows,
            "Terminal",
            *terminal_rows,
        ],
        "Behaviour": behaviour_rows,
        "Themes": themes.render_text().splitlines(),
        "Agent Hub": hub.render_text().splitlines(),
        "Models": models.render_text().splitlines(),
        "Status": [" | ".join(segment.text for segment in segments if segment), *activity_rows],
        "Help": help_overlay.text.splitlines(),
    }


__all__ = ["surface_fixtures"]
