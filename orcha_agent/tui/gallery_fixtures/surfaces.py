"""Deterministic picker and status surfaces for gallery and golden review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from orcha_agent.core.config import StatusLineConfig
from orcha_agent.tui.overlays.model import ModelOverlay
from orcha_agent.tui.overlays.settings import SettingsOverlay
from orcha_agent.tui.statusline import brand_segment, cache_hit_segment, token_rate_segment


@dataclass(frozen=True)
class _Config:
    user_config_path: Path = Path("/tmp/orcha-gallery-config.toml")
    composer: str = "box"
    theme: str = "dark"
    notify: bool = False
    model: str = "demo:small"
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
    return {
        "Settings": [
            "Appearance | Status | Behaviour | Terminal",
            *settings.render_text().splitlines(),
        ],
        "Models": models.render_text().splitlines(),
        "Status": [" | ".join(segment.text for segment in segments if segment)],
    }


__all__ = ["surface_fixtures"]
