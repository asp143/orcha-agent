from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth

from orcha_agent.core.registry import Registry
from orcha_agent.tui.statusline import PRESETS, Segment, render_statusline, visible_segments
from orcha_agent.tui.symbols import resolve_symbols


class _Theme:
    symbols = resolve_symbols("nerd")
    colors = {
        "statusLineBg": "#111111",
        "statusLineSep": "#333333",
        "statusLineModel": "#00ffff",
        "statusLinePath": "#0088ff",
        "statusLineGitClean": "#00ff00",
        "statusLineContext": "#ff00ff",
        "statusLineCost": "#00ff00",
        "statusLineSubagents": "#00ffff",
        "success": "#00ff00",
        "warning": "#ffff00",
        "error": "#ff0000",
        "muted": "#888888",
        "text": "#ffffff",
    }


def _ctx(
    tmp_path: Path,
    *,
    transparent: bool = False,
    preset: str = "default",
    model_text: str = "MODEL",
    context_text: str = "50.0%/100k",
) -> SimpleNamespace:
    registry = Registry()
    values = {
        "model": Segment(model_text, "statusLineModel"),
        "mode": Segment("MODE", "warning"),
        "path": Segment("PATH", "statusLinePath"),
        "git": Segment("GIT", "statusLineGitClean"),
        "session": Segment("SESSION", "text"),
        "subagents": Segment("2", "statusLineSubagents"),
        "tokens": Segment("10k in 2k out", "text"),
        "cache": Segment("8k read 1k write", "muted"),
        "cost": Segment("$1.25", "statusLineCost"),
        "context": Segment(context_text, "statusLineContext"),
        "time": Segment("3.2s", "muted"),
    }
    for name, value in values.items():
        registry._add_status_segment("test", name, lambda _ctx, value=value: value)
    statusline = SimpleNamespace(
        preset=preset,
        separator="powerline-thin",
        left=None,
        right=None,
        transparent=transparent,
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(
            statusbar=True,
            statusline=statusline,
            composer="box",
            cwd=tmp_path,
        ),
        registry=registry,
        console=SimpleNamespace(width=120, encoding="utf-8"),
    )


def _plain(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in fragments)


def _cell_width(fragments: list[tuple[str, str]]) -> int:
    return sum(get_cwidth(text) for _style, text in fragments)


def test_default_statusline_matches_omp_order_separator_and_colors(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    assert PRESETS["default"] == (
        ("brand", "vim", "model", "cost", "mode", "path", "git", "context"),
        ("subagents", "session"),
    )
    assert [name for name, _segment in visible_segments(ctx)] == [
        "model",
        "cost",
        "mode",
        "path",
        "git",
        "context",
        "subagents",
        "session",
    ]

    fragments = render_statusline(ctx, _Theme(), width=120, composer_shape="borderless")
    plain = _plain(fragments)
    assert plain.index(" MODEL ") < plain.index(" MODE ") < plain.index(" PATH ")
    assert plain.index("$1.25") < plain.index("GIT") < plain.index("50.0%/100k")
    assert plain.index(" 2 ") < plain.index(" SESSION ")
    assert "·" in plain
    for label, token in (
        ("MODEL", "statuslinemodel"),
        ("PATH", "statuslinepath"),
        ("GIT", "statuslinegitclean"),
        ("50.0%/100k", "statuslinecontext"),
        ("$1.25", "statuslinecost"),
        ("SESSION", "text"),
    ):
        assert any(label in text and f"class:{token}" in style for style, text in fragments)


def test_transparent_box_keeps_fixed_plain_context_gauge(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, transparent=True)
    ctx.cfg.statusline.left = ("model",)
    ctx.cfg.statusline.right = ("context", "session")

    fragments = render_statusline(ctx, _Theme(), width=120, composer_shape="box")
    plain = _plain(fragments)

    assert _cell_width(fragments) == 120
    assert plain.index("MODEL") < plain.index("50%") < plain.index("SESSION")
    assert plain.count("━") == 10
    assert plain.count("─") == 10
    assert all("bg:" not in style for style, _text in fragments)


@pytest.mark.parametrize(
    ("width", "gauge_visible"),
    [
        (99, False),
        (100, True),
    ],
)
def test_context_gauge_is_hidden_atomically_below_its_minimum_width(
    width: int,
    gauge_visible: bool,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, transparent=True)
    ctx.cfg.statusline.separator = "none"
    ctx.cfg.statusline.left = ("model",)
    ctx.cfg.statusline.right = ("context", "session")

    fragments = render_statusline(ctx, _Theme(), width=width, composer_shape="box")
    plain = _plain(fragments)

    assert _cell_width(fragments) == width
    if gauge_visible:
        assert "━━━━━━━━━━────────── 50%" in plain
    else:
        assert "MODEL" in plain
        assert "SESSION" in plain
        assert "50.0%/100k" in plain
        assert "━" not in plain
        assert "─" not in plain


@pytest.mark.parametrize(
    ("percent", "token", "filled_cells"),
    [
        (69.9, "success", 14),
        (70.0, "warning", 14),
        (89.9, "warning", 18),
        (90.0, "error", 18),
    ],
)
def test_context_gauge_has_twenty_cells_and_threshold_color(
    percent: float,
    token: str,
    filled_cells: int,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, context_text=f"{percent}%/100k")
    ctx.cfg.statusline.left = ("model",)
    ctx.cfg.statusline.right = ("context", "session")

    fragments = render_statusline(ctx, _Theme(), width=120, composer_shape="box")
    plain = _plain(fragments)

    assert plain.count("━") == filled_cells
    assert plain.count("─") == 20 - filled_cells
    assert f"{percent:g}%" in plain
    assert (
        sum(text.count("━") for style, text in fragments if f"class:{token}" in style)
        == filled_cells
    )
    assert any(f"{percent:g}%" in text and f"class:{token}" in style for style, text in fragments)


@pytest.mark.parametrize("preset", tuple(PRESETS))
@pytest.mark.parametrize("width", (60, 80, 120))
def test_presets_keep_fixed_gauge_without_overflow(
    preset: str,
    width: int,
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path, preset=preset)

    fragments = render_statusline(ctx, _Theme(), width=width, composer_shape="box")
    plain = _plain(fragments)
    filled, empty = ("#", "-") if preset == "ascii" else ("━", "─")

    assert _cell_width(fragments) == width
    if width >= 100:
        assert plain.count(filled) == 10
        assert plain.count(empty) == 10
        assert "50%" in plain
    else:
        assert filled not in plain
        assert empty not in plain


@pytest.mark.parametrize("transparent", (False, True))
def test_pressure_drops_rightmost_whole_segments(tmp_path: Path, transparent: bool) -> None:
    ctx = _ctx(tmp_path, transparent=transparent)
    ctx.cfg.statusline.separator = "none"
    ctx.cfg.statusline.left = ("model", "mode", "path", "git")
    ctx.cfg.statusline.right = ("context", "session")
    text = _plain(render_statusline(ctx, _Theme(), width=25, composer_shape="box"))
    assert "MODEL" in text and "MODE" in text and "PATH" in text
    assert "SESSION" not in text and "50.0%" not in text
    assert len(text) == 25


@pytest.mark.parametrize("width", (80, 120))
def test_real_brand_idle_and_running_keep_segments_stable(
    tmp_path: Path, monkeypatch, width: int
) -> None:
    from orcha_agent.tui.statusline import brand_segment

    ctx = _ctx(tmp_path)
    ctx.plugin_states = {"statusbar": {}}
    ctx.registry._add_status_segment("test", "brand", brand_segment)
    monkeypatch.setattr("orcha_agent.tui.statusline.monotonic", lambda: 1000)
    idle = _plain(render_statusline(ctx, _Theme(), width=width))
    assert "ready" in idle and "orcha          " not in idle
    for started in (999, 990, 1):
        ctx.plugin_states["statusbar"]["_turn_started"] = started
        running = _plain(render_statusline(ctx, _Theme(), width=width))
        for label in ("orcha", "MODEL", "$1.25", "MODE"):
            assert idle.index(label) == running.index(label)
        assert len(idle) == len(running) == width
