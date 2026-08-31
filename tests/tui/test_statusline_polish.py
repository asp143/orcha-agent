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
        ("model", "mode", "path", "git", "context", "cost"),
        ("subagents", "session"),
    )
    assert [name for name, _segment in visible_segments(ctx)] == [
        "model",
        "mode",
        "path",
        "git",
        "context",
        "cost",
        "subagents",
        "session",
    ]

    fragments = render_statusline(ctx, _Theme(), width=120, composer_shape="borderless")
    plain = _plain(fragments)
    assert plain.index(" MODEL ") < plain.index(" MODE ") < plain.index(" PATH ")
    assert plain.index("GIT") < plain.index("50.0%/100k") < plain.index("$1.25")
    assert plain.index(" 2 ") < plain.index(" SESSION ")
    assert "│" in plain
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

    fragments = render_statusline(ctx, _Theme(), width=80, composer_shape="box")
    plain = _plain(fragments)

    assert _cell_width(fragments) == 80
    assert plain.index("MODEL") < plain.index("50%") < plain.index("SESSION")
    assert plain.count("━") == 10
    assert plain.count("─") == 10
    assert all("bg:" not in style for style, _text in fragments)


@pytest.mark.parametrize(
    ("width", "gauge_visible"),
    [
        (23, False),
        (24, True),
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
        assert plain == "━━━━━━━━━━────────── 50%"
    else:
        assert "MODEL" in plain
        assert "SESSION" in plain
        assert "50%" not in plain
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

    fragments = render_statusline(ctx, _Theme(), width=80, composer_shape="box")
    plain = _plain(fragments)

    assert plain.count("━") == filled_cells
    assert plain.count("─") == 20 - filled_cells
    assert f"{percent:g}%" in plain
    assert sum(text.count("━") for style, text in fragments if f"class:{token}" in style) == filled_cells
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
    assert plain.count(filled) == 10
    assert plain.count(empty) == 10
    assert "50%" in plain


@pytest.mark.parametrize("transparent", (False, True))
def test_pressure_removes_provider_before_truncating_model_and_keeps_gauge(
    tmp_path: Path,
    transparent: bool,
) -> None:
    ctx = _ctx(
        tmp_path,
        transparent=transparent,
        model_text="provider-name:extremely-long-model-name",
    )
    ctx.cfg.statusline.separator = "none"
    ctx.cfg.statusline.left = ("model",)
    ctx.cfg.statusline.right = ("context",)

    provider_trimmed = render_statusline(ctx, _Theme(), width=60, composer_shape="box")
    model_trimmed = render_statusline(ctx, _Theme(), width=40, composer_shape="box")
    provider_plain = _plain(provider_trimmed)
    model_plain = _plain(model_trimmed)

    assert "provider-name:" not in provider_plain
    assert "extremely-long-model-name" in provider_plain
    assert "provider-name:" not in model_plain
    assert "extremely-long-model-name" not in model_plain
    assert "extremely" in model_plain
    for fragments, plain, width in (
        (provider_trimmed, provider_plain, 60),
        (model_trimmed, model_plain, 40),
    ):
        assert _cell_width(fragments) == width
        assert plain.count("━") == 10
        assert plain.count("─") == 10
        assert "50%" in plain
