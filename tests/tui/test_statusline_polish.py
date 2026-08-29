from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
        "warning": "#ffff00",
        "text": "#ffffff",
    }


def _ctx(tmp_path: Path, *, transparent: bool = False) -> SimpleNamespace:
    registry = Registry()
    values = {
        "model": Segment("MODEL", "statusLineModel"),
        "mode": Segment("MODE", "warning"),
        "path": Segment("PATH", "statusLinePath"),
        "git": Segment("GIT", "statusLineGitClean"),
        "context": Segment("50.0%/100k", "statusLineContext"),
        "cost": Segment("$1.25", "statusLineCost"),
        "subagents": Segment("2", "statusLineSubagents"),
        "session": Segment("SESSION", "text"),
    }
    for name, value in values.items():
        registry._add_status_segment("test", name, lambda _ctx, value=value: value)
    statusline = SimpleNamespace(
        preset="default",
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


def test_transparent_box_uses_a_compact_context_gauge(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, transparent=True)
    ctx.cfg.statusline.left = ("model",)
    ctx.cfg.statusline.right = ("context", "session")

    fragments = render_statusline(ctx, _Theme(), width=80, composer_shape="box")
    plain = _plain(fragments)

    assert len(plain) == 80
    assert plain.index("MODEL") < plain.index("50%") < plain.index("SESSION")
    assert plain.count("─") <= 12
    assert " " * 10 in plain
    assert all("bg:" not in style for style, _text in fragments)
