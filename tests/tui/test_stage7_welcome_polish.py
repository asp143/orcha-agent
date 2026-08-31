from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from rich.text import Text

from orcha_agent.builtin.banner import build_welcome
from orcha_agent.tui.blocks.welcome import render
from orcha_agent.tui.frame import Block


WIDE_LOGO = [
    " ██████╗ ██████╗  ██████╗██╗  ██╗ █████╗ ",
    "██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗",
    "██║   ██║██████╔╝██║     ███████║███████║",
    "██║   ██║██╔══██╗██║     ██╔══██║██╔══██║",
    "╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║",
]


def _context(tmp_path: Path, *, trusted: bool) -> Any:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            model="anthropic:claude-opus-5",
            mode="ask",
            cwd=tmp_path,
            symbols="unicode",
            trust_cwd=trusted,
        ),
        console=SimpleNamespace(console=SimpleNamespace(width=80, encoding="utf-8")),
        session=SimpleNamespace(list=lambda: []),
        session_id="current",
        plugins=[],
        registry=SimpleNamespace(providers={}, auth={}),
    )


def _capture(data: dict[str, object], *, width: int = 80, theme: Any = None) -> str:
    stream = StringIO()
    console = Console(file=stream, width=width, force_terminal=False, color_system=None)
    console.print(render(Block("welcome", "welcome", data=data), theme, width, 100, False))
    return stream.getvalue()


@pytest.mark.parametrize(
    ("trusted", "expected"),
    [
        (True, "✓ Trusted folder"),
        (False, "Untrusted folder · project config skipped"),
    ],
)
def test_welcome_uses_precise_project_trust_status(
    tmp_path: Path,
    trusted: bool,
    expected: str,
) -> None:
    assert build_welcome(_context(tmp_path, trusted=trusted))["hints"][0] == expected


def test_welcome_key_hint_comes_from_effective_bindings(tmp_path: Path) -> None:
    ctx = _context(tmp_path, trusted=True)
    ctx.ui = SimpleNamespace(effective_keys={"expand_tools": ("escape p",)})

    assert build_welcome(ctx)["hints"][3] == (("escape p",), "expand tool output")


def test_welcome_key_hint_uses_dim_key_and_muted_description() -> None:
    theme = {"colors": {"border": "green", "dim": "red", "muted": "blue"}}
    data = {
        "logo": ["ORCHA", "", "", "", ""],
        "model": "model",
        "mode": "ask",
        "cwd": "~/project",
        "sessions": [],
        "hints": [(("c-o",), "expand tool output")],
        "tip": "",
    }
    panel = render(Block("welcome", "welcome", data=data), theme, 80, 100, False)
    console = Console(width=80, force_terminal=True, color_system="standard")
    segments = list(console.render(panel, console.options))

    dim_text = "".join(
        segment.text
        for segment in segments
        if segment.style is not None
        and segment.style.color is not None
        and segment.style.color.name == "red"
    )
    muted_text = "".join(
        segment.text
        for segment in segments
        if segment.style is not None
        and segment.style.color is not None
        and segment.style.color.name == "blue"
    )
    assert "Ctrl+O" in dim_text
    assert "expand tool output" in muted_text


def test_untrusted_welcome_wraps_at_eighty_columns_without_changing_height() -> None:
    data = {
        "logo": WIDE_LOGO,
        "model": "anthropic:claude-opus-5",
        "mode": "ask",
        "cwd": "~/project",
        "sessions": [],
        "hints": [
            "Untrusted folder · project config skipped",
            "3 plugins loaded",
            "anthropic provider ready",
            "",
        ],
        "tip": "Use Ctrl+O to expand tool output.",
    }
    rendered = _capture(data)
    lines = rendered.splitlines()

    assert "Untrusted folder · project" in rendered
    assert "config skipped" in rendered
    assert len(lines) == 14
    assert all(Text(line).cell_len == 78 for line in lines)
