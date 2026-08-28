from __future__ import annotations

import pytest
from rich import box

from orcha_agent.tui.blocks.hud import render_subagents, render_todo
from orcha_agent.tui.blocks.thinking import render as render_thinking
from orcha_agent.tui.blocks.tool import render as render_tool
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.symbols import SYMBOL_KEYS, SYMBOL_PRESETS, resolve_symbols


def test_all_symbol_presets_cover_the_complete_surface() -> None:
    assert set(SYMBOL_PRESETS) == {"unicode", "nerd", "ascii"}
    for preset in SYMBOL_PRESETS.values():
        assert set(preset) == set(SYMBOL_KEYS)


def test_ascii_symbols_are_terminal_safe_and_supply_rich_boxes() -> None:
    symbols = resolve_symbols("ascii")

    assert all(value.isascii() for key, value in symbols.items() if key in SYMBOL_KEYS)
    assert symbols["status.success"] == "+"
    assert symbols["boxRound"] is box.ASCII
    assert symbols["boxSharp"] is box.ASCII


def test_symbol_overrides_are_validated_and_fall_back_to_preset() -> None:
    symbols = resolve_symbols(
        "unicode",
        {
            "status.success": "YES",
            "icon.model": "M",
            "boxRound.topLeft": "@",
            "boxRound.horizontal": "=",
        },
    )

    assert symbols["boxRound"].top_left == "@"
    assert symbols["boxRound"].top == "="
    assert symbols["status.success"] == "YES"
    assert symbols["icon.model"] == "M"
    assert symbols["status.error"] == SYMBOL_PRESETS["unicode"]["status.error"]

    with pytest.raises(ValueError, match="unknown symbol"):
        resolve_symbols("unicode", {"missing.key": "x"})
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_symbols("unicode", {"status.success": ""})


def test_non_utf_terminal_forces_ascii_symbols() -> None:
    symbols = resolve_symbols("nerd", encoding="ascii")

    assert symbols["icon.model"] == SYMBOL_PRESETS["ascii"]["icon.model"]


def test_renderers_consume_resolved_ascii_status_box_and_spinner_symbols() -> None:
    symbols = resolve_symbols(
        "ascii",
        {
            "status.success": "S",
            "status.error": "E",
            "status.pending": "P",
            "spinner.status": "XY",
            "spinner.activity": "AB",
            "boxRound.topLeft": "@",
            "boxRound.horizontal": "=",
        },
    )
    theme = {
        "colors": {
            "accent": "cyan",
            "error": "red",
            "thinkingOff": "cyan",
            "toolOutput": "default",
            "toolPendingBg": "default",
            "toolSuccessBg": "default",
            "toolErrorBg": "default",
            "toolTitle": "cyan",
        },
        "symbols": symbols,
    }

    pending = render_tool(
        Block(
            id="pending",
            kind="tool",
            state=BlockState.ACTIVE,
            data={"name": "execute", "spinner_frame": 1},
        ),
        theme,
        80,
        1,
        False,
    )
    success = render_tool(
        Block(
            id="success",
            kind="tool",
            data={"name": "execute", "result": {"output": "ok"}},
        ),
        theme,
        80,
        1,
        False,
    )
    error = render_tool(
        Block(
            id="error",
            kind="tool",
            data={"name": "execute", "result": {"error": "bad"}},
        ),
        theme,
        80,
        1,
        False,
    )
    folded = render_tool(
        Block(id="folded", kind="tool", data={"name": "read", "result": "ok"}),
        theme,
        80,
        2,
        False,
    )
    grouped = render_tool(
        Block(
            id="grouped",
            kind="tool",
            data={
                "name": "execute",
                "calls": [
                    {"args": {}, "result": "one"},
                    {"args": {}, "result": "two"},
                ],
            },
        ),
        theme,
        80,
        1,
        False,
    )
    thinking = render_thinking(
        Block(
            id="thinking",
            kind="thinking",
            data={"visible": False, "spinner_frame": 1},
        ),
        theme,
        80,
        1,
        False,
    )
    todo = render_todo(
        Block(
            id="todo",
            kind="todo",
            data={"items": [{"text": "done", "done": True}, {"text": "wait"}]},
        ),
        theme,
        80,
        3,
        False,
    )
    subagents = render_subagents(
        Block(
            id="agents",
            kind="subagents",
            data={"agents": [{"name": "worker"}], "spinner_frame": 1},
        ),
        theme,
        80,
        2,
        False,
    )

    assert pending is not None and pending.plain.startswith("B ")
    assert success is not None and success.plain.startswith("S ")
    assert error is not None and error.plain.startswith("E ")
    assert folded is not None and folded.plain.startswith("@= ")
    assert grouped is not None and "execute x2" in grouped.plain
    assert thinking.plain.startswith("B ")
    assert todo is not None and "\nS done\nP wait" in todo.plain
    assert subagents is not None and "\nY worker" in subagents.plain
    assert all(
        rendered.plain.isascii()
        for rendered in (
            pending,
            success,
            error,
            folded,
            grouped,
            thinking,
            todo,
            subagents,
        )
        if rendered is not None
    )
