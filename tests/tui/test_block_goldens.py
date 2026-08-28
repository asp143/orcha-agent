from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich import box
from rich.console import Console

from orcha_agent.tui.blocks import DEFAULT_RENDERERS
from orcha_agent.tui.frame import Block, BlockState

GOLDEN_DIR = Path(__file__).with_name("golden")

DARK_THEME: dict[str, Any] = {
    "id": "dark-like",
    "colors": {
        "accent": "#89b4fa",
        "dim": "#6c7086",
        "error": "#f38ba8",
        "muted": "#6c7086",
        "success": "#a6e3a1",
        "text": "#cdd6f4",
        "thinkingOff": "#89dceb",
        "thinkingText": "#7f849c",
        "toolDiffAdded": "#a6e3a1",
        "toolDiffContext": "#7f849c",
        "toolDiffRemoved": "#f38ba8",
        "toolErrorBg": "#321d2a",
        "toolOutput": "#cdd6f4",
        "toolPendingBg": "#1e293b",
        "toolSuccessBg": "#1f2d28",
        "toolTitle": "#89b4fa",
        "userMessageBg": "#313244",
        "userMessageText": "#cdd6f4",
        "warning": "#f9e2af",
    },
    "symbols": {"boxRound": box.ROUNDED},
}

ANSI_THEME: dict[str, Any] = {
    "id": "ansi-like",
    "colors": {
        "accent": "cyan",
        "dim": "bright_black",
        "error": "red",
        "muted": "bright_black",
        "success": "green",
        "text": "white",
        "thinkingOff": "cyan",
        "thinkingText": "bright_black",
        "toolDiffAdded": "green",
        "toolDiffContext": "bright_black",
        "toolDiffRemoved": "red",
        "toolErrorBg": "black",
        "toolOutput": "white",
        "toolPendingBg": "black",
        "toolSuccessBg": "black",
        "toolTitle": "cyan",
        "userMessageBg": "black",
        "userMessageText": "white",
        "warning": "yellow",
    },
    "symbols": {"boxRound": box.ASCII},
}


@dataclass(frozen=True)
class Sample:
    name: str
    block: Block
    expanded: bool = False


def _block(
    block_id: str,
    kind: str,
    *,
    state: BlockState = BlockState.SETTLED,
    **data: object,
) -> Block:
    return Block(id=block_id, kind=kind, state=state, data=dict(data))


SAMPLES = (
    Sample("user", _block("user", "user", text="Run **focused** tests for `blocks`.")),
    Sample(
        "assistant",
        _block(
            "assistant",
            "assistant",
            text=(
                "# Result\n\n> Deterministic output\n\n"
                "- alpha\n- beta\n\n"
                "| Kind | State |\n| --- | --- |\n| tool | done |\n\n"
                "---\n\n```python\nprint('ok')\n```"
            ),
        ),
    ),
    Sample(
        "thinking-visible",
        _block("thinking-visible", "thinking", text="**Inspect** the renderer.", visible=True),
    ),
    Sample(
        "thinking-hidden",
        _block(
            "thinking-hidden",
            "thinking",
            text="hidden",
            visible=False,
            spinner_frame=3,
            reasoning_tokens=144,
            tokens_per_second=18.0,
        ),
    ),
    Sample(
        "tool-pending",
        _block(
            "tool-pending",
            "tool",
            state=BlockState.ACTIVE,
            name="read_file",
            args={"path": "orcha_agent/tui/runtime.py"},
            elapsed=0.8,
        ),
    ),
    Sample(
        "tool-collapsed",
        _block(
            "tool-collapsed",
            "tool",
            name="execute",
            args={"command": "uv run pytest -q tests/tui"},
            result={
                "stdout": "\n".join(f"case {index}: passed" for index in range(24)),
                "exit_code": 0,
            },
            elapsed=1.4,
        ),
    ),
    Sample(
        "tool-expanded",
        _block(
            "tool-expanded",
            "tool",
            name="execute",
            args={"command": "uv run pytest -q tests/tui"},
            result={
                "stdout": "\n".join(f"case {index}: passed" for index in range(24)),
                "exit_code": 0,
            },
            elapsed=1.4,
        ),
        expanded=True,
    ),
    Sample(
        "tool-error",
        _block(
            "tool-error",
            "tool",
            name="write_file",
            args={"path": "readonly.txt"},
            result={"status": "error", "error": "permission denied"},
        ),
    ),
    Sample(
        "diff",
        _block(
            "diff",
            "diff",
            text=(
                "--- demo.py\n+++ demo.py\n@@ -1,3 +1,3 @@\n"
                " def run():\n-  return 'old'\n+\treturn 'new'\n context"
            ),
        ),
    ),
    Sample(
        "banner",
        _block("banner", "banner", level="warning", message="Network is slow.\nRetrying once."),
    ),
    Sample("marker", _block("marker", "marker", reason="branch", new="thread.2")),
    Sample(
        "todo",
        _block(
            "todo",
            "todo",
            items=[
                {"text": "add renderers", "done": True},
                {"text": "stabilize goldens", "done": False},
            ],
        ),
    ),
    Sample(
        "subagents",
        _block(
            "subagents",
            "subagents",
            agents=[
                {"name": "Research", "status": "running"},
                {"name": "Review", "status": "idle"},
            ],
        ),
    ),
)


def _capture(sample: Sample, theme: dict[str, Any], width: int) -> str:
    output = StringIO()
    console = Console(
        file=output,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=width,
        legacy_windows=False,
    )
    renderable = DEFAULT_RENDERERS[sample.block.kind](
        sample.block,
        theme,
        width,
        20,
        sample.expanded,
    )
    if renderable is None:
        console.print("<hidden>")
    else:
        console.print(renderable)
    return output.getvalue().replace("\x1b", "<ESC>")


@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize(
    ("theme_name", "theme"),
    [("dark", DARK_THEME), ("ansi", ANSI_THEME)],
)
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample.name)
def test_block_golden(
    sample: Sample,
    theme_name: str,
    theme: dict[str, Any],
    width: int,
    update_goldens: bool,
) -> None:
    actual = _capture(sample, theme, width)
    golden = GOLDEN_DIR / f"{sample.name}.{theme_name}.{width}.txt"
    if update_goldens:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
    assert golden.read_text() == actual
