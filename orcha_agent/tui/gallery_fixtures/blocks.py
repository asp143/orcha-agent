"""One sample per built-in block renderer and gallery lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from orcha_agent.tui.frame import BlockState

GalleryState = Literal["streaming", "progress", "success", "error"]
GALLERY_STATES: tuple[GalleryState, ...] = (
    "streaming",
    "progress",
    "success",
    "error",
)


@dataclass(frozen=True, slots=True)
class GalleryBlockFixture:
    data: dict[str, Any]
    state: BlockState = BlockState.SETTLED


def _active(**data: Any) -> GalleryBlockFixture:
    return GalleryBlockFixture(data, BlockState.ACTIVE)


def _settled(**data: Any) -> GalleryBlockFixture:
    return GalleryBlockFixture(data, BlockState.SETTLED)


_DIFF = (
    "--- orcha_agent/tui/gallery.py\n"
    "+++ orcha_agent/tui/gallery.py\n"
    "@@ -12,3 +12,3 @@\n"
    " def render():\n"
    "-  return 'before'\n"
    "+  return 'after'"
)

_WELCOME = {
    "logo": [
        "████████████",
        "   ██  ██   ",
        "   ██  ██   ",
        "   ▒▒  ██   ",
        "       ██   ",
    ],
    "model": "claude-opus-5",
    "mode": "ask",
    "cwd": "~/src/orcha-agent",
    "sessions": ["• gallery polish (now)", "• renderer parity (2h ago)"],
    "hints": ["✓ Trusted folder", "11 plugins loaded", "anthropic provider ready"],
}


GALLERY_FIXTURES: dict[
    str,
    dict[GalleryState, GalleryBlockFixture],
] = {
    "user": {
        "streaming": _active(text="Drafting **gallery** prompt…", synthetic=True),
        "progress": _active(text="Queued `gallery` prompt", queued=True),
        "success": _settled(text="Render every **block** in the gallery."),
        "error": _settled(text="Retry the `gallery` render.", synthetic=True),
    },
    "assistant": {
        "streaming": _active(text="Rendering the **assistant** block…"),
        "progress": _active(text="## Gallery\n\n- renderer\n- lifecycle"),
        "success": _settled(text="## Done\n\nEvery renderer produced output."),
        "error": _settled(text="> Gallery render recovered from an error."),
    },
    "advisory": {
        "streaming": _active(
            note="Check the active implementation assumption.",
            severity="nit",
            advisor_id="advisor",
        ),
        "progress": _active(
            note="The current change may skip an edge case.",
            severity="concern",
            advisor_id="advisor",
        ),
        "success": _settled(
            note="No blocking issue; retain the verification step.",
            severity="nit",
            advisor_id="advisor",
        ),
        "error": _settled(
            note="Stop before shipping this unsafe path.",
            severity="blocker",
            advisor_id="advisor",
        ),
    },
    "thinking": {
        "streaming": _active(
            text="Inspecting renderer state",
            visible=False,
            spinner_frame=2,
            reasoning_tokens=128,
            tokens_per_second=24.0,
        ),
        "progress": _active(text="**Comparing** the active frame.", visible=True),
        "success": _settled(text="**Verified** renderer output.", visible=True),
        "error": _settled(
            text="Reasoning hidden after failure",
            visible=False,
            spinner_frame=6,
            reasoning_tokens=256,
            tokens_per_second=12.0,
        ),
    },
    "tool": {
        "streaming": _active(
            name="execute",
            args={"command": "uv run pytest tests/tui"},
            spinner_frame=1,
            elapsed=0.4,
        ),
        "progress": _active(
            name="execute",
            args={"command": "uv run pytest tests/tui"},
            spinner_frame=4,
            elapsed=2.3,
        ),
        "success": _settled(
            name="execute",
            args={"command": "uv run pytest tests/tui"},
            result={"stdout": "753 passed in 13.91s", "exit_code": 0},
            elapsed=13.9,
        ),
        "error": _settled(
            name="execute",
            args={"command": "uv run pytest tests/tui"},
            result={"stderr": "gallery assertion failed", "exit_code": 1},
            elapsed=1.2,
        ),
    },
    "diff": {
        "streaming": _active(text=f"{_DIFF}\n-unfinished"),
        "progress": _active(text=_DIFF),
        "success": _settled(text=_DIFF),
        "error": _settled(text=_DIFF.replace("after", "failed render")),
    },
    "banner": {
        "streaming": _active(level="info", message="Gallery stream started."),
        "progress": _active(level="warning", message="Gallery render is still running."),
        "success": _settled(level="info", message="Gallery render completed."),
        "error": _settled(level="error", message="Gallery renderer failed.\nFixture preserved."),
    },
    "marker": {
        "streaming": _active(text="⊟ preparing gallery"),
        "progress": _active(reason="compact"),
        "success": _settled(reason="branch", new="gallery.success"),
        "error": _settled(reason="clear"),
    },
    "todo": {
        "streaming": _active(items=[{"text": "render fixtures"}, {"text": "inspect output"}]),
        "progress": _active(items=[{"text": "render fixtures", "done": True}, {"text": "inspect output"}]),
        "success": _settled(items=[{"text": "render fixtures", "done": True}, {"text": "inspect output", "done": True}]),
        "error": _settled(items=[{"text": "render fixtures", "done": True}, {"text": "fix failed output"}]),
    },
    "subagents": {
        "streaming": _active(agents=[{"id": "scan", "name": "Scan", "status": "starting"}], spinner_frame=1),
        "progress": _active(agents=[{"id": "scan", "name": "Scan", "status": "running", "requests": 2}], spinner_frame=4),
        "success": _settled(agents=[{"id": "scan", "name": "Scan", "status": "success", "requests": 4, "elapsed": 8}]),
        "error": _settled(agents=[{"id": "scan", "name": "Scan", "status": "error", "requests": 3, "elapsed": 5}]),
    },
    "queue": {
        "streaming": _active(prompts=["finish renderer fixtures"]),
        "progress": _active(prompts=["finish renderer fixtures", "run focused tests"]),
        "success": _settled(prompts=["run full suite"]),
        "error": _settled(prompts=["repair gallery failure"]),
    },
    "welcome": {
        "streaming": _active(**_WELCOME, tip="Gallery fixtures are loading."),
        "progress": _active(**_WELCOME, tip="Use --tool to focus one renderer."),
        "success": _settled(**_WELCOME, tip="Use --plain when redirecting output."),
        "error": _settled(**_WELCOME, tip="Use --state error to inspect failures."),
    },
}


__all__ = [
    "GALLERY_FIXTURES",
    "GALLERY_STATES",
    "GalleryBlockFixture",
    "GalleryState",
]
