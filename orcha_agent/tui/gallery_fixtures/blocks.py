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


def _tool_states(
    name: str,
    *,
    args: dict[str, Any],
    result: Any,
    error: str,
) -> dict[GalleryState, GalleryBlockFixture]:
    return {
        "streaming": _active(
            name=name,
            args=args,
            spinner_frame=1,
            elapsed=0.4,
        ),
        "progress": _active(
            name=name,
            args=args,
            spinner_frame=4,
            elapsed=3.0,
        ),
        "success": _settled(
            name=name,
            args=args,
            result=result,
            duration=1.2,
        ),
        "error": _settled(
            name=name,
            args=args,
            result={"status": "error", "error": error},
            duration=0.8,
        ),
    }


_DIFF = (
    "--- orcha_agent/tui/gallery.py\n"
    "+++ orcha_agent/tui/gallery.py\n"
    "@@ -12,3 +12,3 @@\n"
    " def render():\n"
    "-  return 'before'\n"
    "+  return 'after'"
)


TOOL_GALLERY_FIXTURES: dict[
    str,
    dict[GalleryState, GalleryBlockFixture],
] = {
    "execute": _tool_states(
        "execute",
        args={"command": "uv run pytest -q tests/tui"},
        result={"stdout": "753 passed in 13.91s", "exit_code": 0},
        error="gallery assertion failed",
    ),
    "ls": _tool_states(
        "ls",
        args={"path": "orcha_agent/tui"},
        result={
            "entries": [
                {"path": "blocks", "type": "directory"},
                {"path": "gallery.py", "type": "file"},
                {"path": "runtime.py", "type": "file"},
            ],
            "count": 3,
        },
        error="directory is not readable",
    ),
    "read_file": _tool_states(
        "read_file",
        args={"path": "orcha_agent/tui/blocks/tool.py", "offset": 40},
        result="\n".join(f"{line:>2}  source line {line}" for line in range(41, 56)),
        error="file is not readable",
    ),
    "write_file": _tool_states(
        "write_file",
        args={
            "path": "tmp/gallery.py",
            "content": "\n".join(f"line {line}" for line in range(15)),
        },
        result="Wrote 15 lines",
        error="destination is read-only",
    ),
    "edit_file": _tool_states(
        "edit_file",
        args={"path": "orcha_agent/tui/gallery.py"},
        result={"diff": _DIFF},
        error="edit did not apply",
    ),
    "delete": _tool_states(
        "delete",
        args={"path": "tmp/obsolete.txt"},
        result="Deleted tmp/obsolete.txt",
        error="file is protected",
    ),
    "glob": _tool_states(
        "glob",
        args={"pattern": "**/*.py"},
        result={"matches": [f"src/{index}.py" for index in range(10)], "count": 12},
        error="glob root is not readable",
    ),
    "grep": _tool_states(
        "grep",
        args={"pattern": "renderer", "path": "orcha_agent/tui"},
        result={
            "matches": [
                {"path": "gallery.py", "line": 29, "text": "def _block(renderer, state):"},
                {"path": "blocks/tool.py", "line": 648, "text": "def _render_impl(...):"},
            ],
            "match_count": 2,
            "file_count": 2,
        },
        error="search path is not readable",
    ),
}

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
    "review": {
        "streaming": _active(
            findings=[
                {
                    "title": "Clarify the fallback",
                    "body": "The fallback behavior is not documented by the changed code.",
                    "priority": "P3",
                    "confidence": 0.72,
                    "file": "orcha_agent/tui/gallery.py",
                    "line_start": 47,
                    "line_end": 47,
                }
            ],
            overall="correct",
            explanation="No correctness issue found in the rendered path.",
        ),
        "progress": _active(
            findings=[
                {
                    "title": "Preserve the row budget",
                    "body": "This content can render beyond the rows assigned by the viewport.",
                    "priority": "P2",
                    "confidence": 0.88,
                    "file": "orcha_agent/tui/blocks/review.py",
                    "line_start": 118,
                    "line_end": 126,
                }
            ],
            overall="incorrect",
            explanation="The review is still collecting findings.",
        ),
        "success": _settled(
            findings=[],
            overall="correct",
            explanation="No findings; the change preserves existing behavior.",
        ),
        "error": _settled(
            findings=[
                {
                    "title": "Card can hide the verdict",
                    "body": "A cropped card must retain the overall review result.",
                    "priority": "P1",
                    "confidence": 0.97,
                    "file": "orcha_agent/tui/blocks/review.py",
                    "line_start": 142,
                    "line_end": 151,
                },
                {
                    "title": "Unbounded finding output",
                    "body": "Large reviews must respect the viewport budget.",
                    "priority": "P0",
                    "confidence": 0.99,
                    "file": "orcha_agent/tui/blocks/review.py",
                    "line_start": 132,
                    "line_end": 151,
                },
            ],
            overall="incorrect",
            explanation="Blocking findings remain in the review renderer.",
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
    "tool": TOOL_GALLERY_FIXTURES["execute"],
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
    "working": {
        "streaming": _active(message="Working… (Esc to interrupt)", spinner_frame=1),
        "progress": _active(message="Retrying in 2s… (Esc to cancel)", spinner_frame=4, level="warning"),
        "success": _settled(message="Turn completed.", spinner_frame=7),
        "error": _settled(message="Retry failed.", spinner_frame=9, level="warning"),
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
    "TOOL_GALLERY_FIXTURES",
    "GalleryState",
]
