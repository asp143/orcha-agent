from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from orcha_agent.tui.blocks.review import render
from orcha_agent.tui.frame import Block, BlockState, Frame
from orcha_agent.tui.transcript import Transcript


THEME = {
    "id": "review-test",
    "colors": {
        "text": "white",
        "muted": "bright_black",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "dim": "bright_black",
    },
}


def _capture(renderable: object, width: int = 100) -> str:
    output = StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    ).print(renderable)
    return output.getvalue()


def _block(**data: object) -> Block:
    return Block(id="review-1", kind="review", data=data)


def test_review_groups_findings_by_priority_and_renders_review_details() -> None:
    block = _block(
        findings=[
            {
                "priority": "P2",
                "title": "Handle expired credentials",
                "body": "Refresh before retrying the request.",
                "file": "src/auth.py",
                "line_start": 20,
                "line_end": 24,
                "confidence": 0.75,
            },
            {
                "priority": "P0",
                "title": "Prevent data loss",
                "body": "The write can replace committed state.",
                "file": "src/store.py",
                "line_start": 9,
                "line_end": 9,
                "confidence": 0.95,
            },
            {
                "priority": "P2",
                "title": "Close the response",
                "body": "Release the connection after reading.",
                "file": "src/client.py",
                "line_start": 31,
                "confidence": "not-scored",
            },
        ],
        overall="incorrect",
        explanation="  Critical findings must be fixed.  ",
    )

    rendered = render(block, THEME, width=100, budget_rows=30, expanded=False)

    assert isinstance(rendered, Panel)
    output = _capture(rendered)
    assert output.index("P0 · 1 finding") < output.index("P2 · 2 findings")
    assert output.index("Handle expired credentials") < output.index("Close the response")
    assert "src/store.py:9 · confidence 95%" in output
    assert "src/auth.py:20-24 · confidence 75%" in output
    assert "src/client.py:31 · confidence unknown" in output
    assert "Verdict · Incorrect" in output
    assert "Critical findings must be fixed." in output
    assert isinstance(rendered.title, Text)
    assert rendered.title.plain == "Review · Incorrect"
    assert rendered.title.style == Style(color="red", bold=True)
    assert rendered.border_style == Style(color="red", bold=True)


def test_review_without_findings_renders_a_correct_verdict() -> None:
    rendered = render(
        _block(
            findings=[],
            overall="correct",
            explanation="The implementation preserves the required invariants.",
        ),
        THEME,
        width=80,
        budget_rows=10,
        expanded=False,
    )

    assert isinstance(rendered, Panel)
    output = _capture(rendered, width=80)
    assert "finding" not in output.casefold()
    assert "Verdict · Correct" in output
    assert "The implementation preserves the required invariants." in output
    assert isinstance(rendered.title, Text)
    assert rendered.title.plain == "Review · Correct"
    assert rendered.title.style == Style(color="green", bold=True)
    assert rendered.border_style == Style(color="green", bold=True)


def test_review_preserves_verdict_when_row_budget_omits_findings() -> None:
    rendered = render(
        _block(
            findings=[
                {
                    "priority": "P1",
                    "title": f"Finding {index}",
                    "body": "Detailed review guidance",
                    "file": f"src/file_{index}.py",
                    "line_start": index + 1,
                    "confidence": 0.8,
                }
                for index in range(3)
            ],
            overall="incorrect",
            explanation="The findings require changes.",
        ),
        THEME,
        width=48,
        budget_rows=6,
        expanded=False,
    )

    assert isinstance(rendered, Panel)
    output = _capture(rendered, width=48)
    assert "… additional findings omitted" in output
    assert "Verdict · Incorrect" in output
    assert "The findings require changes." in output
    assert len(output.splitlines()) <= 6
    assert max(map(len, output.splitlines())) <= 48


def test_review_collapses_to_a_width_clipped_summary_under_tight_constraints() -> None:
    rendered = render(
        _block(
            findings=[{"priority": "P0", "title": "Hidden by summary"}],
            overall="incorrect",
            explanation="A deliberately long explanation that cannot fit.",
        ),
        THEME,
        width=24,
        budget_rows=2,
        expanded=False,
    )

    assert isinstance(rendered, Text)
    assert rendered.plain.startswith("Review · Incorrect:")
    assert rendered.plain.endswith("…")
    assert len(rendered.plain) <= 24
    assert "Hidden by summary" not in rendered.plain


def test_append_review_commits_a_stable_block_ahead_of_active_tasks() -> None:
    frame = Frame()
    active_task = frame.add("task", {"description": "Reviewing changes"})
    transcript = Transcript(frame)
    data = {
        "findings": [],
        "overall": "correct",
        "explanation": "No findings.",
    }

    review = transcript.append_review(data)
    data["overall"] = "incorrect"

    assert frame.blocks == [review, active_task]
    assert review.kind == "review"
    assert review.state is BlockState.COMMITTED
    assert review.data == {
        "findings": [],
        "overall": "correct",
        "explanation": "No findings.",
    }
    assert active_task.state is BlockState.ACTIVE
    with pytest.raises(RuntimeError, match="cannot update committed block"):
        review.update(overall="incorrect")
