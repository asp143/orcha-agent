from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.builtin import commands_review


def _diff(changed_lines: int) -> str:
    additions = "".join(f"+change-{index}\n" for index in range(changed_lines))
    return (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        f"@@ -0,0 +1,{changed_lines} @@\n"
        f"{additions}"
    )


class _Run:
    def __init__(self, run_id: str, *, terminal: bool, status: str) -> None:
        self.id = run_id
        self.result = (
            {"overall": "correct", "explanation": "No issues", "findings": []}
            if status == "done"
            else None
        )
        self.terminal = terminal
        self.status = status
        self.delivered = False


class _Transcript:
    def __init__(self) -> None:
        self.reviews: list[dict[str, Any]] = []

    def append_review(self, review: dict[str, Any]) -> None:
        self.reviews.append(review)


def _context(tmp_path: Path, agents: Any) -> Any:
    return SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path),
        agents=agents,
        transcript=_Transcript(),
        console=SimpleNamespace(
            error=lambda _message: None,
            print=lambda _message: None,
        ),
    )


@pytest.mark.asyncio
async def test_cancellation_during_wait_cancels_settles_and_claims_reviewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run("review-1", terminal=False, status="running")
    wait_started = asyncio.Event()

    class Agents:
        def __init__(self) -> None:
            self.cancelled: list[tuple[str, str]] = []
            self.waited: list[tuple[str, ...]] = []
            self.delivered: list[tuple[str, tuple[str, ...]]] = []

        async def spawn(self, *_args: Any, **_kwargs: Any) -> _Run:
            return run

        async def wait_all(self, ids: Any, *, timeout_s: int) -> None:
            assert timeout_s == 300
            captured = tuple(ids)
            self.waited.append(captured)
            if not run.terminal:
                wait_started.set()
                await asyncio.Future()

        async def cancel(self, run_id: str, *, reason: str) -> None:
            self.cancelled.append((run_id, reason))
            run.status = "aborted"
            run.terminal = True

        async def deliver(self, parent: str, ids: Any) -> None:
            captured = tuple(ids)
            self.delivered.append((parent, captured))
            run.delivered = True

    agents = Agents()
    ctx = _context(tmp_path, agents)
    notifications: list[str] = []
    monkeypatch.setattr(commands_review, "select_diff", lambda *_args: _diff(1))

    async def fake_run_turn(_ctx: Any, notification: str) -> None:
        notifications.append(notification)

    monkeypatch.setattr(commands_review, "run_turn", fake_run_turn)

    review_task = asyncio.create_task(commands_review.review(ctx, "--uncommitted"))
    await wait_started.wait()
    review_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await review_task

    assert agents.cancelled == [("review-1", "cancel")]
    assert agents.waited == [("review-1",), ("review-1",)]
    assert agents.delivered == [("main", ("review-1",))]
    assert run.terminal
    assert run.delivered
    assert ctx.transcript.reviews == []
    assert notifications == []


@pytest.mark.asyncio
async def test_cancellation_during_partial_spawn_claims_settled_and_late_reviewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settled = _Run("review-done", terminal=True, status="done")
    late = _Run("review-late", terminal=False, status="running")
    late_registered = asyncio.Event()
    release_late_spawn = asyncio.Event()

    class Agents:
        def __init__(self) -> None:
            self.cancelled: list[tuple[str, str]] = []
            self.waited: list[tuple[str, ...]] = []
            self.delivered: list[tuple[str, tuple[str, ...]]] = []

        async def spawn(self, _agent_type: str, _prompt: str, **kwargs: Any) -> _Run:
            if kwargs["name"] == "Reviewer 1":
                return settled
            late_registered.set()
            await release_late_spawn.wait()
            return late

        async def wait_all(self, ids: Any, *, timeout_s: int) -> None:
            assert timeout_s == 300
            self.waited.append(tuple(ids))

        async def cancel(self, run_id: str, *, reason: str) -> None:
            self.cancelled.append((run_id, reason))
            assert run_id == late.id
            late.status = "aborted"
            late.terminal = True

        async def deliver(self, parent: str, ids: Any) -> None:
            captured = tuple(ids)
            self.delivered.append((parent, captured))
            for run in (settled, late):
                if run.id in captured:
                    run.delivered = True

    agents = Agents()
    ctx = _context(tmp_path, agents)
    notifications: list[str] = []
    monkeypatch.setattr(
        commands_review,
        "select_diff",
        lambda *_args: _diff(51) + _diff(50).replace("src/app.py", "src/other.py"),
    )

    async def fake_run_turn(_ctx: Any, notification: str) -> None:
        notifications.append(notification)

    monkeypatch.setattr(commands_review, "run_turn", fake_run_turn)

    review_task = asyncio.create_task(commands_review.review(ctx, "--uncommitted"))
    await late_registered.wait()
    review_task.cancel()
    release_late_spawn.set()

    with pytest.raises(asyncio.CancelledError):
        await review_task

    assert agents.cancelled == [("review-late", "cancel")]
    assert agents.waited == [("review-late",)]
    assert agents.delivered == [("main", ("review-done", "review-late"))]
    assert settled.delivered
    assert late.terminal
    assert late.delivered
    assert ctx.transcript.reviews == []
    assert notifications == []
