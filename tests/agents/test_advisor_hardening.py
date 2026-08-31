from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.agents.test_advisor import _Run, _service


@pytest.mark.asyncio
async def test_cancelled_accepted_look_restores_previous_cursor(
    tmp_path: Path,
) -> None:
    run = _Run()
    service, _ctx, agents, _run, _followups = _service(tmp_path, run=run)
    state = service._state("session")
    state.run = run
    state.cursor = "turn-1"
    task = asyncio.create_task(service._look("session", state, "review", cursor="turn-2"))
    service._look_tasks["session"] = task

    await agents.sent_event.wait()
    assert state.cursor == "turn-2"

    service.before_user_prompt()
    await task

    assert state.cursor == "turn-1"


@pytest.mark.asyncio
async def test_timed_out_accepted_look_restores_previous_cursor(
    tmp_path: Path,
) -> None:
    run = _Run()
    service, _ctx, agents, _run, _followups = _service(
        tmp_path,
        run=run,
        timeout_s=0.0,
    )
    state = service._state("session")
    state.run = run
    state.cursor = "turn-1"

    await service._look("session", state, "review", cursor="turn-2")

    assert agents.sent_event.is_set()
    assert state.cursor == "turn-1"


def test_untrusted_cwd_loads_only_user_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    fallback = home / ".config" / "orcha-agent" / "WATCHDOG.md"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("user rules", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    project = tmp_path / "malicious-project"
    cwd = project / "src"
    cwd.mkdir(parents=True)
    (cwd / "WATCHDOG.md").write_text("malicious local rules", encoding="utf-8")
    (project / "WATCHDOG.md").write_text(
        "malicious ancestor rules",
        encoding="utf-8",
    )

    service, ctx, _agents, _run, _followups = _service(tmp_path)
    ctx.cfg.cwd = cwd
    ctx.cfg.trust_cwd = False
    ctx.cfg.trusted_dirs = (project,)

    assert service._watchdog(service._state("session")) == "user rules"


def test_trusted_watchdog_lookup_stops_at_most_specific_trusted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    fallback = home / ".config" / "orcha-agent" / "WATCHDOG.md"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("user rules", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    outer_root = tmp_path / "trusted"
    project_root = outer_root / "project"
    cwd = project_root / "src" / "package"
    cwd.mkdir(parents=True)
    outer_watchdog = outer_root / "WATCHDOG.md"
    project_watchdog = project_root / "WATCHDOG.md"
    outer_watchdog.write_text("out-of-bound rules", encoding="utf-8")
    project_watchdog.write_text("project rules", encoding="utf-8")

    service, ctx, _agents, _run, _followups = _service(tmp_path)
    ctx.cfg.cwd = cwd
    ctx.cfg.trust_cwd = True
    ctx.cfg.trusted_dirs = (outer_root, project_root)
    assert service._watchdog(service._state("with-project-rules")) == "project rules"

    project_watchdog.unlink()
    assert service._watchdog(service._state("fallback-only")) == "user rules"
