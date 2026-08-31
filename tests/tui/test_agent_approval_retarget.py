from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.ledger import CustomEntry, Ledger
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.context import AppContext


def _config(tmp_path: Path) -> Config:
    return Config(
        model="fake:main",
        subagent_model="fake:task",
        summarizer_model=None,
        mode="default",
        backend="test",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )


@pytest.mark.asyncio
async def test_resume_retargets_restored_children_with_target_approvals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_with: list[set[str]] = []

    async def capture_build(*_args: Any, **kwargs: Any) -> object:
        compiled_with.append(set(kwargs["always_allowed"]))
        return object()

    monkeypatch.setattr("orcha_agent.core.agents.build_agent", capture_build)

    with SessionStore(tmp_path / "sessions.db") as store:
        source = store.create(tmp_path, "fake:main", thread_id="source")
        target = store.create(tmp_path, "fake:main", thread_id="target")
        child = store.create(
            tmp_path,
            "fake:task",
            thread_id="target-child",
            parent_session=target.thread_id,
        )
        store.set_plugin_state(
            target.thread_id,
            "approval",
            {"always_allowed": ["target_tool"]},
        )
        Ledger(store).append(
            target.thread_id,
            CustomEntry(
                custom_type="agent_job",
                data={
                    "run_id": "restored-child",
                    "agent_type": "task",
                    "name": "RestoredChild",
                    "description": "restored child",
                    "parent_id": "main",
                    "parent_session": target.thread_id,
                    "session_id": child.thread_id,
                    "thread_id": child.current_thread,
                    "depth": 0,
                    "status": "idle",
                },
            ),
        )
        assert source.current_thread is not None
        ctx = AppContext(
            cfg=_config(tmp_path),
            registry=Registry(),
            bus=EventBus(),
            session=store,
            plugins=[],
            plugin_states={
                "approval": {"always_allowed": ["source_tool"]},
            },
            console=SimpleNamespace(error=lambda _message: None),
            session_id=source.thread_id,
            thread_id=source.current_thread,
        )
        assert ctx.agent is None
        assert ctx.agents is not None
        assert ctx.agents.always_allowed == {"source_tool"}

        await ctx.resume(target.thread_id)
        assert ctx.agents.always_allowed == {"target_tool"}

        restored = ctx.agents.get("restored-child")
        assert restored is not None
        await restored.ensure_agent()

        assert compiled_with == [{"target_tool"}]
