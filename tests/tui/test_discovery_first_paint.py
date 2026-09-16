"""Discovery must never gate the interactive terminal's first rendered frame."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.builtin import mcp, skills
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.runtime import ApplicationRuntime, _run_runtime


@pytest.mark.asyncio
async def test_first_paint_while_skills_and_mcp_discovery_are_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    skills_started, mcp_started, painted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_skills = threading.Event()
    release_mcp = asyncio.Event()

    def blocked_skills(*_args, **_kwargs):
        loop.call_soon_threadsafe(skills_started.set)
        if not release_skills.wait(5):
            raise TimeoutError("test did not release skill discovery")
        return {}, []

    async def blocked_mcp(_manager):
        mcp_started.set()
        await release_mcp.wait()

    monkeypatch.setattr(skills, "discover_skills", blocked_skills)
    monkeypatch.setattr(mcp.MCPManager, "start", blocked_mcp)
    registry, bus = Registry(), EventBus()
    for module in (skills, mcp):
        module.register(
            PluginAPI(
                name=module.PLUGIN.name,
                config={},
                state={},
                registry=registry,
                bus=bus,
                request_rebuild=lambda: None,
            )
        )
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False, resume=False),
        registry=registry,
        agents=None,
        agent=None,
        rebuild_requested=False,
        _reseed_pending=lambda: False,
        console=Mock(),
        persist_plugin_states=Mock(),
        record_exit=Mock(),
    )

    async def submit(_text: str) -> None:
        raise AssertionError("startup must not submit a provider turn")

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit, ctx=ctx, input=pipe, output=DummyOutput(), status=lambda: "ready"
        )
        runtime.application.after_render += lambda _app: painted.set()
        task = asyncio.create_task(_run_runtime(ctx, runtime, bus))
        try:
            await asyncio.wait_for(
                asyncio.gather(skills_started.wait(), mcp_started.wait(), painted.wait()), 2
            )
            assert not release_skills.is_set()
            assert not release_mcp.is_set()
            assert runtime.application.is_running
            pipe.send_bytes(b"\x04")
            assert await asyncio.wait_for(task, 2) == 0
            ctx.record_exit.assert_called_once_with("normal")
        finally:
            release_skills.set()
            release_mcp.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
