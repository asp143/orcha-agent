from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.core.events import AppExit, AppStart
from orcha_agent.tui.runtime import _run_runtime


@pytest.mark.asyncio
async def test_startup_failure_still_shuts_down_agents() -> None:
    shutdown = False

    class Agents:
        async def shutdown(self) -> None:
            nonlocal shutdown
            shutdown = True

    class Runtime:
        async def run(self) -> None:
            raise AssertionError("runtime must not start")

    class Bus:
        async def emit(self, event: Any) -> None:
            assert isinstance(event, AppStart)
            raise RuntimeError("boom")

    ctx = SimpleNamespace(agents=Agents())

    with pytest.raises(RuntimeError, match="boom"):
        await _run_runtime(ctx, Runtime(), Bus())

    assert shutdown is True

@pytest.mark.asyncio
async def test_failed_shutdown_is_retried_in_finally() -> None:
    order: list[str] = []
    shutdown_attempts = 0

    class Agents:
        async def shutdown(self) -> None:
            nonlocal shutdown_attempts
            shutdown_attempts += 1
            order.append("shutdown")
            if shutdown_attempts == 1:
                raise RuntimeError("shutdown failed")

    class Runtime:
        async def run(self) -> None:
            order.append("run")

    class Bus:
        async def emit(self, event: Any) -> None:
            if isinstance(event, AppStart):
                order.append("start")
            elif isinstance(event, AppExit):
                order.append("exit")

    ctx = SimpleNamespace(
        agents=Agents(),
        agent=None,
        cfg=SimpleNamespace(resume=False),
        rebuild_requested=False,
        persist_plugin_states=lambda: order.append("persist"),
        record_exit=lambda _kind: order.append("record"),
        _reseed_pending=lambda: False,
    )

    with pytest.raises(RuntimeError, match="shutdown failed"):
        await _run_runtime(ctx, Runtime(), Bus())

    assert order == [
        "start",
        "run",
        "persist",
        "record",
        "shutdown",
        "shutdown",
    ]


@pytest.mark.asyncio
async def test_normal_runtime_shuts_down_agents_before_app_exit() -> None:
    order: list[str] = []

    class Agents:
        async def shutdown(self) -> None:
            order.append("shutdown")

    class Runtime:
        async def run(self) -> None:
            order.append("run")

    class Bus:
        async def emit(self, event: Any) -> None:
            if isinstance(event, AppStart):
                order.append("start")
            elif isinstance(event, AppExit):
                order.append("exit")

    ctx = SimpleNamespace(
        agents=Agents(),
        agent=None,
        cfg=SimpleNamespace(resume=False),
        rebuild_requested=False,
        persist_plugin_states=lambda: order.append("persist"),
        record_exit=lambda _kind: order.append("record"),
        _reseed_pending=lambda: False,
    )

    assert await _run_runtime(ctx, Runtime(), Bus()) == 0
    assert order == ["start", "run", "persist", "record", "shutdown", "exit"]
