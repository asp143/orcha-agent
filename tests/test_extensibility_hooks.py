from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orcha_agent.core.events import AppExit, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.context import AppContext
from orcha_agent.tui.runtime import _run_runtime


def test_remove_tool_cannot_remove_another_plugins_tool():
    registry, bus = Registry(), EventBus()
    owner = PluginAPI(
        name="owner", config={}, state={}, registry=registry, bus=bus, request_rebuild=lambda: None
    )
    other = PluginAPI(
        name="other", config={}, state={}, registry=registry, bus=bus, request_rebuild=lambda: None
    )

    def tool():
        return "ok"

    owner.add_tool(tool)
    assert not other.remove_tool("tool")
    assert "tool" in registry.tools
    assert owner.remove_tool("tool")
    assert "tool" not in registry.tools
    assert not owner.remove_tool("tool")


@pytest.mark.asyncio
async def test_expanded_prompt_bypasses_dispatch_without_model_change(monkeypatch):
    turn = AsyncMock(side_effect=ValueError("turn failed"))
    monkeypatch.setattr("orcha_agent.tui.turn._run_cancellable_turn", turn)
    ctx = SimpleNamespace(cfg=SimpleNamespace(model="original"), switch_model=AsyncMock())
    with pytest.raises(ValueError, match="turn failed"):
        await AppContext.submit_prompt(ctx, "/literal body")
    turn.assert_awaited_once_with(ctx, "/literal body")
    ctx.switch_model.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_event_emitted_when_runtime_raises():
    bus = EventBus()
    exits = []

    async def exited(event):
        exits.append(event)

    bus.on(AppExit, exited)
    ctx = SimpleNamespace(
        agents=None,
        _reseed_pending=lambda: False,
        rebuild_requested=False,
        cfg=SimpleNamespace(resume=False),
    )
    runtime = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        await _run_runtime(ctx, runtime, bus)
    assert len(exits) == 1
