from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orcha_agent.tui.context import AppContext


@dataclass
class TurnConfig:
    model: str = "original:model"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, ValueError("turn failure")])
async def test_model_override_builds_transient_agent_without_persisting(monkeypatch, failure):
    original_cfg, original_agent, temporary_agent = TurnConfig(), object(), object()
    build = AsyncMock(return_value=temporary_agent)
    monkeypatch.setattr("orcha_agent.tui.context.build_agent", build)
    monkeypatch.setattr("orcha_agent.tui.context.agent_tools", lambda _ctx: [])
    monkeypatch.setattr("orcha_agent.tui.context._compat", lambda _name, default: default)
    cfg_seen = []

    async def run(ctx, text):
        assert ctx.agent is temporary_agent
        assert text == "/literal expanded body"
        cfg_seen.append(ctx.cfg.model)
        if failure:
            raise failure

    monkeypatch.setattr("orcha_agent.tui.turn._run_cancellable_turn", run)
    ctx = SimpleNamespace(
        cfg=original_cfg,
        agent=original_agent,
        summarizer=object(),
        registry=object(),
        session=Mock(),
        _bus=Mock(),
        _always_allowed=lambda: set(),
        _resolve_summarizer=lambda cfg: None,
        _clean_history_for_model=Mock(),
        _reseed_pending=lambda: False,
        history_model=None,
        rebuild_requested=False,
        switch_model=AsyncMock(),
        request_rebuild=Mock(),
    )
    if failure:
        with pytest.raises(ValueError, match="turn failure"):
            await AppContext.submit_prompt(ctx, "/literal expanded body", model="temporary:model")
    else:
        await AppContext.submit_prompt(ctx, "/literal expanded body", model="temporary:model")
    assert build.call_count == 1
    assert build.call_args.args[1].model == "temporary:model"
    assert cfg_seen == ["temporary:model"]
    assert ctx.cfg is original_cfg and ctx.agent is original_agent
    ctx.switch_model.assert_not_called()
    assert ctx.session.mock_calls == []
    assert ctx._bus.mock_calls == []
