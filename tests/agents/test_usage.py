from __future__ import annotations

from types import SimpleNamespace

import pytest

from orcha_agent.core.agents import _RunEventBus
from orcha_agent.core.events import EventBus, ModelChunk, TurnStart


@pytest.mark.asyncio
async def test_streamed_chunks_count_each_model_response_once() -> None:
    persisted: list[object] = []
    run = SimpleNamespace(
        id="worker",
        requests=0,
        model_label="custom:model",
        tokens_in=0,
        tokens_out=0,
        cost=0.0,
        visible=True,
        cfg=SimpleNamespace(pricing={}),
        owner=SimpleNamespace(_persist_job=persisted.append),
    )
    bus = _RunEventBus(run, EventBus())
    await bus.emit(TurnStart(thread_id="thread", text="go", source_id="worker"))

    for request_id in ("response-a", "response-a", "response-b", "response-b"):
        await bus.emit(
            ModelChunk(
                SimpleNamespace(id=request_id, content="delta", usage_metadata=None),
                role="subagent",
                source_id="worker",
                request_id=request_id,
            )
        )

    assert run.requests == 2
    assert persisted == [run, run]


@pytest.mark.asyncio
async def test_agent_model_usage_updates_tokens_cost_and_persistence() -> None:
    persisted: list[object] = []
    run = SimpleNamespace(
        id="worker",
        requests=0,
        model_label="custom:model",
        tokens_in=0,
        tokens_out=0,
        cost=0.0,
        cfg=SimpleNamespace(pricing={"custom:model": {"input": 2.0, "output": 10.0}}),
        owner=SimpleNamespace(_persist_job=persisted.append),
    )
    bus = _RunEventBus(run, EventBus())

    await bus.emit(
        ModelChunk(
            SimpleNamespace(
                content="done",
                usage_metadata={"input_tokens": 1_000, "output_tokens": 500},
            ),
            role="subagent",
            model_name="custom:model",
            source_id="worker",
        )
    )

    assert run.tokens_in == 1_000
    assert run.tokens_out == 500
    assert run.cost == pytest.approx(0.007)
    assert persisted == [run]
