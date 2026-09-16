import asyncio

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from orcha_agent.core.compaction import CompactionMiddleware, Compactor
from orcha_agent.core.compaction_config import CompactionConfig
from orcha_agent.core.events import CompactionStatus, EventBus


class Summary:
    async def ainvoke(self, messages):
        return AIMessage(content="Earlier work completed.")


class Request:
    def __init__(self, messages):
        self.messages = messages

    def override(self, **kwargs):
        return Request(kwargs["messages"])


@pytest.mark.asyncio
async def test_sync_compaction_emits_status_without_touching_async_lifecycle():
    bus = EventBus()
    statuses = []

    async def record(event):
        statuses.append(event.active)

    bus.on(CompactionStatus, record)
    controller = Compactor(
        Summary(),
        CompactionConfig(threshold_tokens=1000, keep_recent_tokens=100),
        bus=bus,
    )
    messages = [
        HumanMessage(content="old context " * 2000, id="old"),
        AIMessage(content="Earlier answer", id="answer"),
        HumanMessage(content="continue", id="latest"),
    ]
    controller.speculate(messages)
    pending = controller._speculation
    key = controller._speculation_key
    subscriptions = list(bus.handlers)
    assert pending is not None and not pending.done()
    response = CompactionMiddleware(controller).wrap_model_call(
        Request(messages),
        lambda request: ModelResponse(result=[AIMessage(content="done")]),
    )
    assert response.model_response.result[0].content == "done"
    assert statuses == [True, False]
    assert bus.handlers == subscriptions
    assert controller._speculation is pending
    assert controller._speculation_key == key
    assert not pending.done()
    await pending
    assert statuses == [True, False]
    await controller.close()
    assert len(bus.handlers) == 1


def test_sync_compaction_emits_status_without_running_event_loop():
    bus = EventBus()
    statuses = []

    async def record(event):
        statuses.append(event.active)

    bus.on(CompactionStatus, record)
    controller = Compactor(
        Summary(), CompactionConfig(threshold_tokens=1000, keep_recent_tokens=100), bus=bus
    )
    subscriptions = list(bus.handlers)
    CompactionMiddleware(controller).wrap_model_call(
        Request(
            [
                HumanMessage(content="old context " * 2000, id="old"),
                AIMessage(content="Earlier answer", id="answer"),
                HumanMessage(content="continue", id="latest"),
            ]
        ),
        lambda request: ModelResponse(result=[AIMessage(content="done")]),
    )
    assert statuses == [True, False]
    assert bus.handlers == subscriptions
    assert controller._speculation is None
    asyncio.run(controller.close())
