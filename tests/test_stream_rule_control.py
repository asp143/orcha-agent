from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.callbacks.manager import ahandle_event
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.errors import GraphBubbleUp

from orcha_agent.core.agent import ModelFallbackMiddleware
from orcha_agent.extensibility.stream_rules import (
    StreamInterrupt,
    StreamRetry,
    _model_monitor,
    install_model_callback,
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_interrupt_never_calls_fallback_model(asynchronous):
    primary = FakeListChatModel(responses=["primary"])
    fallback = FakeListChatModel(responses=["fallback"])
    middleware = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[])
    interrupt = StreamInterrupt(StreamRetry([]))
    called = []

    def handler(request):
        called.append(request.model)
        if request.model is primary:
            raise interrupt
        return "fallback"

    async def async_handler(request):
        return handler(request)

    with pytest.raises(StreamInterrupt) as caught:
        if asynchronous:
            await middleware.awrap_model_call(request, async_handler)
        else:
            middleware.wrap_model_call(request, handler)
    assert caught.value is interrupt
    assert not isinstance(caught.value, GraphBubbleUp)
    assert called == [primary]


@pytest.mark.parametrize("expected_interrupt", [False, True])
@pytest.mark.asyncio
async def test_callback_logs_real_errors_but_not_stream_interrupts(caplog, expected_interrupt):
    model = FakeListChatModel(responses=["x"])
    install_model_callback(model)
    error = (
        StreamInterrupt(StreamRetry([])) if expected_interrupt else ValueError("broken callback")
    )

    async def inspect(message):
        raise error

    token = _model_monitor.set(SimpleNamespace(inspect=inspect))
    try:
        with pytest.raises(type(error)):
            await ahandle_event(
                model.callbacks,
                "on_llm_new_token",
                "ignore_llm",
                "x",
                chunk=SimpleNamespace(message=AIMessageChunk(content="x")),
                run_id=uuid4(),
            )
    finally:
        _model_monitor.reset(token)
    assert bool(caplog.records) is not expected_interrupt
    if not expected_interrupt:
        assert "broken callback" in caplog.text


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_provider_errors_still_call_fallback_model(asynchronous):
    primary = FakeListChatModel(responses=["primary"])
    fallback = FakeListChatModel(responses=["fallback"])
    middleware = ModelFallbackMiddleware(fallback)
    request = ModelRequest(model=primary, messages=[])
    called = []

    def handler(request):
        called.append(request.model)
        if request.model is primary:
            raise ConnectionError("provider unavailable")
        return "fallback succeeded"

    async def async_handler(request):
        return handler(request)

    result = (
        await middleware.awrap_model_call(request, async_handler)
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert result == "fallback succeeded"
    assert called == [primary, fallback]


@pytest.mark.parametrize("fallback_streaming", [False, True])
@pytest.mark.asyncio
async def test_fallback_response_is_checked_after_partial_primary_failure(
    tmp_path, fallback_streaming
):
    import re

    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.tools import StructuredTool
    from langgraph.checkpoint.memory import InMemorySaver

    from orcha_agent.core.events import EventBus
    from orcha_agent.extensibility.rules import Rule, RulesMiddleware
    from orcha_agent.extensibility.stream_rules import (
        StreamInspect,
        StreamRules,
        intercepted_stream,
    )

    calls = []

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            response = next(self.messages)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=response.content, tool_calls=response.tool_calls)
            )
            if response.content == "partial primary":
                raise ConnectionError("primary failed after streaming")

    def operation() -> str:
        """Count executions that rules should prevent."""
        calls.append(True)
        return "done"

    rule = Rule("r", "Avoid forbidden output", conditions=(re.compile("forbidden"),))
    manager = StreamRules({"r": rule}, {})
    bus = EventBus()

    async def inspect(event):
        if event.item is not None and manager.inspect(event.item)[0]:
            return StreamRetry([rule.reminder()])
        return None

    bus.on(StreamInspect, inspect)
    primary = Model(
        messages=iter([AIMessage(content="partial primary"), AIMessage(content="safe")])
    )
    fallback = Model(
        disable_streaming=not fallback_streaming,
        messages=iter(
            [
                AIMessage(
                    content="forbidden", tool_calls=[{"name": "operation", "args": {}, "id": "bad"}]
                )
            ]
        ),
    )
    graph = create_agent(
        model=primary,
        tools=[StructuredTool.from_function(operation)],
        middleware=[RulesMiddleware({"r": rule}, tmp_path), ModelFallbackMiddleware(fallback)],
        checkpointer=InMemorySaver(),
    )
    host = SimpleNamespace(
        agent=graph, bus=bus, thread_config={"configurable": {"thread_id": "main"}}
    )
    async for _ in intercepted_stream(
        host,
        {"messages": [{"role": "user", "content": "start"}]},
        config=host.thread_config,
        stream_mode=["messages", "updates"],
    ):
        pass
    assert calls == []
    assert manager.fired == {"r": 0}
    assert (await graph.aget_state(host.thread_config)).values["messages"][-1].text == "safe"
