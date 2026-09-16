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
