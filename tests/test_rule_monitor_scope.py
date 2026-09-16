from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from orcha_agent.core.events import EventBus
from orcha_agent.extensibility.rules import Rule, RulesMiddleware
from orcha_agent.extensibility.stream_rules import (
    StreamInspect,
    StreamRetry,
    StreamRules,
    intercepted_stream,
)


def host_for(tmp_path: Path):
    rule = Rule("r", "Use safe text.", conditions=(re.compile("forbidden"),))
    manager = StreamRules({"r": rule}, {})
    bus = EventBus()
    inspected = []

    async def inspect(event):
        if event.item is None:
            return None
        inspected.append(event.item[-1][0].text)
        matches, _ = manager.inspect(event.item)
        if matches:
            return StreamRetry(messages=[rule.reminder()])
        return None

    bus.on(StreamInspect, inspect)
    host = SimpleNamespace(bus=bus, thread_config={"configurable": {"thread_id": "main"}})
    return host, RulesMiddleware({"r": rule}, tmp_path), manager, inspected


@pytest.mark.asyncio
@pytest.mark.parametrize("child_interceptor", [False, True])
async def test_async_tool_child_task_does_not_inherit_main_monitor(tmp_path, child_interceptor):
    host, middleware, manager, inspected = host_for(tmp_path)
    completed = []
    child = create_agent(
        model=FakeListChatModel(responses=["forbidden child answer"]), middleware=[middleware]
    )

    async def delegate() -> str:
        """Run a child graph from a task, just as the task tool does."""

        async def run_child():
            value = {"messages": [{"role": "user", "content": "child"}]}
            config = {"configurable": {"thread_id": "child", "checkpoint_ns": ""}}
            if child_interceptor:
                child_host = SimpleNamespace(bus=EventBus(), agent=child, thread_config=config)
                stream = intercepted_stream(
                    child_host, value, config=config, stream_mode=["messages", "updates"]
                )
            else:
                stream = child.astream(value, config=config, stream_mode=["messages", "updates"])
            async for _ in stream:
                pass
            completed.append(True)

        await asyncio.create_task(run_child())
        return "child completed"

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            response = next(self.messages)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=response.content, tool_calls=response.tool_calls)
            )

    host.agent = create_agent(
        model=Model(
            messages=iter(
                [
                    AIMessage(
                        content="delegating",
                        tool_calls=[{"name": "delegate", "args": {}, "id": "call"}],
                    ),
                    AIMessage(content="main complete"),
                ]
            )
        ),
        tools=[StructuredTool.from_function(coroutine=delegate)],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    async for _ in intercepted_stream(
        host,
        {"messages": [{"role": "user", "content": "start"}]},
        config=host.thread_config,
        stream_mode=["messages", "updates"],
    ):
        pass
    assert completed == [True]
    assert manager.fired == {}
    assert not any("forbidden" in text for text in inspected)
    assert (await host.agent.aget_state(host.thread_config)).values["messages"][
        -1
    ].text == "main complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["disabled", "updates", "no_stream"])
async def test_final_nonstreaming_message_interrupts_before_tools(tmp_path, kind):
    host, middleware, manager, _ = host_for(tmp_path)
    calls = []

    def operation() -> str:
        """Count the rejected operation."""
        calls.append(True)
        return "done"

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            response = next(self.messages)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=response.content, tool_calls=response.tool_calls)
            )

    responses = [
        AIMessage(content="forbidden", tool_calls=[{"name": "operation", "args": {}, "id": "bad"}]),
        AIMessage(content="safe retry"),
    ]
    if kind == "no_stream":
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult

        class NoStreamModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "nonstream-test"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(generations=[ChatGeneration(message=responses.pop(0))])

        model = NoStreamModel()
    else:
        model = Model(messages=iter(responses), disable_streaming=kind == "disabled")
    host.agent = create_agent(
        model=model,
        tools=[StructuredTool.from_function(operation)],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    async for _ in intercepted_stream(
        host,
        {"messages": [{"role": "user", "content": "start"}]},
        config=host.thread_config,
        stream_mode=["updates"] if kind == "updates" else ["messages", "updates"],
    ):
        pass
    assert calls == []
    assert manager.fired == {"r": 0}
    assert (await host.agent.aget_state(host.thread_config)).values["messages"][
        -1
    ].text == "safe retry"


@pytest.mark.asyncio
async def test_direct_model_message_and_after_model_fallback_are_inspected(tmp_path, monkeypatch):
    from orcha_agent.extensibility import rules as rules_module
    from orcha_agent.extensibility.stream_rules import (
        ModelStreamMonitor,
        StreamInterrupt,
        _stream_host,
    )

    host, middleware, _, _ = host_for(tmp_path)
    monkeypatch.setattr(rules_module, "get_config", lambda: host.thread_config)
    request = SimpleNamespace(model=FakeListChatModel(responses=["unused"]), messages=[], state={})

    async def handler(request):
        return AIMessage(content="forbidden direct response")

    token = _stream_host.set(host)
    try:
        with pytest.raises(StreamInterrupt):
            await middleware.awrap_model_call(request, handler)
    finally:
        _stream_host.reset(token)
    assert middleware.model_monitors == {}

    # The after-model guard also covers middleware-provided state without a
    # conventional model response, while regular responses are checked precommit.
    host, middleware, _, _ = host_for(tmp_path)
    middleware.model_monitors["main"] = ModelStreamMonitor(host)
    with pytest.raises(StreamInterrupt):
        await middleware.aafter_model({"messages": [AIMessage(content="forbidden")]}, None)
