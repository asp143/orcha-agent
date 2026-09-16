from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langgraph.checkpoint.memory import InMemorySaver

from orcha_agent.builtin import rules as plugin
from orcha_agent.core.events import AppStart, EventBus, TurnEnd, TurnStart
from orcha_agent.core.ledger import CompactionEntry, MessageEntry, ResetBoundaryEntry, build_context
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.rules import MARKER, Rule, RulesMiddleware, discover_rules, rulebook
from orcha_agent.extensibility.stream_rules import StreamRules, intercepted_stream


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_discovery_imports_and_precedence(tmp_path: Path) -> None:
    cwd, home = tmp_path / "project", tmp_path / "home"
    write(cwd, "RULES.md", "Always follow project conventions")
    write(cwd, ".orcha-agent/rules/python.md", '---\nglobs: ["**/*.py"]\n---\nNative body')
    write(home, ".config/orcha-agent/rules/python.md", "User body")
    write(cwd, ".claude/rules/imported.md", '---\npaths: ["src/**"]\n---\nClaude body')
    write(cwd, ".cursor/rules/web.mdc", '---\nglobs: "*.js, *.ts"\n---\nCursor body')
    write(cwd, ".orcha-agent/rules/bad.md", '---\ncondition: "["\n---\nBad expression')
    rules, warnings = discover_rules(cwd, home, trust_cwd=True)
    assert rules["RULES"].always_apply
    assert rules["python"].body == "Native body"
    assert rules["imported"].globs == ("src/**",)
    assert rules["web"].globs == ("*.js", "*.ts")
    assert len(warnings) == 1
    prompt = rulebook(rules)
    assert "rule://python" in prompt and "**/*.py" in prompt
    assert "Native body" not in prompt
    assert "Always follow project conventions" in prompt


@pytest.mark.asyncio
async def test_file_rule_attaches_once_and_is_model_system_context(tmp_path: Path) -> None:
    rule = Rule("python", "Use type annotations", globs=("**/*.py",))
    middleware = RulesMiddleware({rule.name: rule}, tmp_path)
    call = AIMessage(
        content="", tool_calls=[{"name": "read_file", "id": "a", "args": {"file_path": "/main.py"}}]
    )
    result = await middleware.abefore_model(
        {"messages": [call, ToolMessage(content="ok", tool_call_id="a")]}, None
    )
    assert result is not None
    reminder = result["messages"][0]
    assert (
        await middleware.abefore_model(
            {"messages": [call, reminder], "rule_reminders": {"python": reminder}}, None
        )
        is None
    )
    captured = []

    class Request:
        messages = [call, reminder]
        system_message = SystemMessage(content="Base prompt")

        def override(self, **kwargs):
            return kwargs

    async def handler(request):
        captured.append(request)
        return "ok"

    assert await middleware.awrap_model_call(Request(), handler) == "ok"
    assert captured[0]["messages"] == [call]
    assert "Use type annotations" in captured[0]["system_message"].text


def test_injections_survive_compaction_but_not_reset() -> None:
    reminder = Rule("r", "Never lose this").reminder()
    entries = [
        MessageEntry(id="a", message=message_to_dict(reminder)),
        CompactionEntry(id="b", summary="summary"),
    ]
    rebuilt = build_context(entries)
    assert rebuilt.messages[0].additional_kwargs[MARKER] == ["r"]
    assert "Never lose this" in rebuilt.messages[0].text
    assert len(build_context([*entries, ResetBoundaryEntry(id="c")]).messages) == 0


def chunk(text: str, identifier: str = "one"):
    return ((), "messages", (AIMessageChunk(content=text, id=identifier), {}))


def test_split_chunk_matching_scope_and_repeat_gap() -> None:
    rule = Rule("r", "Avoid forbidden text", conditions=(re.compile("forbidden"),))
    manager = StreamRules({"r": rule}, {"repeatMode": "after-gap", "repeatGap": 2})
    assert manager.inspect(chunk("forbid"))[0] == []
    assert manager.inspect(chunk("den"))[0] == [rule]
    manager.finish()
    manager.start()
    assert manager.inspect(chunk("forbidden"))[0] == []
    manager.finish()
    manager.start()
    assert manager.inspect(chunk("forbidden"))[0] == [rule]
    scoped = Rule("tool", "Only tools", conditions=(re.compile("x"),), scope=("tool",))
    assert StreamRules({"tool": scoped}, {}).inspect(chunk("x"))[0] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("context_mode", ["discard", "keep"])
async def test_real_stream_abort_retry_and_reminder(
    tmp_path: Path, monkeypatch, context_mode: str
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    write(
        tmp_path,
        ".orcha-agent/rules/no_bad.md",
        "---\ncondition: forbidden\n---\nUse safe wording instead.",
    )
    registry, bus = Registry(), EventBus()
    plugin.register(
        PluginAPI(
            name="rules",
            config={"contextMode": context_mode},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=Mock(),
        )
    )
    host = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=True),
        session_id="session",
        console=Mock(),
        bus=bus,
        thread_config={"configurable": {"thread_id": "t"}},
    )
    await bus.emit(AppStart(host))
    model = FakeListChatModel(responses=["forbidden trailing text", "safe response"], sleep=0.001)
    middleware = next(
        item.middleware
        for item in registry.middleware
        if isinstance(item.middleware, RulesMiddleware)
    )
    host.agent = create_agent(model=model, middleware=[middleware], checkpointer=InMemorySaver())
    await bus.emit(TurnStart(thread_id="t", text="hello"))
    items = [
        item
        async for item in intercepted_stream(
            host,
            {"messages": [{"role": "user", "content": "hello"}]},
            config=host.thread_config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )
    ]
    await bus.emit(TurnEnd(thread_id="t"))
    messages = (await host.agent.aget_state(host.thread_config)).values["messages"]
    assert any(isinstance(m, SystemMessage) and "safe wording" in m.text for m in messages)
    assert messages[-1].text == "safe response"
    assert sum(m.type == "human" for m in messages) == 1
    assert all("trailing text" not in m.text for m in messages)
    assert any(isinstance(m, AIMessage) and m.text == "forbidden" for m in messages) == (
        context_mode == "keep"
    )
    streamed = "".join(item[2][0].text for item in items if item[1] == "messages")
    assert "trailing text" not in streamed and streamed.endswith("safe response")
    assert host.console.print.call_count == 1


def test_rule_symlinks_never_read(tmp_path: Path) -> None:
    outside = tmp_path / "Credentials" / "hidden"
    outside.parent.mkdir()
    outside.write_text("private")
    directory = tmp_path / "project/.orcha-agent/rules"
    directory.mkdir(parents=True)
    (directory / "bad.md").symlink_to(outside)
    found, warnings = discover_rules(tmp_path / "project", tmp_path / "home")
    assert not found
    assert warnings == ["Skipping symlink rule: bad.md"]


def test_tool_paths_scopes_and_thinking() -> None:
    tool = Rule(
        "tool",
        "no placeholders",
        globs=("*.py",),
        conditions=(re.compile("TODO"),),
        scope=("tool:write(*.py)",),
    )
    thinking = Rule(
        "thinking", "think safely", conditions=(re.compile("unsafe"),), scope=("thinking",)
    )
    manager = StreamRules({"tool": tool, "thinking": thinking}, {})
    item = (
        (),
        "messages",
        (
            AIMessageChunk(
                content="",
                id="tool",
                tool_call_chunks=[
                    {
                        "name": "write",
                        "args": '{"path": "main.py", "content": "TODO"}',
                        "id": "c",
                        "index": 0,
                    }
                ],
            ),
            {},
        ),
    )
    assert manager.inspect(item)[0] == [tool]
    item = (
        (),
        "messages",
        (AIMessageChunk(content=[{"type": "thinking", "thinking": "unsafe"}], id="think"), {}),
    )
    assert manager.inspect(item)[0] == [thinking]
    assert manager.match_sources == {"tool": "tool", "thinking": "thinking"}


@pytest.mark.asyncio
async def test_unsuccessful_or_unrelated_tools_do_not_attach(tmp_path: Path) -> None:
    middleware = RulesMiddleware({"p": Rule("p", "body", globs=("*.py",))}, tmp_path)
    for name, status in (("read_file", "error"), ("task", "success")):
        call = AIMessage(
            content="", tool_calls=[{"name": name, "id": "a", "args": {"path": "main.py"}}]
        )
        result = ToolMessage(content="result", tool_call_id="a", status=status)
        assert await middleware.abefore_model({"messages": [call, result]}, None) is None


@pytest.mark.asyncio
async def test_deferred_rule_and_restored_once_policy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    write(
        tmp_path,
        ".orcha-agent/rules/r.md",
        "---\ncondition: forbidden\ninterruptMode: never\n---\nRemember this.",
    )
    registry, bus = Registry(), EventBus()
    plugin.register(
        PluginAPI(
            name="rules", config={}, state={}, registry=registry, bus=bus, request_rebuild=Mock()
        )
    )
    host = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=True),
        session_id="s",
        console=Mock(),
        bus=bus,
        thread_config={"configurable": {"thread_id": "t"}},
    )
    await bus.emit(AppStart(host))
    middleware = next(
        item.middleware
        for item in registry.middleware
        if isinstance(item.middleware, RulesMiddleware)
    )
    host.agent = create_agent(
        model=FakeListChatModel(responses=["forbidden complete answer"]),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    for _ in range(2):
        await bus.emit(TurnStart(thread_id="t", text="hello"))
        _ = [
            item
            async for item in intercepted_stream(
                host,
                {"messages": [{"role": "user", "content": "hello"}]},
                config=host.thread_config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            )
        ]
        await bus.emit(TurnEnd(thread_id="t"))
    messages = (await host.agent.aget_state(host.thread_config)).values["messages"]
    assert sum(isinstance(m, SystemMessage) for m in messages) == 1
    assert sum(m.text == "forbidden complete answer" for m in messages) == 2
    assert host.console.print.call_count == 1


@pytest.mark.asyncio
async def test_automatic_summarization_preserves_injection(tmp_path: Path) -> None:
    from typing import Any
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.messages import HumanMessage

    received: list[Any] = []

    class CapturingModel(FakeListChatModel):
        def _call(self, messages, stop=None, run_manager=None, **kwargs):
            received.append(messages)
            return super()._call(messages, stop, run_manager, **kwargs)

    rule = Rule("durable", "Keep this exact instruction.")
    middleware = RulesMiddleware({"durable": rule}, tmp_path)
    graph = create_agent(
        model=CapturingModel(responses=["answer"]),
        middleware=[
            middleware,
            SummarizationMiddleware(
                model=FakeListChatModel(responses=["short summary"]),
                trigger=("messages", 2),
                keep=("messages", 1),
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="old"),
                rule.reminder(),
                AIMessage(content="old response"),
                HumanMessage(content="new question"),
            ]
        },
        config={"configurable": {"thread_id": "t"}},
    )
    assert "durable" in result["rule_reminders"]
    assert "Keep this exact instruction." in received[-1][0].text
    assert isinstance(received[-1][0], SystemMessage)
    assert any("summary" in message.text for message in result["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger_source", ["text", "tool", "deferred"])
async def test_retry_does_not_replay_completed_tool(
    tmp_path: Path, monkeypatch, trigger_source: str
) -> None:
    from typing import Any
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.tools import StructuredTool

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    condition = "forbidden" if trigger_source == "text" else "\\{\\}"
    write(
        tmp_path,
        ".orcha-agent/rules/r.md",
        f"---\ncondition: '{condition}'\ninterruptMode: {'never' if trigger_source == 'deferred' else 'always'}\n---\nUse safe words.",
    )
    registry, bus = Registry(), EventBus()
    plugin.register(
        PluginAPI(
            name="rules", config={}, state={}, registry=registry, bus=bus, request_rebuild=Mock()
        )
    )
    host = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=True),
        session_id="s",
        console=Mock(),
        bus=bus,
        thread_config={"configurable": {"thread_id": "t"}},
    )
    await bus.emit(AppStart(host))
    calls = []
    received = []

    def side_effect() -> str:
        """Perform a counted test operation."""
        calls.append(True)
        return "done"

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            import asyncio
            from langchain_core.outputs import ChatGenerationChunk

            received.append(messages)
            response = next(self.messages)
            assert isinstance(response, AIMessage)
            if response.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=response.tool_calls)
                )
            else:
                for character in response.text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=character))
                    await asyncio.sleep(0.001)

    model = Model(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "side_effect", "args": {}, "id": "c"}]),
                *(
                    [AIMessage(content="forbidden discarded words")]
                    if trigger_source == "text"
                    else []
                ),
                AIMessage(content="safe answer"),
            ]
        )
    )
    middleware = next(
        item.middleware
        for item in registry.middleware
        if isinstance(item.middleware, RulesMiddleware)
    )
    host.agent = create_agent(
        model=model,
        tools=[StructuredTool.from_function(side_effect)],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    await bus.emit(TurnStart(thread_id="t", text="hello"))
    _ = [
        item
        async for item in intercepted_stream(
            host,
            {"messages": [{"role": "user", "content": "hello"}]},
            config=host.thread_config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )
    ]
    messages = (await host.agent.aget_state(host.thread_config)).values["messages"]
    assert calls == ([] if trigger_source == "tool" else [True])
    assert messages[-1].text == "safe answer"
    assert sum(isinstance(m, ToolMessage) for m in messages) == (
        0 if trigger_source == "tool" else 1
    )
    if trigger_source == "deferred":
        assert "Use safe words." in received[-1][0].text
    from rich.console import Console
    from io import StringIO

    output = StringIO()
    Console(file=output, width=60, force_terminal=False).print(host.console.print.call_args.args[0])
    assert "⚠ Injecting rule: r" in output.getvalue()


def test_auto_compaction_checkpoint_rules_survive_ledger_reload(tmp_path: Path) -> None:
    from orcha_agent.core.capture import capture_graph_values
    from orcha_agent.core.ledger import Ledger
    from orcha_agent.core.session import SessionStore
    from langchain_core.messages import HumanMessage

    database = tmp_path / "session.db"
    with SessionStore(database) as store:
        session = store.create(cwd=str(tmp_path), model="fake", mode="normal")
        values = {
            "messages": [HumanMessage(content="summary after automatic compaction")],
            "rule_reminders": {"r": Rule("r", "Exact durable instruction").reminder()},
        }
        capture_graph_values(
            store, session.thread_id, session.current_thread or "", values, only_if_new=False
        )
        session_id = session.thread_id
    with SessionStore(database) as store:
        context = build_context(Ledger(store).path(session_id))
        assert context.messages[0].additional_kwargs[MARKER] == ["r"]
        assert "Exact durable instruction" in context.messages[0].text


@pytest.mark.asyncio
async def test_after_gap_restores_injection_age_on_resume(tmp_path: Path, monkeypatch) -> None:
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    write(
        tmp_path,
        ".orcha-agent/rules/r.md",
        "---\ncondition: forbidden\ninterruptMode: never\n---\nRemember this.",
    )
    registry, bus = Registry(), EventBus()
    plugin.register(
        PluginAPI(
            name="rules",
            config={"repeatMode": "after-gap", "repeatGap": 2},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=Mock(),
        )
    )
    host = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=True),
        session_id="s",
        console=Mock(),
        bus=bus,
        thread_config={"configurable": {"thread_id": "t"}},
    )
    await bus.emit(AppStart(host))
    middleware = next(
        item.middleware
        for item in registry.middleware
        if isinstance(item.middleware, RulesMiddleware)
    )
    host.agent = create_agent(
        model=FakeListChatModel(responses=["forbidden answer"]),
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    await host.agent.ainvoke(
        {"messages": [HumanMessage(content="prior turn"), Rule("r", "Remember this.").reminder()]},
        config=host.thread_config,
    )
    for index in range(3):
        await bus.emit(TurnStart(thread_id="t", text="hello"))
        _ = [
            item
            async for item in intercepted_stream(
                host,
                {"messages": [HumanMessage(content="hello")]},
                config=host.thread_config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            )
        ]
        await bus.emit(TurnEnd(thread_id="t"))
        assert host.console.print.call_count == (1 if index == 2 else 0)


def test_compaction_keeps_latest_rule_body_only() -> None:
    older = MessageEntry(id="a", message=message_to_dict(Rule("r", "old body").reminder()))
    newer = MessageEntry(id="b", message=message_to_dict(Rule("r", "new body").reminder()))
    context = build_context(
        [older, newer, CompactionEntry(id="c", summary="summary", first_kept_id="a")]
    )
    assert sum(isinstance(m, SystemMessage) for m in context.messages) == 1
    assert all("old body" not in m.text for m in context.messages)
