import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Command

from orcha_agent.builtin import approval_prompt, filesystem, modes, provider_codex, statusbar
from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import (
    AgentBuildAfter,
    Event,
    EventBus,
    ModelChunk,
    InterruptRaised,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
)
from orcha_agent.core.plugin import PluginAPI, ProviderCaps, Resolved
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.app import AppContext, _run_turn


class ToolCallingFakeModel(GenericFakeChatModel):
    disable_streaming: bool = True

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> "ToolCallingFakeModel":
        return self


class _UsageStreamGraph:
    def __init__(self, turns: list[list[tuple[Any, ...]]]) -> None:
        self.turns = list(turns)

    async def astream(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[tuple[Any, ...]]:
        for item in self.turns.pop(0):
            yield item


class _UsageContext:
    def __init__(self, agent: Any, registry: Registry, bus: EventBus) -> None:
        self.agent = agent
        self.registry = registry
        self.bus = bus
        self._bus = bus
        self.console = _RecordingConsole()
        self.thread_id = "usage-thread"
        self.thread_config = {"configurable": {"thread_id": self.thread_id}}
        self.rebuild_requested = False
        self.session = SimpleNamespace(get=lambda _thread_id: None)

    async def ensure_agent(self) -> bool:
        return True

    async def rebuild(self) -> None:
        raise AssertionError("usage streaming must not request a rebuild")


class _FakeCodexTokenSource:
    def get_token(self) -> tuple[str, str]:
        return "fake-codex-access-token", "fake-codex-account"


def _successful_codex_sse() -> httpx.Response:
    event = {
        "type": "response.output_text.delta",
        "sequence_number": 0,
        "item_id": "fake-message",
        "output_index": 0,
        "content_index": 0,
        "delta": "ok",
        "logprobs": [],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


def _api(
    name: str,
    registry: Registry,
    bus: EventBus,
    *,
    state: dict[str, Any] | None = None,
) -> PluginAPI:
    return PluginAPI(
        name=name,
        registry=registry,
        bus=bus,
        config={},
        state={} if state is None else state,
        request_rebuild=lambda: None,
    )


def _config(tmp_path: Path, mode: str) -> Config:
    return Config(
        model="fake:test",
        subagent_model="fake:test",
        summarizer_model="fake:test",
        mode=mode,
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "sessions.sqlite",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )


def _script(*writes: tuple[str, str]) -> Iterator[AIMessage]:
    messages: list[AIMessage] = []
    for index, (file_path, content) in enumerate(writes, start=1):
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": file_path, "content": content},
                            "id": f"write-{index}",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The file was written."),
            ]
        )
    return iter(messages)


class _RecordingConsole:
    def __init__(self) -> None:
        self.output: list[tuple[object, ...]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def print(self, *objects: object, **_kwargs: Any) -> None:
        self.output.append(objects)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _runtime(
    tmp_path: Path,
    *,
    mode: str,
    writes: tuple[tuple[str, str], ...],
) -> tuple[Registry, EventBus, Config]:
    registry = Registry()
    bus = EventBus()
    model = ToolCallingFakeModel(messages=_script(*writes))

    filesystem.register(_api("filesystem", registry, bus))
    modes.register(_api("modes", registry, bus))
    _api("fake-provider", registry, bus).add_provider(
        "fake",
        lambda model_name, provider_config: model,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=False,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
    )
    return registry, bus, _config(tmp_path, mode)


async def _build_harness(
    tmp_path: Path,
    *,
    mode: str,
    thread_id: str,
    content: str,
) -> tuple[Any, SessionStore, EventBus, dict[str, dict[str, str]]]:
    registry, bus, cfg = _runtime(
        tmp_path,
        mode=mode,
        writes=(("/approved.txt", content),),
    )
    session = SessionStore(cfg.db_path)
    created = session.create(cwd=tmp_path, model=cfg.model, thread_id=thread_id)
    assert created.thread_id == thread_id
    graph = await build_agent(registry, cfg, session, bus)
    thread_config = {"configurable": {"thread_id": thread_id}}
    return graph, session, bus, thread_config


async def _build_context(
    tmp_path: Path,
    *,
    thread_id: str,
    writes: tuple[tuple[str, str], ...],
    approval_state: dict[str, Any] | None = None,
) -> tuple[AppContext, SessionStore]:
    registry, bus, cfg = _runtime(tmp_path, mode="ask", writes=writes)
    session = SessionStore(cfg.db_path)
    session.create(cwd=tmp_path, model=cfg.model, thread_id=thread_id)
    plugin_states: dict[str, dict[str, Any]] = {}
    holder: dict[str, AppContext] = {}

    if approval_state is not None:
        plugin_states["approval_prompt"] = approval_state
        approval_prompt.register(
            PluginAPI(
                name="approval_prompt",
                registry=registry,
                bus=bus,
                config={},
                state=approval_state,
                request_rebuild=lambda: holder["ctx"].request_rebuild(),
            )
        )

    ctx = AppContext(
        cfg=cfg,
        registry=registry,
        bus=bus,
        session=session,
        plugins=[],
        plugin_states=plugin_states,
        console=_RecordingConsole(),
        thread_id=thread_id,
    )
    holder["ctx"] = ctx
    await ctx.rebuild()
    return ctx, session


def _interrupt_payload(result: dict[str, Any], expected_content: str) -> dict[str, Any]:
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    actions = payload["action_requests"]
    assert len(actions) == 1
    assert actions[0]["name"] == "write_file"
    assert actions[0]["args"] == {
        "file_path": "/approved.txt",
        "content": expected_content,
    }
    assert isinstance(actions[0].get("description"), (str, type(None)))
    assert len(payload["review_configs"]) == 1
    review = payload["review_configs"][0]
    assert review["action_name"] == "write_file"
    assert "approve" in review["allowed_decisions"]
    return payload


@pytest.mark.asyncio
async def test_ask_mode_interrupts_then_approval_writes_under_tmp_cwd(tmp_path: Path) -> None:
    graph, session, _, thread_config = await _build_harness(
        tmp_path,
        mode="ask",
        thread_id="ask-thread",
        content="approved by user\n",
    )
    approved_path = tmp_path / "approved.txt"

    try:
        interrupted = graph.invoke(
            {"messages": [{"role": "user", "content": "Write the approved file."}]},
            config=thread_config,
        )
        _interrupt_payload(interrupted, "approved by user\n")
        assert not approved_path.exists()

        resumed = graph.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=thread_config,
        )

        assert "__interrupt__" not in resumed
        assert approved_path.read_text() == "approved by user\n"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_yolo_mode_writes_without_interrupt(tmp_path: Path) -> None:
    graph, session, _, thread_config = await _build_harness(
        tmp_path,
        mode="yolo",
        thread_id="yolo-thread",
        content="written without approval\n",
    )

    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Write the file."}]},
            config=thread_config,
        )

        assert "__interrupt__" not in result
        assert (tmp_path / "approved.txt").read_text() == "written without approval\n"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_async_graph_invocation_succeeds_with_session_store_saver(
    tmp_path: Path,
) -> None:
    graph, session, _, thread_config = await _build_harness(
        tmp_path,
        mode="yolo",
        thread_id="async-thread",
        content="written asynchronously\n",
    )

    try:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Write the file asynchronously."}]},
            config=thread_config,
        )

        assert "__interrupt__" not in result
        assert result["messages"][-1].content == "The file was written."
        assert (tmp_path / "approved.txt").read_text() == "written asynchronously\n"
    finally:
        session.close()


@pytest.mark.asyncio
async def test_approval_always_writes_then_skips_the_next_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_state: dict[str, Any] = {}
    ctx, session = await _build_context(
        tmp_path,
        thread_id="always-thread",
        writes=(
            ("/first.txt", "first approved write\n"),
            ("/second.txt", "second approved write\n"),
        ),
        approval_state=approval_state,
    )
    events: list[Event] = []
    prompt_calls: list[str] = []

    async def choose_always(
        name: str,
        _args: dict[str, Any],
        _description: str | None,
    ) -> str:
        prompt_calls.append(name)
        return "always"

    async def record(event: Event) -> None:
        events.append(event)

    monkeypatch.setattr(approval_prompt, "_prompt_action", choose_always)
    ctx._bus.on(Event, record, priority=0)

    try:
        await _run_turn(ctx, "Write the first file.")

        assert (tmp_path / "first.txt").read_text() == "first approved write\n"
        assert approval_state["always_allowed"] == ["write_file"]
        assert session.get_plugin_state("always-thread", "approval_prompt") == {
            "always_allowed": ["write_file"]
        }
        rebuild_index = max(
            index for index, event in enumerate(events) if isinstance(event, AgentBuildAfter)
        )
        turn_end_index = next(
            index for index, event in enumerate(events) if isinstance(event, TurnEnd)
        )
        assert rebuild_index > turn_end_index

        events.clear()
        await _run_turn(ctx, "Write the second file.")

        assert (tmp_path / "second.txt").read_text() == "second approved write\n"
        assert not any(isinstance(event, InterruptRaised) for event in events)
        assert prompt_calls == ["write_file"]
        assert ctx.console.errors == []
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_result",
    [pytest.param("n", id="n"), pytest.param(EOFError(), id="eof")],
)
async def test_rejected_ask_write_ends_without_creating_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_result: str | EOFError,
) -> None:
    ctx, session = await _build_context(
        tmp_path,
        thread_id="reject-thread",
        writes=(("/rejected.txt", "must not be written\n"),),
        approval_state={},
    )
    prompt_calls: list[str] = []

    async def reject(
        _name: str,
        _args: dict[str, Any],
        _description: str | None,
    ) -> str:
        prompt_calls.append(_name)
        if isinstance(prompt_result, EOFError):
            raise prompt_result
        return prompt_result

    monkeypatch.setattr(approval_prompt, "_prompt_action", reject)

    try:
        await _run_turn(ctx, "Try to write a rejected file.")

        assert not (tmp_path / "rejected.txt").exists()
        assert ctx.console.errors == []
        assert ctx.console.warnings == []
        assert prompt_calls == ["write_file"]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_run_turn_resumes_one_real_graph_interrupt_and_renders_write(
    tmp_path: Path,
) -> None:
    ctx, session = await _build_context(
        tmp_path,
        thread_id="plugin-thread",
        writes=(("/approved.txt", "approved by plugin\n"),),
    )
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    async def approve(_event: InterruptRaised) -> Resolved:
        return Resolved(resume_value={"decisions": [{"type": "approve"}]})

    ctx._bus.on(Event, record, priority=0)
    ctx._bus.on(InterruptRaised, approve, plugin="auto-approve", priority=1)

    try:
        await _run_turn(ctx, "Write the file.")

        interrupts = [event for event in events if isinstance(event, InterruptRaised)]
        starts = [event for event in events if isinstance(event, ToolCallStart)]
        ends = [event for event in events if isinstance(event, ToolCallEnd)]
        turn_ends = [event for event in events if isinstance(event, TurnEnd)]
        assert len(interrupts) == 1
        assert [(event.name, event.id) for event in starts] == [
            ("write_file", "write-1")
        ]
        assert [(event.name, event.id) for event in ends] == [
            ("write_file", "write-1")
        ]
        assert len(turn_ends) == 1
        assert (tmp_path / "approved.txt").read_text() == "approved by plugin\n"
        assert ctx.console.errors == []
    finally:
        session.close()


@pytest.mark.asyncio
async def test_real_deepagent_sends_codex_compatible_one_turn_payload(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, Any]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _successful_codex_sse()

    registry = Registry()
    bus = EventBus()
    transport = httpx.MockTransport(capture)
    tokens = _FakeCodexTokenSource()
    filesystem.register(_api("filesystem", registry, bus))
    modes.register(_api("modes", registry, bus))
    _api("codex-provider", registry, bus).add_provider(
        "codex",
        lambda model_name, provider_config: provider_codex.create_model(
            model_name,
            provider_config,
            tokens,
            transport=transport,
        ),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
    )
    cfg = replace(
        _config(tmp_path, "yolo"),
        model="codex:gpt-5.6-sol",
        subagent_model="codex:gpt-5.6-sol",
        summarizer_model="codex:gpt-5.6-sol",
    )
    session = SessionStore(cfg.db_path)
    session.create(
        cwd=tmp_path,
        model="codex:gpt-5.6-sol",
        thread_id="codex-payload-thread",
    )

    try:
        graph = await build_agent(registry, cfg, session, bus)
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Answer with ok."}]},
            config={"configurable": {"thread_id": "codex-payload-thread"}},
        )
    finally:
        session.close()

    final_content = result["messages"][-1].content
    if isinstance(final_content, list):
        assert final_content[0]["text"] == "ok"
    else:
        assert final_content == "ok"
    assert len(payloads) == 1
    payload = payloads[0]
    required_keys = {
        "model",
        "input",
        "stream",
        "store",
        "include",
        "tools",
        "instructions",
        "reasoning",
    }
    assert set(payload) in (
        required_keys,
        required_keys | {"parallel_tool_calls"},
    )
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert not {"max_output_tokens", "text", "truncation"} & payload.keys()
    assert payload["instructions"].startswith(
        "You are a careful terminal coding agent. "
        "Use tools deliberately and report concrete results."
    )
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["tools"]
    assert all(tool["type"] == "function" for tool in payload["tools"])
    assert {"read_file", "write_file"} <= {
        tool["name"] for tool in payload["tools"]
    }
    assert payload["input"]
    for item in payload["input"]:
        assert item["type"] in {
            "message",
            "function_call",
            "function_call_output",
        }
        if item["type"] == "message":
            assert item["role"] in {"user", "assistant"}
        else:
            assert "role" not in item


@pytest.mark.asyncio
async def test_streamed_main_and_subagent_usage_accumulates_once_across_turns_and_resume(
    tmp_path: Path,
) -> None:
    usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_token_details": {"cache_read": 10},
    }
    subagent_usage = {
        "input_tokens": 30,
        "output_tokens": 5,
        "total_tokens": 35,
        "input_token_details": {"cache_read": 4},
    }
    resumed_usage = {
        "input_tokens": 50,
        "output_tokens": 10,
        "total_tokens": 60,
        "input_token_details": {"cache_read": 5},
    }
    graph = _UsageStreamGraph(
        [
            [
                (
                    "messages",
                    (
                        AIMessageChunk(content="main response"),
                        {"langgraph_node": "agent"},
                    ),
                ),
                (
                    "messages",
                    (
                        AIMessageChunk(content="", usage_metadata=usage_metadata),
                        {"langgraph_node": "agent"},
                    ),
                ),
                (
                    ("tools:research",),
                    "messages",
                    (
                        AIMessageChunk(
                            content="subagent response",
                            usage_metadata=subagent_usage,
                        ),
                        {"langgraph_node": "agent", "ls_agent_type": "subagent"},
                    ),
                ),
            ],
            [
                (
                    "messages",
                    (
                        AIMessageChunk(
                            content="resumed response",
                            usage_metadata=resumed_usage,
                        ),
                        {"langgraph_node": "agent"},
                    ),
                )
            ],
        ]
    )
    registry = Registry()
    bus = EventBus()
    state: dict[str, Any] = {}
    statusbar.register(_api("statusbar", registry, bus, state=state))
    observed: list[ModelChunk] = []

    async def record_usage(event: ModelChunk) -> None:
        if getattr(event.chunk, "usage_metadata", None):
            observed.append(event)

    bus.on(ModelChunk, record_usage, plugin="test", priority=1000)
    ctx = _UsageContext(graph, registry, bus)

    await _run_turn(ctx, "first")

    assert [
        (event.role, event.chunk.usage_metadata["input_tokens"])
        for event in observed
    ] == [("main", 100), ("subagent", 30)]
    assert state == {
        "input_tokens": 130,
        "output_tokens": 25,
        "cache_read_tokens": 14,
        "last_input_tokens": 30,
    }

    # AppContext.resume preserves the PluginAPI state object by replacing its
    # contents. Using unrelated saved totals catches handlers that retained
    # local counters instead of reading the live state mapping.
    state.clear()
    state.update(
        {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "cache_read_tokens": 40,
            "last_input_tokens": 900,
        }
    )
    await _run_turn(ctx, "after resume")

    assert [
        (event.role, event.chunk.usage_metadata["input_tokens"])
        for event in observed
    ] == [("main", 100), ("subagent", 30), ("main", 50)]
    assert state == {
        "input_tokens": 1_050,
        "output_tokens": 110,
        "cache_read_tokens": 45,
        "last_input_tokens": 50,
    }
    assert ctx.console.errors == []
