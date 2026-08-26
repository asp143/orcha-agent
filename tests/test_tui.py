import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from prompt_toolkit.keys import Keys

import orcha_agent.tui.app as app_module
from orcha_agent.builtin import commands_core, commands_model
from orcha_agent.core.config import Config
from orcha_agent.core.events import (
    AppExit,
    EventBus,
    InterruptRaised,
    ModelChunk,
    ModelSwitch,
    SessionSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps, Resolved
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionInfo, SessionStore
from orcha_agent.tui.app import (
    AppContext,
    _ModelLabelBuffer,
    _bindings,
    _run_turn,
    _stored_model,
    _ToolCallBuffer,
    run_app,
)


def test_enter_submits_and_alt_enter_inserts_newline() -> None:
    bindings = _bindings()
    enter = bindings.get_bindings_for_keys((Keys.Enter,))[-1]
    alt_enter = bindings.get_bindings_for_keys((Keys.Escape, Keys.Enter))[-1]

    class Buffer:
        def __init__(self) -> None:
            self.submissions = 0
            self.insertions: list[str] = []

        def validate_and_handle(self) -> None:
            self.submissions += 1

        def insert_text(self, text: str) -> None:
            self.insertions.append(text)

    buffer = Buffer()
    event = SimpleNamespace(current_buffer=buffer)

    enter.handler(event)
    alt_enter.handler(event)

    assert buffer.submissions == 1
    assert buffer.insertions == ["\n"]


def test_stored_model_keeps_fallback_lists_and_decodes_legacy_strings() -> None:
    chain = ["anthropic:primary", "openai:fallback"]

    assert _stored_model(chain) == chain
    assert _stored_model("anthropic:primary,openai:fallback") == chain


def test_model_label_is_emitted_once_per_streamed_response() -> None:
    labels = _ModelLabelBuffer()
    metadata = {"ls_model_name": "fallback:model"}

    assert labels.take(AIMessageChunk(content="one", id="response-1"), metadata) == "fallback:model"
    assert labels.take(AIMessageChunk(content="two", id="response-1"), metadata) is None
    assert labels.take(AIMessageChunk(content="next", id="response-2"), metadata) == "fallback:model"


@pytest.mark.asyncio
async def test_compact_uses_the_configured_summarizer_model(tmp_path: Path) -> None:
    history = _HistoryGraph([AIMessage(content="prior answer")])
    ctx = _context(tmp_path, agent=history)
    seen: list[list[Any]] = []

    class Summarizer:
        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            seen.append(messages)
            return AIMessage(content="compact summary")

    ctx.summarizer = Summarizer()

    await ctx.compact()

    assert seen and seen[0][-1].content.startswith("Summarize the conversation")
    assert len(history.update_calls) == 1
    replacement, as_node = history.update_calls[0]
    assert isinstance(replacement[0], RemoveMessage)
    assert replacement[0].id == REMOVE_ALL_MESSAGES
    assert replacement[1:] == [
        HumanMessage(content="[Conversation summary]\ncompact summary")
    ]
    assert as_node == "model"
    assert history.messages == replacement[1:]


@pytest.mark.asyncio
async def test_requested_rebuild_runs_after_the_current_stream_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class OrderedStreamGraph:
        async def astream(
            self, *_args: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[str, Any]]:
            order.append("stream started")
            yield "updates", {}
            order.append("stream finished")

    ctx = _context(tmp_path, agent=OrderedStreamGraph())

    async def rebuild(self: AppContext) -> None:
        order.append("rebuild")
        self.rebuild_requested = False

    monkeypatch.setattr(AppContext, "rebuild", rebuild)
    ctx.rebuild_requested = True

    await _run_turn(ctx, "continue")

    assert order == ["stream started", "stream finished", "rebuild"]


def test_tool_call_chunks_are_buffered_until_arguments_form_complete_json() -> None:
    buffer = _ToolCallBuffer()
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "write_file",
                "args": '{"file_path":"/notes.txt",',
                "id": "call-1",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    second = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": None,
                "args": '"content":"hello"}',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )

    assert buffer.add(first) == []
    events = buffer.add(second)

    assert len(events) == 1
    assert events[0].name == "write_file"
    assert events[0].id == "call-1"
    assert events[0].args == {"file_path": "/notes.txt", "content": "hello"}


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


class _RecordingConsole:
    def __init__(self) -> None:
        self.output: list[tuple[object, ...]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.clear_calls = 0
        self.console = self

    def print(self, *objects: object, **_kwargs: Any) -> None:
        self.output.append(objects)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def clear(self) -> None:
        self.clear_calls += 1


class _PromptScript:
    def __init__(self, *responses: str) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self._next_response = 0

    async def prompt_async(self, message: str) -> str:
        self.prompts.append(message)
        if self._next_response == len(self.responses):
            raise EOFError
        response = self.responses[self._next_response]
        self._next_response += 1
        return response


def _register_lazy_runtime(
    registry: Registry,
    bus: EventBus,
    *,
    provider_factory: Any,
    available: Any = lambda: None,
) -> PluginAPI:
    api = PluginAPI(
        name="lazy-runtime",
        registry=registry,
        bus=bus,
        config={},
        state={},
        request_rebuild=lambda: None,
    )
    commands_core.register(api)
    commands_model.register(api)
    api.add_mode(
        "ask",
        ModeSpec(description="Ask mode", interrupt_on={}, allowed_tools=None),
    )
    api.add_backend("local_shell", lambda _cfg: object())
    api.add_provider(
        "fake",
        provider_factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
        available=available,
    )
    return api


class _SessionDouble:
    def __init__(
        self,
        records: dict[str, SessionInfo] | None = None,
        plugin_states: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.records = records if records is not None else {}
        self.plugin_states = plugin_states if plugin_states is not None else {}
        self.operations: list[tuple[Any, ...]] = []

    def get(self, thread_id: str) -> SessionInfo | None:
        return self.records.get(thread_id)

    def exists(self, thread_id: str) -> bool:
        return thread_id in self.records

    def create(
        self,
        cwd: Path,
        model: str | list[str],
        *,
        mode: str = "ask",
    ) -> SessionInfo:
        thread_id = "new-thread"
        stored_model = model if isinstance(model, str) else list(model)
        record = SessionInfo(
            thread_id=thread_id,
            cwd=str(cwd),
            model=stored_model,
            created="2026-08-27T00:00:00+00:00",
            title=None,
            mode=mode,
        )
        self.operations.append(("create", thread_id))
        self.records[thread_id] = record
        return record


    def set_title(self, thread_id: str, title: str) -> None:
        record = self.records[thread_id]
        self.records[thread_id] = SessionInfo(
            record.thread_id,
            record.cwd,
            record.model,
            record.created,
            title,
            record.mode,
        )

    def set_model(self, thread_id: str, model: str | list[str]) -> None:
        if thread_id not in self.records:
            self.operations.append(("set_model", thread_id, model))
            return
        record = self.records[thread_id]
        self.records[thread_id] = SessionInfo(
            record.thread_id,
            record.cwd,
            model,
            record.created,
            record.title,
            record.mode,
        )
        self.operations.append(("set_model", thread_id, model))

    def set_mode(self, thread_id: str, mode: str) -> None:
        if thread_id not in self.records:
            self.operations.append(("set_mode", thread_id, mode))
            return
        record = self.records[thread_id]
        self.records[thread_id] = SessionInfo(
            record.thread_id,
            record.cwd,
            record.model,
            record.created,
            record.title,
            mode,
        )
        self.operations.append(("set_mode", thread_id, mode))

    def set_plugin_state(self, thread_id: str, plugin: str, state: dict[str, Any]) -> None:
        snapshot = dict(state)
        self.operations.append(("set_plugin_state", thread_id, plugin, snapshot))
        self.plugin_states.setdefault(thread_id, {})[plugin] = snapshot

    def all_plugin_state(self, thread_id: str) -> dict[str, dict[str, Any]]:
        self.operations.append(("all_plugin_state", thread_id))
        return self.plugin_states.get(thread_id, {})


class _SessionStoreDouble(_SessionDouble):
    def __enter__(self) -> "_SessionStoreDouble":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _StreamGraph:
    def __init__(
        self,
        events: list[tuple[str, Any]],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.error = error

    def stream(self, *_args: Any, **_kwargs: Any) -> Iterator[tuple[str, Any]]:
        yield from self.events
        if self.error is not None:
            raise self.error

    async def astream(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error


def _cancelled_graph() -> _StreamGraph:
    return _StreamGraph(
        [
            (
                "messages",
                (
                    AIMessageChunk(content="partial"),
                    {"langgraph_node": "agent"},
                ),
            )
        ],
        error=asyncio.CancelledError(),
    )


class _HistoryGraph:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.update_calls: list[tuple[list[Any], str | None]] = []

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": self.messages})

    def update_state(
        self,
        _config: Any,
        values: dict[str, list[Any]],
        *,
        as_node: str | None = None,
    ) -> None:
        replacement = list(values["messages"])
        self.update_calls.append((replacement, as_node))
        if (
            replacement
            and isinstance(replacement[0], RemoveMessage)
            and replacement[0].id == REMOVE_ALL_MESSAGES
        ):
            replacement = replacement[1:]
        self.messages = replacement


class _ObservedHistoryGraph(_HistoryGraph):
    def __init__(self, messages: list[Any]) -> None:
        super().__init__(messages)
        self.first_use_messages: list[Any] | None = None

    async def astream(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[tuple[str, Any]]:
        self.first_use_messages = list(self.messages)
        yield "updates", {}


def _config(tmp_path: Path, *, model: str = "old:model", cwd: Path | None = None) -> Config:
    return Config(
        model=model,
        subagent_model=model,
        summarizer_model=model,
        mode="ask",
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "sessions.sqlite",
        cwd=cwd or tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )


def _context(
    tmp_path: Path,
    *,
    agent: Any = None,
    session: _SessionDouble | None = None,
    plugin_states: dict[str, dict[str, Any]] | None = None,
) -> AppContext:
    return AppContext(
        cfg=_config(tmp_path),
        registry=Registry(),
        bus=_RecordingBus(),
        session=session or _SessionDouble(),
        plugins=[],
        plugin_states=plugin_states or {},
        console=_RecordingConsole(),
        thread_id="current",
        agent=agent,
    )


@pytest.mark.asyncio
async def test_ensure_agent_builds_at_most_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = object()
    builds: list[str | list[str]] = []

    async def build_once(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> object:
        builds.append(cfg.model)
        return graph

    monkeypatch.setattr(app_module, "build_agent", build_once)
    ctx = _context(tmp_path)

    await ctx.ensure_agent()
    await ctx.ensure_agent()

    assert ctx.agent is graph
    assert builds == ["old:model"]


@pytest.mark.asyncio
async def test_tool_result_in_message_and_update_streams_emits_one_end_event(
    tmp_path: Path,
) -> None:
    result = ToolMessage(
        content="wrote file",
        tool_call_id="call-1",
        name="write_file",
    )
    graph = _StreamGraph(
        [
            ("messages", (result, {"langgraph_node": "tools"})),
            ("updates", {"tools": {"messages": [result]}}),
        ]
    )
    ctx = _context(tmp_path, agent=graph)

    await _run_turn(ctx, "Write the file")

    ends = [event for event in ctx.bus.events if isinstance(event, ToolCallEnd)]
    assert [(event.name, event.id) for event in ends] == [("write_file", "call-1")]


@pytest.mark.asyncio
async def test_unhandled_interrupt_rejects_every_action_and_resumes_turn(
    tmp_path: Path,
) -> None:
    actions = [
        {"name": "write_file", "args": {"file_path": "one.txt"}},
        {"name": "execute", "args": {"command": "false"}},
    ]

    class InterruptGraph:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        async def astream(
            self, next_input: Any, **_kwargs: Any
        ) -> AsyncIterator[tuple[str, Any]]:
            self.inputs.append(next_input)
            if len(self.inputs) == 1:
                yield (
                    "updates",
                    {
                        "__interrupt__": [
                            SimpleNamespace(value={"action_requests": actions})
                        ]
                    },
                )
            else:
                yield (
                    "messages",
                    (
                        AIMessageChunk(content="continued safely"),
                        {"langgraph_node": "agent"},
                    ),
                )

    graph = InterruptGraph()
    ctx = _context(tmp_path, agent=graph)

    await _run_turn(ctx, "Try both actions")

    assert len(graph.inputs) == 2
    assert isinstance(graph.inputs[1], Command)
    assert graph.inputs[1].resume == {
        "decisions": [{"type": "reject"}, {"type": "reject"}]
    }
    assert len(ctx.console.warnings) == 1
    assert "reject" in ctx.console.warnings[0].lower()
    chunks = [event for event in ctx.bus.events if isinstance(event, ModelChunk)]
    assert [(event.chunk.content, event.role) for event in chunks] == [
        ("continued safely", "main")
    ]


@pytest.mark.asyncio
async def test_non_dict_interrupt_payload_is_rejected_without_error(
    tmp_path: Path,
) -> None:
    class MalformedInterruptGraph:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        async def astream(
            self,
            next_input: Any,
            **_kwargs: Any,
        ) -> AsyncIterator[tuple[str, Any]]:
            self.inputs.append(next_input)
            if len(self.inputs) == 1:
                yield (
                    "updates",
                    {
                        "__interrupt__": [
                            SimpleNamespace(id="malformed", value="not-a-dict")
                        ]
                    },
                )

    graph = MalformedInterruptGraph()
    ctx = _context(tmp_path, agent=graph)

    await _run_turn(ctx, "continue")

    assert len(graph.inputs) == 2
    assert isinstance(graph.inputs[1], Command)
    assert graph.inputs[1].resume == {"decisions": []}
    assert len(ctx.console.warnings) == 1
    assert ctx.console.errors == []


@pytest.mark.asyncio
async def test_duplicate_interrupt_id_is_approved_once(
    tmp_path: Path,
) -> None:
    interrupt = SimpleNamespace(
        id="interrupt-1",
        value={"action_requests": [{"name": "write_file", "args": {}}]},
    )

    class DuplicateInterruptGraph:
        async def astream(
            self,
            next_input: Any,
            **_kwargs: Any,
        ) -> AsyncIterator[tuple[str, Any]]:
            if isinstance(next_input, Command):
                return
            yield ("updates", {"__interrupt__": [interrupt]})
            yield ("updates", {"__interrupt__": [interrupt]})

    ctx = _context(tmp_path, agent=DuplicateInterruptGraph())
    bus = EventBus()
    ctx._bus = bus
    ctx.bus = app_module.EventBusView(bus)
    approvals: list[str] = []
    raised: list[InterruptRaised] = []

    async def record(event: InterruptRaised) -> None:
        raised.append(event)

    async def approve(event: InterruptRaised) -> Resolved:
        approvals.append(event.payload["action_requests"][0]["name"])
        return Resolved(resume_value={"decisions": [{"type": "approve"}]})

    bus.on(InterruptRaised, record, plugin="record", priority=0)
    bus.on(InterruptRaised, approve, plugin="approve", priority=1)

    await _run_turn(ctx, "write")

    assert approvals == ["write_file"]
    assert len(raised) == 1


@pytest.mark.asyncio
async def test_subgraph_stream_namespace_marks_model_chunk_as_subagent(
    tmp_path: Path,
) -> None:
    class SubgraphStream:
        def __init__(self) -> None:
            self.stream_kwargs: dict[str, Any] = {}

        async def astream(
            self, *_args: Any, **kwargs: Any
        ) -> AsyncIterator[tuple[Any, str, Any]]:
            self.stream_kwargs = kwargs
            yield (
                ("research:4d3f",),
                "messages",
                (
                    AIMessageChunk(content="subagent finding"),
                    {"langgraph_node": "agent"},
                ),
            )

    graph = SubgraphStream()
    ctx = _context(tmp_path, agent=graph)

    await _run_turn(ctx, "Delegate research")

    assert graph.stream_kwargs.get("subgraphs") is True
    chunks = [event for event in ctx.bus.events if isinstance(event, ModelChunk)]
    assert [(event.chunk.content, event.role) for event in chunks] == [
        ("subagent finding", "subagent")
    ]


@pytest.mark.asyncio
async def test_stream_exception_is_rendered_and_still_ends_turn(tmp_path: Path) -> None:
    ctx = _context(
        tmp_path,
        agent=_StreamGraph([], error=RuntimeError("stream disconnected")),
    )

    await _run_turn(ctx, "Continue")

    assert [type(event) for event in ctx.bus.events] == [TurnStart, TurnEnd]
    assert ctx.console.errors == ["RuntimeError: stream disconnected"]



@pytest.mark.asyncio
async def test_cancelled_stream_prints_interrupted_and_still_ends_turn(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, agent=_cancelled_graph())

    await _run_turn(ctx, "Continue")

    assert isinstance(ctx.bus.events[0], TurnStart)
    assert sum(isinstance(event, TurnEnd) for event in ctx.bus.events) == 1
    assert isinstance(ctx.bus.events[-1], TurnEnd)
    assert ctx.console.output == [()]
    assert ctx.console.warnings == ["interrupted"]
    assert ctx.console.errors == []


@pytest.mark.asyncio
async def test_run_app_continues_after_cancelled_turn_and_emits_app_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    app_events: list[object] = []

    class _Prompt:
        calls = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def prompt_async(self, _message: str) -> str:
            type(self).calls += 1
            if type(self).calls == 1:
                return "Cancel this turn"
            raise EOFError

    async def build_cancelled_graph(*_args: Any, **_kwargs: Any) -> _StreamGraph:
        return _cancelled_graph()

    def load_test_plugins(
        _registry: Registry,
        bus: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        async def record_app_exit(event: AppExit) -> None:
            app_events.append(event)

        bus.on(AppExit, record_app_exit)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", _Prompt)
    monkeypatch.setattr(app_module, "build_agent", build_cancelled_graph)
    monkeypatch.setattr(app_module, "load_plugins", load_test_plugins)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    result = await run_app(_config(tmp_path))

    assert result == 0
    assert _Prompt.calls == 2
    assert [type(event) for event in app_events] == [AppExit]
    assert console.errors == []
    assert console.warnings == ["interrupted"]


@pytest.mark.asyncio
async def test_run_app_continues_after_cancelled_command_and_emits_app_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    app_events: list[AppExit] = []

    class Prompt:
        calls = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def prompt_async(self, _message: str) -> str:
            type(self).calls += 1
            if type(self).calls == 1:
                return "/compact"
            raise EOFError

    async def cancelled_command(*_args: Any, **_kwargs: Any) -> bool:
        raise asyncio.CancelledError

    def load_test_plugins(
        _registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        async def record_exit(event: AppExit) -> None:
            app_events.append(event)

        bus.on(AppExit, record_exit)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", Prompt)
    monkeypatch.setattr(app_module, "dispatch_command", cancelled_command)
    monkeypatch.setattr(app_module, "load_plugins", load_test_plugins)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")
    monkeypatch.setattr(
        app_module,
        "build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=_StreamGraph([])),
    )

    status = await run_app(_config(tmp_path))

    assert status == 0
    assert Prompt.calls == 2
    assert console.warnings == ["interrupted"]
    assert len(app_events) == 1


@pytest.mark.asyncio
async def test_run_app_provider_free_commands_never_construct_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript("/providers", "/help", "/plugins", "/mode")
    factory_calls: list[str] = []

    def forbidden_provider_factory(model_name: str, _config: Any) -> Any:
        factory_calls.append(model_name)
        raise AssertionError("provider factory must not run before a model turn")

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        _register_lazy_runtime(
            registry,
            bus,
            provider_factory=forbidden_provider_factory,
        )
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:unconfigured"))

    assert status == 0
    assert prompt.prompts == ["> "] * 5
    assert factory_calls == []


@pytest.mark.asyncio
async def test_run_app_failed_first_turn_prints_provider_hints_and_reprompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript("Try a turn without credentials")
    factory_calls: list[str] = []

    def forbidden_provider_factory(model_name: str, _config: Any) -> Any:
        factory_calls.append(model_name)
        raise AssertionError("unavailable providers must not be constructed")

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        _register_lazy_runtime(
            registry,
            bus,
            provider_factory=forbidden_provider_factory,
            available=lambda: "no credentials configured",
        )
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:unconfigured"))

    rendered_errors = "\n".join(console.errors)
    assert status == 0
    assert prompt.prompts == ["> ", "> "]
    assert "/login codex" in rendered_errors
    assert "/model" in rendered_errors
    assert factory_calls == []


@pytest.mark.asyncio
async def test_run_app_failed_model_command_prints_provider_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript("/model fake:unconfigured")

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        _register_lazy_runtime(
            registry,
            bus,
            provider_factory=lambda name, config: FakeListChatModel(
                responses=[name]
            ),
            available=lambda: "provider unavailable",
        )
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:unconfigured"))

    rendered_errors = "\n".join(console.errors)
    assert status == 0
    assert "/login codex" in rendered_errors
    assert "/model" in rendered_errors


@pytest.mark.asyncio
async def test_run_app_model_switch_builds_before_the_following_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript("/model fake:ready", "Stream after switching")
    order: list[str] = []

    class FakeGraph:
        async def astream(
            self,
            *_args: Any,
            **_kwargs: Any,
        ) -> AsyncIterator[tuple[str, Any]]:
            order.append("turn")
            yield (
                "messages",
                (
                    AIMessageChunk(content="response from selected model"),
                    {"langgraph_node": "agent"},
                ),
            )

    graph = FakeGraph()

    async def build_selected_model(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> FakeGraph:
        assert cfg.model == "fake:ready", "agent built before /model selected it"
        order.append("switch")
        return graph

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        api = _register_lazy_runtime(
            registry,
            bus,
            provider_factory=lambda model_name, _config: FakeListChatModel(
                responses=[model_name]
            ),
        )
        api.add_renderer(ModelChunk, lambda event: event.chunk.content)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "build_agent", build_selected_model)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:unconfigured"))

    rendered = " ".join(str(value) for row in console.output for value in row)
    assert status == 0
    assert prompt.prompts == ["> "] * 3
    assert order == ["switch", "turn"]
    assert "response from selected model" in rendered


@pytest.mark.asyncio
async def test_first_model_command_cleans_candidate_history_before_lazy_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    candidate_graph = _HistoryGraph([private_history])
    store = _SessionStoreDouble()
    console = _RecordingConsole()
    prompt = _PromptScript("/model new:ready")
    switches: list[tuple[str, str]] = []

    async def build_candidate(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> _HistoryGraph:
        assert cfg.model == "new:ready"
        return candidate_graph

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        api = PluginAPI(
            name="model-command-runtime",
            registry=registry,
            bus=bus,
            config={},
            state={},
            request_rebuild=lambda: None,
        )
        commands_model.register(api)
        registry.providers["old"] = SimpleNamespace(
            foreign_block_types=frozenset({"reasoning"})
        )

        async def record_switch(event: ModelSwitch) -> None:
            switches.append((event.old, event.new))

        bus.on(ModelSwitch, record_switch, plugin="test", priority=0)
        return []

    monkeypatch.setattr(app_module, "SessionStore", lambda _path: store)
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "build_agent", build_candidate)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: object(),
    )
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="old:configured"))

    assert status == 0
    assert console.errors == []
    assert switches == [("old:configured", "new:ready")]
    assert store.records["new-thread"].model == "new:ready"
    assert candidate_graph.messages == [
        private_history.model_copy(
            update={"content": [{"type": "text", "text": "visible"}]}
        )
    ]
    assert len(candidate_graph.update_calls) == 1
    assert candidate_graph.update_calls[0][1] == "model"


@pytest.mark.asyncio
async def test_failed_model_switch_preserves_config_graph_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "thinking", "thinking": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    old_graph = _HistoryGraph([private_history])
    ctx = _context(tmp_path, agent=old_graph)
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"thinking"})
    )
    old_cfg = ctx.cfg

    async def fail_build(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(app_module, "build_agent", fail_build)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await ctx.switch_model("new:model")

    assert ctx.cfg is old_cfg
    assert ctx.agent is old_graph
    assert old_graph.messages == [private_history]
    assert not any(isinstance(event, ModelSwitch) for event in ctx.bus.events)


@pytest.mark.asyncio
async def test_same_provider_model_switch_keeps_provider_private_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "thinking", "thinking": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    old_graph = _HistoryGraph([private_history])
    replacement_graph = object()
    ctx = _context(tmp_path, agent=old_graph)
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"thinking"})
    )
    cleanup_calls: list[tuple[Any, ...]] = []

    async def build_replacement(*_args: Any, **_kwargs: Any) -> object:
        return replacement_graph

    monkeypatch.setattr(app_module, "build_agent", build_replacement)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: "replacement summarizer",
    )
    monkeypatch.setattr(
        app_module,
        "strip_foreign_blocks",
        lambda *args: cleanup_calls.append(args),
    )

    await ctx.switch_model("old:replacement")

    assert cleanup_calls == []
    assert old_graph.messages == [private_history]
    assert ctx.cfg.model == "old:replacement"
    assert ctx.agent is replacement_graph
    assert ctx.summarizer == "replacement summarizer"
    switches = [event for event in ctx.bus.events if isinstance(event, ModelSwitch)]
    assert [(event.old, event.new) for event in switches] == [
        ("old:model", "old:replacement")
    ]


@pytest.mark.asyncio
async def test_cross_provider_model_switch_cleans_history_before_atomic_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "thinking", "thinking": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    old_graph = _HistoryGraph([private_history])
    replacement_graph = object()
    ctx = _context(tmp_path, agent=old_graph)
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"thinking"})
    )
    cleanup_order: list[tuple[Any, str | list[str]]] = []
    strip_foreign_blocks = app_module.strip_foreign_blocks

    async def build_replacement(*_args: Any, **_kwargs: Any) -> object:
        return replacement_graph

    def record_cleanup(
        graph: Any,
        thread_config: Any,
        foreign_types: set[str] | frozenset[str],
    ) -> None:
        cleanup_order.append((ctx.agent, ctx.cfg.model))
        strip_foreign_blocks(graph, thread_config, foreign_types)

    monkeypatch.setattr(app_module, "build_agent", build_replacement)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: "replacement summarizer",
    )
    monkeypatch.setattr(app_module, "strip_foreign_blocks", record_cleanup)

    await ctx.switch_model("new:model")

    assert cleanup_order == [(old_graph, "old:model")]
    assert old_graph.messages == [
        private_history.model_copy(
            update={"content": [{"type": "text", "text": "visible"}]}
        )
    ]
    assert ctx.cfg.model == "new:model"
    assert ctx.cfg.model_overridden is False
    assert ctx.agent is replacement_graph
    assert ctx.summarizer == "replacement summarizer"
    switches = [event for event in ctx.bus.events if isinstance(event, ModelSwitch)]
    assert [(event.old, event.new) for event in switches] == [
        ("old:model", "new:model")
    ]


@pytest.mark.asyncio
async def test_model_persistence_failure_keeps_live_context_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "visible"},
        ]
    )
    old_graph = _HistoryGraph([private_history])
    current = SessionInfo(
        thread_id="current",
        cwd=str(tmp_path),
        model="old:model",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    session = _SessionDouble(records={"current": current})
    ctx = _context(tmp_path, agent=old_graph, session=session)
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"thinking"})
    )
    old_cfg = ctx.cfg

    async def build_replacement(*_args: Any, **_kwargs: Any) -> object:
        return object()

    def fail_persistence(_thread_id: str, _model: str | list[str]) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(app_module, "build_agent", build_replacement)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: object(),
    )
    session.set_model = fail_persistence

    with pytest.raises(RuntimeError, match="database unavailable"):
        await ctx.switch_model("new:model")

    assert ctx.cfg is old_cfg
    assert ctx.agent is old_graph
    assert old_graph.messages == [private_history]
    assert ctx.summarizer is None
    assert not any(isinstance(event, ModelSwitch) for event in ctx.bus.events)


@pytest.mark.asyncio
async def test_unknown_mode_preserves_config_and_graph(tmp_path: Path) -> None:
    graph = object()
    ctx = _context(tmp_path, agent=graph)
    old_cfg = ctx.cfg

    await ctx.switch_mode("missing")

    assert ctx.console.errors == ["Unknown mode: missing"]
    assert ctx.cfg is old_cfg
    assert ctx.agent is graph


@pytest.mark.asyncio
async def test_valid_mode_switch_is_transactional_when_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = object()
    ctx = _context(tmp_path, agent=graph)
    old_cfg = ctx.cfg
    ctx._registry.modes["plan"] = ModeSpec(
        description="Planning mode",
        interrupt_on={},
        allowed_tools=None,
    )

    async def fail_build(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("mode build failed")

    monkeypatch.setattr(app_module, "build_agent", fail_build)

    with pytest.raises(RuntimeError, match="mode build failed"):
        await ctx.switch_mode("plan")

    assert ctx.cfg is old_cfg
    assert ctx.agent is graph


@pytest.mark.asyncio
async def test_valid_mode_switch_builds_candidate_before_swapping_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_graph = object()
    replacement_graph = object()
    ctx = _context(tmp_path, agent=old_graph)
    old_cfg = ctx.cfg
    ctx._registry.modes["plan"] = ModeSpec(
        description="Planning mode",
        interrupt_on={},
        allowed_tools=None,
    )
    build_observations: list[tuple[str, str, Any]] = []

    async def build_candidate(
        _registry: Registry,
        cfg: Config,
        _session: Any,
        _bus: Any,
        **_kwargs: Any,
    ) -> object:
        build_observations.append((cfg.mode, ctx.cfg.mode, ctx.agent))
        return replacement_graph

    monkeypatch.setattr(app_module, "build_agent", build_candidate)

    await ctx.switch_mode("plan")

    assert build_observations == [("plan", "ask", old_graph)]
    assert ctx.cfg is not old_cfg
    assert ctx.cfg.mode == "plan"
    assert ctx.agent is replacement_graph


@pytest.mark.asyncio
async def test_model_switch_preserves_always_approved_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(
        tmp_path,
        agent=_HistoryGraph([]),
        plugin_states={"approval": {"always_allowed": ["execute", "write_file"]}},
    )
    captured: list[set[str]] = []

    async def capture_build(*_args: Any, **kwargs: Any) -> object:
        captured.append(set(kwargs["always_allowed"]))
        return object()

    monkeypatch.setattr(app_module, "build_agent", capture_build)

    await ctx.switch_model("new:model")

    assert captured == [{"execute", "write_file"}]


@pytest.mark.asyncio
async def test_resume_restores_saved_cwd_and_model_before_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_cwd = tmp_path / "saved-project"
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(saved_cwd),
        model="saved:model",
        created="2026-08-27T00:00:00+00:00",
        title=None,
    )
    session = _SessionDouble(
        records={"saved": saved},
        plugin_states={"saved": {"approval": {"always_allowed": ["write_file"]}}},
    )
    approval_state = {"always_allowed": ["execute"]}
    scratch_state = {"draft": "keep before resume"}
    ctx = _context(
        tmp_path,
        agent=object(),
        session=session,
        plugin_states={"approval": approval_state, "scratch": scratch_state},
    )
    rebuilt: list[tuple[Path, str | list[str], str, set[str]]] = []
    replacement_graph = object()

    async def capture_build(
        _registry: Registry,
        cfg: Config,
        _session: Any,
        _bus: Any,
        **_kwargs: Any,
    ) -> Any:
        rebuilt.append(
            (cfg.cwd, cfg.model, ctx.thread_id, set(_kwargs["always_allowed"]))
        )
        return replacement_graph

    monkeypatch.setattr(app_module, "build_agent", capture_build)

    await ctx.resume("saved")

    assert rebuilt == [
        (saved_cwd, "saved:model", "saved", {"write_file"})
    ]
    assert ctx.cfg.cwd == saved_cwd
    assert ctx.cfg.model == "saved:model"
    assert ctx.agent is replacement_graph
    assert ctx.plugin_states["approval"] is approval_state
    assert approval_state == {"always_allowed": ["write_file"]}
    assert ctx.plugin_states["scratch"] is scratch_state
    assert scratch_state == {}
    assert session.plugin_states["current"] == {
        "approval": {"always_allowed": ["execute"]},
        "scratch": {"draft": "keep before resume"},
    }
    assert session.operations[:3] == [
        (
            "set_plugin_state",
            "current",
            "approval",
            {"always_allowed": ["execute"]},
        ),
        (
            "set_plugin_state",
            "current",
            "scratch",
            {"draft": "keep before resume"},
        ),
        ("all_plugin_state", "saved"),
    ]


@pytest.mark.asyncio
async def test_resume_build_failure_rolls_back_thread_config_agent_and_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "saved"),
        model="saved:model",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="plan",
    )
    session = _SessionDouble(
        records={"saved": saved},
        plugin_states={"saved": {"approval": {"always_allowed": ["write_file"]}}},
    )
    approval_state = {"always_allowed": ["execute"]}
    old_graph = object()
    ctx = _context(
        tmp_path,
        agent=old_graph,
        session=session,
        plugin_states={"approval": approval_state},
    )
    old_cfg = ctx.cfg

    async def fail_build(*_args: Any, **_kwargs: Any) -> object:
        raise RuntimeError("resume build failed")

    monkeypatch.setattr(app_module, "build_agent", fail_build)

    with pytest.raises(RuntimeError, match="resume build failed"):
        await ctx.resume("saved")

    assert ctx.thread_id == "current"
    assert ctx.cfg is old_cfg
    assert ctx.agent is old_graph
    assert approval_state == {"always_allowed": ["execute"]}


@pytest.mark.asyncio
async def test_resume_keeps_explicit_cli_model_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "saved"),
        model="stored:model",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    session = _SessionDouble(records={"saved": saved})
    ctx = _context(tmp_path, agent=object(), session=session)
    ctx.cfg = replace(
        ctx.cfg,
        model="cli:model",
        model_overridden=True,
    )
    built_models: list[str | list[str]] = []

    async def capture_build(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> object:
        built_models.append(cfg.model)
        return object()

    monkeypatch.setattr(app_module, "build_agent", capture_build)

    await ctx.resume("saved")

    assert built_models == ["cli:model"]
    assert ctx.cfg.model == "cli:model"
    switches = [event for event in ctx.bus.events if isinstance(event, SessionSwitch)]
    assert [(event.old, event.new) for event in switches] == [("current", "saved")]


@pytest.mark.asyncio
async def test_resume_with_cross_provider_override_cleans_candidate_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "saved"),
        model="old:stored",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    private_history = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    candidate_graph = _ObservedHistoryGraph([private_history])
    ctx = _context(
        tmp_path,
        agent=object(),
        session=_SessionDouble(records={"saved": saved}),
    )
    ctx.cfg = replace(
        ctx.cfg,
        model="new:cli",
        model_overridden=True,
    )
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"reasoning"})
    )

    async def build_candidate(*_args: Any, **_kwargs: Any) -> _ObservedHistoryGraph:
        return candidate_graph

    monkeypatch.setattr(app_module, "build_agent", build_candidate)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: object(),
    )

    await ctx.resume("saved")
    await _run_turn(ctx, "continue")

    cleaned_history = [
        private_history.model_copy(
            update={"content": [{"type": "text", "text": "visible"}]}
        )
    ]
    assert candidate_graph.messages == cleaned_history
    assert candidate_graph.first_use_messages == cleaned_history
    assert len(candidate_graph.update_calls) == 1
    assert candidate_graph.update_calls[0][1] == "model"
    assert ctx.cfg.model == "new:cli"
    assert ctx.agent is candidate_graph


@pytest.mark.asyncio
async def test_resume_with_same_provider_override_leaves_candidate_history_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "saved"),
        model="old:stored",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    private_history = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    candidate_graph = _ObservedHistoryGraph([private_history])
    ctx = _context(
        tmp_path,
        agent=object(),
        session=_SessionDouble(records={"saved": saved}),
    )
    ctx.cfg = replace(
        ctx.cfg,
        model="old:cli",
        model_overridden=True,
    )
    ctx._registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"reasoning"})
    )

    async def build_candidate(*_args: Any, **_kwargs: Any) -> _ObservedHistoryGraph:
        return candidate_graph

    monkeypatch.setattr(app_module, "build_agent", build_candidate)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: object(),
    )

    await ctx.resume("saved")
    await _run_turn(ctx, "continue")

    assert candidate_graph.messages == [private_history]
    assert candidate_graph.first_use_messages == [private_history]
    assert candidate_graph.update_calls == []
    assert ctx.cfg.model == "old:cli"
    assert ctx.agent is candidate_graph


@pytest.mark.asyncio
async def test_resume_rechecks_trust_for_saved_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "untrusted"),
        model="stored:model",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    session = _SessionDouble(records={"saved": saved})
    ctx = _context(tmp_path, agent=object(), session=session)
    ctx.cfg = replace(ctx.cfg, trust_cwd=True)
    built_trust: list[bool] = []

    async def capture_build(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> object:
        built_trust.append(cfg.trust_cwd)
        return object()

    monkeypatch.setattr(app_module, "build_agent", capture_build)

    await ctx.resume("saved")

    assert built_trust == [False]
    assert ctx.cfg.trust_cwd is False
    assert ctx.cfg.model == "stored:model"


@pytest.mark.asyncio
async def test_clear_persists_current_plugin_state_before_clearing_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _SessionDouble()
    approval_state = {"always_allowed": ["execute"]}
    old_graph = object()
    replacement_graph = object()
    ctx = _context(
        tmp_path,
        agent=old_graph,
        session=session,
        plugin_states={"approval": approval_state},
    )
    build_allowed: list[set[str]] = []

    async def build_replacement(*_args: Any, **kwargs: Any) -> object:
        build_allowed.append(set(kwargs["always_allowed"]))
        return replacement_graph

    monkeypatch.setattr(app_module, "build_agent", build_replacement)

    await ctx.clear()

    assert session.plugin_states.get("current") == {
        "approval": {"always_allowed": ["execute"]}
    }
    assert session.operations[:2] == [
        (
            "set_plugin_state",
            "current",
            "approval",
            {"always_allowed": ["execute"]},
        ),
        ("create", "new-thread"),
    ]
    assert ctx.thread_id == "new-thread"
    assert ctx.plugin_states["approval"] is approval_state
    assert approval_state == {}
    assert build_allowed == [set()]
    assert ctx.agent is replacement_graph
    assert ctx.console.clear_calls == 1
    switches = [event for event in ctx.bus.events if isinstance(event, SessionSwitch)]
    assert [(event.old, event.new) for event in switches] == [
        ("current", "new-thread")
    ]


@pytest.mark.asyncio
async def test_app_context_exposes_live_read_only_registry_and_bus_views(
    tmp_path: Path,
) -> None:
    registry = Registry()
    bus = EventBus()
    ctx = AppContext(
        cfg=_config(tmp_path),
        registry=registry,
        bus=bus,
        session=_SessionDouble(),
        plugins=[],
        plugin_states={},
        console=_RecordingConsole(),
        thread_id="current",
        agent=None,
    )
    api = PluginAPI(
        name="late-plugin",
        registry=registry,
        bus=bus,
        config={},
        state={},
        request_rebuild=lambda: None,
    )
    emitted: list[str] = []

    async def command_handler(_ctx: object, _args: str) -> None:
        return None

    async def event_handler(event: TurnStart) -> None:
        emitted.append(event.text)

    api.add_command("late", command_handler, help="registered after context creation")
    api.on(TurnStart, event_handler)

    assert ctx.registry.commands["late"].handler is command_handler
    assert any(entry.handler is event_handler for entry in ctx.bus.handlers)
    await ctx.bus.emit(TurnStart(thread_id="current", text="plugin event"))
    assert emitted == ["plugin event"]
    with pytest.raises(TypeError):
        ctx.registry.commands["forbidden"] = object()
    with pytest.raises(AttributeError):
        ctx.registry.renderers.append(object())
    with pytest.raises(AttributeError):
        ctx.bus.handlers.append(object())
    with pytest.raises(AttributeError):
        ctx.bus.on(TurnStart, event_handler)


def _fail_if_prompt_is_constructed(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("non-interactive startup paths must not construct a prompt")


@pytest.mark.asyncio
async def test_run_app_lists_sessions_and_returns_zero_without_prompting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path, cwd=tmp_path / "workspace")
    with SessionStore(cfg.db_path) as store:
        saved = store.create(
            cfg.cwd,
            "fake:model",
            title="Saved work",
            thread_id="saved-session",
        )
    console = _RecordingConsole()
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", _fail_if_prompt_is_constructed)

    status = await run_app(replace(cfg, list_sessions=True))

    rendered = " ".join(str(value) for row in console.output for value in row)
    assert status == 0
    assert saved.thread_id in rendered
    assert saved.cwd in rendered
    assert "Saved work" in rendered


@pytest.mark.parametrize(
    ("configured_model", "should_clean"),
    [("new:cli", True), ("old:cli", False)],
    ids=["cross-provider", "same-provider"],
)
@pytest.mark.asyncio
async def test_run_app_resume_override_cleans_only_across_providers_before_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_model: str,
    should_clean: bool,
) -> None:
    saved = SessionInfo(
        thread_id="saved",
        cwd=str(tmp_path / "saved"),
        model="old:stored",
        created="2026-08-27T00:00:00+00:00",
        title=None,
        mode="ask",
    )
    private_history = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "provider-private"},
            {"type": "text", "text": "visible"},
        ]
    )
    candidate_graph = _ObservedHistoryGraph([private_history])
    store = _SessionStoreDouble(records={"saved": saved})
    console = _RecordingConsole()
    prompt = _PromptScript("continue")
    built_models: list[str | list[str]] = []

    async def build_candidate(
        _registry: Registry,
        cfg: Config,
        *_args: Any,
        **_kwargs: Any,
    ) -> _ObservedHistoryGraph:
        built_models.append(cfg.model)
        return candidate_graph

    def load_runtime(
        registry: Registry,
        _bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        registry.providers["old"] = SimpleNamespace(
            foreign_block_types=frozenset({"reasoning"})
        )
        return []

    monkeypatch.setattr(app_module, "SessionStore", lambda _path: store)
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", lambda **_kwargs: prompt)
    monkeypatch.setattr(app_module, "build_agent", build_candidate)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(
        AppContext,
        "_resolve_summarizer",
        lambda _self, _cfg: object(),
    )
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")
    cfg = replace(
        _config(tmp_path, model=configured_model),
        resume="saved",
        model_overridden=True,
    )

    status = await run_app(cfg)

    expected_history = (
        [
            private_history.model_copy(
                update={"content": [{"type": "text", "text": "visible"}]}
            )
        ]
        if should_clean
        else [private_history]
    )
    assert status == 0
    assert console.errors == []
    assert built_models == [configured_model]
    assert candidate_graph.messages == expected_history
    assert candidate_graph.first_use_messages == expected_history
    assert [as_node for _messages, as_node in candidate_graph.update_calls] == (
        ["model"] if should_clean else []
    )


@pytest.mark.asyncio
async def test_run_app_unknown_resume_returns_one_without_prompting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(_config(tmp_path), resume="missing-session")
    console = _RecordingConsole()
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", _fail_if_prompt_is_constructed)

    status = await run_app(cfg)

    assert status == 1
    assert console.errors == ["Unknown session: missing-session"]


@pytest.mark.asyncio
async def test_run_app_unopenable_database_returns_one_and_names_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "sessions-directory"
    database_path.mkdir()
    cfg = replace(_config(tmp_path), db_path=database_path)
    console = _RecordingConsole()
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "PromptSession", _fail_if_prompt_is_constructed)

    status = await run_app(cfg)

    assert status == 1
    assert len(console.errors) == 1
    assert "cannot open session database" in console.errors[0].lower()
    assert str(database_path) in console.errors[0]
