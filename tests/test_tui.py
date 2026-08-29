import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.keys import Keys

import orcha_agent.tui.app as app_module
from orcha_agent.builtin import commands_core, commands_model
from orcha_agent.core.config import Config
from orcha_agent.core.events import (
    AppExit,
    AppStart,
    EventBus,
    InterruptRaised,
    ModelChunk,
    ModelSwitch,
    SessionSwitch,
    ThreadSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.ledger import (
    CompactionEntry,
    CustomEntry,
    Ledger,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
    ResetBoundaryEntry,
    build_context,
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
    ledger = Ledger(ctx.session)
    ledger.append(
        ctx.session_id,
        MessageEntry(message=message_to_dict(AIMessage(content="prior answer"))),
    )
    seen: list[list[Any]] = []

    class Summarizer:
        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            seen.append(messages)
            return AIMessage(content="compact summary")

    ctx.summarizer = Summarizer()
    old_thread = ctx.thread_id

    await ctx.compact()

    assert seen and seen[0][-1].content.startswith("Summarize the conversation")
    compaction = Ledger(ctx.session).path(ctx.session_id)[-1]
    assert isinstance(compaction, CompactionEntry)
    assert compaction.summary == "compact summary"
    assert compaction.first_kept_id is None
    assert history.update_calls == []
    assert history.seed_calls[-1] == (
        ctx.thread_config,
        {
            "messages": [
                HumanMessage(content="[Conversation summary]\ncompact summary")
            ],
            "todos": [],
            "files": {},
        },
    )
    switch = _thread_switches(ctx)
    assert len(switch) == 1
    assert (switch[0].session_id, switch[0].old, switch[0].new, switch[0].reason) == (
        ctx.session_id,
        old_thread,
        ctx.thread_id,
        "compact",
    )
    assert _session_switches(ctx) == []


@pytest.mark.asyncio
async def test_compact_excludes_provider_thinking_from_summary_input(
    tmp_path: Path,
) -> None:
    history = _HistoryGraph(
        [
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "private reasoning"}
                        ],
                    },
                    {"type": "thinking", "thinking": "private thinking"},
                    {"type": "text", "text": "visible answer"},
                ],
                additional_kwargs={
                    "reasoning": {"summary": "legacy private reasoning"},
                    "keep": "additional",
                },
                response_metadata={
                    "thinking": "metadata private thinking",
                    "keep": "metadata",
                },
            )
        ]
    )
    ctx = _context(tmp_path, agent=history)
    Ledger(ctx.session).append(
        ctx.session_id,
        MessageEntry(message=message_to_dict(history.messages[0])),
    )
    seen: list[list[Any]] = []

    class Summarizer:
        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            seen.append(messages)
            return AIMessage(content="compact summary")

    ctx.summarizer = Summarizer()

    await ctx.compact()

    summarized = seen[0][0]
    assert isinstance(summarized, AIMessage)
    assert summarized.content == [{"type": "text", "text": "visible answer"}]
    assert summarized.additional_kwargs == {"keep": "additional"}
    assert summarized.response_metadata == {"keep": "metadata"}


@pytest.mark.asyncio
async def test_requested_rebuild_runs_after_the_current_stream_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class OrderedStreamGraph:
        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

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


@pytest.mark.asyncio
async def test_run_turn_requires_capture_turn_on_its_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(tmp_path, agent=_StreamGraph([]))
    monkeypatch.delattr(AppContext, "capture_turn")

    with pytest.raises(AttributeError, match="capture_turn"):
        await _run_turn(ctx, "continue")


@pytest.mark.asyncio
async def test_cancelled_run_turn_requires_record_exit_on_its_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(tmp_path, agent=_cancelled_graph())
    monkeypatch.delattr(AppContext, "record_exit")

    with pytest.raises(AttributeError, match="record_exit"):
        await _run_turn(ctx, "continue")


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
        self._submit: Any = None
        self.application = SimpleNamespace(exit=lambda: None)

    def runtime(self, submit: Any, **_kwargs: Any) -> "_PromptScript":
        self._submit = submit
        return self

    async def run(self) -> None:
        while True:
            try:
                response = await self.prompt_async("> ")
            except EOFError:
                return
            await self._submit(response)

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


class _SessionDouble(SessionStore):
    def __init__(
        self,
        records: dict[str, SessionInfo] | None = None,
        plugin_states: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self._temporary_directory = TemporaryDirectory()
        super().__init__(Path(self._temporary_directory.name) / "sessions.sqlite")
        self.operations: list[tuple[Any, ...]] = []
        for record in (records or {}).values():
            created = super().create(
                record.cwd,
                record.model,
                mode=record.mode,
                title=record.title,
                thread_id=record.thread_id,
            )
            requested_thread = record.current_thread or created.current_thread
            if requested_thread != created.current_thread:
                self.create_thread(record.thread_id, thread_id=requested_thread)
                self.set_current_thread(record.thread_id, requested_thread)
            self._put_live_checkpoint(requested_thread)
        for session_id, states in (plugin_states or {}).items():
            for plugin, state in states.items():
                super().set_plugin_state(session_id, plugin, state)

    @property
    def records(self) -> dict[str, SessionInfo]:
        return {record.thread_id: record for record in self.list()}

    @property
    def plugin_states(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            record.thread_id: SessionStore.all_plugin_state(self, record.thread_id)
            for record in self.list()
        }

    def _put_live_checkpoint(self, thread_id: str) -> None:
        self.saver.put(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            empty_checkpoint(),
            {},
            {},
        )

    def create(
        self,
        cwd: Path,
        model: str | list[str],
        mode: str = "ask",
        *,
        title: str | None = None,
        thread_id: str | None = None,
    ) -> SessionInfo:
        identifier = thread_id or "new-thread"
        record = super().create(
            cwd,
            model,
            mode=mode,
            title=title,
            thread_id=identifier,
        )
        self.operations.append(("create", identifier))
        return record

    def set_title(self, session_id: str, title: str) -> None:
        super().set_title(session_id, title)

    def set_model(self, session_id: str, model: str | list[str]) -> None:
        super().set_model(session_id, model)
        self.operations.append(("set_model", session_id, model))

    def set_mode(self, session_id: str, mode: str) -> None:
        super().set_mode(session_id, mode)
        self.operations.append(("set_mode", session_id, mode))

    def set_plugin_state(self, session_id: str, plugin: str, state: dict[str, Any]) -> None:
        snapshot = dict(state)
        super().set_plugin_state(session_id, plugin, snapshot)
        self.operations.append(("set_plugin_state", session_id, plugin, snapshot))

    def all_plugin_state(self, session_id: str) -> dict[str, dict[str, Any]]:
        self.operations.append(("all_plugin_state", session_id))
        return super().all_plugin_state(session_id)


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
        values: dict[str, Any] | None = None,
    ) -> None:
        self.events = events
        self.error = error
        self.values = {
            "messages": [],
            "todos": [],
            "files": {},
            **(values or {}),
        }
        self.seed_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values=self.values)

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        snapshot = {
            "messages": list(values.get("messages", [])),
            "todos": list(values.get("todos", [])),
            "files": dict(values.get("files", {})),
        }
        self.seed_calls.append((config, snapshot))
        self.values.update(snapshot)

    def stream(self, *_args: Any, **_kwargs: Any) -> Iterator[tuple[str, Any]]:
        yield from self.events
        if self.error is not None:
            raise self.error

    async def astream(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error


def _empty_graph_state() -> SimpleNamespace:
    return SimpleNamespace(values={"messages": [], "todos": [], "files": {}})


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
        self.todos: list[Any] = []
        self.files: dict[str, Any] = {}
        self.update_calls: list[tuple[list[Any], str | None]] = []
        self.seed_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(
            values={
                "messages": self.messages,
                "todos": self.todos,
                "files": self.files,
            }
        )

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str,
    ) -> None:
        assert as_node == "model"
        snapshot = {
            "messages": list(values.get("messages", [])),
            "todos": list(values.get("todos", [])),
            "files": dict(values.get("files", {})),
        }
        self.seed_calls.append((config, snapshot))
        self.messages = snapshot["messages"]
        self.todos = snapshot["todos"]
        self.files = snapshot["files"]

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
    active_session = session or _SessionDouble()
    current = active_session.get("current")
    if current is None:
        current = active_session.create(
            tmp_path,
            "old:model",
            mode="ask",
            thread_id="current",
        )
        active_session.operations.clear()
    return AppContext(
        cfg=_config(tmp_path),
        registry=Registry(),
        bus=_RecordingBus(),
        session=active_session,
        plugins=[],
        plugin_states=plugin_states or {},
        console=_RecordingConsole(),
        session_id=current.thread_id,
        thread_id=current.current_thread,
        agent=agent,
    )


def test_bottom_toolbar_joins_nonempty_segments_and_isolates_failures(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path)
    api = PluginAPI(
        name="toolbar-test",
        config={},
        state={},
        registry=ctx._registry,
        bus=ctx._bus,
        request_rebuild=ctx.request_rebuild,
    )

    def fail(_ctx: AppContext) -> str:
        raise RuntimeError("segment failed")

    api.add_status_segment("model", lambda _ctx: "model", priority=10)
    api.add_status_segment("broken", fail, priority=20)
    api.add_status_segment("empty", lambda _ctx: "", priority=30)
    api.add_status_segment("missing", lambda _ctx: None, priority=35)
    api.add_status_segment("mode", lambda _ctx: "ask", priority=40)

    fragments = to_formatted_text(app_module._bottom_toolbar(ctx))
    plain = "".join(text for _style, text in fragments)

    assert plain.index("model") < plain.index("ask") < plain.index("!broken")
    assert "empty" not in plain
    assert "<style" not in plain
    model_styles = [style for style, text in fragments if "model" in text]
    assert len(model_styles) == 1
    assert "class:text" in model_styles[0]


@pytest.mark.asyncio
async def test_run_app_configures_live_bottom_toolbar_without_building_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    prompt = _PromptScript()

    def application_runtime(submit: Any, **kwargs: Any) -> _PromptScript:
        captured.update(kwargs)
        return prompt.runtime(submit)

    monkeypatch.setattr(app_module, "ConsoleOutput", _RecordingConsole)
    monkeypatch.setattr(app_module, "ApplicationRuntime", application_runtime)
    monkeypatch.setattr(app_module, "load_plugins", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path))

    assert status == 0
    assert callable(captured["status"])


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

        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

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

        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

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
        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

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

        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

        async def astream(
            self, *_args: Any, **kwargs: Any
        ) -> AsyncIterator[tuple[Any, str, Any]]:
            self.stream_kwargs = kwargs
            yield (
                ("research:4d3f",),
                "messages",
                (
                    AIMessageChunk(
                        content="subagent finding",
                        tool_call_chunks=[
                            {
                                "name": "read_file",
                                "args": '{"path":"README.md"}',
                                "id": "call-1",
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    ),
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
    assert [getattr(event, "source_id", None) for event in chunks] == [
        "research:4d3f"
    ]
    tool_starts = [
        event for event in ctx.bus.events if isinstance(event, ToolCallStart)
    ]
    assert [getattr(event, "source_id", None) for event in tool_starts] == [
        "research:4d3f"
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

        def __init__(self, submit: Any, **_kwargs: Any) -> None:
            self._submit = submit
            self.application = SimpleNamespace(exit=lambda: None)

        async def run(self) -> None:
            type(self).calls += 1
            await self._submit("Cancel this turn")
            type(self).calls += 1

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
    monkeypatch.setattr(app_module, "ApplicationRuntime", _Prompt)
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

        def __init__(self, submit: Any, **_kwargs: Any) -> None:
            self._submit = submit
            self.application = SimpleNamespace(exit=lambda: None)

        async def run(self) -> None:
            type(self).calls += 1
            await self._submit("/compact")
            type(self).calls += 1

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
    monkeypatch.setattr(app_module, "ApplicationRuntime", Prompt)
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


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("model x", "Did you mean /model x?"),
        ("mode plan", "Did you mean /mode plan?"),
        ("help", "Did you mean /help?"),
    ],
)
@pytest.mark.parametrize("has_agent", [False, True])
@pytest.mark.asyncio
async def test_run_app_hints_for_registered_commands_missing_a_slash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    hint: str,
    has_agent: bool,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript(text)
    ensure_calls: list[Any] = []
    build_calls: list[Any] = []

    class RecordingGraph:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

        async def astream(
            self,
            next_input: Any,
            **_kwargs: Any,
        ) -> AsyncIterator[tuple[str, Any]]:
            self.inputs.append(next_input)
            yield "updates", {}

    graph = RecordingGraph()

    async def record_ensure(self: AppContext) -> bool:
        ensure_calls.append(self.agent)
        return self.agent is not None

    async def record_build(*args: Any, **kwargs: Any) -> RecordingGraph:
        build_calls.append((args, kwargs))
        return graph

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        _register_lazy_runtime(
            registry,
            bus,
            provider_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider factory must not run for a command hint")
            ),
        )

        async def install_agent(event: AppStart) -> None:
            if has_agent:
                event.ctx.agent = graph

        bus.on(AppStart, install_agent)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "build_agent", record_build)
    monkeypatch.setattr(AppContext, "ensure_agent", record_ensure)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:ready"))

    assert status == 0
    assert prompt.prompts == ["> ", "> "]
    assert console.warnings == [hint]
    assert console.errors == []
    assert ensure_calls == []
    assert build_calls == []
    assert graph.inputs == []


@pytest.mark.asyncio
async def test_run_app_sends_a_non_command_first_word_to_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _RecordingConsole()
    prompt = _PromptScript("modeling the problem")

    class RecordingGraph:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

        async def astream(
            self,
            next_input: Any,
            **_kwargs: Any,
        ) -> AsyncIterator[tuple[str, Any]]:
            self.inputs.append(next_input)
            yield "updates", {}

    graph = RecordingGraph()

    def load_runtime(
        registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        _register_lazy_runtime(
            registry,
            bus,
            provider_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("provider factory must not replace the usable agent")
            ),
        )

        async def install_agent(event: AppStart) -> None:
            event.ctx.agent = graph

        bus.on(AppStart, install_agent)
        return []

    async def forbidden_build(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("usable agent must not be rebuilt")

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "build_agent", forbidden_build)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path, model="fake:ready"))

    assert status == 0
    assert graph.inputs == [
        {"messages": [{"role": "user", "content": "modeling the problem"}]}
    ]
    assert console.warnings == []
    assert console.errors == []


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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
        def get_state(self, _config: Any) -> SimpleNamespace:
            return _empty_graph_state()

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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
async def test_model_switch_retargets_unset_role_models_to_selected_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _SessionDouble()
    ctx = _context(tmp_path, session=session)
    ctx.cfg = replace(
        ctx.cfg,
        model="anthropic:old",
        subagent_model=None,
        summarizer_model=None,
    )
    api = PluginAPI(
        name="switch-runtime",
        config={},
        state={},
        registry=ctx._registry,
        bus=ctx._bus,
        request_rebuild=lambda: None,
    )
    api.add_mode(
        "ask",
        ModeSpec(description="Ask mode", interrupt_on={}, allowed_tools=None),
    )
    api.add_backend("local_shell", lambda _cfg: object())
    caps = ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=False,
        structured_output=False,
        max_context=None,
    )
    anthropic_availability_calls: list[None] = []
    anthropic_factory_calls: list[str] = []
    codex_availability_calls: list[None] = []
    codex_factory_calls: list[str] = []
    created: list[FakeListChatModel] = []

    def anthropic_available() -> str:
        anthropic_availability_calls.append(None)
        return "ANTHROPIC_API_KEY is missing"

    def anthropic_factory(model_name: str, _provider_config: Any) -> FakeListChatModel:
        anthropic_factory_calls.append(model_name)
        return FakeListChatModel(responses=[model_name])

    def codex_available() -> None:
        codex_availability_calls.append(None)

    def codex_factory(model_name: str, _provider_config: Any) -> FakeListChatModel:
        codex_factory_calls.append(model_name)
        model = FakeListChatModel(responses=[f"codex:{model_name}"])
        created.append(model)
        return model

    api.add_provider(
        "anthropic",
        anthropic_factory,
        capabilities=caps,
        available=anthropic_available,
    )
    api.add_provider(
        "codex",
        codex_factory,
        capabilities=caps,
        available=codex_available,
    )
    captured: dict[str, Any] = {}
    candidate_graph = object()
    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or candidate_graph,
    )

    await ctx.switch_model("codex:x")

    general_purpose = next(
        spec for spec in captured["subagents"] if spec["name"] == "general-purpose"
    )
    summarizer = next(
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, SummarizationMiddleware)
    )
    assert ctx.cfg.model == "codex:x"
    assert ctx.cfg.subagent_model is None
    assert ctx.cfg.summarizer_model is None
    assert ctx.agent is candidate_graph
    assert isinstance(captured["model"], BaseChatModel)
    assert isinstance(general_purpose["model"], BaseChatModel)
    assert isinstance(summarizer.model, BaseChatModel)
    assert isinstance(ctx.summarizer, BaseChatModel)
    assert captured["model"] is created[0]
    assert general_purpose["model"] is created[1]
    assert summarizer.model is created[2]
    assert ctx.summarizer is created[3]
    assert codex_availability_calls == [None, None, None, None]
    assert codex_factory_calls == ["x", "x", "x", "x"]
    assert anthropic_availability_calls == []
    assert anthropic_factory_calls == []


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
    ctx.ui = SimpleNamespace(
        prepare_session_switch=lambda: scratch_state.update(
            {"draft": "captured by runtime"}
        )
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
        (saved_cwd, "saved:model", "saved.0", {"write_file"})
    ]
    assert ctx.session_id == "saved"
    assert ctx.thread_id == "saved.0"
    assert ctx.cfg.cwd == saved_cwd
    assert ctx.cfg.model == "saved:model"
    assert ctx.agent is replacement_graph
    assert ctx.plugin_states["approval"] is approval_state
    assert approval_state == {"always_allowed": ["write_file"]}
    assert ctx.plugin_states["scratch"] is scratch_state
    assert scratch_state == {}
    assert session.plugin_states["current"] == {
        "approval": {"always_allowed": ["execute"]},
        "scratch": {"draft": "captured by runtime"},
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
            {"draft": "captured by runtime"},
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
    composer_state = {"draft": "before resume"}
    old_graph = object()
    ctx = _context(
        tmp_path,
        agent=old_graph,
        session=session,
        plugin_states={
            "approval": approval_state,
            "composer": composer_state,
        },
    )
    ctx.ui = SimpleNamespace(
        prepare_session_switch=lambda: composer_state.update(
            {"draft": "captured before failure"}
        )
    )
    old_cfg = ctx.cfg

    async def fail_build(*_args: Any, **_kwargs: Any) -> object:
        raise RuntimeError("resume build failed")

    monkeypatch.setattr(app_module, "build_agent", fail_build)

    with pytest.raises(RuntimeError, match="resume build failed"):
        await ctx.resume("saved")

    assert ctx.session_id == "current"
    assert ctx.thread_id == "current.0"
    assert ctx.cfg is old_cfg
    assert ctx.agent is old_graph
    assert approval_state == {"always_allowed": ["execute"]}
    assert composer_state == {"draft": "captured before failure"}


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
async def test_clear_keeps_session_and_plugin_state_while_seeding_empty_thread(
    tmp_path: Path,
) -> None:
    graph = _HistoryGraph([HumanMessage(content="stale graph state")])
    approval_state = {"always_allowed": ["execute"]}
    ctx = _context(
        tmp_path,
        agent=graph,
        plugin_states={"approval": approval_state},
    )
    ledger = Ledger(ctx.session)
    ledger.append(
        ctx.session_id,
        MessageEntry(message=message_to_dict(HumanMessage(content="prior"))),
    )
    old_session = ctx.session_id
    old_thread = ctx.thread_id
    runtime_clears = 0

    async def clear_scrollback() -> None:
        nonlocal runtime_clears
        runtime_clears += 1

    ctx.ui = SimpleNamespace(clear=clear_scrollback)

    await ctx.clear()

    assert ctx.session_id == old_session
    assert ctx.thread_id != old_thread
    assert ctx.plugin_states["approval"] is approval_state
    assert approval_state == {"always_allowed": ["execute"]}
    assert isinstance(ledger.path(ctx.session_id)[-1], ResetBoundaryEntry)
    assert graph.seed_calls[-1] == (
        ctx.thread_config,
        {"messages": [], "todos": [], "files": {}},
    )
    assert runtime_clears == 1
    assert ctx.console.clear_calls == 0
    thread_switches = [
        event for event in ctx.bus.events if isinstance(event, ThreadSwitch)
    ]
    assert thread_switches == [
        ThreadSwitch(
            session_id=old_session,
            old=old_thread,
            new=ctx.thread_id,
            reason="clear",
        )
    ]
    assert not any(isinstance(event, SessionSwitch) for event in ctx.bus.events)


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
        session_id="current",
        thread_id="current.0",
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
    monkeypatch.setattr(app_module, "ApplicationRuntime", _fail_if_prompt_is_constructed)

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
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
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
    monkeypatch.setattr(app_module, "ApplicationRuntime", _fail_if_prompt_is_constructed)

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
    monkeypatch.setattr(app_module, "ApplicationRuntime", _fail_if_prompt_is_constructed)

    status = await run_app(cfg)

    assert status == 1
    assert len(console.errors) == 1
    assert "cannot open session database" in console.errors[0].lower()
    assert str(database_path) in console.errors[0]


def _real_context(
    tmp_path: Path,
    store: SessionStore,
    graph: Any,
    *,
    session_id: str = "runtime-session",
    plugin_states: dict[str, dict[str, Any]] | None = None,
) -> AppContext:
    info = store.get(session_id)
    if info is None:
        info = store.create(
            tmp_path,
            "old:model",
            mode="ask",
            thread_id=session_id,
        )
    assert info.current_thread is not None
    return AppContext(
        cfg=_config(tmp_path),
        registry=Registry(),
        bus=_RecordingBus(),
        session=store,
        plugins=[],
        plugin_states=plugin_states or {},
        console=_RecordingConsole(),
        session_id=info.thread_id,
        thread_id=info.current_thread,
        agent=graph,
    )


def _thread_switches(ctx: AppContext) -> list[ThreadSwitch]:
    return [event for event in ctx.bus.events if isinstance(event, ThreadSwitch)]


def _session_switches(ctx: AppContext) -> list[SessionSwitch]:
    return [event for event in ctx.bus.events if isinstance(event, SessionSwitch)]


def test_app_context_keys_persistent_state_by_session_and_graph_state_by_thread(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "identity.sqlite") as store:
        state = {"enabled": True}
        ctx = _real_context(
            tmp_path,
            store,
            _StreamGraph([]),
            session_id="identity",
            plugin_states={"example": state},
        )

        ctx.persist_plugin_states()

        assert ctx.session_id == "identity"
        assert ctx.thread_id == "identity.0"
        assert ctx.thread_config == {
            "configurable": {"thread_id": "identity.0"}
        }
        assert store.get_plugin_state("identity", "example") == state
        assert store.get_plugin_state("identity.0", "example") == {}


def test_app_context_requires_explicit_session_id(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "required-session-id.sqlite") as store:
        created = store.create(tmp_path, "old:model", thread_id="identity")

        with pytest.raises(TypeError, match="session_id"):
            AppContext(
                cfg=_config(tmp_path),
                registry=Registry(),
                bus=_RecordingBus(),
                session=store,
                plugins=[],
                plugin_states={},
                console=_RecordingConsole(),
                thread_id=created.current_thread,
            )


@pytest.mark.parametrize("outcome", ["success", "exception", "cancel"])
@pytest.mark.asyncio
async def test_turn_capture_runs_in_finally_and_snapshots_non_message_state(
    tmp_path: Path,
    outcome: str,
) -> None:
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="answer"),
    ]
    error: BaseException | None = None
    if outcome == "exception":
        error = RuntimeError("stream failed")
    elif outcome == "cancel":
        error = asyncio.CancelledError()
    graph = _StreamGraph(
        [],
        error=error,
        values={
            "messages": messages,
            "todos": [{"content": "ship tests", "status": "pending"}],
            "files": {"notes.txt": "draft"},
        },
    )
    with SessionStore(tmp_path / f"capture-{outcome}.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)

        await _run_turn(ctx, "question")

        path = Ledger(store).path(ctx.session_id)
        captured_messages = [entry for entry in path if isinstance(entry, MessageEntry)]
        turn_state = next(
            entry
            for entry in path
            if isinstance(entry, CustomEntry) and entry.custom_type == "turn_state"
        )
        assert build_context(path).messages == messages
        assert len(captured_messages) == 2
        assert turn_state.data == {
            "todos": [{"content": "ship tests", "status": "pending"}],
            "files": {"notes.txt": "draft"},
        }
        assert store.get_thread(ctx.thread_id).captured == 2


@pytest.mark.asyncio
async def test_capture_counter_advances_only_after_ledger_append_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        HumanMessage(content="retry me", id="retry-user"),
        AIMessage(content="done", id="retry-assistant"),
    ]
    graph = _StreamGraph([], values={"messages": messages})
    with SessionStore(tmp_path / "capture-retry.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)
        original_capture = Ledger.capture
        failed = False

        def fail_once(
            ledger: Ledger,
            session_id: str,
            thread_id: str,
            entries: Any,
            *,
            captured_message_ids: tuple[str, ...],
            captured: int,
        ) -> Any:
            nonlocal failed
            assert captured_message_ids == ("retry-user", "retry-assistant")
            assert captured == 2
            if not failed:
                failed = True
                raise RuntimeError("ledger unavailable")
            return original_capture(
                ledger,
                session_id,
                thread_id,
                entries,
                captured_message_ids=captured_message_ids,
                captured=captured,
            )

        monkeypatch.setattr(Ledger, "capture", fail_once)

        with pytest.raises(RuntimeError, match="ledger unavailable"):
            await _run_turn(ctx, "retry me")

        assert store.get_thread(ctx.thread_id).captured == 0
        assert Ledger(store).path(ctx.session_id) == []

        await _run_turn(ctx, "retry me")

        assert store.get_thread(ctx.thread_id).captured == 2
        path = Ledger(store).path(ctx.session_id)
        assert len([entry for entry in path if isinstance(entry, MessageEntry)]) == 2
        assert len(
            [
                entry
                for entry in path
                if isinstance(entry, CustomEntry)
                and entry.custom_type == "turn_state"
            ]
        ) == 1


def test_capture_recognizes_summarization_after_message_state_shrinks(
    tmp_path: Path,
) -> None:
    graph = _HistoryGraph(
        [
            HumanMessage(content="before", id="before-user"),
            AIMessage(content="before reply", id="before-assistant"),
        ]
    )
    with SessionStore(tmp_path / "capture-summary-shrink.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)
        ctx.capture_turn()
        summary = HumanMessage(
            content=(
                "Here is a summary of the conversation to date:\n\n"
                "Condensed history."
            ),
            id="summary-message",
            additional_kwargs={"lc_source": "summarization"},
        )
        post_summary = [
            HumanMessage(content="after summary", id="after-user"),
            AIMessage(content="after summary reply", id="after-assistant"),
        ]
        graph.update_state(
            ctx.thread_config,
            {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    summary,
                    *post_summary,
                ]
            },
            as_node="summarization",
        )

        ctx.capture_turn()

        path = Ledger(store).path(ctx.session_id)
        compactions = [entry for entry in path if isinstance(entry, CompactionEntry)]
        captured_messages = [
            messages_from_dict([entry.message])[0]
            for entry in path
            if isinstance(entry, MessageEntry)
        ]
        assert len(compactions) == 1
        assert compactions[0].summary == "Condensed history."
        assert [message.id for message in captured_messages] == [
            "before-user",
            "before-assistant",
            "after-user",
            "after-assistant",
        ]
        thread = store.get_thread(ctx.thread_id)
        assert thread.captured == 3
        assert thread.captured_message_ids == (
            "summary-message",
            "after-user",
            "after-assistant",
        )
        captured_entry_ids = [
            entry.id
            for entry in path
            if isinstance(entry, (CompactionEntry, MessageEntry))
        ]

        ctx.capture_turn()

        assert [
            entry.id
            for entry in Ledger(store).path(ctx.session_id)
            if isinstance(entry, (CompactionEntry, MessageEntry))
        ] == captured_entry_ids


@pytest.mark.asyncio
async def test_branch_seeds_context_and_emits_only_branch_thread_switch(
    tmp_path: Path,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "branch.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="branching")
        ledger = Ledger(store)
        root = ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="root"))),
        )
        ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="old reply"))),
        )
        old_thread = ctx.thread_id

        await ctx.branch(root.id)

        assert ledger.leaf(ctx.session_id) == root.id
        assert ctx.thread_id != old_thread
        assert graph.seed_calls == [
            (
                ctx.thread_config,
                {
                    "messages": [HumanMessage(content="root")],
                    "todos": [],
                    "files": {},
                },
            )
        ]
        thread = store.get_thread(ctx.thread_id)
        assert thread.session_id == ctx.session_id
        assert thread.seeded_from == root.id
        assert thread.captured == 1
        switch = _thread_switches(ctx)
        assert len(switch) == 1
        assert (switch[0].session_id, switch[0].old, switch[0].new, switch[0].reason) == (
            ctx.session_id,
            old_thread,
            ctx.thread_id,
            "branch",
        )
        assert _session_switches(ctx) == []


@pytest.mark.asyncio
async def test_fork_copies_current_path_seeds_it_and_switches_session(
    tmp_path: Path,
) -> None:
    graph = _HistoryGraph([])
    approval = {"always_allowed": ["execute"]}
    navigation = {"recent_paths": ["/one", "/two"]}
    live_states = {"approval": approval, "navigation": navigation}
    with SessionStore(tmp_path / "fork.sqlite") as store:
        ctx = _real_context(
            tmp_path,
            store,
            graph,
            session_id="source",
            plugin_states=live_states,
        )
        store.set_plugin_state(
            ctx.session_id,
            "dormant",
            {"retained_without_live_plugin": True},
        )
        source_ledger = Ledger(store)
        source_ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="copy me"))),
        )
        source_ledger.append(
            ctx.session_id,
            CustomEntry(
                custom_type="turn_state",
                data={"todos": ["one"], "files": {"a.txt": "A"}},
            ),
        )
        old_session = ctx.session_id
        old_thread = ctx.thread_id
        source_path = source_ledger.path(old_session)

        await ctx.fork()

        assert ctx.session_id != old_session
        forked = store.get(ctx.session_id)
        assert forked.parent_session == old_session
        assert ctx.thread_id == f"{ctx.session_id}.0"
        assert forked.current_thread == ctx.thread_id
        assert store.all_plugin_state(ctx.session_id) == {
            "approval": {"always_allowed": ["execute"]},
            "navigation": {"recent_paths": ["/one", "/two"]},
            "dormant": {"retained_without_live_plugin": True},
        }
        assert ctx.plugin_states is live_states
        assert set(ctx.plugin_states) == {"approval", "navigation"}
        assert ctx.plugin_states["approval"] is approval
        assert ctx.plugin_states["navigation"] is navigation
        forked_path = Ledger(store).path(ctx.session_id)
        assert forked_path == source_path
        assert graph.seed_calls[-1] == (
            ctx.thread_config,
            {
                "messages": [HumanMessage(content="copy me")],
                "todos": ["one"],
                "files": {"a.txt": "A"},
            },
        )
        session_switch = _session_switches(ctx)
        assert [(event.old, event.new) for event in session_switch] == [
            (old_session, ctx.session_id)
        ]
        thread_switch = _thread_switches(ctx)
        assert len(thread_switch) == 1
        assert (
            thread_switch[0].session_id,
            thread_switch[0].old,
            thread_switch[0].new,
            thread_switch[0].reason,
        ) == (ctx.session_id, old_thread, ctx.thread_id, "reseed")


@pytest.mark.asyncio
async def test_new_session_clears_live_plugin_state_but_clear_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    approval = {"always_allowed": ["execute"]}
    with SessionStore(tmp_path / "new-session.sqlite") as store:
        ctx = _real_context(
            tmp_path,
            store,
            graph,
            session_id="old-session",
            plugin_states={"approval": approval},
        )
        old_session = ctx.session_id
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )

        await ctx.new_session()

        assert ctx.session_id != old_session
        assert approval == {}
        assert store.get_plugin_state(old_session, "approval") == {
            "always_allowed": ["execute"]
        }
        assert store.get_plugin_state(ctx.session_id, "approval") == {}
        assert [(event.old, event.new) for event in _session_switches(ctx)] == [
            (old_session, ctx.session_id)
        ]
        assert _thread_switches(ctx) == []
        assert graph.seed_calls == []


@pytest.mark.asyncio
async def test_resume_without_checkpoint_reseeds_ledger_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "resume-reseed.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="target")
        Ledger(store).append(
            target.thread_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="saved"))),
        )
        old_thread = ctx.thread_id
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )

        await ctx.resume("target")

        assert ctx.session_id == "target"
        assert ctx.thread_id == target.current_thread
        assert graph.seed_calls[-1] == (
            ctx.thread_config,
            {
                "messages": [HumanMessage(content="saved")],
                "todos": [],
                "files": {},
            },
        )
        assert [(event.old, event.new) for event in _session_switches(ctx)] == [
            ("current", "target")
        ]
        thread_switch = _thread_switches(ctx)
        assert len(thread_switch) == 1
        assert (
            thread_switch[0].session_id,
            thread_switch[0].old,
            thread_switch[0].new,
            thread_switch[0].reason,
        ) == ("target", old_thread, target.current_thread, "reseed")


@pytest.mark.asyncio
async def test_plain_resume_reuses_a_live_checkpoint_without_seeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "resume-live.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="target")
        store.saver.put(
            {
                "configurable": {
                    "thread_id": target.current_thread,
                    "checkpoint_ns": "",
                }
            },
            empty_checkpoint(),
            {},
            {},
        )
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )

        await ctx.resume("target")

        assert ctx.session_id == "target"
        assert ctx.thread_id == target.current_thread
        assert graph.seed_calls == []
        assert _thread_switches(ctx) == []
        assert [(event.old, event.new) for event in _session_switches(ctx)] == [
            ("current", "target")
        ]


@pytest.mark.asyncio
async def test_model_switch_appends_model_change_audit_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "model-audit.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )
        monkeypatch.setattr(
            AppContext, "_resolve_summarizer", lambda _self, _cfg: object()
        )

        await ctx.switch_model(["new:primary", "new:fallback"])

        audit = Ledger(store).path(ctx.session_id)[-1]
        assert isinstance(audit, ModelChangeEntry)
        assert audit.model == ["new:primary", "new:fallback"]
        assert store.get(ctx.session_id).model == ["new:primary", "new:fallback"]


@pytest.mark.asyncio
async def test_mode_switch_appends_mode_change_audit_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "mode-audit.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)
        ctx._registry.modes["plan"] = ModeSpec(
            description="Plan", interrupt_on={}, allowed_tools=None
        )
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )
        monkeypatch.setattr(
            AppContext, "_resolve_summarizer", lambda _self, _cfg: object()
        )

        await ctx.switch_mode("plan")

        audit = Ledger(store).path(ctx.session_id)[-1]
        assert isinstance(audit, ModeChangeEntry)
        assert audit.mode == "plan"
        assert store.get(ctx.session_id).mode == "plan"


def test_normal_exit_diagnostic_requires_an_assistant_message(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "normal-exit.sqlite") as store:
        ctx = _real_context(tmp_path, store, _StreamGraph([]))
        ledger = Ledger(store)

        ctx.record_exit("normal")
        assert ledger.path(ctx.session_id) == []

        ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="hello"))),
        )
        ctx.record_exit("normal")
        assert not any(
            isinstance(entry, CustomEntry) and entry.custom_type == "session_exit"
            for entry in ledger.path(ctx.session_id)
        )

        ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="goodbye"))),
        )
        ctx.record_exit("normal")

        exit_entry = ledger.path(ctx.session_id)[-1]
        assert isinstance(exit_entry, CustomEntry)
        assert exit_entry.custom_type == "session_exit"
        assert exit_entry.data == {"kind": "normal", "pending_tool_calls": []}


@pytest.mark.asyncio
async def test_run_app_app_exit_records_normal_session_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _StreamGraph(
        [],
        values={
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="goodbye"),
            ]
        },
    )
    console = _RecordingConsole()
    prompt = _PromptScript("hello")
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
    monkeypatch.setattr(
        app_module,
        "build_agent",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
    )
    monkeypatch.setattr(app_module, "load_plugins", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(_config(tmp_path))

    assert status == 0
    with SessionStore(tmp_path / "sessions.sqlite") as store:
        session = store.list()[0]
        exit_entry = Ledger(store).path(session.thread_id)[-1]
        assert isinstance(exit_entry, CustomEntry)
        assert exit_entry.custom_type == "session_exit"
        assert exit_entry.data == {"kind": "normal", "pending_tool_calls": []}


@pytest.mark.asyncio
async def test_cancelled_turn_records_signal_exit_with_pending_tool_calls(
    tmp_path: Path,
) -> None:
    messages = [
        HumanMessage(content="write it"),
        AIMessage(
            content="calling tool",
            tool_calls=[{"id": "call-1", "name": "write_file", "args": {}}],
        ),
    ]
    graph = _StreamGraph(
        [], error=asyncio.CancelledError(), values={"messages": messages}
    )
    with SessionStore(tmp_path / "signal-exit.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph)

        await _run_turn(ctx, "write it")

        path = Ledger(store).path(ctx.session_id)
        exit_entry = path[-1]
        assert isinstance(path[-2], CustomEntry)
        assert path[-2].custom_type == "turn_state"
        assert isinstance(exit_entry, CustomEntry)
        assert exit_entry.custom_type == "session_exit"
        assert exit_entry.data == {
            "kind": "signal",
            "pending_tool_calls": [{"id": "call-1", "name": "write_file"}],
        }


@pytest.mark.asyncio
async def test_resume_warns_exactly_once_for_interrupted_final_assistant_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    with SessionStore(tmp_path / "interrupted-resume.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="interrupted")
        ledger = Ledger(store)
        ledger.append(
            target.thread_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="run both"))),
        )
        ledger.append(
            target.thread_id,
            MessageEntry(
                message=message_to_dict(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "one", "name": "read_file", "args": {}},
                            {"id": "two", "name": "execute", "args": {}},
                        ],
                    )
                )
            ),
        )
        monkeypatch.setattr(
            app_module,
            "build_agent",
            lambda *_args, **_kwargs: asyncio.sleep(0, result=graph),
        )

        await ctx.resume("interrupted")

        assert ctx.console.warnings == [
            "Previous turn was interrupted; 2 pending tool call(s) dropped."
        ]


@pytest.mark.asyncio
async def test_run_app_resume_accepts_unique_session_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    with SessionStore(cfg.db_path) as store:
        selected = store.create(tmp_path, "old:model", thread_id="alpha-1234")
        store.create(tmp_path, "old:model", thread_id="beta-1234")
        store.saver.put(
            {
                "configurable": {
                    "thread_id": selected.current_thread,
                    "checkpoint_ns": "",
                }
            },
            empty_checkpoint(),
            {},
            {},
        )
    console = _RecordingConsole()
    prompt = _PromptScript()
    started: list[tuple[str, str]] = []

    def load_runtime(
        _registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        async def record_start(event: AppStart) -> None:
            started.append((event.ctx.session_id, event.ctx.thread_id))

        bus.on(AppStart, record_start)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(replace(cfg, resume="alpha"))

    assert status == 0
    assert console.errors == []
    assert started == [("alpha-1234", selected.current_thread)]


@pytest.mark.asyncio
async def test_run_app_resume_rejects_ambiguous_session_prefix_without_prompting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    with SessionStore(cfg.db_path) as store:
        store.create(tmp_path, "old:model", thread_id="team-b")
        store.create(tmp_path, "old:model", thread_id="team-a")
    console = _RecordingConsole()
    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", _fail_if_prompt_is_constructed)

    status = await run_app(replace(cfg, resume="team-"))

    assert status == 1
    assert console.errors == [
        "Ambiguous session prefix team-: team-a, team-b"
    ]


class _FailingSeedGraph(_HistoryGraph):
    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str,
    ) -> None:
        assert as_node == "model"
        raise RuntimeError("seed failed")


@pytest.mark.asyncio
async def test_branch_seed_failure_restores_prior_leaf_and_thread_identity(
    tmp_path: Path,
) -> None:
    graph = _FailingSeedGraph([])
    with SessionStore(tmp_path / "branch-seed-failure.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="branch-source")
        ledger = Ledger(store)
        root = ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="root"))),
        )
        prior_leaf = ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="reply"))),
        ).id
        prior_thread = ctx.thread_id

        with pytest.raises(RuntimeError, match="seed failed"):
            await ctx.branch(root.id)

        assert ctx.session_id == "branch-source"
        assert ctx.thread_id == prior_thread
        assert ledger.leaf(ctx.session_id) == prior_leaf
        assert ledger.path(ctx.session_id)[-1].id == prior_leaf
        assert store.get(ctx.session_id).current_thread == prior_thread
        assert _thread_switches(ctx) == []
        assert _session_switches(ctx) == []


@pytest.mark.parametrize(
    ("operation", "entry_type"),
    [("clear", ResetBoundaryEntry), ("compact", CompactionEntry)],
)
@pytest.mark.asyncio
async def test_reset_seed_failure_restores_position_and_deletes_new_entry(
    tmp_path: Path,
    operation: str,
    entry_type: type[ResetBoundaryEntry] | type[CompactionEntry],
) -> None:
    graph = _FailingSeedGraph([])
    with SessionStore(tmp_path / f"{operation}-seed-failure.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id=f"{operation}-source")
        ledger = Ledger(store)
        prior_leaf = ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="prior"))),
        ).id
        prior_thread = ctx.thread_id

        if operation == "compact":
            class Summarizer:
                async def ainvoke(self, _messages: list[Any]) -> AIMessage:
                    return AIMessage(content="summary")

            ctx.summarizer = Summarizer()

        with pytest.raises(RuntimeError, match="seed failed"):
            if operation == "clear":
                await ctx.clear()
            else:
                await ctx.compact()

        assert ctx.thread_id == prior_thread
        assert ledger.leaf(ctx.session_id) == prior_leaf
        assert ledger.path(ctx.session_id)[-1].id == prior_leaf
        assert not any(
            isinstance(entry, entry_type)
            for entry in ledger.all(ctx.session_id)
        )
        assert store.get(ctx.session_id).current_thread == prior_thread
        assert _thread_switches(ctx) == []
        assert _session_switches(ctx) == []


@pytest.mark.asyncio
async def test_model_switch_rollback_deletes_unreferenced_model_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    candidate = _HistoryGraph([])
    with SessionStore(tmp_path / "model-change-rollback.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="model-source")
        ctx._registry.providers["old"] = SimpleNamespace(
            foreign_block_types=frozenset({"reasoning"})
        )
        ledger = Ledger(store)
        prior_leaf = ledger.append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="prior"))),
        ).id

        async def build_candidate(*_args: Any, **_kwargs: Any) -> _HistoryGraph:
            return candidate

        def fail_history_cleanup(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("history cleanup failed")

        monkeypatch.setattr(app_module, "build_agent", build_candidate)
        monkeypatch.setattr(
            AppContext, "_resolve_summarizer", lambda _self, _cfg: object()
        )
        monkeypatch.setattr(app_module, "strip_foreign_blocks", fail_history_cleanup)

        with pytest.raises(RuntimeError, match="history cleanup failed"):
            await ctx.switch_model("new:model")

        assert ledger.leaf(ctx.session_id) == prior_leaf
        assert not any(
            isinstance(entry, ModelChangeEntry)
            for entry in ledger.all(ctx.session_id)
        )


@pytest.mark.asyncio
async def test_fork_seed_failure_restores_source_and_removes_failed_child_session(
    tmp_path: Path,
) -> None:
    graph = _FailingSeedGraph([])
    with SessionStore(tmp_path / "fork-seed-failure.sqlite") as store:
        ctx = _real_context(tmp_path, store, graph, session_id="fork-source")
        Ledger(store).append(
            ctx.session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="copy"))),
        )
        prior_thread = ctx.thread_id
        prior_sessions = [session.thread_id for session in store.list()]

        with pytest.raises(RuntimeError, match="seed failed"):
            await ctx.fork()

        assert ctx.session_id == "fork-source"
        assert ctx.thread_id == prior_thread
        assert [session.thread_id for session in store.list()] == prior_sessions
        assert store.get("fork-source").current_thread == prior_thread
        assert _thread_switches(ctx) == []
        assert _session_switches(ctx) == []


@pytest.mark.asyncio
async def test_new_session_clears_console_after_success(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "new-session-console.sqlite") as store:
        ctx = _real_context(tmp_path, store, None, session_id="old-session")
        old_session = ctx.session_id
        runtime_clears = 0

        async def clear_scrollback() -> None:
            nonlocal runtime_clears
            runtime_clears += 1

        ctx.ui = SimpleNamespace(clear=clear_scrollback)

        await ctx.new_session()

        assert ctx.session_id != old_session
        assert runtime_clears == 1
        assert ctx.console.clear_calls == 0
        assert [(event.old, event.new) for event in _session_switches(ctx)] == [
            (old_session, ctx.session_id)
        ]


@pytest.mark.asyncio
async def test_new_session_build_failure_removes_fresh_row_and_does_not_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    approval = {"always_allowed": ["execute"]}
    with SessionStore(tmp_path / "new-session-failure.sqlite") as store:
        ctx = _real_context(
            tmp_path,
            store,
            graph,
            session_id="old-session",
            plugin_states={"approval": approval},
        )
        prior_thread = ctx.thread_id
        prior_sessions = [session.thread_id for session in store.list()]

        async def fail_build(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("build failed")

        monkeypatch.setattr(app_module, "build_agent", fail_build)

        with pytest.raises(RuntimeError, match="build failed"):
            await ctx.new_session()

        assert ctx.session_id == "old-session"
        assert ctx.thread_id == prior_thread
        assert ctx.agent is graph
        assert approval == {"always_allowed": ["execute"]}
        assert [session.thread_id for session in store.list()] == prior_sessions
        assert _session_switches(ctx) == []
        assert _thread_switches(ctx) == []
        assert ctx.console.clear_calls == 0


@pytest.mark.asyncio
async def test_model_audit_failure_restores_pre_strip_history_and_old_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_history = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "provider-private"},
            {"type": "text", "text": "visible"},
        ],
        additional_kwargs={"keep": "additional"},
        response_metadata={"keep": "metadata"},
    )
    old_graph = _HistoryGraph([private_history])
    candidate_graph = _HistoryGraph([])
    with SessionStore(tmp_path / "model-audit-failure.sqlite") as store:
        ctx = _real_context(tmp_path, store, old_graph)
        ctx._registry.providers["old"] = SimpleNamespace(
            foreign_block_types=frozenset({"reasoning"})
        )
        old_cfg = ctx.cfg
        original_append = Ledger.append

        async def build_candidate(*_args: Any, **_kwargs: Any) -> _HistoryGraph:
            return candidate_graph

        def fail_model_audit(
            ledger: Ledger,
            session_id: str,
            entry: Any,
        ) -> Any:
            if isinstance(entry, ModelChangeEntry):
                raise RuntimeError("audit failed")
            return original_append(ledger, session_id, entry)

        monkeypatch.setattr(app_module, "build_agent", build_candidate)
        monkeypatch.setattr(
            AppContext, "_resolve_summarizer", lambda _self, _cfg: object()
        )
        monkeypatch.setattr(Ledger, "append", fail_model_audit)

        with pytest.raises(RuntimeError, match="audit failed"):
            await ctx.switch_model("new:model")

        assert ctx.cfg is old_cfg
        assert ctx.cfg.model == "old:model"
        assert ctx.agent is old_graph
        assert old_graph.messages == [private_history]
        assert store.get(ctx.session_id).model == "old:model"
        assert Ledger(store).path(ctx.session_id) == []
        assert not any(isinstance(event, ModelSwitch) for event in ctx.bus.events)


@pytest.mark.asyncio
async def test_capture_storage_error_names_session_and_thread_and_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        HumanMessage(content="question", id="storage-user"),
        AIMessage(content="answer", id="storage-assistant"),
    ]
    graph = _StreamGraph([], values={"messages": messages})
    with SessionStore(tmp_path / "capture-context-error.sqlite") as store:
        ctx = _real_context(
            tmp_path, store, graph, session_id="capture-context-session"
        )
        original_capture = Ledger.capture

        def fail_capture(
            _ledger: Ledger,
            _session_id: str,
            _thread_id: str,
            _entries: Any,
            *,
            captured_message_ids: tuple[str, ...],
            captured: int,
        ) -> Any:
            assert captured_message_ids == ("storage-user", "storage-assistant")
            assert captured == 2
            raise RuntimeError("storage offline")

        monkeypatch.setattr(Ledger, "capture", fail_capture)

        with pytest.raises(RuntimeError) as raised:
            await _run_turn(ctx, "question")

        error_text = str(raised.value)
        assert ctx.session_id in error_text
        assert ctx.thread_id in error_text
        assert "storage offline" in error_text
        assert store.get_thread(ctx.thread_id).captured == 0
        assert Ledger(store).path(ctx.session_id) == []

        monkeypatch.setattr(Ledger, "capture", original_capture)
        await _run_turn(ctx, "question")

        assert store.get_thread(ctx.thread_id).captured == 2
        path = Ledger(store).path(ctx.session_id)
        assert len([entry for entry in path if isinstance(entry, MessageEntry)]) == 2
        assert any(
            isinstance(entry, CustomEntry) and entry.custom_type == "turn_state"
            for entry in path
        )


def _put_checkpoint(
    store: SessionStore,
    thread_id: str,
    *,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    checkpoint = empty_checkpoint()
    if messages is not None:
        checkpoint["channel_values"] = {"messages": messages}
    return store.saver.put(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        checkpoint,
        {},
        {},
    )


@pytest.mark.asyncio
async def test_cross_provider_missing_checkpoint_reseed_strips_only_source_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_private = {"type": "source_private", "value": "remove"}
    target_native = {"type": "target_native", "value": "preserve"}
    visible = {"type": "text", "text": "visible"}
    stored_message = AIMessage(
        content=[source_private, target_native, visible],
        additional_kwargs={"keep": "additional"},
        response_metadata={"keep": "metadata"},
    )
    candidate = _HistoryGraph([])
    with SessionStore(tmp_path / "cross-provider-reseed.sqlite") as store:
        ctx = _real_context(tmp_path, store, _HistoryGraph([]), session_id="current")
        target = store.create(tmp_path, "source:stored", thread_id="target")
        Ledger(store).append(
            target.thread_id,
            MessageEntry(message=message_to_dict(stored_message)),
        )
        ctx.cfg = replace(ctx.cfg, model="target:cli", model_overridden=True)
        ctx._registry.providers["source"] = SimpleNamespace(
            foreign_block_types=frozenset({"source_private"})
        )
        ctx._registry.providers["target"] = SimpleNamespace(
            foreign_block_types=frozenset({"target_native"})
        )

        async def build_candidate(*_args: Any, **_kwargs: Any) -> _HistoryGraph:
            return candidate

        monkeypatch.setattr(app_module, "build_agent", build_candidate)
        monkeypatch.setattr(
            AppContext, "_resolve_summarizer", lambda _self, _cfg: object()
        )

        await ctx.resume("target")

        expected = stored_message.model_copy(
            update={"content": [target_native, visible]}
        )
        assert candidate.seed_calls == [
            (
                ctx.thread_config,
                {"messages": [expected], "todos": [], "files": {}},
            )
        ]
        assert expected.additional_kwargs == {"keep": "additional"}
        assert expected.response_metadata == {"keep": "metadata"}


@pytest.mark.asyncio
async def test_lazy_resume_reseed_exposes_target_thread_before_agent_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _HistoryGraph([])
    build_calls: list[str] = []
    with SessionStore(tmp_path / "lazy-resume-reseed.sqlite") as store:
        ctx = _real_context(tmp_path, store, None, session_id="current")
        source_thread = ctx.thread_id
        store.create(tmp_path, "old:model", thread_id="target")

        async def build_target(*_args: Any, **_kwargs: Any) -> _HistoryGraph:
            build_calls.append(ctx.thread_id)
            return graph

        monkeypatch.setattr(app_module, "build_agent", build_target)

        await ctx.resume("target")

        pending_thread = ctx.thread_id
        assert ctx.session_id == "target"
        assert pending_thread != source_thread
        assert pending_thread.startswith("target.")
        assert build_calls == []
        assert graph.seed_calls == []
        assert _thread_switches(ctx) == []

        assert await ctx.ensure_agent() is True

        assert build_calls == [pending_thread]
        assert graph.seed_calls[-1][0] == ctx.thread_config
        switch = _thread_switches(ctx)
        assert len(switch) == 1
        assert switch[0].session_id == "target"
        assert switch[0].new == ctx.thread_id
        assert switch[0].reason == "reseed"


@pytest.mark.asyncio
async def test_slash_resume_recovers_uncaptured_checkpoint_messages_before_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    human = HumanMessage(content="run tool", id="slash-human")
    assistant = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
        id="slash-assistant",
    )
    checkpoint_graph = _HistoryGraph([human, assistant])
    with SessionStore(tmp_path / "slash-resume-recovery.sqlite") as store:
        ctx = _real_context(tmp_path, store, _HistoryGraph([]), session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="target")
        Ledger(store).capture(
            target.thread_id,
            target.current_thread,
            [MessageEntry(message=message_to_dict(human))],
            captured_message_ids=(human.id,),
            captured=1,
        )
        _put_checkpoint(
            store, target.current_thread, messages=[human, assistant]
        )

        async def build_checkpoint_graph(
            *_args: Any, **_kwargs: Any
        ) -> _HistoryGraph:
            return checkpoint_graph

        monkeypatch.setattr(app_module, "build_agent", build_checkpoint_graph)

        await ctx.resume("target")

        thread = store.get_thread(target.current_thread)
        path = Ledger(store).path(target.thread_id)
        assert thread.captured == 2
        assert [
            type(entry)
            for entry in path
        ] == [MessageEntry, MessageEntry, CustomEntry]
        assert path[-1].custom_type == "turn_state"
        assert ctx.console.warnings == [
            "Previous turn was interrupted; 1 pending tool call(s) dropped."
        ]


@pytest.mark.asyncio
async def test_startup_resume_recovers_checkpoint_before_warning_and_normal_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    human = HumanMessage(content="run tool", id="startup-human")
    assistant = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
        id="startup-assistant",
    )
    checkpoint_graph = _HistoryGraph([human, assistant])
    with SessionStore(cfg.db_path) as store:
        target = store.create(tmp_path, "old:model", thread_id="startup-target")
        Ledger(store).capture(
            target.thread_id,
            target.current_thread,
            [MessageEntry(message=message_to_dict(human))],
            captured_message_ids=(human.id,),
            captured=1,
        )
        _put_checkpoint(
            store, target.current_thread, messages=[human, assistant]
        )
    console = _RecordingConsole()
    prompt = _PromptScript()

    def load_runtime(
        _registry: Registry,
        bus: EventBus,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[Any]:
        async def attach_checkpoint_graph(event: AppStart) -> None:
            event.ctx.agent = checkpoint_graph

        bus.on(AppStart, attach_checkpoint_graph)
        return []

    monkeypatch.setattr(app_module, "ConsoleOutput", lambda: console)
    monkeypatch.setattr(app_module, "ApplicationRuntime", prompt.runtime)
    monkeypatch.setattr(app_module, "load_plugins", load_runtime)
    monkeypatch.setattr(app_module, "_history_path", lambda: tmp_path / "history")

    status = await run_app(replace(cfg, resume="startup-target"))

    assert status == 0
    assert console.warnings == [
        "Previous turn was interrupted; 1 pending tool call(s) dropped."
    ]
    with SessionStore(cfg.db_path) as store:
        thread = store.get_thread(target.current_thread)
        path = Ledger(store).path(target.thread_id)
        assert thread.captured == 2
        assert isinstance(path[-2], CustomEntry)
        assert path[-2].custom_type == "turn_state"
        assert isinstance(path[-1], CustomEntry)
        assert path[-1].custom_type == "session_exit"
        assert path[-1].data["kind"] == "normal"


@pytest.mark.asyncio
async def test_signal_exit_with_live_checkpoint_forces_sanitized_reseed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    human = HumanMessage(content="run tool", id="signal-human")
    dangling = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
        id="signal-assistant",
    )
    candidate = _HistoryGraph([])
    with SessionStore(tmp_path / "signal-live-reseed.sqlite") as store:
        ctx = _real_context(tmp_path, store, _HistoryGraph([]), session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="target")
        Ledger(store).capture(
            target.thread_id,
            target.current_thread,
            [
                MessageEntry(message=message_to_dict(human)),
                MessageEntry(message=message_to_dict(dangling)),
            ],
            captured_message_ids=(human.id, dangling.id),
            captured=2,
        )
        Ledger(store).append(
            target.thread_id,
            CustomEntry(
                custom_type="session_exit",
                data={
                    "kind": "signal",
                    "pending_tool_calls": [{"id": "call-1", "name": "execute"}],
                },
            ),
        )
        _put_checkpoint(store, target.current_thread, messages=[human, dangling])

        async def build_candidate(*_args: Any, **_kwargs: Any) -> _HistoryGraph:
            return candidate

        monkeypatch.setattr(app_module, "build_agent", build_candidate)

        await ctx.resume("target")

        assert ctx.thread_id != target.current_thread
        assert candidate.seed_calls[-1] == (
            ctx.thread_config,
            {"messages": [human], "todos": [], "files": {}},
        )
        assert _thread_switches(ctx)[-1].reason == "reseed"


@pytest.mark.asyncio
async def test_live_checkpoint_with_pending_approval_remains_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    human = HumanMessage(content="run tool", id="approval-human")
    pending = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
        id="approval-assistant",
    )
    checkpoint_graph = _HistoryGraph([human, pending])
    with SessionStore(tmp_path / "approval-live-resume.sqlite") as store:
        ctx = _real_context(tmp_path, store, _HistoryGraph([]), session_id="current")
        target = store.create(tmp_path, "old:model", thread_id="target")
        Ledger(store).capture(
            target.thread_id,
            target.current_thread,
            [
                MessageEntry(message=message_to_dict(human)),
                MessageEntry(message=message_to_dict(pending)),
            ],
            captured_message_ids=(human.id, pending.id),
            captured=2,
        )
        checkpoint_config = _put_checkpoint(
            store, target.current_thread, messages=[human, pending]
        )
        store.saver.put_writes(
            checkpoint_config,
            [
                (
                    "__interrupt__",
                    {
                        "action_requests": [
                            {"name": "execute", "args": {"command": "true"}}
                        ]
                    },
                )
            ],
            "approval-task",
        )

        async def build_checkpoint_graph(
            *_args: Any, **_kwargs: Any
        ) -> _HistoryGraph:
            return checkpoint_graph

        monkeypatch.setattr(app_module, "build_agent", build_checkpoint_graph)

        await ctx.resume("target")

        assert ctx.thread_id == target.current_thread
        assert checkpoint_graph.seed_calls == []
        assert _thread_switches(ctx) == []
        assert ctx.console.warnings == []


class _InspectingFailingSeedGraph(_HistoryGraph):
    def __init__(self, store: SessionStore, session_id: str) -> None:
        super().__init__([])
        self.store = store
        self.session_id = session_id
        self.observed_current_threads: list[str | None] = []

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str,
    ) -> None:
        assert as_node == "model"
        self.observed_current_threads.append(
            self.store.get(self.session_id).current_thread
        )
        raise RuntimeError("seed inspection failed")


@pytest.mark.parametrize("operation", ["branch", "clear", "compact"])
@pytest.mark.asyncio
async def test_seed_marks_session_reseed_required_during_async_crash_window(
    tmp_path: Path,
    operation: str,
) -> None:
    with SessionStore(tmp_path / f"{operation}-crash-window.sqlite") as store:
        session_id = f"{operation}-source"
        ctx = _real_context(tmp_path, store, None, session_id=session_id)
        graph = _InspectingFailingSeedGraph(store, session_id)
        ctx.agent = graph
        ledger = Ledger(store)
        root = ledger.append(
            session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="root"))),
        )
        prior_leaf = ledger.append(
            session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="reply"))),
        ).id
        prior_thread = ctx.thread_id
        if operation == "compact":
            class Summarizer:
                async def ainvoke(self, _messages: list[Any]) -> AIMessage:
                    return AIMessage(content="summary")

            ctx.summarizer = Summarizer()

        with pytest.raises(RuntimeError, match="seed inspection failed"):
            if operation == "branch":
                await ctx.branch(root.id)
            elif operation == "clear":
                await ctx.clear()
            else:
                await ctx.compact()

        assert graph.observed_current_threads == [None]
        assert ctx.thread_id == prior_thread
        assert store.get(session_id).current_thread == prior_thread
        assert ledger.leaf(session_id) == prior_leaf
        assert _thread_switches(ctx) == []


class _OrphanCheckpointInspectingGraph(_HistoryGraph):
    def __init__(self, store: SessionStore) -> None:
        super().__init__([])
        self.store = store
        self.checkpoint_exists_during_update: list[bool] = []

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str,
    ) -> None:
        assert as_node == "model"
        thread_id = config["configurable"]["thread_id"]
        self.checkpoint_exists_during_update.append(
            self.store.checkpoint_exists(thread_id)
        )
        await super().aupdate_state(config, values, as_node=as_node)


@pytest.mark.asyncio
async def test_restart_reseed_deletes_orphan_checkpoint_before_state_update(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "restart-orphan-reseed.sqlite") as store:
        session_id = "restart-target"
        ctx = _real_context(tmp_path, store, None, session_id=session_id)
        ledger = Ledger(store)
        ledger.append(
            session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="old history"))),
        )
        ledger.append(
            session_id,
            CompactionEntry(summary="rebuilt once", first_kept_id=None),
        )
        pending_thread = store.next_thread_id(session_id)
        store._write(
            "UPDATE sessions SET current_thread = NULL WHERE thread_id = ?",
            (session_id,),
        )
        _put_checkpoint(
            store,
            pending_thread,
            messages=[
                HumanMessage(content="[Conversation summary]\nobsolete checkpoint copy")
            ],
        )
        assert store.get_thread(pending_thread) is None
        assert store.checkpoint_exists(pending_thread) is True
        assert store.get(session_id).current_thread is None
        graph = _OrphanCheckpointInspectingGraph(store)
        ctx.agent = graph
        ctx.thread_id = pending_thread

        assert await ctx.ensure_agent() is True

        assert graph.checkpoint_exists_during_update == [False]
        assert graph.seed_calls == [
            (
                ctx.thread_config,
                {
                    "messages": [
                        HumanMessage(
                            content="[Conversation summary]\nrebuilt once"
                        )
                    ],
                    "todos": [],
                    "files": {},
                },
            )
        ]
        assert store.get(session_id).current_thread == ctx.thread_id
        assert store.get_thread(ctx.thread_id).session_id == session_id


@pytest.mark.parametrize("operation", ["branch", "clear", "compact"])
@pytest.mark.asyncio
async def test_lazy_pending_reseed_failure_keeps_persisted_thread_null(
    tmp_path: Path,
    operation: str,
) -> None:
    with SessionStore(tmp_path / f"{operation}-lazy-pending-failure.sqlite") as store:
        session_id = f"{operation}-target"
        ctx = _real_context(tmp_path, store, None, session_id=session_id)
        ledger = Ledger(store)
        root = ledger.append(
            session_id,
            MessageEntry(message=message_to_dict(HumanMessage(content="root"))),
        )
        prior_leaf = ledger.append(
            session_id,
            MessageEntry(message=message_to_dict(AIMessage(content="reply"))),
        ).id
        pending_thread = store.next_thread_id(session_id)
        store._write(
            "UPDATE sessions SET current_thread = NULL WHERE thread_id = ?",
            (session_id,),
        )
        ctx.thread_id = pending_thread
        ctx.agent = _FailingSeedGraph([])
        if operation == "compact":
            class Summarizer:
                async def ainvoke(self, _messages: list[Any]) -> AIMessage:
                    return AIMessage(content="summary")

            ctx.summarizer = Summarizer()

        with pytest.raises(RuntimeError, match="seed failed"):
            if operation == "branch":
                await ctx.branch(root.id)
            elif operation == "clear":
                await ctx.clear()
            else:
                await ctx.compact()

        assert ctx.thread_id == pending_thread
        assert store.get(session_id).current_thread is None
        assert store.get_thread(pending_thread) is None
        assert ledger.leaf(session_id) == prior_leaf
        assert _thread_switches(ctx) == []


def test_application_runtime_accepts_real_app_context(tmp_path: Path) -> None:
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from orcha_agent.tui.runtime import ApplicationRuntime

    async def submit(_text: str) -> None:
        return None

    ctx = _context(tmp_path)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            input=pipe,
            output=DummyOutput(),
            ctx=ctx,
        )
    assert ctx.queue is runtime.queue
    assert ctx.ui is runtime.ui
