import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

import orcha_agent.tui.app as app_module
from orcha_agent.core.config import Config
from orcha_agent.core.events import AppExit, ToolCallEnd, TurnEnd, TurnStart
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionInfo
from orcha_agent.tui.app import (
    AppContext,
    _ModelLabelBuffer,
    _run_turn,
    _stored_model,
    _ToolCallBuffer,
    run_app,
)



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
        def invoke(self, messages: list[Any]) -> AIMessage:
            seen.append(messages)
            return AIMessage(content="compact summary")

    ctx.summarizer = Summarizer()

    await ctx.compact()

    assert seen and seen[0][-1].content.startswith("Summarize the conversation")
    assert [message.content for message in history.messages] == ["compact summary"]


@pytest.mark.asyncio
async def test_requested_rebuild_runs_after_the_current_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _context(tmp_path, agent=_StreamGraph([]))
    rebuilt: list[bool] = []

    async def rebuild(self: AppContext) -> None:
        rebuilt.append(True)
        self.rebuild_requested = False

    monkeypatch.setattr(AppContext, "rebuild", rebuild)
    ctx.rebuild_requested = True

    await _run_turn(ctx, "continue")

    assert rebuilt == [True]

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

    def print(self, *objects: object, **_kwargs: Any) -> None:
        self.output.append(objects)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _SessionDouble:
    def __init__(
        self,
        records: dict[str, SessionInfo] | None = None,
        plugin_states: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.records = records or {}
        self.plugin_states = plugin_states or {}

    def get(self, thread_id: str) -> SessionInfo | None:
        return self.records.get(thread_id)

    def exists(self, thread_id: str) -> bool:
        return thread_id in self.records

    def set_title(self, thread_id: str, title: str) -> None:
        record = self.records[thread_id]
        self.records[thread_id] = SessionInfo(
            record.thread_id,
            record.cwd,
            record.model,
            record.created,
            title,
        )

    def set_plugin_state(self, thread_id: str, plugin: str, state: dict[str, Any]) -> None:
        self.plugin_states.setdefault(thread_id, {})[plugin] = dict(state)

    def all_plugin_state(self, thread_id: str) -> dict[str, dict[str, Any]]:
        return self.plugin_states.get(thread_id, {})


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

    def get_state(self, _config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": self.messages})

    def update_state(self, _config: Any, values: dict[str, list[Any]]) -> None:
        self.messages = list(values["messages"][1:])


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
    agent: Any,
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
    ctx.registry.providers["old"] = SimpleNamespace(
        foreign_block_types=frozenset({"thinking"})
    )
    old_cfg = ctx.cfg

    async def fail_build(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(app_module, "build_agent", fail_build)

    try:
        await ctx.switch_model("new:model")
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"

    assert ctx.cfg == old_cfg
    assert ctx.agent is old_graph
    assert old_graph.messages == [private_history]


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
    ctx = _context(
        tmp_path,
        agent=object(),
        session=session,
        plugin_states={"approval": {"always_allowed": ["execute"]}},
    )
    rebuilt: list[tuple[Path, str | list[str], str]] = []
    replacement_graph = object()

    async def capture_build(
        _registry: Registry,
        cfg: Config,
        _session: Any,
        _bus: Any,
        **_kwargs: Any,
    ) -> Any:
        rebuilt.append((cfg.cwd, cfg.model, ctx.thread_id))
        return replacement_graph

    monkeypatch.setattr(app_module, "build_agent", capture_build)

    await ctx.resume("saved")

    assert rebuilt == [(saved_cwd, "saved:model", "saved")]
    assert ctx.cfg.cwd == saved_cwd
    assert ctx.cfg.model == "saved:model"
    assert ctx.agent is replacement_graph
