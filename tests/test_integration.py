from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from orcha_agent.builtin import approval_prompt, filesystem, modes
from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import (
    AgentBuildAfter,
    Event,
    EventBus,
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


def _api(name: str, registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name=name,
        registry=registry,
        bus=bus,
        config={},
        state={},
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
