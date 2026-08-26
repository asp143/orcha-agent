from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from orcha_agent.builtin import filesystem, modes
from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus, InterruptRaised
from orcha_agent.core.plugin import PluginAPI, ProviderCaps, Resolved
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


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


def _script(content: str) -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/approved.txt", "content": content},
                        "id": "write-approved",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The file was written."),
        ]
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


async def _build_harness(
    tmp_path: Path,
    *,
    mode: str,
    thread_id: str,
    content: str,
) -> tuple[Any, SessionStore, EventBus, dict[str, dict[str, str]]]:
    registry = Registry()
    bus = EventBus()
    model = ToolCallingFakeModel(messages=_script(content))

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

    cfg = _config(tmp_path, mode)
    session = SessionStore(cfg.db_path)
    created = session.create(cwd=tmp_path, model=cfg.model, thread_id=thread_id)
    assert created.thread_id == thread_id
    graph = await build_agent(registry, cfg, session, bus)
    thread_config = {"configurable": {"thread_id": thread_id}}
    return graph, session, bus, thread_config


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
async def test_plugin_interrupt_handler_can_auto_approve_and_resume(tmp_path: Path) -> None:
    graph, session, bus, thread_config = await _build_harness(
        tmp_path,
        mode="ask",
        thread_id="plugin-thread",
        content="approved by plugin\n",
    )
    observed_payloads: list[dict[str, Any]] = []

    async def approve(event: InterruptRaised) -> Resolved:
        observed_payloads.append(event.payload)
        return Resolved(resume_value={"decisions": [{"type": "approve"}]})

    _api("auto-approve", Registry(), bus).on(InterruptRaised, approve, priority=1)

    try:
        interrupted = graph.invoke(
            {"messages": [{"role": "user", "content": "Write the file."}]},
            config=thread_config,
        )
        payload = _interrupt_payload(interrupted, "approved by plugin\n")
        resolution = await bus.emit(InterruptRaised(payload=payload))

        assert isinstance(resolution, Resolved)
        assert observed_payloads == [payload]

        resumed = graph.invoke(
            Command(resume=resolution.resume_value),
            config=thread_config,
        )

        assert "__interrupt__" not in resumed
        assert (tmp_path / "approved.txt").read_text() == "approved by plugin\n"
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
