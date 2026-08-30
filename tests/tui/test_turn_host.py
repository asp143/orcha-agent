from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from orcha_agent.core.events import ToolCallEnd, ToolCallStart, TurnEnd, TurnStart
from orcha_agent.tui.turn import TurnHost, run_turn


class _Bus:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


class _Console:
    def __init__(self) -> None:
        self.prints = 0
        self.warnings: list[str] = []

    def print(self, *_args: object, **_kwargs: object) -> None:
        self.prints += 1

    def error(self, _message: str) -> None:
        raise AssertionError("unexpected turn error")

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Session:
    def __init__(self) -> None:
        self.title: str | None = None

    def get(self, _session_id: str) -> SimpleNamespace:
        return SimpleNamespace(title=self.title)

    def set_title(self, _session_id: str, title: str) -> None:
        self.title = title


class _Graph:
    def __init__(self, items: list[tuple[Any, ...]] | None = None) -> None:
        self.inputs: list[Any] = []
        self.items = items or []

    async def astream(self, value: Any, **_kwargs: Any):
        self.inputs.append(value)
        for item in self.items:
            yield item


class _CancelledGraph(_Graph):
    async def astream(self, value: Any, **kwargs: Any):
        self.inputs.append(value)
        raise __import__("asyncio").CancelledError
        yield


@dataclass
class _Host:
    agent: Any = field(default_factory=_Graph)
    thread_config: dict[str, dict[str, str]] = field(
        default_factory=lambda: {"configurable": {"thread_id": "worker.0"}}
    )
    bus: Any = field(default_factory=_Bus)
    source_id: str = "worker-a1b2"
    session_id: str = "worker-session"
    session: Any = field(default_factory=_Session)
    console: Any = field(default_factory=_Console)
    captured: int = 0
    exits: list[str] = field(default_factory=list)

    def capture_turn(self) -> None:
        self.captured += 1

    def record_exit(self, kind: str) -> None:
        self.exits.append(kind)


@pytest.mark.asyncio
async def test_run_turn_accepts_a_non_app_turn_host_and_tags_boundary_events() -> None:
    host = _Host()

    await run_turn(host, "inspect the worker")

    assert isinstance(host, TurnHost)
    assert host.agent.inputs == [
        {"messages": [{"role": "user", "content": "inspect the worker"}]}
    ]
    assert host.captured == 1
    assert host.session.title == "inspect the worker"
    assert [type(event) for event in host.bus.events] == [TurnStart, TurnEnd]
    assert [event.source_id for event in host.bus.events] == [host.source_id, host.source_id]
    assert [event.thread_id for event in host.bus.events] == ["worker.0", "worker.0"]


@pytest.mark.asyncio
async def test_run_turn_keeps_nested_tool_events_on_the_same_worker_source() -> None:
    tool = ToolMessage(content="done", tool_call_id="call-1", name="read_file")
    host = _Host(
        agent=_Graph(
            [
                (
                    ("nested:1",),
                    "messages",
                    (
                        AIMessageChunk(
                            content="working",
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
                ),
                (("nested:1",), "updates", {"tools": {"messages": [tool]}}),
            ]
        )
    )

    await run_turn(host, "inspect")

    starts = [event for event in host.bus.events if isinstance(event, ToolCallStart)]
    ends = [event for event in host.bus.events if isinstance(event, ToolCallEnd)]
    assert [event.source_id for event in starts] == ["worker-a1b2/nested:1"]
    assert [event.source_id for event in ends] == ["worker-a1b2/nested:1"]


@pytest.mark.asyncio
async def test_run_turn_records_cancelled_hosts_that_satisfy_the_protocol() -> None:
    host = _Host(agent=_CancelledGraph())
    incomplete = SimpleNamespace(
        agent=host.agent,
        thread_config=host.thread_config,
        bus=host.bus,
        source_id=host.source_id,
        session_id=host.session_id,
        session=host.session,
        console=host.console,
        capture_turn=host.capture_turn,
    )

    assert not isinstance(incomplete, TurnHost)
    await run_turn(host, "stop")

    assert isinstance(host, TurnHost)
    assert host.captured == 1
    assert host.exits == ["signal"]
    assert host.console.warnings == ["interrupted"]
