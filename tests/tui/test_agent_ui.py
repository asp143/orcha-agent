from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from orcha_agent.core.events import (
    AgentSpawned,
    AgentStatus,
    ModelChunk,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.registry import Registry
from orcha_agent.tui.blocks.hud import subagent_hud_data
from orcha_agent.tui.blocks.task import render_delivery, render_task
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.overlays.hub import HubOverlay
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.statusline import agent_counts, subagents_segment
from orcha_agent.tui.title import TerminalTitle
from orcha_agent.tui.transcript import Transcript


THEME = {
    "id": "agent-ui-test",
    "colors": {
        "accent": "cyan",
        "borderMuted": "bright_black",
        "error": "red",
        "muted": "bright_black",
        "success": "green",
        "text": "white",
        "toolOutput": "white",
        "toolTitle": "cyan",
        "warning": "yellow",
    },
    "symbols": {
        "sep.thin": "·",
        "spinner.activity": ("⟳",),
        "status.error": "✘",
        "status.pending": "○",
        "status.success": "✔",
    },
}


@dataclass
class FakeRun:
    id: str
    name: str
    status: str
    parent_id: str = "main"
    depth: int = 0
    description: str = "focused assignment"
    agent_type: Any = field(default_factory=lambda: SimpleNamespace(name="task"))
    model_label: str = "fake:worker"
    session_id: str = "session"
    thread_id: str = "thread"
    current_tool: str | None = None
    current_tool_args: str | None = None
    last_tool: str | None = None
    last_tool_args: str | None = None
    requests: int = 1
    tokens_in: int = 10
    tokens_out: int = 5
    cost: float = 0.01
    result: Any = None
    partial_findings: list[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC) - timedelta(seconds=12))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "name": self.name,
            "agent_type": self.agent_type.name,
            "description": self.description,
            "model_label": self.model_label,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "current_tool": self.current_tool,
            "current_tool_args": self.current_tool_args,
            "last_tool": self.last_tool,
            "last_tool_args": self.last_tool_args,
            "requests": self.requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "result": self.result,
            "partial_findings": self.partial_findings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FakeRegistry:
    def __init__(self, runs: list[FakeRun]) -> None:
        self.runs = runs
        self.cancelled: list[str] = []
        self.revived: list[str] = []
        self.messages: list[tuple[str, str]] = []

    def list(self, status: str | None = None) -> list[FakeRun]:
        if status is None:
            return list(self.runs)
        return [run for run in self.runs if run.status == status]

    def tree(self) -> list[FakeRun]:
        ordered: list[FakeRun] = []

        def append_children(parent: str) -> None:
            for run in self.runs:
                if run.parent_id == parent:
                    ordered.append(run)
                    append_children(run.id)

        append_children("main")
        return ordered

    def get(self, run_id: str) -> FakeRun | None:
        return next((run for run in self.runs if run.id == run_id), None)

    async def cancel(self, run_id: str, reason: str = "cancel") -> FakeRun:
        del reason
        run = self.get(run_id)
        if run is None:
            raise LookupError(run_id)
        self.cancelled.append(run_id)
        run.status = "aborted"
        return run

    async def revive(self, session_id: str) -> FakeRun:
        run = next((item for item in self.runs if item.session_id == session_id), None)
        if run is None:
            raise LookupError(session_id)
        self.revived.append(session_id)
        run.status = "idle"
        return run

    async def send(self, run_id: str, text: str, *, interrupt: bool = False) -> FakeRun:
        del interrupt
        run = self.get(run_id)
        if run is None:
            raise LookupError(run_id)
        self.messages.append((run_id, text))
        return run


class TitleOutput(DummyOutput):
    def __init__(self) -> None:
        super().__init__()
        self.titles: list[str] = []

    def set_title(self, title: str) -> None:
        self.titles.append(title)


def plain(renderable: object, width: int = 120) -> str:
    output = StringIO()
    Console(file=output, width=width, force_terminal=False, color_system=None).print(renderable)
    return output.getvalue()


async def drive_overlay(overlay: HubOverlay, keys: bytes | str) -> Any:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime_task = asyncio.create_task(runtime.run())
        shown = asyncio.create_task(runtime.ui.show(overlay))
        await asyncio.sleep(0.02)
        if isinstance(keys, bytes):
            pipe.send_bytes(keys)
        else:
            pipe.send_text(keys)
        result = await asyncio.wait_for(shown, 1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(runtime_task, 1)
        return result


def task_block(status: str) -> Block:
    agents = []
    for index in range(5):
        finding = f"partial-{index}" if status == "running" else None
        result = None if status == "running" else f"result-{index}"
        agents.append(
            {
                "run_id": f"run-{index}",
                "name": f"Worker-{index}",
                "description": f"assignment {index}",
                "status": status,
                "requests": index + 1,
                "tokens": 100 + index,
                "cost": 0.01 * index,
                "elapsed": 12 + index,
                "last_tool": "read" if index == 4 else None,
                "last_tool_args": "x" * 60 if index == 4 else None,
                "partial_findings": [finding] if finding else [],
                "result": result,
            }
        )
    return Block("task-1", "task", data={"agents": agents, "elapsed": 63})


@pytest.mark.parametrize(
    ("status", "marker", "footer"),
    [
        ("running", "⟳", "0 succeeded · 0 failed"),
        ("done", "✔", "5 succeeded · 0 failed"),
        ("failed", "✘", "0 succeeded · 5 failed"),
    ],
)
@pytest.mark.parametrize("expanded", [False, True], ids=["collapsed", "expanded"])
def test_task_cards_cover_running_done_and_failed_states(
    status: str,
    marker: str,
    footer: str,
    expanded: bool,
) -> None:
    output = plain(render_task(task_block(status), THEME, 120, 100, expanded))

    assert "⇶ Task · 5 agents" in output
    assert f"{marker} Worker-4: assignment 4 ⟦{status}⟧" in output
    assert footer in output
    assert "└ read:" in output
    assert "x" * 40 not in output
    assert "x" * 39 + "…" in output
    if expanded:
        assert "Worker-0: assignment 0" in output
        assert ("partial-0" if status == "running" else "result-0") in output
        assert "earlier agents" not in output
    else:
        assert "Worker-0: assignment 0" not in output
        assert "… 1 earlier agents" in output
        assert "partial-4" not in output
        assert "result-4" not in output


def test_delivered_result_is_a_collapsible_system_card() -> None:
    value = Block(
        "delivery-1",
        "delivery",
        data={
            "job": {
                "run_id": "worker",
                "name": "Researcher",
                "status": "done",
                "result": "one\ntwo\nthree\nfour\nfive\nsix",
            }
        },
    )

    collapsed = plain(render_delivery(value, THEME, 80, 20, False), 80)
    expanded = plain(render_delivery(value, THEME, 80, 20, True), 80)

    assert "↩ Researcher finished" in collapsed
    assert "one" in collapsed and "four" in collapsed
    assert "five" not in collapsed
    assert "… 2 more lines" in collapsed
    assert "six" in expanded
    assert "more lines" not in expanded


def test_registry_drives_hud_status_and_title_counts() -> None:
    registry = FakeRegistry(
        [
            FakeRun(
                "running",
                "Runner",
                "running",
                current_tool="read",
                current_tool_args="{'path': 'a.py'}",
            ),
            FakeRun("queued", "Queued", "queued"),
            FakeRun("idle", "Idle", "idle"),
            FakeRun("parked", "Parked", "parked"),
            FakeRun("done", "Done", "done"),
            FakeRun("failed", "Failed", "failed"),
            FakeRun("aborted", "Aborted", "aborted"),
        ]
    )
    ctx = SimpleNamespace(agents=registry)

    data = subagent_hud_data(ctx, spinner_frame=3)
    assert data is not None
    assert data["running"] == 1
    assert data["idle"] == 1
    assert data["queued"] == 1
    assert data["spinner_frame"] == 3
    assert [row["name"] for row in data["agents"]] == [
        "Runner",
        "Queued",
        "Idle",
        "Parked",
        "Done",
        "Failed",
        "Aborted",
    ]
    assert data["agents"][0]["last_tool"] == "read"
    assert "result" not in data["agents"][0]
    assert "partial_findings" not in data["agents"][0]
    assert agent_counts(ctx) == (1, 1, 4)
    assert subagents_segment(ctx).text == "1 running · 1 idle · 1 queued"

    output = TitleOutput()
    title = TerminalTitle(output, unicode=False)
    title.set_session("orchestration")
    title.set_agents(agent_counts(ctx)[2])
    title.set_agents(3)
    title.set_agents(0)
    assert output.titles == [
        "orcha - orchestration",
        "orcha - orchestration - 4 agents",
        "orcha - orchestration - 3 agents",
        "orcha - orchestration",
    ]


def test_empty_registry_hides_agent_hud_and_status() -> None:
    ctx = SimpleNamespace(agents=FakeRegistry([]))

    assert subagent_hud_data(ctx) is None
    assert agent_counts(ctx) == (0, 0, 0)
    assert subagents_segment(ctx) is None


def hub_context(registry: FakeRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        agents=registry,
        ledger=SimpleNamespace(all=lambda _thread_id: (), leaf=lambda _thread_id: None),
        session=SimpleNamespace(get=lambda _session_id: None),
    )


def test_hub_roster_inspector_tree_navigation_and_refresh_throttle() -> None:
    root = FakeRun(
        "root",
        "Runner",
        "running",
        description="inspect orchestration",
        current_tool="read",
        current_tool_args="{'path': 'orcha_agent/tui/runtime.py'}",
        result={"summary": "root result"},
    )
    child = FakeRun(
        "child",
        "Reviewer",
        "idle",
        parent_id="root",
        depth=1,
        description="review changes",
        result={"summary": "review complete"},
    )
    parked = FakeRun(
        "parked",
        "Sleeper",
        "parked",
        description="waiting for follow-up",
        session_id="parked-session",
    )
    registry = FakeRegistry([root, parked, child])
    now = [10.0]
    overlay = HubOverlay(hub_context(registry), clock=lambda: now[0])

    roster = overlay.render_roster_text()
    inspector = overlay.render_inspector_text()
    assert "3 agents" in overlay.render_text()
    assert "fake:" in roster
    assert "⟳" in roster and "Runner" in roster
    assert "•" in roster and "Reviewer" in roster
    assert "⏸" in roster and "Sleeper" in roster
    assert "read" in inspector
    assert "orcha_agent/tui/runtime.py" in inspector

    overlay.move(1)
    assert overlay.selected_run is parked
    overlay.toggle_tree()
    assert overlay.refresh_from_event() is True
    assert overlay.tree_mode is True
    assert [run.id for run in overlay.filtered_runs] == ["root", "child", "parked"]
    assert "└" in overlay.render_roster_text()

    initial_refreshes = overlay.refresh_count
    registry.runs.append(FakeRun("new", "New", "running"))
    assert overlay.refresh_from_event() is False
    now[0] += 0.249
    assert overlay.refresh_from_event() is False
    now[0] += 0.001
    assert overlay.refresh_from_event() is True
    assert overlay.refresh_count == initial_refreshes + 1
    assert any(run.id == "new" for run in overlay.filtered_runs)


@pytest.mark.asyncio
async def test_hub_filter_tree_and_drill_are_headlessly_driven() -> None:
    root = FakeRun("root", "Runner", "running")
    child = FakeRun("child", "Reviewer", "idle", parent_id="root", depth=1)
    parked = FakeRun("parked", "Sleeper", "parked", session_id="parked-session")
    registry = FakeRegistry([root, parked, child])

    filtered = HubOverlay(hub_context(registry))
    assert await drive_overlay(filtered, b"/sleep\r\x1b") is None
    assert [run.id for run in filtered.filtered_runs] == ["parked"]

    drilled = HubOverlay(hub_context(registry))
    assert await drive_overlay(drilled, b"tj\r") == "child"
    assert drilled.tree_mode is True
    assert drilled.selected_run is child


@pytest.mark.asyncio
async def test_hub_cancel_revive_message_and_copy_actions_target_selection() -> None:
    running = FakeRun("running", "Runner", "running")
    parked = FakeRun(
        "parked",
        "Sleeper",
        "parked",
        session_id="parked-session",
        result={"summary": "usable result"},
    )
    registry = FakeRegistry([running, parked])
    overlay = HubOverlay(
        hub_context(registry),
        clipboard=lambda text: text,
    )

    assert await overlay.cancel_selected() is True
    assert registry.cancelled == ["running"]
    overlay.move(1)
    assert await overlay.revive_selected() is True
    assert registry.revived == ["parked-session"]
    assert await overlay.send_message("continue with the review") is True
    assert registry.messages == [("parked", "continue with the review")]
    copied = overlay.copy_selected()
    assert copied is not None
    assert "usable result" in copied


@pytest.mark.asyncio
async def test_hub_action_uses_selection_captured_by_key_handler() -> None:
    first = FakeRun("first", "First", "running")
    second = FakeRun("second", "Second", "running")
    registry = FakeRegistry([first, second])
    overlay = HubOverlay(hub_context(registry))

    action = overlay.cancel_selected(overlay.selected_run)
    overlay.move(1)
    assert await action is True

    assert registry.cancelled == ["first"]


@pytest.mark.asyncio
async def test_task_card_preserves_indexes_after_earlier_spawn_failure() -> None:
    transcript = Transcript()
    await transcript.handle(
        ToolCallStart(
            "task",
            {
                "tasks": [
                    {"task": "fails before spawning"},
                    {"task": "succeeds with generated name"},
                ]
            },
            "call",
        )
    )
    await transcript.handle(AgentSpawned("run-2", "main", "GeneratedName", "task"))
    await transcript.handle(
        ToolCallEnd(
            "task",
            "call",
            {
                "spawned": [
                    {
                        "id": "run-2",
                        "name": "GeneratedName",
                        "type": "task",
                        "status": "running",
                    }
                ],
                "errors": [{"index": 0, "error": "boom"}],
            },
        )
    )

    block = next(block for block in transcript.frame.blocks if block.data.get("id") == "call")
    agents = block.data["agents"]
    assert agents[0]["status"] == "failed"
    assert agents[0]["run_id"].startswith("pending:call:0")
    assert agents[0]["name"] == "agent 1"
    assert agents[0]["description"] == "fails before spawning"
    assert agents[1]["run_id"] == "run-2"
    assert agents[1]["name"] == "GeneratedName"


@pytest.mark.asyncio
async def test_machine_delivery_turn_is_not_rendered_as_a_user_card() -> None:
    transcript = Transcript()

    await transcript.handle(
        TurnStart(
            "main",
            "<system-notification>\nJob run finished: done\n</system-notification>",
        )
    )

    assert transcript.frame.blocks == []


@pytest.mark.asyncio
async def test_active_task_card_does_not_block_later_transcript_commits() -> None:
    transcript = Transcript()
    await transcript.handle(
        ToolCallStart(
            "task",
            {"tasks": [{"task": "wait for follow-up"}]},
            "call",
            source_id="child",
        )
    )
    await transcript.handle(TurnStart("child-thread", "continue", source_id="child"))

    assert [block.kind for block in transcript.frame.blocks] == ["user", "task"]
    assert transcript.frame.blocks[0].source_id == "child"
    assert transcript.frame.blocks[0].state.value == "committed"
    assert transcript.frame.blocks[1].state.value == "active"


@pytest.mark.asyncio
async def test_runtime_drill_send_and_back_restore_main_view() -> None:
    child = FakeRun("child", "Reviewer", "idle", parent_id="root", depth=1)
    registry = FakeRegistry([child])
    output = TitleOutput()
    ctx = SimpleNamespace(
        agents=registry,
        cfg=SimpleNamespace(
            cwd=Path.cwd(),
            models={},
            notify=False,
            statusbar=False,
            symbols="ascii",
        ),
        plugin_states={},
        session=SimpleNamespace(get=lambda _session_id: None),
        session_id="main-session",
    )
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        ctx=ctx,
        output=output,
    )

    assert runtime._drill_in("child") is True
    assert runtime.drilled_run_id == "child"
    assert runtime.ui.active_agent is child
    assert "Reviewer" in output.titles[-1]
    assert await runtime._send_to_drilled("inspect the failure") is True
    assert registry.messages == [("child", "inspect the failure")]

    chunk = ModelChunk(
        SimpleNamespace(content="live child output"),
        role="subagent",
        source_id="child",
    )
    await runtime.transcript.handle(chunk)
    await runtime.handle_presentation(chunk)
    await asyncio.sleep(0.26)
    assert runtime._drilled_frame is not None
    assert any(
        block.data.get("text") == "live child output" for block in runtime._drilled_frame.blocks
    )

    trailing = ModelChunk(
        SimpleNamespace(content=" more"),
        role="subagent",
        source_id="child",
    )
    await runtime.transcript.handle(trailing)
    await runtime.handle_presentation(trailing)
    assert runtime._drill_refresh_task is not None
    pending_refresh = runtime._drill_refresh_task
    end = TurnEnd("thread", source_id="child")
    await runtime.transcript.handle(end)
    await runtime.handle_presentation(end)
    assert runtime._drill_refresh_task is None
    assert pending_refresh not in runtime._pending

    assert runtime._leave_agent() is True
    assert runtime.drilled_run_id is None
    assert runtime.ui.active_agent is None
    assert runtime._leave_agent() is False
    await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_agents_command_and_alt_a_binding_open_the_hub() -> None:
    from orcha_agent.builtin import commands_core
    from orcha_agent.core.events import EventBus
    from orcha_agent.core.plugin import PluginAPI
    from orcha_agent.tui.runtime import dispatch_command

    registry = Registry()
    api = PluginAPI(
        name="commands-core",
        registry=registry,
        bus=EventBus(),
        config={},
        state={},
        request_rebuild=lambda: None,
    )
    commands_core.register(api)
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        registry=registry,
        output=DummyOutput(),
    )
    assert "agents" in registry.commands
    shown: list[str] = []

    async def show(name: str) -> None:
        shown.append(name)

    ctx = SimpleNamespace(ui=SimpleNamespace(show=show))
    assert await dispatch_command(registry, ctx, "/agents") is True
    binding = registry.keybindings["agents"]
    await binding.handler(ctx, SimpleNamespace())

    assert binding.default == "escape a"
    assert shown == ["hub", "hub"]
    await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_advisor_followup_is_dropped_after_session_switch() -> None:
    submitted: list[str] = []

    async def submit(text: str) -> None:
        submitted.append(text)

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=Path.cwd(),
            models={},
            notify=False,
            statusbar=False,
            symbols="ascii",
        ),
        plugin_states={},
        session=SimpleNamespace(get=lambda _session_id: None),
        session_id="new-session",
    )
    runtime = ApplicationRuntime(submit, ctx=ctx, output=DummyOutput())

    await runtime._submit_advisor_followup("old-session", "advice")

    assert submitted == []
    await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_late_agent_event_after_settled_task_card_does_not_duplicate_it() -> None:
    transcript = Transcript()
    await transcript.handle(
        ToolCallStart("task", {"tasks": [{"task": "count things"}]}, "call-1")
    )
    await transcript.handle(AgentSpawned("run-9", "main", "Scout", "scout"))
    await transcript.handle(
        AgentStatus("run-9", "main", "Scout", "scout", status="done")
    )
    block = transcript._agent_tasks["run-9"]
    block.data["tool_complete"] = True
    for agent in block.data.get("agents", []):
        agent["result"] = {"count": 1}
        agent["status"] = "done"
        agent["delivered"] = True
    transcript._settle_task_if_complete(block)
    assert block.state is not BlockState.ACTIVE
    task_blocks_before = [b for b in transcript.frame.blocks if b.kind == "task"]

    # A late per-agent update (e.g. delivered-flag refresh) must not create a
    # second aggregate task card.
    await transcript.handle(
        AgentStatus("run-9", "main", "Scout", "scout", status="done")
    )
    task_blocks_after = [b for b in transcript.frame.blocks if b.kind == "task"]
    assert task_blocks_after == task_blocks_before
