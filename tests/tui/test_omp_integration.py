"""Exercise the merged native-tool, skill, MCP, and inline TUI paths together."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from mcp.server.mcpserver import MCPServer
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from orcha_agent.builtin import filesystem, modes, render_default, skills, tools_native
from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import load_config
from orcha_agent.core.events import AppExit, AppStart, Event, EventBus, ToolCallEnd
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.extensibility.mcp import MCPManager
from orcha_agent.tui.console import ConsoleOutput
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.turn import run_turn


class ScriptedToolsModel(GenericFakeChatModel):
    disable_streaming: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolsModel:
        return self


class TerminalOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=40, columns=120)


@pytest.mark.asyncio
async def test_native_skill_and_mcp_turn_through_inline_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait_for_render: Any
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = replace(
        load_config([], env={}, cwd=tmp_path, user_config_path=tmp_path / "config.toml"),
        model="fake:integration",
        mode="yolo",
        memory=(),
        db_path=tmp_path / "session.db",
    )
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n")
    skill = tmp_path / ".config/orcha-agent/skills/integration/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: integration\ndescription: Integration instructions\n---\nSKILL_BODY_OK\n"
    )
    calls = [
        ("read", {"path": "sample.txt"}),
        ("edit", {"path": "sample.txt", "old_string": "beta", "new_string": "gamma"}),
        ("bash", {"command": "printf SHELL_BODY_OK"}),
        ("skill", {"name": "integration"}),
        ("mcp__integration__echo", {"value": "MCP_BODY_OK"}),
    ]
    model = ScriptedToolsModel(
        messages=iter(
            [
                *[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}
                        ],
                    )
                    for i, (name, args) in enumerate(calls)
                ],
                AIMessage(content="INTEGRATION_TURN_DONE"),
            ]
        )
    )
    registry, bus = Registry(), EventBus()

    def api(name: str) -> PluginAPI:
        return PluginAPI(
            name=name,
            config={"cwd": tmp_path},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )

    for module in (filesystem, modes, tools_native, render_default, skills):
        module.register(api(module.PLUGIN.name))
    api("fake").add_provider(
        "fake",
        lambda *_: model,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=False,
            thinking=False,
            structured_output=False,
            max_context=100_000,
        ),
    )
    server = MCPServer("integration")
    received: list[str] = []

    @server.tool()
    def echo(value: str) -> str:
        received.append(value)
        return value

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    runner = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="critical"))
    serving = asyncio.create_task(runner.serve(sockets=[listener]))
    settings = tmp_path / ".config/orcha-agent/mcp.json"
    settings.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "integration": {
                        "type": "http",
                        "url": f"http://127.0.0.1:{listener.getsockname()[1]}/mcp",
                        "timeout": 3,
                    }
                }
            }
        )
    )
    manager = MCPManager(api("mcp"), tmp_path, tmp_path, False, {})
    results: list[ToolCallEnd] = []

    async def record(event: ToolCallEnd) -> None:
        results.append(event)

    bus.on(ToolCallEnd, record, plugin="test", priority=20_000)
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    finished = asyncio.Event()
    try:
        await manager.start()
        await asyncio.wait_for(manager.connections["integration"].ready.wait(), 8)
        with SessionStore(cfg.db_path) as session, create_pipe_input() as pipe:
            info = session.create(tmp_path, cfg.model, cfg.mode)
            host = SimpleNamespace(
                cfg=cfg,
                registry=registry,
                bus=bus,
                session=session,
                session_id=info.thread_id,
                source_id="main",
                thread_config={"configurable": {"thread_id": info.thread_id}},
                console=ConsoleOutput(console),
                capture_turn=lambda: None,
                record_exit=lambda _: None,
                rebuild_requested=False,
                raise_turn_errors=True,
                plugin_states={},
            )
            await bus.emit(AppStart(host))
            host.agent = await build_agent(
                registry, cfg, session, bus, exclude_general_purpose=True
            )

            async def submit(text: str) -> None:
                try:
                    await run_turn(host, text)
                finally:
                    finished.set()

            runtime = ApplicationRuntime(
                submit,
                registry=registry,
                ctx=host,
                input=pipe,
                output=TerminalOutput(),
                console=console,
            )
            host.transcript = runtime.transcript
            host.console = ConsoleOutput(console, transcript=runtime.transcript)
            bus.on(Event, runtime.handle_presentation, plugin="presentation", priority=10_000)
            bus.on(Event, runtime.transcript.handle, plugin="transcript", priority=9_000)
            running = asyncio.create_task(runtime.run())
            try:
                pipe.send_text("Exercise the integration tools.\n")
                await asyncio.wait_for(finished.wait(), 15)
                await wait_for_render(runtime, lambda: not runtime.streaming)
                screen = runtime.application.renderer._last_screen
                rows = [
                    "".join(screen.data_buffer[y][x].char for x in range(120))
                    for y in range(screen.height)
                ]
                assert any(row.startswith("╭──") and row.endswith("╮") for row in rows)
            finally:
                pipe.send_bytes(b"\x04")
                await asyncio.wait_for(running, 5)
        assert [event.name for event in results] == [name for name, _ in calls]
        assert all(
            isinstance(event.result, ToolMessage) and event.result.status == "success"
            for event in results
        )
        assert target.read_text() == "alpha\ngamma\n"
        assert received == ["MCP_BODY_OK"]
        text = output.getvalue()
        for marker in (
            "sample.txt",
            "gamma",
            "SHELL_BODY_OK",
            "SKILL_BODY_OK",
            "MCP_BODY_OK",
            "INTEGRATION_TURN_DONE",
        ):
            assert marker in text
        assert "╭" in text and "╰" in text
        assert "Traceback" not in text and "\x1b" not in text
    finally:
        await bus.emit(AppExit())
        await manager.close()
        runner.should_exit = True
        await serving
        listener.close()


@pytest.mark.asyncio
async def test_rules_hooks_compaction_roles_and_usage_through_inline_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    from orcha_agent.builtin import hooks, rules
    from orcha_agent.core.compaction_config import CompactionConfig
    from orcha_agent.core.config import HookConfig
    from orcha_agent.core.ledger import CompactionEntry
    from orcha_agent.core.usage_store import UsageStore
    from orcha_agent.extensibility.stream_rules import StreamAborted
    from orcha_agent.tui.context import AppContext
    from orcha_agent.tui.frame import BlockState
    from orcha_agent.tui.transcript import Transcript

    cards: list[Any] = []
    append_compaction = Transcript.append_compaction

    def record_compaction(self: Any, data: dict[str, Any]) -> Any:
        block = append_compaction(self, data)
        cards.append(block)
        return block

    monkeypatch.setattr(Transcript, "append_compaction", record_compaction)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    config_dir = tmp_path / ".config/orcha-agent"
    rule_dir = config_dir / "rules"
    rule_dir.mkdir(parents=True)
    (rule_dir / "safe.md").write_text("---\ncondition: forbidden\n---\nUse safe wording instead.\n")
    cfg = replace(
        load_config([], env={}, cwd=tmp_path, user_config_path=config_dir / "config.toml"),
        model="@smol",
        model_roles={"smol": "fake:small", "summarizer": "fake:summary"},
        mode="yolo",
        memory=(),
        db_path=tmp_path / "wave2.db",
        hooks=(
            HookConfig(
                "tool_call_before",
                matcher="write",
                command="printf 'WRITE_BLOCKED_BY_HOOK' >&2; exit 2",
            ),
        ),
        compaction=CompactionConfig(
            threshold_tokens=1000,
            keep_recent_tokens=120,
            speculative=False,
            method_order=("summary",),
        ),
    )
    received: list[list[Any]] = []

    class StreamingModel(ScriptedToolsModel):
        disable_streaming: bool = False

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            received.append(messages)
            response = next(self.messages)
            assert isinstance(response, AIMessage)
            if response.tool_calls:
                yield ChatGenerationChunk(message=AIMessageChunk(content="Preparing operation."))
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_calls=response.tool_calls)
                )
            else:
                # Split the trigger across chunks to exercise the authoritative
                # callback interrupt before the rejected suffix can be produced.
                parts = (
                    ["forbid", "den", " REJECTED_SUFFIX"]
                    if response.content == "forbidden REJECTED_SUFFIX"
                    else [response.text]
                )
                for part in parts:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=part))
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                    )
                )

    main = StreamingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "bash",
                            "args": {"command": "printf 'once\\n' >> executed.txt"},
                            "id": "once",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="forbidden REJECTED_SUFFIX"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write",
                            "args": {"path": "blocked.txt", "content": "must not exist"},
                            "id": "blocked",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="WAVE2_TURN_DONE"),
            ]
        )
    )
    summary = ScriptedToolsModel(
        messages=iter(
            [
                AIMessage(
                    content="WAVE2_COMPACTION_SUMMARY",
                    usage_metadata={"input_tokens": 1100, "output_tokens": 8, "total_tokens": 1108},
                )
            ]
        )
    )
    registry, bus = Registry(), EventBus()

    def api(name: str) -> PluginAPI:
        return PluginAPI(
            name=name,
            config={"cwd": tmp_path},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )

    for module in (filesystem, modes, tools_native, render_default, hooks, rules):
        module.register(api(module.PLUGIN.name))
    resolved: list[str] = []

    def provider(name: str, _options: Any) -> Any:
        resolved.append(name)
        return summary if name == "summary" else main

    api("fake").add_provider(
        "fake",
        provider,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=100_000,
        ),
    )
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    bus.on(Event, record, plugin="test", priority=20_000)
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    completed = asyncio.Event()
    painted = asyncio.Event()
    errors: list[BaseException] = []
    with SessionStore(cfg.db_path) as session, create_pipe_input() as pipe:
        info = session.create(tmp_path, cfg.model, cfg.mode)
        assert info.current_thread is not None
        host = AppContext(
            cfg=cfg,
            registry=registry,
            bus=bus,
            session=session,
            plugins=[],
            plugin_states={},
            console=ConsoleOutput(console),
            session_id=info.thread_id,
            thread_id=info.current_thread,
        )
        await bus.emit(AppStart(host))
        assert "Use safe wording" in await registry.tools["rule"].ainvoke({"name": "safe"})
        host.agent = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)

        async def submit(text: str) -> None:
            try:
                await run_turn(host, text)
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        runtime = ApplicationRuntime(
            submit,
            registry=registry,
            ctx=host,
            input=pipe,
            output=TerminalOutput(),
            console=console,
        )
        host.transcript = runtime.transcript
        host.console = ConsoleOutput(console, transcript=runtime.transcript)
        bus.on(Event, runtime.handle_presentation, plugin="presentation", priority=10_000)
        bus.on(Event, runtime.transcript.handle, plugin="transcript", priority=9_000)

        def after_render(_application: Any) -> None:
            if (
                completed.is_set()
                and not runtime.streaming
                and cards
                and cards[0].state is BlockState.COMMITTED
                and "WAVE2_COMPACTION_SUMMARY" in output.getvalue()
            ):
                painted.set()

        runtime.application.after_render += after_render
        running = asyncio.create_task(runtime.run())
        try:
            # A large input crosses the actual middleware threshold; every model
            # request remains local to the scripted provider above.
            pipe.send_text("Exercise wave two. " + "historical context " * 700 + "\n")
            await asyncio.wait_for(completed.wait(), 30)
            runtime.application.invalidate()
            await asyncio.wait_for(painted.wait(), 30)
            assert not errors
            assert (tmp_path / "executed.txt").read_text() == "once\n"
            assert not (tmp_path / "blocked.txt").exists()
            assert sum(isinstance(event, StreamAborted) for event in events) == 1
            results = [event for event in events if isinstance(event, ToolCallEnd)]
            assert [event.name for event in results] == ["bash", "write"]
            assert isinstance(results[1].result, ToolMessage)
            assert results[1].result.status == "error"
            assert "WRITE_BLOCKED_BY_HOOK" in results[1].result.content
            entries = [
                entry
                for entry in host.ledger.path(info.thread_id)
                if isinstance(entry, CompactionEntry)
            ]
            assert len(entries) == 1
            assert entries[0].tokens_before >= cfg.compaction.threshold_tokens
            assert len(cards) == 1 and cards[0].state is BlockState.COMMITTED
            assert cards[0].data["summary"] == "WAVE2_COMPACTION_SUMMARY"
            rows = UsageStore(session).report("session", info.thread_id)
            assert {row["model"] for row in rows} == {"fake:small", "fake:summary"}
            assert sum(row["requests"] for row in rows) == 5
            assert sum(row["input_tokens"] for row in rows) >= 1100
            assert "small" in resolved and "summary" in resolved
            assert any("Use safe wording instead." in messages[0].text for messages in received)
            rendered = output.getvalue()
            assert "WAVE2_COMPACTION_SUMMARY" in rendered
            assert "WAVE2_TURN_DONE" in rendered
            assert "REJECTED_SUFFIX" not in rendered
            assert "Traceback" not in rendered and "\x1b" not in rendered
        finally:
            pipe.send_bytes(b"\x04")
            await asyncio.wait_for(running, 10)
            await bus.emit(AppExit())
