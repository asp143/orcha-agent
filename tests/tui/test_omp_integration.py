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
