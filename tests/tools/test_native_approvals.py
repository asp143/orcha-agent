"""Approval interrupts expose real native targets and accumulated shell state."""

from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from test_native_integration import config, kernel

from orcha_agent.core.agent import build_agent
from orcha_agent.core.events import AppExit
from orcha_agent.core.session import SessionStore


def approval_description(result):
    requests = result["__interrupt__"][0].value["action_requests"]
    assert len(requests) == 1
    return requests[0]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["write", "edit"])
async def test_file_approval_shows_resolved_symlink_target(tmp_path: Path, name: str):
    actual = tmp_path / "actual"
    actual.mkdir()
    destination = actual / "sample.txt"
    destination.write_text("before\n")
    (tmp_path / "alias").symlink_to(actual, target_is_directory=True)
    args = (
        {"path": "alias/sample.txt", "content": "after\n"}
        if name == "write"
        else {"path": "alias/sample.txt", "old_string": "before", "new_string": "after"}
    )
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(cfg, [("read", {"path": "alias/sample.txt"}), (name, args)])
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            thread = {"configurable": {"thread_id": f"resolved-{name}"}}
            interrupted = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Update the file."}]}, thread
            )
            description = approval_description(interrupted)
            assert f"{name} resolved target(s):\n{destination}" == description
            assert destination.read_text() == "before\n"
            await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), thread)
            assert destination.read_text() == "after\n"
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_shell_approval_tracks_live_state_and_excludes_parent_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-parent-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-parent-openai")
    (tmp_path / "nested").mkdir()
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(
        cfg,
        [
            (
                "bash",
                {
                    "command": "cd nested; export BUILD_MODE=debug; export APPROVAL_SECRET=synthetic-export-secret"
                },
            ),
            (
                "bash",
                {
                    "command": 'printf "%s:%s:%s:%s" "$PWD" "$BUILD_MODE" "${ANTHROPIC_API_KEY-unset}" "${OPENAI_API_KEY-unset}"',
                    "env": {"ONCE_ONLY": "temporary", "TEMP_TOKEN": "synthetic-temporary-secret"},
                },
            ),
        ],
    )
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            thread = {"configurable": {"thread_id": "shell-approval"}}
            first = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Run the commands."}]}, thread
            )
            first_description = approval_description(first)
            assert f"Working directory (resolved): {tmp_path}\n" in first_description
            assert "Session environment delta: {}" in first_description
            second = await graph.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}), thread
            )
            description = approval_description(second)
            assert f"Working directory (resolved): {tmp_path / 'nested'}\n" in description
            assert '"BUILD_MODE": "debug"' in description
            assert '"APPROVAL_SECRET": "[redacted]"' in description
            assert '"ONCE_ONLY": "temporary"' in description
            assert '"TEMP_TOKEN": "[redacted]"' in description
            assert "synthetic-" not in description
            finished = await graph.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}), thread
            )
            results = [
                message for message in finished["messages"] if isinstance(message, ToolMessage)
            ]
            assert all(message.status == "success" for message in results)
            assert f"{tmp_path / 'nested'}:debug:unset:unset" in results[-1].content
            for result in (first, second, finished):
                assert "synthetic-parent-anthropic" not in repr(result)
                assert "synthetic-parent-openai" not in repr(result)
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_file_approval_marks_symlink_escape_blocked_and_execution_refuses(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(cfg, [("write", {"path": "escape/new.txt", "content": "blocked"})])
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            thread = {"configurable": {"thread_id": "blocked-target"}}
            interrupted = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Write the file."}]}, thread
            )
            description = approval_description(interrupted)
            assert "BLOCKED:" in description
            assert str(outside / "new.txt") in description
            finished = await graph.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}), thread
            )
            results = [
                message for message in finished["messages"] if isinstance(message, ToolMessage)
            ]
            assert results[-1].status == "error"
            assert not (outside / "new.txt").exists()
    finally:
        await bus.emit(AppExit())
