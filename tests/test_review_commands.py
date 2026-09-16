from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from orcha_agent.builtin import commands_core, file_commands
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.commands import discover_commands
from orcha_agent.tui.context import AppContext


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_untrusted_commands_cannot_shadow_user_aliases(tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "repo"
    put(home / ".claude/commands", "team/review.md", "USER")
    put(cwd / ".orcha-agent/commands", "review.md", "PROJECT")
    put(cwd / ".orcha-agent/commands", "team:review.md", "PROJECT")
    commands = discover_commands(cwd, home=home, trust_cwd=False)
    assert commands["review"].body == "USER"
    assert commands["team:review"].body == "USER"


def test_nested_alias_requires_trust_and_unambiguous_stem(tmp_path):
    root = tmp_path / ".claude/commands"
    put(root, "a/run.md", "A")
    assert set(discover_commands(tmp_path, home=tmp_path / "home")) == {"a:run"}
    assert set(discover_commands(tmp_path, home=tmp_path / "home", trust_cwd=True)) == {
        "a:run",
        "run",
    }
    put(root, "b/run.md", "B")
    assert set(discover_commands(tmp_path, home=tmp_path / "home", trust_cwd=True)) == {
        "a:run",
        "b:run",
    }


@pytest.mark.asyncio
async def test_untrusted_model_ignored_with_one_warning_per_file(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    put(
        tmp_path / ".orcha-agent/commands",
        "explain.md",
        "---\nmodel: expensive:model\n---\nExplain",
    )
    found = discover_commands(tmp_path, home=tmp_path / "home")
    assert found["explain"].model is None
    registry, bus = Registry(), EventBus()
    file_commands.register(
        PluginAPI(
            name="file_commands",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False),
        console=Mock(),
        submit_prompt=AsyncMock(),
        _command_discovery_tasks=[],
    )
    ctx.add_command_discovery_task = lambda task: AppContext.add_command_discovery_task(ctx, task)
    await bus.emit(AppStart(ctx))
    await AppContext.wait_command_discovery(ctx)
    for _ in range(2):
        await registry.commands["explain"].handler(ctx, "")
    assert ctx.console.warning.call_count == 1
    assert all(call.kwargs["model"] is None for call in ctx.submit_prompt.await_args_list)


@pytest.mark.asyncio
async def test_help_renders_repository_markup_literally():
    output = StringIO()
    console = Console(file=output, width=150, force_terminal=False)
    ctx = SimpleNamespace(
        registry=SimpleNamespace(
            commands={
                "[red]name[/red]": SimpleNamespace(help="[link=https://example.com]click[/link]")
            }
        ),
        console=console,
    )
    await commands_core._help(ctx, "all")
    text = output.getvalue()
    assert "[red]name[/red]" in text
    assert "[link=https://example.com]click[/link]" in text
