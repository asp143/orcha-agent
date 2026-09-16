from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent

from orcha_agent.builtin import skills as plugin
from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.skills import discover_skills, parse_skill, render_skills
from orcha_agent.tui.complete import ComposerCompleter


def write(root: Path, name: str, body: str = "Instructions", metadata: str = "") -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: About {name}\n{metadata}---\n{body}")
    return path


def test_discovery_precedence_importers_and_nonrecursive(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = home / "project" / "nested"
    native = cwd / ".orcha-agent/skills"
    write(home / ".config/orcha-agent/skills", "same", "user")
    write(cwd.parent / ".orcha-agent/skills", "same", "parent")
    write(cwd / ".claude/skills", "same", "imported")
    write(native, "same", "nearest native")
    write(cwd / ".codex/skills", "codex")
    write(cwd / ".github/skills", "github")
    write(home / ".claude/skills", "global")
    write(native / "nested", "not-recursive")
    found, warnings = discover_skills(cwd, home)
    assert not warnings
    assert set(found) == {"same", "codex", "github", "global"}
    assert found["same"].read() == "nearest native"
    found, _ = discover_skills(cwd, home, {"import_claude": False, "import_codex": False})
    assert set(found) == {"same", "github"}
    assert discover_skills(cwd, home, {"enabled": False}) == ({}, [])


def test_native_user_precedes_user_import_even_beneath_home(tmp_path: Path) -> None:
    write(tmp_path / ".claude/skills", "same", "claude")
    write(tmp_path / ".config/orcha-agent/skills", "same", "native")
    found, _ = discover_skills(tmp_path / "project", tmp_path)
    assert found["same"].read() == "native"


def test_frontmatter_and_progressive_prompt(tmp_path: Path) -> None:
    path = write(
        tmp_path, "folded", "BODY SECRET", "globs: ['*.py', 'src/**']\nalwaysApply: false\n"
    )
    skill = parse_skill(path)
    assert skill.globs == ("*.py", "src/**")
    assert not skill.always_apply
    prompt = render_skills({skill.name: skill})
    assert "About folded" in prompt
    assert "BODY SECRET" not in prompt
    always = parse_skill(write(tmp_path, "always", "Inline instructions", "alwaysApply: true\n"))
    hidden = parse_skill(write(tmp_path, "hidden", "Hidden body", "hide: true\n"))
    user_only = parse_skill(
        write(tmp_path, "user-only", "User only", "disableModelInvocation: true\n")
    )
    prompt = render_skills({s.name: s for s in (skill, always, hidden, user_only)})
    assert "Inline instructions" in prompt
    assert "hidden" not in prompt
    assert "user-only" not in prompt
    assert hidden.read() == "Hidden body"


def test_folded_description_and_fresh_on_demand_body(tmp_path: Path) -> None:
    path = tmp_path / "folder-name" / "SKILL.md"
    path.parent.mkdir()
    path.write_text("---\ndescription: >\n  First line\n  second line\n---\nOriginal body")
    skill = parse_skill(path)
    assert skill.name == "folder-name"
    assert skill.description == "First line second line"
    path.write_text("---\ndescription: New metadata\n---\nUpdated body")
    assert skill.read() == "Updated body"


@pytest.mark.parametrize("metadata", ["hide: 'false'\n", "globs: 12\n", "name: '../bad'\n"])
def test_malformed_metadata_is_skipped(tmp_path: Path, metadata: str) -> None:
    write(tmp_path / ".orcha-agent/skills", "bad", metadata=metadata)
    found, warnings = discover_skills(tmp_path, tmp_path / "home")
    assert not found
    assert len(warnings) == 1


def test_sensitive_symlink_and_size_bounds(tmp_path: Path) -> None:
    root = tmp_path / ".orcha-agent/skills"
    secret = write(tmp_path / "Credentials", "secret")
    root.mkdir(parents=True)
    (root / "secret").symlink_to(secret.parent, target_is_directory=True)
    write(root, "huge", "x" * (256 * 1024 + 1))
    found, warnings = discover_skills(tmp_path, tmp_path / "home")
    assert not found
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_plugin_commands_tool_completions_and_first_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    write(tmp_path / ".orcha-agent/skills", "demo", "Do the work")
    write(
        tmp_path / ".orcha-agent/skills", "private", "Only user", "disableModelInvocation: true\n"
    )
    registry, bus = Registry(), EventBus()
    rebuild = Mock()
    plugin.register(
        PluginAPI(
            name="skills", config={}, state={}, registry=registry, bus=bus, request_rebuild=rebuild
        )
    )
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path),
        console=Mock(),
        submit_prompt=AsyncMock(),
        add_command_discovery_task=Mock(),
    )
    await bus.emit(AppStart(ctx))
    ctx.add_command_discovery_task.assert_called_once()
    event = AgentBuildBefore({"system_prompt": "Base"})
    await bus.emit(event)
    assert "About demo" in event.kwargs["system_prompt"]
    assert "Do the work" not in event.kwargs["system_prompt"]
    assert "private" not in event.kwargs["system_prompt"]
    assert rebuild.called
    await registry.commands["skill:demo"].handler(ctx, "one two")
    injected = ctx.submit_prompt.call_args.args[0]
    assert "Do the work" in injected and injected.endswith("one two")
    await registry.commands["skill"].handler(ctx, "private user args")
    assert "Only user" in ctx.submit_prompt.call_args.args[0]
    await registry.commands["skills"].handler(ctx, "")
    assert any("demo" in str(call) for call in ctx.console.print.call_args_list)
    await registry.commands["skill"].handler(ctx, "missing")
    assert ctx.console.error.called
    assert "Do the work" in await registry.tools["skill"].ainvoke({"name": "demo"})
    assert "user invocation only" in await registry.tools["skill"].ainvoke({"name": "private"})
    assert "unknown skill" in await registry.tools["skill"].ainvoke({"name": "missing"})
    assert registry.completers[0].fn(Document("/skill de"))[0].text == "demo"
    completer = ComposerCompleter(registry, tmp_path)
    completions = list(completer.get_completions(Document("/skill:d"), CompleteEvent()))
    assert "skill:demo" in [item.text for item in completions]
    again = AgentBuildBefore(dict(event.kwargs))
    await bus.emit(again)
    assert again.kwargs == event.kwargs
    await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_start_does_not_wait_for_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def slow_to_thread(*_args):
        entered.set()
        await finish.wait()
        return {}, []

    monkeypatch.setattr(plugin.asyncio, "to_thread", slow_to_thread)
    registry, bus = Registry(), EventBus()
    plugin.register(
        PluginAPI(
            name="skills", config={}, state={}, registry=registry, bus=bus, request_rebuild=Mock()
        )
    )
    await asyncio.wait_for(
        bus.emit(AppStart(SimpleNamespace(cfg=SimpleNamespace(cwd=tmp_path)))), timeout=0.2
    )
    await entered.wait()
    assert not finish.is_set()
    await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_collision_and_disappearing_always_apply_do_not_poison_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".orcha-agent/skills"
    healthy = parse_skill(write(root, "healthy", "Healthy instructions"))
    missing_path = write(root, "missing", "Gone", "alwaysApply: true\n")
    missing = parse_skill(missing_path)
    missing_path.unlink()
    monkeypatch.setattr(
        plugin,
        "discover_skills",
        lambda *_args: ({"healthy": healthy, "missing": missing}, []),
    )
    registry, bus = Registry(), EventBus()
    existing = AsyncMock()
    registry._add_command("external", "skill:healthy", existing, "External command")
    plugin.register(
        PluginAPI(
            name="skills", config={}, state={}, registry=registry, bus=bus, request_rebuild=Mock()
        )
    )
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path), console=Mock(), submit_prompt=AsyncMock()
    )
    await bus.emit(AppStart(ctx))
    event = AgentBuildBefore({"system_prompt": "Base"})
    await bus.emit(event)
    assert "About healthy" in event.kwargs["system_prompt"]
    assert "Gone" not in event.kwargs["system_prompt"]
    assert registry.commands["skill:healthy"].handler is existing
    assert "skill:missing" in registry.commands
    warnings = str(ctx.console.warning.call_args_list)
    assert "Could not register /skill:healthy" in warnings
    assert "Could not load always-apply skill missing" in warnings
    await registry.commands["skill"].handler(ctx, "healthy arguments")
    assert "Healthy instructions" in ctx.submit_prompt.call_args.args[0]
    assert "Healthy instructions" in await registry.tools["skill"].ainvoke({"name": "healthy"})
    assert "Error reading skill missing" in await registry.tools["skill"].ainvoke(
        {"name": "missing"}
    )
    await bus.emit(AppExit())
