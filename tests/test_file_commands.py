import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import pytest

from orcha_agent.builtin import file_commands
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.commands import (
    FileCommand,
    discover_commands,
    render_command,
    substitute_arguments,
)
from orcha_agent.extensibility.frontmatter import parse_frontmatter
from orcha_agent.tui.complete import ComposerCompleter
from orcha_agent.tui.context import AppContext
from orcha_agent.tui.runtime import dispatch_command


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_frontmatter_yaml_and_body():
    metadata, body = parse_frontmatter(
        '---\nname: "a:b"\ndescription: >-\n  first line\n  second line\n'
        'globs: ["*.py", "*.md"]\nalwaysApply: true\n---\n\nHello\n'
    )
    assert metadata == {
        "name": "a:b",
        "description": "first line second line",
        "globs": ["*.py", "*.md"],
        "alwaysApply": True,
    }
    assert body == "Hello\n"
    assert parse_frontmatter("plain") == ({}, "plain")


@pytest.mark.parametrize("text", ["---\nno end", "---\n- list\n---", "---\nx: [\n---"])
def test_frontmatter_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_frontmatter(text)


def test_discovery_precedence_aliases_metadata_and_import_toggle(tmp_path):
    cwd, home = tmp_path / "project", tmp_path / "home"
    write(
        cwd / ".orcha-agent/commands",
        "review.md",
        "---\ndescription: Review code\nargument-hint: <path>\nmodel: openai:test\n---\nNative $1",
    )
    write(home / ".config/orcha-agent/commands", "review.md", "User")
    write(cwd / ".claude/commands", "team/review.md", "Imported")
    write(home / ".claude/commands", "personal/foo.md", "Personal")
    write(cwd / ".orcha-agent/commands", "nested/hidden.md", "Hidden")
    found = discover_commands(cwd, home=home)
    assert set(found) == {"review", "team:review", "foo", "personal:foo"}
    assert found["review"].body == "Native $1"
    assert found["review"].description == "Review code"
    assert found["review"].argument_hint == "<path>"
    assert found["review"].model == "openai:test"
    assert not found["review"].trusted
    assert found["foo"].trusted
    assert set(discover_commands(cwd, home=home, import_claude=False)) == {"review"}
    assert discover_commands(cwd, home=home, trust_cwd=True)["review"].trusted


def test_discovery_skips_invalid_and_external_symlinks(tmp_path):
    root = tmp_path / ".orcha-agent/commands"
    write(root, "bad.md", "---\nbad: [\n---")
    external = write(tmp_path, "external.md", "secret")
    (root / "linked.md").symlink_to(external)
    assert discover_commands(tmp_path, home=tmp_path / "home") == {}


@pytest.mark.parametrize(
    ("body", "args", "expected"),
    [
        ("$1|$2|$9", 'one "two words"', "one|two words|"),
        ("$@|$ARGUMENTS", "one two", "one two|one two"),
        ("$@[2:2]|$@[2:]|$@[0:2]|$@[1:0]", "a b c d", "b c|b c d||"),
        ("$1 $2", "'$2' '$@'", "$2 $@"),
        ("Explain", "this file", "Explain\n\nthis file"),
        ("$1|$2", 'C:\\folder "unfinished word', "C:\\folder|unfinished word"),
    ],
)
def test_substitution(body, args, expected):
    assert substitute_arguments(body, args) == expected


@pytest.mark.asyncio
async def test_shell_interpolation_trust_and_argument_injection(tmp_path):
    command = FileCommand("test", tmp_path / "test.md", "Value !`printf hello` $1", "test")
    with pytest.raises(ValueError, match="trusted"):
        await render_command(command, "", cwd=tmp_path)
    trusted = FileCommand("test", command.path, command.body, "test", trusted=True)
    assert await render_command(trusted, "world", cwd=tmp_path) == "Value hello world"
    argument = "!`touch should-not-exist`"
    assert await render_command(trusted, repr(argument), cwd=tmp_path) == f"Value hello {argument}"
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.asyncio
async def test_shell_error_and_timeout(tmp_path):
    for body, error in [("!`exit 4`", ValueError), ("!`sleep 5`", TimeoutError)]:
        command = FileCommand("test", tmp_path / "test.md", body, "test", trusted=True)
        with pytest.raises(error):
            await render_command(command, "", cwd=tmp_path, timeout=0.05)


@pytest.mark.asyncio
async def test_plugin_registration_submission_model_and_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    write(
        tmp_path / ".orcha-agent/commands",
        "explain.md",
        "---\ndescription: Explain code\nargument-hint: <file>\nmodel: openai:test\n---\nExplain $1",
    )
    write(tmp_path / ".orcha-agent/commands", "help.md", "overwrite")
    registry, bus, state = Registry(), EventBus(), {}
    api = PluginAPI(
        name="file_commands",
        config={},
        state=state,
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )
    original = AsyncMock()
    api.add_command("help", original, help="Original")
    file_commands.register(api)
    completer = ComposerCompleter(registry, tmp_path)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False),
        submit_prompt=AsyncMock(),
        console=SimpleNamespace(error=Mock(), warning=Mock()),
        _command_discovery_tasks=[],
    )
    ctx.add_command_discovery_task = lambda task: AppContext.add_command_discovery_task(ctx, task)
    ctx.wait_command_discovery = lambda: AppContext.wait_command_discovery(ctx)
    await bus.emit(AppStart(ctx))
    # Dispatch immediately: it must wait for registration, without delaying AppStart.
    assert "explain" not in registry.commands
    assert await dispatch_command(registry, ctx, "/explain test.py")
    assert registry.commands["help"].handler is original
    assert registry.commands["explain"].help == "Explain code (<file>)"
    ctx.submit_prompt.assert_awaited_once_with("Explain test.py", model="openai:test")
    completions = list(completer.get_completions(Document("/exp"), CompleteEvent()))
    assert [entry.text for entry in completions] == ["explain"]
    assert "Explain code" in str(completions[0].display_meta)
    assert state == {}  # Runtime tasks must never enter persisted plugin state.


@pytest.mark.asyncio
async def test_command_discovery_survives_cancelled_dispatch():
    ready = asyncio.Event()
    discovery = asyncio.create_task(ready.wait())
    ctx = SimpleNamespace(_command_discovery_tasks=[discovery], console=Mock())
    waiter = asyncio.create_task(AppContext.wait_command_discovery(ctx))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not discovery.cancelled()
    ready.set()
    await AppContext.wait_command_discovery(ctx)
    assert ctx._command_discovery_tasks == []


@pytest.mark.asyncio
async def test_shell_output_is_bounded(tmp_path):
    command = FileCommand("test", tmp_path / "test.md", "!`yes data`", "test", trusted=True)
    with pytest.raises(ValueError, match="exceeds"):
        await render_command(command, "", cwd=tmp_path)


@pytest.mark.asyncio
async def test_shell_output_is_not_treated_as_argument_template(tmp_path):
    command = FileCommand("test", tmp_path / "test.md", "!`printf '$1'` $1", "test", trusted=True)
    assert await render_command(command, "value", cwd=tmp_path) == "$1 value"
