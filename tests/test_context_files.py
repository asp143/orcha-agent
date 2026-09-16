from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent.builtin.context_files import (
    ancestor_dirs,
    expand_imports,
    register,
    render_context,
)
from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_ladder_ancestors_and_global(tmp_path: Path):
    home = tmp_path / "home"
    root = tmp_path / "repo"
    cwd = root / "src"
    cwd.mkdir(parents=True)
    put(root / ".git", "gitdir: elsewhere")
    put(tmp_path / "AGENTS.md", "outside repository")
    put(home / ".config/orcha-agent/AGENTS.md", "global native")
    put(home / ".claude/CLAUDE.md", "global claude")
    put(root / "AGENTS.md", "root agent")
    put(root / "CLAUDE.md", "shadowed")
    put(cwd / ".orcha-agent/AGENTS.md", "closest native")
    put(cwd / "AGENTS.md", "shadowed child")
    result = render_context(cwd, {}, home=home)
    assert ancestor_dirs(cwd) == [cwd, root]
    assert (
        result.index("global native") < result.index("root agent") < result.index("closest native")
    )
    assert "global claude" in result
    assert "shadowed" not in result
    assert "outside repository" not in result
    assert result.startswith("<repo-rules>") and result.endswith("</repo-rules>")


def test_imports_bound_cycles_and_preserve_code(tmp_path: Path):
    first = put(
        tmp_path / "AGENTS.md", "See @second.md.\n`@second.md`\n```\n@second.md\n```\na@second.md"
    )
    put(tmp_path / "second.md", "Imported @third.md")
    put(tmp_path / "third.md", "deep @AGENTS.md")
    result = expand_imports(first, max_bytes=4096)
    assert "See Imported deep @AGENTS.md." in result
    assert "`@second.md`" in result
    assert "```\n@second.md\n```" in result
    assert "a@second.md" in result
    assert "deep" not in expand_imports(first, max_bytes=4096, max_depth=1)
    assert len(expand_imports(first, max_bytes=10).encode()) <= 10


def test_pointer_only_and_byte_cap(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    put(root / ".git", "")
    put(root / "AGENTS.md", "root")
    child = put(root / "sub/AGENTS.md", "must never inline")
    result = render_context(root, {}, home=tmp_path / "home")
    assert f'<dir-context path="{child}" />' in result
    assert "must never inline" not in result
    put(root / "AGENTS.md", "<&😀" * 200)
    capped = render_context(root, {"max_bytes": 300}, home=tmp_path / "home")
    assert len(capped.encode()) <= 300
    assert capped.endswith("</repo-rules>")


def test_importer_toggles_and_sensitive_imports(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    put(root / ".git", "")
    put(root / "CLAUDE.md", "claude")
    put(root / ".cursorrules", "cursor")
    put(root / ".github/copilot-instructions.md", "copilot")
    result = render_context(
        root, {"import_claude": False, "import_cursor": False}, home=tmp_path / "home"
    )
    assert "copilot" in result and ">claude<" not in result
    file = put(root / "AGENTS.md", "@.env.secret @Credentials/password @missing.md")
    assert expand_imports(file, max_bytes=1000) == file.read_text()


@pytest.mark.asyncio
async def test_plugin_background_build_and_legacy_replacement(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    put(tmp_path / ".git", "")
    put(tmp_path / "AGENTS.md", "native instructions")
    registry, bus = Registry(), EventBus()
    register(
        PluginAPI(
            name="context_files",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    cfg = SimpleNamespace(cwd=tmp_path, memory=("AGENTS.md", "CLAUDE.md"))
    await bus.emit(AppStart(SimpleNamespace(cfg=cfg)))
    event = AgentBuildBefore({"system_prompt": "base", "memory": ["/AGENTS.md"]})
    await bus.emit(event)
    assert "native instructions" in event.kwargs["system_prompt"]
    assert event.kwargs["memory"] == []
    assert len(registry.prompt_fragments) == 1
    await bus.emit(event)
    assert event.kwargs["system_prompt"].count("<repo-rules>") == 1
    cfg.memory = ("CUSTOM.md",)
    event.kwargs["memory"] = ["/CUSTOM.md"]
    await bus.emit(event)
    assert event.kwargs["memory"] == ["/CUSTOM.md"]
    await bus.emit(AppExit())


def test_imports_preserve_matching_and_multiline_code_spans(tmp_path: Path):
    text = "``literal ` @included.md text``\n`multiline\n@included.md`\n@included.md"
    source = put(tmp_path / "AGENTS.md", text)
    put(tmp_path / "included.md", "EXPANDED")
    assert expand_imports(source, max_bytes=4096) == text.rsplit("@included.md", 1)[0] + "EXPANDED"


def test_hidden_descendant_context_gets_pointer(tmp_path: Path):
    put(tmp_path / ".git", "")
    path = put(tmp_path / ".github/workflows/AGENTS.md", "not loaded")
    result = render_context(tmp_path, {}, home=tmp_path / "home")
    assert f'<dir-context path="{path}" />' in result
    assert "not loaded" not in result


@pytest.mark.asyncio
async def test_turso_keeps_structured_only_memory(tmp_path: Path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Turso must not discover filesystem instructions")

    monkeypatch.setattr("orcha_agent.builtin.context_files.render_context", forbidden)
    registry, bus = Registry(), EventBus()
    register(
        PluginAPI(
            name="context_files",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    cfg = SimpleNamespace(cwd=tmp_path, memory_store=SimpleNamespace(backend="turso"))
    await bus.emit(AppStart(SimpleNamespace(cfg=cfg)))
    event = AgentBuildBefore({"system_prompt": "stored", "memory": []})
    await bus.emit(event)
    assert event.kwargs == {"system_prompt": "stored", "memory": []}
    await bus.emit(AppExit())
