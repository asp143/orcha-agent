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


def test_untrusted_context_contains_ladder_and_imports(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    put(root / ".git", "")
    outside = put(tmp_path / "outside.md", "OUTSIDE CONTENT")
    (root / "CLAUDE.md").symlink_to(outside)
    home = tmp_path / "home"
    put(home / ".config/orcha-agent/AGENTS.md", "HOME RULES")
    result = render_context(root, {}, home=home, trust_cwd=False)
    assert "OUTSIDE CONTENT" not in result
    assert "HOME RULES" in result
    put(root / "AGENTS.md", "@~/.ssh/x @/etc/hostname @../outside.md @linked.md @local.md")
    (root / "linked.md").symlink_to(outside)
    put(root / "local.md", "LOCAL CONTENT")
    result = render_context(root, {}, home=home, trust_cwd=False)
    assert "@~/.ssh/x @/etc/hostname @../outside.md @linked.md LOCAL CONTENT" in result
    assert "OUTSIDE CONTENT" not in result


@pytest.mark.parametrize(
    "name",
    [
        ".ssh/x",
        ".aws/config",
        ".gnupg/x",
        ".netrc",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        ".docker/config.json",
        ".kube/config",
        "a.pfx",
        "a.pem",
        "a.key",
        "a.p12",
        "credentials.json",
        "secrets.txt",
        ".env.fake",
        "Credentials/x",
    ],
)
@pytest.mark.parametrize("trusted", [True, False])
def test_sensitive_context_never_read(tmp_path: Path, name: str, trusted: bool, monkeypatch):
    # No sensitive file contents are needed: reject before attempting any open.
    source = put(tmp_path / "AGENTS.md", f"@{name}")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path == source, f"Attempted sensitive read: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert (
        expand_imports(source, max_bytes=4096, trust_cwd=trusted, project_root=tmp_path)
        == f"@{name}"
    )


def test_non_repo_ancestor_walk_stops_at_home(tmp_path: Path):
    home = tmp_path / "home"
    cwd = home / "a/b"
    cwd.mkdir(parents=True)
    put(tmp_path / "AGENTS.md", "OUTSIDE")
    put(home / "AGENTS.md", "HOME")
    assert ancestor_dirs(cwd, home=home) == [cwd, cwd.parent, home]
    assert "OUTSIDE" not in render_context(cwd, {}, home=home)
    outside = tmp_path / "elsewhere/child"
    outside.mkdir(parents=True)
    assert ancestor_dirs(outside, home=home) == [outside]


def test_untrusted_non_repo_import_boundary_is_cwd_under_home(tmp_path: Path):
    home = tmp_path / "home"
    cwd = home / "Downloads/evil"
    put(cwd / "AGENTS.md", "@../../.netrc @../../ordinary.md @docs/local.md")
    put(home / ".netrc", "PRIVATE NETRC FIXTURE")
    put(home / "ordinary.md", "OUTSIDE ORDINARY FIXTURE")
    put(home / "AGENTS.md", "ANCESTOR RULES @ordinary.md")
    put(cwd / "docs/local.md", "LOCAL @nested.md")
    put(cwd / "docs/nested.md", "NESTED RULES")
    put(home / ".config/orcha-agent/AGENTS.md", "GLOBAL @local.md")
    put(home / ".config/orcha-agent/local.md", "GLOBAL IMPORT")

    result = render_context(cwd, {}, home=home, trust_cwd=False)

    assert "@../../.netrc @../../ordinary.md LOCAL NESTED RULES" in result
    assert "PRIVATE NETRC FIXTURE" not in result
    assert "OUTSIDE ORDINARY FIXTURE" not in result
    assert "ANCESTOR RULES" not in result
    assert "GLOBAL GLOBAL IMPORT" in result


def test_untrusted_nested_repo_imports_keep_git_root_boundary(tmp_path: Path):
    home = tmp_path / "home"
    root = home / "repo"
    cwd = root / "src/nested"
    put(root / ".git", "gitdir: elsewhere")
    put(root / "AGENTS.md", "ROOT @shared.md")
    put(root / "shared.md", "SHARED RULES")
    put(cwd / "AGENTS.md", "NESTED @../../shared.md @../../../ordinary.md")
    put(home / "ordinary.md", "OUTSIDE ORDINARY FIXTURE")

    result = render_context(cwd, {}, home=home, trust_cwd=False)

    assert "ROOT SHARED RULES" in result
    assert "NESTED SHARED RULES @../../../ordinary.md" in result
    assert "OUTSIDE ORDINARY FIXTURE" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [True, False])
async def test_discovery_gate_preserves_background_and_disables_legacy(
    tmp_path: Path, monkeypatch, fails: bool, caplog
):
    import asyncio
    import threading

    release = threading.Event()

    def discover(*args, **kwargs):
        release.wait(3)
        if fails:
            raise OSError("discovery failed")
        assert kwargs["trust_cwd"] is False
        return "<repo-rules>READY</repo-rules>"

    monkeypatch.setattr("orcha_agent.builtin.context_files.render_context", discover)
    registry, bus = Registry(), EventBus()
    rebuilt = asyncio.Event()
    register(
        PluginAPI(
            name="context_files",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=rebuilt.set,
        )
    )
    await bus.emit(AppStart(SimpleNamespace(cfg=SimpleNamespace(cwd=tmp_path, trust_cwd=False))))
    event = AgentBuildBefore({"system_prompt": "base", "memory": ["AGENTS.md"]})
    try:
        await asyncio.wait_for(bus.emit(event), timeout=0.7)
        assert event.kwargs == {"system_prompt": "base", "memory": []}
    finally:
        release.set()
    await asyncio.wait_for(rebuilt.wait(), timeout=1)
    await bus.emit(event)
    if fails:
        assert "discovery failed" in caplog.text
    else:
        assert "READY" in event.kwargs["system_prompt"]
    await bus.emit(AppExit())


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("trusted", [True, False])
@pytest.mark.parametrize(
    "name",
    [
        ".ssh/config",
        ".netrc",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        ".docker/config.json",
        ".kube/config",
        "a.pfx",
    ],
)
def test_sensitive_import_symlink_cannot_disguise_path(
    tmp_path: Path, monkeypatch, reverse: bool, trusted: bool, name: str
):
    harmless = put(tmp_path / "ordinary.md", "ordinary fixture")
    sensitive = tmp_path / name
    sensitive.parent.mkdir(parents=True, exist_ok=True)
    if reverse:
        alias = tmp_path / "alias.md"
        alias.symlink_to(sensitive)
    else:
        sensitive.symlink_to(harmless)
        alias = sensitive
    source = put(tmp_path / "AGENTS.md", f"@{alias.relative_to(tmp_path)}")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path == source, f"Attempted sensitive read: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert expand_imports(
        source, max_bytes=4096, trust_cwd=trusted, project_root=tmp_path
    ).startswith("@")
