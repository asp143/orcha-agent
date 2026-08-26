from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import AgentBuildBefore, EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


MODE_SPECS = {
    "ask": ModeSpec(
        description="Ask before destructive actions",
        interrupt_on={
            "write_file": True,
            "edit_file": True,
            "delete": True,
            "execute": True,
        },
        allowed_tools=None,
    ),
    "edit": ModeSpec(
        description="Allow file changes but ask before execution",
        interrupt_on={"execute": True},
        allowed_tools=None,
    ),
    "yolo": ModeSpec(
        description="Allow all actions",
        interrupt_on={},
        allowed_tools=None,
    ),
    "plan": ModeSpec(
        description="Read-only planning",
        interrupt_on={},
        allowed_tools={"ls", "read_file", "glob", "grep"},
    ),
}


def _caps() -> ProviderCaps:
    return ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=False,
        structured_output=False,
        max_context=None,
    )


def _config(tmp_path: Path, mode: str) -> Config:
    return Config(
        model="fake:main",
        subagent_model="fake:subagent",
        summarizer_model="fake:summarizer",
        mode=mode,
        backend="test",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )


def _kernel() -> tuple[Registry, EventBus, PluginAPI]:
    registry = Registry()
    bus = EventBus()
    api = PluginAPI(
        name="test",
        config={},
        state={},
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )
    api.add_provider(
        "fake",
        lambda model_name, provider_config: FakeListChatModel(responses=[model_name]),
        capabilities=_caps(),
    )
    api.add_backend("test", lambda config: StateBackend())
    for name, spec in MODE_SPECS.items():
        api.add_mode(name, spec)
    return registry, bus, api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_interrupts"),
    [
        (
            "ask",
            {
                "write_file": True,
                "edit_file": True,
                "delete": True,
                "execute": True,
            },
        ),
        ("edit", {"execute": True}),
        ("yolo", {}),
        ("plan", {}),
    ],
)
async def test_build_agent_uses_exact_mode_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_interrupts: dict[str, bool],
) -> None:
    registry, bus, _ = _kernel()
    captured: list[dict[str, Any]] = []
    graphs: list[object] = []

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.append(kwargs)
        graph = object()
        graphs.append(graph)
        return graph

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", fake_create_deep_agent)

    with SessionStore(tmp_path / "sessions.db") as session:
        graph = await build_agent(registry, _config(tmp_path, mode), session, bus)

    assert graph is graphs[0]
    assert captured[0]["interrupt_on"] == expected_interrupts


@pytest.mark.asyncio
async def test_plan_mode_passes_explicit_read_only_filesystem_middleware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, _ = _kernel()
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", fake_create_deep_agent)

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "plan"), session, bus)

    filesystem = [
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, FilesystemMiddleware)
    ]
    assert len(filesystem) == 1
    assert {tool.name for tool in filesystem[0].tools} == {
        "ls",
        "read_file",
        "glob",
        "grep",
    }


@pytest.mark.asyncio
async def test_agent_build_before_mutation_reaches_create_deep_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    captured: dict[str, Any] = {}

    async def mutate_build(event: AgentBuildBefore) -> None:
        event.kwargs["system_prompt"] = "mutated by AgentBuildBefore"

    api.on(AgentBuildBefore, mutate_build)

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", fake_create_deep_agent)

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "yolo"), session, bus)

    assert captured["system_prompt"] == "mutated by AgentBuildBefore"
