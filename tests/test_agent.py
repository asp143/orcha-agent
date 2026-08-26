import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfileConfig,
    create_deep_agent as real_create_deep_agent,
)
from deepagents.backends import LocalShellBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import AgentBuildBefore, EventBus
from orcha_agent.core.models import ModelResolver
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


class ToolCallingFakeModel(GenericFakeChatModel):
    disable_streaming: bool = True

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> "ToolCallingFakeModel":
        return self


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


def _raising_script() -> Iterator[AIMessage]:
    raise RuntimeError("primary model unavailable")
    yield


@pytest.mark.asyncio
async def test_main_model_fallback_is_middleware_and_role_models_remain_chat_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    created: dict[str, list[ToolCallingFakeModel]] = {
        "primary": [],
        "fallback": [],
    }

    def model_factory(
        model_name: str,
        provider_config: dict[str, Any],
    ) -> ToolCallingFakeModel:
        del provider_config
        messages = (
            _raising_script()
            if model_name == "primary"
            else iter([AIMessage(content="fallback succeeded")])
        )
        model = ToolCallingFakeModel(messages=messages)
        created[model_name].append(model)
        return model

    api.add_provider("fake", model_factory, capabilities=_caps(), replace=True)
    fallback_chain = ["fake:primary", "fake:fallback"]
    cfg = replace(
        _config(tmp_path, "yolo"),
        model=fallback_chain,
        subagent_model=list(fallback_chain),
        summarizer_model=list(fallback_chain),
    )
    captured: dict[str, Any] = {}

    def capture_and_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_create_deep_agent(**kwargs)

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", capture_and_create)

    with SessionStore(cfg.db_path) as session:
        session.create(cwd=tmp_path, model=cfg.model, thread_id="fallback-thread")
        graph = await build_agent(registry, cfg, session, bus)
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Answer with the fallback."}]},
            config={"configurable": {"thread_id": "fallback-thread"}},
        )

    fallback_middleware = [
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, ModelFallbackMiddleware)
    ]
    assert len(fallback_middleware) == 1
    assert isinstance(captured["model"], BaseChatModel)
    assert captured["model"] is created["primary"][0]

    subagent_models = [spec["model"] for spec in captured["subagents"]]
    assert subagent_models
    assert all(isinstance(model, BaseChatModel) for model in subagent_models)
    assert all(
        any(model is primary for primary in created["primary"])
        for model in subagent_models
    )

    summarizers = [
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, SummarizationMiddleware)
    ]
    assert len(summarizers) == 1
    assert isinstance(summarizers[0].model, BaseChatModel)
    assert any(
        summarizers[0].model is primary for primary in created["primary"]
    )
    assert result["messages"][-1].content == "fallback succeeded"


@pytest.mark.asyncio
async def test_plan_graph_rejects_scripted_write_and_keeps_read_only_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    model = ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/blocked.txt",
                                "content": "must not be written\n",
                            },
                            "id": "blocked-write",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Plan completed without changing files."),
            ]
        )
    )
    api.add_provider(
        "fake",
        lambda model_name, provider_config: model,
        capabilities=_caps(),
        replace=True,
    )
    api.add_backend(
        "test",
        lambda config: LocalShellBackend(root_dir=config.cwd),
        replace=True,
    )
    captured: dict[str, Any] = {}

    def capture_and_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_create_deep_agent(**kwargs)

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", capture_and_create)
    cfg = _config(tmp_path, "plan")

    with SessionStore(cfg.db_path) as session:
        session.create(cwd=tmp_path, model=cfg.model, thread_id="plan-thread")
        graph = await build_agent(registry, cfg, session, bus)
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "Write blocked.txt."}]},
            config={"configurable": {"thread_id": "plan-thread"}},
        )

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
    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "Plan completed without changing files."
    assert not (tmp_path / "blocked.txt").exists()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
async def test_build_agent_uses_configured_model_for_general_purpose_subagent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    registry, bus, api = _kernel()
    if explicit:
        api.add_subagent(
            {
                "name": "general-purpose",
                "description": "Explicit general-purpose subagent",
                "system_prompt": "Handle delegated work.",
            }
        )
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", fake_create_deep_agent)

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "yolo"), session, bus)

    general_purpose = [
        spec
        for spec in captured["subagents"]
        if spec["name"] == "general-purpose"
    ]
    assert len(general_purpose) == 1
    model = general_purpose[0]["model"]
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["subagent"]


@pytest.mark.asyncio
async def test_disabled_general_purpose_harness_profile_is_honored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    api.add_provider(
        "fake",
        lambda model_name, provider_config: FakeListChatModel(responses=[model_name]),
        capabilities=_caps(),
        harness=HarnessProfileConfig(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
        replace=True,
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "yolo"), session, bus)

    assert all(spec["name"] != "general-purpose" for spec in captured["subagents"])



@pytest.mark.asyncio
async def test_plan_mode_omits_unrestrictable_compiled_subagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    api.add_subagent(
        {
            "name": "compiled-writer",
            "description": "Has its own unrestricted graph",
            "runnable": object(),
        }
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "plan"), session, bus)

    assert all(spec["name"] != "compiled-writer" for spec in captured["subagents"])



@pytest.mark.asyncio
async def test_plan_mode_excludes_plugin_tools_outside_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()

    def mutate_workspace() -> str:
        return "mutated"

    api.add_tool(mutate_workspace)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    with SessionStore(tmp_path / "sessions.db") as session:
        await build_agent(registry, _config(tmp_path, "plan"), session, bus)

    assert captured["tools"] == []
@pytest.mark.asyncio
async def test_build_agent_uses_configured_model_for_main_summarization(
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
        await build_agent(registry, _config(tmp_path, "yolo"), session, bus)

    summarizers = [
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, SummarizationMiddleware)
    ]
    assert len(summarizers) == 1
    model = summarizers[0].model
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["summarizer"]


@pytest.mark.asyncio
async def test_build_agent_sanitizes_absolute_memory_sources_for_backend_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested_memory = workspace / "docs" / "AGENTS.md"
    nested_memory.parent.mkdir()
    nested_memory.write_text("Use repository conventions.")
    outside_memory = tmp_path / "outside-memory.md"
    outside_memory.write_text("Do not load this.")
    cfg = replace(
        _config(workspace, "yolo"),
        memory=(str(nested_memory), str(outside_memory)),
    )
    registry, bus, _ = _kernel()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    caplog.set_level(logging.WARNING, logger="orcha_agent.core.agent")

    with SessionStore(cfg.db_path) as session:
        await build_agent(registry, cfg, session, bus)

    assert captured["memory"] == ["/docs/AGENTS.md"]
    assert str(outside_memory) in caplog.text


@pytest.mark.asyncio
async def test_harness_profiles_are_registered_once_across_resolvers_and_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    alpha_profile = HarnessProfileConfig(system_prompt_suffix="Alpha profile.")
    beta_profile = HarnessProfileConfig(system_prompt_suffix="Beta profile.")

    def provider_factory(
        model_name: str,
        _provider_config: dict[str, Any],
    ) -> FakeListChatModel:
        return FakeListChatModel(responses=[model_name])

    api.add_provider(
        "testalpha",
        provider_factory,
        capabilities=_caps(),
        harness=alpha_profile,
    )
    api.add_provider(
        "testbeta",
        provider_factory,
        capabilities=_caps(),
        harness=beta_profile,
    )
    cfg = replace(
        _config(tmp_path, "yolo"),
        model="testalpha:main",
        subagent_model="testalpha:subagent",
        summarizer_model="testalpha:summarizer",
    )
    registrations: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        "deepagents.register_harness_profile",
        lambda key, profile: registrations.append((key, profile)),
    )
    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: object(),
    )

    ModelResolver(registry, cfg)
    ModelResolver(registry, cfg)
    with SessionStore(cfg.db_path) as session:
        await build_agent(registry, cfg, session, bus)
        await build_agent(registry, cfg, session, bus)

    assert registrations == [
        ("testalpha", alpha_profile),
        ("testbeta", beta_profile),
    ]
