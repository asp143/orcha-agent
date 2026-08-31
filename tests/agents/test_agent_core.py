from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from orcha_agent.core.agent import build_agent
from orcha_agent.core.agents import AgentRegistry
from orcha_agent.core.agent_types import AgentType, builtin_agent_types
from orcha_agent.core.config import AgentsConfig, Config, load_config
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.context import AppContext


def _caps() -> ProviderCaps:
    return ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=False,
        structured_output=False,
        max_context=None,
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
        lambda model_name, _config: FakeListChatModel(responses=[model_name]),
        capabilities=_caps(),
    )
    api.add_backend("test", lambda _cfg: StateBackend())
    api.add_mode(
        "yolo",
        ModeSpec(description="all", interrupt_on={}, allowed_tools=None),
    )
    return registry, bus, api


def _config(tmp_path: Path) -> Config:
    return Config(
        model="fake:main",
        subagent_model="fake:legacy-task",
        summarizer_model="fake:summary",
        mode="yolo",
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
        model_roles={"task": "fake:worker"},
        agents=AgentsConfig(max_concurrency=3, max_depth=4),
    )


def test_builtin_agent_types_define_worker_scopes_and_roles() -> None:
    types = builtin_agent_types()

    assert set(types) == {"task", "scout", "reviewer", "advisor"}
    assert types["task"].model_role == "task"
    assert types["task"].spawns is True
    assert types["task"].tools is None
    assert types["scout"].tools == {"ls", "read_file", "glob", "grep"}
    assert types["scout"].spawns is False
    assert types["reviewer"].output_schema is not None
    assert types["advisor"].tools == {"read_file", "grep", "glob", "advise"}


def test_config_loads_agent_limits_and_model_roles_with_legacy_task_fallback(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[core]
model = "fake:main"
subagent_model = "fake:legacy"

[agents]
max_concurrency = 5
max_live_runs = 17
max_depth = 3
idle_ttl_s = 9
max_runtime_s = 11
soft_request_budget = 13

[models]
fast = "fake:fast"

[models.roles]
scout = "fake:scout"
""".strip()
        + "\n"
    )

    cfg = load_config([], env={"HOME": str(tmp_path)}, user_config_path=config)

    assert cfg.agents == AgentsConfig(
        max_concurrency=5,
        max_live_runs=17,
        max_depth=3,
        idle_ttl_s=9,
        max_runtime_s=11,
        soft_request_budget=13,
    )
    assert cfg.models == {"fast": "fake:fast"}
    assert cfg.model_roles == {"task": "fake:legacy", "scout": "fake:scout"}


def test_plugin_agent_type_registration_and_legacy_subagent_mapping() -> None:
    registry, _, api = _kernel()
    scout = AgentType(
        name="SearchOnly",
        description="Search files",
        system_prompt="Search and report.",
        tools={"grep"},
        model_role="scout",
        spawns=False,
        output_schema=None,
    )

    api.add_agent_type(scout)
    api.add_subagent(
        {
            "name": "LegacyWorker",
            "description": "Legacy plugin worker",
            "system_prompt": "Do legacy work.",
        },
        model="fake:legacy",
    )

    assert registry.agent_types["SearchOnly"] == scout
    legacy = registry.agent_types["LegacyWorker"]
    assert legacy.description == "Legacy plugin worker"
    assert legacy.system_prompt == "Do legacy work."
    assert legacy.model_role == "fake:legacy"


@pytest.mark.asyncio
async def test_agent_build_accepts_scoped_tools_prompt_and_no_deepagent_subagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, bus, api = _kernel()
    api.system_prompt_fragment("Global guidance")
    captured: dict[str, Any] = {}

    @tool
    def task(context: str) -> str:
        """Spawn an orcha task."""
        return context

    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    with SessionStore(tmp_path / "sessions.db") as store:
        await build_agent(
            registry,
            _config(tmp_path),
            store,
            bus,
            extra_tools=[task],
            system_prompt="Worker guidance",
            exclude_general_purpose=True,
            tool_scope={"task"},
        )

    assert [candidate.name for candidate in captured["tools"]] == ["task"]
    filesystem = next(
        item for item in captured["middleware"] if isinstance(item, FilesystemMiddleware)
    )
    assert filesystem._enabled_tools == frozenset()
    assert captured["subagents"] == []
    assert captured["system_prompt"] == "Worker guidance\n\nGlobal guidance"


@pytest.mark.asyncio
async def test_excluded_child_build_disables_deepagents_general_purpose_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, bus, _ = _kernel()
    captured: dict[str, Any] = {}

    def inspect_profile(**kwargs: Any) -> object:
        from deepagents.graph import _harness_profile_for_model

        profile = _harness_profile_for_model(kwargs["model"], None)
        captured["enabled"] = getattr(
            profile.general_purpose_subagent, "enabled", None
        )
        return object()

    monkeypatch.setattr("orcha_agent.core.agent.create_deep_agent", inspect_profile)
    with SessionStore(tmp_path / "sessions.db") as store:
        await build_agent(
            registry,
            _config(tmp_path),
            store,
            bus,
            exclude_general_purpose=True,
        )

    assert captured["enabled"] is False


def test_legacy_subagent_named_like_a_builtin_keeps_core_agent_type() -> None:
    registry, _, api = _kernel()
    builtin = registry.agent_types["task"]

    api.add_subagent(
        {
            "name": "task",
            "description": "Legacy deepagents worker",
            "system_prompt": "Legacy prompt",
        }
    )

    assert registry.agent_types["task"] is builtin
    assert any(entry.name == "task" for entry in registry.subagents)


def test_app_context_exposes_an_agent_registry_for_its_session(tmp_path: Path) -> None:
    registry, bus, _ = _kernel()
    with SessionStore(tmp_path / "sessions.db") as store:
        info = store.create(tmp_path, "fake:main", thread_id="main-session")
        assert info.current_thread is not None
        ctx = AppContext(
            cfg=_config(tmp_path),
            registry=registry,
            bus=bus,
            session=store,
            plugins=[],
            plugin_states={},
            console=SimpleNamespace(),
            session_id=info.thread_id,
            thread_id=info.current_thread,
        )

        assert isinstance(ctx.agents, AgentRegistry)
        assert ctx.agents.parent_session_id == info.thread_id


@pytest.mark.asyncio
async def test_new_session_retargets_agents_before_terminal_clear_failure(
    tmp_path: Path,
) -> None:
    registry, bus, _ = _kernel()

    async def fail_clear() -> None:
        raise RuntimeError("clear failed")

    with SessionStore(tmp_path / "sessions.db") as store:
        info = store.create(tmp_path, "fake:main", thread_id="main-session")
        assert info.current_thread is not None
        ctx = AppContext(
            cfg=_config(tmp_path),
            registry=registry,
            bus=bus,
            session=store,
            plugins=[],
            plugin_states={},
            console=SimpleNamespace(),
            session_id=info.thread_id,
            thread_id=info.current_thread,
            ui=SimpleNamespace(clear=fail_clear),
        )
        assert ctx.agents is not None
        old_session = ctx.session_id

        with pytest.raises(RuntimeError, match="clear failed"):
            await ctx.new_session()

        assert ctx.session_id != old_session
        assert ctx.agents.parent_session_id == ctx.session_id
