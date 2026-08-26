import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("third_party_orcha_plugin", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_external_plugin_contributions_reach_registry_and_agent_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_path = tmp_path / "external_plugin.py"
    plugin_path.write_text(
        """
from orcha_agent.core.plugin import ModeSpec


def external_echo(text: str) -> str:
    return f"external:{text}"


async def external_command(ctx, args: str) -> None:
    ctx.command_args = args


def render_external(event) -> str:
    return f"rendered:{event}"


def register(api) -> None:
    api.state["configured_greeting"] = api.config["greeting"]
    api.add_tool(external_echo)
    api.add_command("external", external_command, help="Run the external command")
    api.add_renderer("external_echo", render_external, priority=25)
    api.add_mode(
        "external",
        ModeSpec(
            description="Only use the external tool",
            interrupt_on={"external_echo": True},
            allowed_tools={"external_echo"},
        ),
    )
    api.system_prompt_fragment("Follow the external plugin instructions.", priority=25)
""".lstrip()
    )
    module = _import_module(plugin_path)
    registry = Registry()
    bus = EventBus()
    state: dict[str, object] = {}
    api = PluginAPI(
        name="external_plugin",
        config={"greeting": "hello"},
        state=state,
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )

    module.register(api)

    assert state == {"configured_greeting": "hello"}
    assert registry.tools["external_echo"] is module.external_echo
    assert registry.tools["external_echo"]("ping") == "external:ping"

    command = registry.commands["external"]
    assert command.plugin == "external_plugin"
    assert command.help == "Run the external command"
    context = SimpleNamespace(command_args=None)
    await command.handler(context, "one two")
    assert context.command_args == "one two"

    renderer = registry.renderers[0]
    assert renderer.plugin == "external_plugin"
    assert renderer.priority == 25
    assert renderer.match == "external_echo"
    assert renderer.render("payload") == "rendered:payload"

    assert registry.modes["external"] == ModeSpec(
        description="Only use the external tool",
        interrupt_on={"external_echo": True},
        allowed_tools={"external_echo"},
    )

    fragment = registry.prompt_fragments[0]
    assert fragment.plugin == "external_plugin"
    assert fragment.priority == 25
    assert fragment.text == "Follow the external plugin instructions."

    api.add_provider(
        "fake",
        lambda model_name, provider_config: FakeListChatModel(
            responses=[model_name]
        ),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
    )
    api.add_backend("test", lambda config: StateBackend())
    cfg = Config(
        model="fake:main",
        subagent_model="fake:subagent",
        summarizer_model="fake:summarizer",
        mode="external",
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
    captured: dict[str, Any] = {}
    built_graph = object()

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return built_graph

    monkeypatch.setattr(
        "orcha_agent.core.agent.create_deep_agent",
        fake_create_deep_agent,
    )

    with SessionStore(cfg.db_path) as session:
        session.create(cwd=tmp_path, model=cfg.model, thread_id="external-thread")
        graph = await build_agent(registry, cfg, session, bus)

    assert graph is built_graph
    assert captured["tools"] == [module.external_echo]
    assert captured["interrupt_on"] == {"external_echo": True}
    assert captured["system_prompt"] == "Follow the external plugin instructions."
