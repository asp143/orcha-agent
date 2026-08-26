import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI
from orcha_agent.core.registry import Registry


def _import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("third_party_orcha_plugin", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_external_plugin_contributions_reach_registry_and_build_facing_collections(
    tmp_path: Path,
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

    build_facing = {
        "tools": list(registry.tools.values()),
        "mode": registry.modes["external"],
        "system_prompt": "\n\n".join(item.text for item in registry.prompt_fragments),
    }
    assert module.external_echo in build_facing["tools"]
    assert build_facing["mode"].allowed_tools == {"external_echo"}
    assert build_facing["system_prompt"] == "Follow the external plugin instructions."
