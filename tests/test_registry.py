from collections.abc import Callable
from typing import Any

import pytest

from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import ModeSpec, PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry


def _api(name: str, registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name=name,
        config={},
        state={},
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )


def _register_named(api: PluginAPI, kind: str, *, replace: bool = False) -> None:
    if kind == "tool":
        def shared() -> str:
            return api.name

        api.add_tool(shared, replace=replace)
    elif kind == "command":
        async def handler(ctx: Any, args: str) -> None:
            del ctx, args

        api.add_command("shared", handler, help=api.name, replace=replace)
    elif kind == "provider":
        def factory(model: str, config: dict[str, Any]) -> tuple[str, str]:
            del config
            return api.name, model

        api.add_provider(
            "shared",
            factory,
            capabilities=ProviderCaps(
                tool_calling=True,
                streaming=True,
                thinking=False,
                structured_output=False,
                max_context=None,
            ),
            replace=replace,
        )
    elif kind == "backend":
        api.add_backend("shared", lambda config: (api.name, config), replace=replace)
    elif kind == "mode":
        api.add_mode(
            "shared",
            ModeSpec(description=api.name, interrupt_on={}, allowed_tools=None),
            replace=replace,
        )
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(f"unknown registry kind: {kind}")


@pytest.mark.parametrize("kind", ["tool", "command", "provider", "backend", "mode"])
def test_named_registries_reject_a_second_plugin_instead_of_silently_overwriting(
    kind: str,
) -> None:
    registry = Registry()
    bus = EventBus()
    _register_named(_api("alpha", registry, bus), kind)

    with pytest.raises(ValueError) as raised:
        _register_named(_api("beta", registry, bus), kind)

    message = str(raised.value)
    assert "alpha" in message
    assert "beta" in message


@pytest.mark.parametrize("kind", ["tool", "command", "provider", "backend", "mode"])
def test_replace_transfers_duplicate_ownership_to_the_replacing_plugin(kind: str) -> None:
    registry = Registry()
    bus = EventBus()
    _register_named(_api("alpha", registry, bus), kind)
    _register_named(_api("beta", registry, bus), kind, replace=True)

    with pytest.raises(ValueError) as raised:
        _register_named(_api("gamma", registry, bus), kind)

    message = str(raised.value)
    assert "beta" in message
    assert "gamma" in message
    assert "alpha" not in message


def test_renderer_priority_order_is_deterministic_even_when_registration_order_differs() -> None:
    registry = Registry()
    bus = EventBus()

    def renderer_for(label: str) -> Callable[[object], str]:
        return lambda event: f"{label}:{event!r}"

    _api("zeta", registry, bus).add_renderer("external", renderer_for("zeta"), priority=50)
    _api("early", registry, bus).add_renderer("external", renderer_for("early"), priority=10)
    _api("alpha", registry, bus).add_renderer("external", renderer_for("alpha"), priority=50)

    assert [entry.plugin for entry in registry.renderers] == ["early", "alpha", "zeta"]
    assert [entry.priority for entry in registry.renderers] == [10, 50, 50]
