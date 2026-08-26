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
    elif kind == "middleware":
        def shared() -> str:
            return api.name

        api.add_middleware(shared, replace=replace)
    elif kind == "renderer":
        api.add_renderer(
            "shared",
            lambda event: f"{api.name}:{event!r}",
            replace=replace,
        )
    elif kind == "subagent":
        api.add_subagent(
            {
                "name": "shared",
                "description": api.name,
                "system_prompt": f"{api.name} system prompt",
            },
            replace=replace,
        )
    elif kind == "mode":
        api.add_mode(
            "shared",
            ModeSpec(description=api.name, interrupt_on={}, allowed_tools=None),
            replace=replace,
        )
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(f"unknown registry kind: {kind}")


def _assert_replacement_is_stored(registry: Registry, kind: str) -> None:
    if kind == "tool":
        assert registry.tools["shared"]() == "beta"
    elif kind == "command":
        entry = registry.commands["shared"]
        assert (entry.plugin, entry.help) == ("beta", "beta")
    elif kind == "provider":
        entry = registry.providers["shared"]
        assert entry.plugin == "beta"
        assert entry.factory("model", {}) == ("beta", "model")
    elif kind == "backend":
        entry = registry.backends["shared"]
        assert entry.plugin == "beta"
        assert entry.factory({"key": "value"}) == (
            "beta",
            {"key": "value"},
        )
    elif kind == "middleware":
        entries = [entry for entry in registry.middleware if entry.name == "shared"]
        assert len(entries) == 1
        assert entries[0].plugin == "beta"
        assert entries[0].middleware() == "beta"
    elif kind == "renderer":
        entries = [entry for entry in registry.renderers if entry.name == "shared"]
        assert len(entries) == 1
        assert entries[0].plugin == "beta"
        assert entries[0].render("event") == "beta:'event'"
    elif kind == "subagent":
        entries = [entry for entry in registry.subagents if entry.name == "shared"]
        assert len(entries) == 1
        assert entries[0].plugin == "beta"
        assert entries[0].spec["description"] == "beta"
    elif kind == "mode":
        assert registry.modes["shared"].description == "beta"
    else:  # pragma: no cover - protects the test helper itself
        raise AssertionError(f"unknown registry kind: {kind}")


@pytest.mark.parametrize(
    "kind",
    [
        "tool",
        "middleware",
        "command",
        "renderer",
        "provider",
        "backend",
        "subagent",
        "mode",
    ],
)
def test_registrations_reject_a_second_plugin_instead_of_silently_overwriting(
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


@pytest.mark.parametrize(
    "kind",
    [
        "tool",
        "middleware",
        "command",
        "renderer",
        "provider",
        "backend",
        "subagent",
        "mode",
    ],
)
def test_replace_transfers_duplicate_ownership_to_the_replacing_plugin(kind: str) -> None:
    registry = Registry()
    bus = EventBus()
    _register_named(_api("alpha", registry, bus), kind)
    _register_named(_api("beta", registry, bus), kind, replace=True)
    _assert_replacement_is_stored(registry, kind)
    if kind in {"middleware", "renderer", "subagent"}:
        collection_name = {
            "middleware": "middleware",
            "renderer": "renderers",
            "subagent": "subagents",
        }[kind]
        registrations = getattr(registry, collection_name)
        assert [
            entry.plugin
            for entry in registrations
            if entry.name == "shared"
        ] == ["beta"]

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

    _api("zeta", registry, bus).add_renderer(
        "zeta-event",
        renderer_for("zeta"),
        priority=50,
    )
    _api("early", registry, bus).add_renderer(
        "early-event",
        renderer_for("early"),
        priority=10,
    )
    _api("alpha", registry, bus).add_renderer(
        "alpha-event",
        renderer_for("alpha"),
        priority=50,
    )

    assert [entry.plugin for entry in registry.renderers] == ["early", "alpha", "zeta"]
    assert [entry.priority for entry in registry.renderers] == [10, 50, 50]


def test_provider_keeps_foreign_history_block_types() -> None:
    registry = Registry()
    api = _api("provider", registry, EventBus())

    api.add_provider(
        "example",
        lambda model, config: (model, config),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=True,
            max_context=None,
        ),
        foreign_block_types=("thinking", "reasoning"),
    )

    assert registry.providers["example"].foreign_block_types == frozenset(
        {"thinking", "reasoning"}
    )


def test_distinct_callable_renderer_predicates_do_not_conflict() -> None:
    registry = Registry()
    bus = EventBus()

    first = lambda event: event == "first"
    second = lambda event: event == "second"
    _api("alpha", registry, bus).add_renderer(first, lambda event: str(event))
    _api("beta", registry, bus).add_renderer(second, lambda event: str(event))

    assert [entry.match for entry in registry.renderers] == [first, second]
