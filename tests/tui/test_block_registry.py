from __future__ import annotations

from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


def _api(registry: Registry) -> PluginAPI:
    return PluginAPI(
        name="test",
        config={},
        state={},
        registry=registry,
        bus=EventBus(),
        request_rebuild=lambda: None,
    )


def test_block_renderers_are_ordered_and_replace_by_kind() -> None:
    registry = Registry()
    api = _api(registry)
    late = lambda *_args: "late"
    early = lambda *_args: "early"
    replacement = lambda *_args: "replacement"

    api.add_block_renderer("tool", late, priority=200)
    api.add_block_renderer("assistant", early, priority=10)
    api.add_block_renderer("tool", replacement, priority=5, replace=True)

    assert [(entry.kind, entry.render) for entry in registry.block_renderers] == [
        ("tool", replacement),
        ("assistant", early),
    ]
