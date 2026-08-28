from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from orcha_agent.builtin import statusbar
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.statusline import BUILTIN_SEGMENTS, Segment


def _api(
    registry: Registry,
    bus: EventBus,
    state: dict[str, Any],
    *,
    enabled: bool = True,
) -> PluginAPI:
    return PluginAPI(
        name="statusbar",
        registry=registry,
        bus=bus,
        config={"statusbar": enabled},
        state=state,
        request_rebuild=lambda: None,
    )


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            model="codex:gpt-5.6-sol",
            mode="ask",
            cwd=tmp_path / "project",
            providers={"codex": {"reasoning_effort": "high"}},
            pricing={},
        ),
        registry=Registry(),
        plugin_states={"statusbar": {}},
        console=SimpleNamespace(width=80, encoding="utf-8"),
        ui=SimpleNamespace(thinking_level="off"),
    )


def test_adapter_registers_every_builtin_as_explicit_segments(tmp_path: Path) -> None:
    registry = Registry()
    bus = EventBus()
    state: dict[str, Any] = {}
    statusbar.register(_api(registry, bus, state))
    ctx = _ctx(tmp_path)
    ctx.registry = registry

    assert [entry.name for entry in registry.status_segments] == [
        name for name, _render in BUILTIN_SEGMENTS
    ]
    assert isinstance(registry.status_segments[0].render(ctx), Segment)
    assert "status" in registry.commands
    assert {entry.event_type.__name__ for entry in bus.handlers} == {
        "ModelChunk",
        "SessionSwitch",
        "ThreadSwitch",
        "TurnStart",
        "TurnEnd",
    }


def test_compatibility_helpers_delegate_to_new_segment_producers(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert statusbar.cwd_segment(ctx) == statusbar.path_segment(ctx)
    assert statusbar.effort_segment(ctx) == Segment(
        "high", "thinkingMedium", "icon.thinking"
    )
    assert statusbar._parse_git("## main\nA  staged.py\n") == "main *1"


def test_disabled_adapter_registers_nothing() -> None:
    registry = Registry()
    bus = EventBus()
    statusbar.register(_api(registry, bus, {}, enabled=False))
    assert registry.status_segments == []
    assert registry.commands == {}
    assert bus.handlers == []
