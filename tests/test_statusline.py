from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.builtin import statusbar
from orcha_agent.core.events import EventBus, ModelChunk, ThreadSwitch, TurnEnd, TurnStart
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.tui.statusline import (
    PRESETS,
    SEPARATORS,
    Segment,
    _parse_git,
    cache_segment,
    context_segment,
    cost_segment,
    git_segment,
    mode_segment,
    model_segment,
    path_segment,
    render_statusline,
    session_segment,
    subagents_segment,
    time_segment,
    tokens_segment,
    visible_segments,
    wrap_segment,
)
from orcha_agent.tui.symbols import resolve_symbols


class _Console:
    def __init__(self, *, width: int = 100, encoding: str = "utf-8") -> None:
        self.width = width
        self.encoding = encoding
        self.console = self
        self.output: list[str] = []

    def print(self, value: object, **_kwargs: Any) -> None:
        self.output.append(str(value))


class _Session:
    def __init__(self, title: str | None = "Status line session") -> None:
        self.title = title

    def get(self, _session_id: str) -> SimpleNamespace:
        return SimpleNamespace(title=self.title)


class _Theme:
    def __init__(self, symbols: str = "nerd") -> None:
        self.symbols = resolve_symbols(symbols)
        self.colors = {
            "statusLineBg": "#111111",
            "statusLineSep": "#333333",
            "statusLineModel": "#00ffff",
            "statusLinePath": "#0088ff",
            "statusLineGitClean": "#00ff00",
            "statusLineGitDirty": "#ffff00",
            "statusLineContext": "#ff00ff",
            "statusLineCost": "#00ff00",
            "statusLineSubagents": "#00ffff",
            "text": "#ffffff",
            "muted": "#888888",
            "warning": "#ffff00",
            "error": "#ff0000",
        }


def _cfg(tmp_path: Path, **values: Any) -> SimpleNamespace:
    statusline = SimpleNamespace(
        preset=values.pop("preset", "default"),
        separator=values.pop("separator", "powerline"),
        left=values.pop("left", None),
        right=values.pop("right", None),
        transparent=values.pop("transparent", False),
    )
    defaults = {
        "model": "codex:gpt-5.6-sol",
        "mode": "ask",
        "cwd": tmp_path,
        "providers": {"codex": {"reasoning_effort": "high"}},
        "pricing": {},
        "statusbar": True,
        "composer": "box",
        "statusline": statusline,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _ctx(tmp_path: Path, **values: Any) -> SimpleNamespace:
    registry = values.pop("registry", Registry())
    state = values.pop("state", {})
    console = values.pop("console", _Console())
    ctx = SimpleNamespace(
        cfg=values.pop("cfg", _cfg(tmp_path)),
        registry=registry,
        plugin_states={"statusbar": state},
        console=console,
        session=values.pop("session", _Session()),
        session_id="session-1",
        ui=values.pop(
            "ui",
            SimpleNamespace(thinking_level="high", subagents=[{"status": "running"}]),
        ),
    )
    for name, value in values.items():
        setattr(ctx, name, value)
    return ctx


def _api(registry: Registry, bus: EventBus, state: dict[str, Any], *, enabled: bool = True) -> PluginAPI:
    return PluginAPI(
        name="statusbar",
        config={"statusbar": enabled},
        state=state,
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )


def _register_provider(registry: Registry) -> None:
    _api(registry, EventBus(), {}).add_provider(
        "codex",
        lambda model, config: (model, config),
        models=("gpt-5.6-sol",),
        capabilities=ProviderCaps(True, True, True, False, 272_000),
    )


def _plain(fragments: Any) -> str:
    return "".join(text for _style, text in fragments)


def test_segment_protocol_wraps_legacy_strings_and_preserves_explicit_segments() -> None:
    explicit = Segment("ready", "success", "status.success")
    assert wrap_segment("legacy") == Segment("legacy", "text")
    assert wrap_segment(None) is None
    assert wrap_segment(explicit) is explicit


def test_preset_groups_are_stable() -> None:
    assert PRESETS == {
        "default": (
            ("model", "mode", "path", "git"),
            ("subagents", "context", "cost"),
        ),
        "minimal": (("model", "path"), ("context",)),
        "compact": (("mode", "path", "git"), ("context", "time")),
        "full": (
            ("model", "mode", "path", "git", "session"),
            ("subagents", "tokens", "cache", "cost", "context", "time"),
        ),
        "nerd": (
            ("model", "mode", "path", "git", "session"),
            ("subagents", "tokens", "cache", "cost", "context", "time"),
        ),
        "ascii": (
            ("model", "mode", "path", "git"),
            ("subagents", "context", "cost"),
        ),
    }


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_resolves_and_renders(preset: str, tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("legacy", "plugin", lambda _ctx: "legacy", priority=999)
    ctx = _ctx(tmp_path, registry=registry, cfg=_cfg(tmp_path, preset=preset))
    statusbar.register(_api(registry, EventBus(), ctx.plugin_states["statusbar"]))

    rendered = _plain(render_statusline(ctx, _Theme(), width=120, composer_shape="borderless"))

    assert rendered
    assert len(rendered) <= 120
    if preset == "ascii":
        assert rendered.isascii()
    if preset == "default":
        assert "legacy" in rendered


@pytest.mark.parametrize(
    ("separator", "marker"),
    [
        ("ascii", "|"),
        ("block", ""),
        ("none", ""),
        ("pipe", ""),
        ("powerline", ""),
        ("powerline-thin", "│"),
        ("slash", "/"),
    ],
)
def test_every_separator_renders_without_exceeding_width(
    separator: str,
    marker: str,
    tmp_path: Path,
) -> None:
    registry = Registry()
    registry._add_status_segment("test", "one", lambda _ctx: Segment("one", "text"))
    registry._add_status_segment("test", "two", lambda _ctx: Segment("two", "text"))
    cfg = _cfg(tmp_path, separator=separator, left=("one", "two"), right=())
    rendered = _plain(render_statusline(_ctx(tmp_path, registry=registry, cfg=cfg), _Theme(), width=24))
    between = rendered.split("one", 1)[1].split("two", 1)[0]
    assert between.strip() == marker
    assert len(rendered) == 24


def test_transparent_mode_drops_status_background_style(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("test", "one", lambda _ctx: Segment("one", "text"))
    opaque = render_statusline(
        _ctx(tmp_path, registry=registry, cfg=_cfg(tmp_path, left=("one",), right=())),
        _Theme(),
        width=20,
    )
    transparent = render_statusline(
        _ctx(
            tmp_path,
            registry=registry,
            cfg=_cfg(tmp_path, left=("one",), right=(), transparent=True),
        ),
        _Theme(),
        width=20,
    )
    assert any("bg:#111111" in style for style, _text in opaque)
    assert all("bg:" not in style for style, _text in transparent)


def test_explicit_groups_are_ordered_and_failures_are_isolated(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("test", "alpha", lambda _ctx: "A")
    registry._add_status_segment("test", "broken", lambda _ctx: 1 / 0)
    registry._add_status_segment("test", "omega", lambda _ctx: Segment("Z", "text"))
    cfg = _cfg(tmp_path, separator="pipe", left=("omega", "broken"), right=("alpha",))
    rendered = _plain(render_statusline(_ctx(tmp_path, registry=registry, cfg=cfg), _Theme(), width=30))
    assert rendered.index("Z") < rendered.index("!broken") < rendered.index("A")


def test_each_omitted_group_falls_back_to_its_preset_default(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("test", "custom", lambda _ctx: Segment("custom", "text"))
    registry._add_status_segment(
        "test",
        "context",
        lambda _ctx: Segment("25.0%/100k", "statusLineContext"),
    )
    cfg = _cfg(tmp_path, preset="minimal", left=("custom",), right=None)
    assert [name for name, _value in visible_segments(
        _ctx(tmp_path, registry=registry, cfg=cfg)
    )] == ["custom", "context"]


def test_box_context_uses_gap_gauge_while_other_shapes_use_segment(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("test", "left", lambda _ctx: Segment("LEFT", "text"))
    registry._add_status_segment("test", "context", lambda _ctx: Segment("50.0%/100k", "statusLineContext"))
    registry._add_status_segment("test", "right", lambda _ctx: Segment("RIGHT", "text"))
    cfg = _cfg(tmp_path, separator="none", left=("left",), right=("context", "right"))
    ctx = _ctx(tmp_path, registry=registry, cfg=cfg)

    box = _plain(render_statusline(ctx, _Theme(), width=48, composer_shape="box"))
    borderless = _plain(render_statusline(ctx, _Theme(), width=48, composer_shape="borderless"))
    narrow = _plain(render_statusline(ctx, _Theme(), width=7, composer_shape="box"))

    assert box.index("LEFT") < box.index("50%") < box.index("RIGHT")
    assert "50.0%/100k" in borderless
    assert len(box) == 48
    assert len(narrow) == 7


def test_non_utf_output_is_ascii_safe(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment("test", "unicode", lambda _ctx: Segment("café ✓", "text", "icon.model"))
    ctx = _ctx(
        tmp_path,
        registry=registry,
        cfg=_cfg(tmp_path, left=("unicode",), right=(), separator="powerline"),
        console=_Console(encoding="ascii"),
    )
    assert _plain(render_statusline(ctx, _Theme(), width=30)).isascii()


def test_git_parser_counts_staged_unstaged_and_untracked_exactly() -> None:
    assert _parse_git(
        "## feat/status...origin/feat/status [ahead 2]\n"
        "A  staged.py\n"
        " M unstaged.py\n"
        "MM both.py\n"
        "?? one.txt\n"
        "?? two.txt\n"
        "!! ignored.txt\n"
    ) == "feat/status *2 !2 ?2"
    assert _parse_git("fatal: not a repository\n") is None


def test_git_refresh_is_nonblocking_cached_and_no_more_frequent_than_two_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[object] = []

    def slow_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls.append(object())
        started.set()
        release.wait(1)
        return SimpleNamespace(returncode=0, stdout="## main\nA  staged.py\n M dirty.py\n?? new.py\n")

    monkeypatch.setattr("orcha_agent.tui.statusline.subprocess.run", slow_run)
    now = [100.0]
    monkeypatch.setattr("orcha_agent.tui.statusline.monotonic", lambda: now[0])
    state: dict[str, Any] = {}
    ctx = _ctx(tmp_path, state=state)

    before = time.perf_counter()
    assert git_segment(ctx) is None
    assert time.perf_counter() - before < 0.1
    assert started.wait(1)
    assert git_segment(ctx) is None
    assert len(calls) == 1
    release.set()
    deadline = time.time() + 1
    while "_git_text" not in state and time.time() < deadline:
        time.sleep(0.005)

    assert git_segment(ctx).text == "main *1 !1 ?1"
    now[0] = 101.999
    assert git_segment(ctx).text == "main *1 !1 ?1"
    assert len(calls) == 1


def test_git_segment_stays_hidden_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def not_a_repository(*_args: object, **_kwargs: object) -> SimpleNamespace:
        calls.append(object())
        return SimpleNamespace(returncode=128, stdout="")

    monkeypatch.setattr(
        "orcha_agent.tui.statusline.subprocess.run",
        not_a_repository,
    )
    state: dict[str, Any] = {}
    ctx = _ctx(tmp_path, state=state)
    assert git_segment(ctx) is None
    deadline = time.time() + 1
    while "_git_at" not in state and time.time() < deadline:
        time.sleep(0.005)
    assert git_segment(ctx) is None
    assert len(calls) == 1


def test_all_builtin_segments_report_runtime_state(tmp_path: Path) -> None:
    registry = Registry()
    _register_provider(registry)
    state = {
        "input_tokens": 136_000,
        "output_tokens": 12_000,
        "last_input_tokens": 136_000,
        "cache_read_tokens": 4_000,
        "cache_write_tokens": 2_000,
        "cache_known": True,
        "_git_text": "main",
        "_git_dirty": False,
        "_git_at": time.monotonic(),
        "_last_turn_elapsed": 3.25,
    }
    cwd = tmp_path / "parent" / "project"
    ctx = _ctx(tmp_path, registry=registry, state=state, cfg=_cfg(cwd))

    assert model_segment(ctx).text == "GPT-5.6 Sol · high"
    assert mode_segment(ctx).text == "ask"
    assert path_segment(ctx).text == "project"
    assert git_segment(ctx).text == "main"
    assert session_segment(ctx).text == "Status line session"
    assert subagents_segment(ctx).text == "1"
    assert tokens_segment(ctx).text == "136k in 12k out"
    assert cache_segment(ctx).text == "4k read 2k write"
    assert cost_segment(ctx).text == "$1.02"
    assert context_segment(ctx).text == "50.0%/272k"
    assert time_segment(ctx).text == "3.2s"


@pytest.mark.asyncio
async def test_accounting_counts_main_and_subagent_once_resets_and_resumes(tmp_path: Path) -> None:
    registry = Registry()
    bus = EventBus()
    state: dict[str, Any] = {}
    statusbar.register(_api(registry, bus, state))
    main_chunk = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "input_token_details": {"cache_read": 10, "cache_creation": 5},
        }
    )
    subagent_chunk = SimpleNamespace(
        usage_metadata={"input_tokens": 40, "output_tokens": 8}
    )
    main = ModelChunk(main_chunk, role="main", source_id="main")
    subagent = ModelChunk(subagent_chunk, role="subagent", source_id="worker")
    await bus.emit(main)
    await bus.emit(main)
    await bus.emit(subagent)
    assert state["input_tokens"] == 140
    assert state["output_tokens"] == 28
    assert state["cache_read_tokens"] == 10
    assert state["cache_write_tokens"] == 5
    assert state["last_input_tokens"] == 100

    preserved = dict(state)
    resumed_registry = Registry()
    resumed_bus = EventBus()
    statusbar.register(_api(resumed_registry, resumed_bus, preserved))
    assert preserved["input_tokens"] == 140

    await resumed_bus.emit(ThreadSwitch("session-1", "thread-1", "thread-2", "compact"))
    assert preserved["input_tokens"] == 0
    assert preserved["output_tokens"] == 0
    assert preserved["cache_read_tokens"] == 0
    assert preserved["cache_write_tokens"] == 0
    assert preserved["last_input_tokens"] == 0


@pytest.mark.asyncio
async def test_turn_time_tracks_active_and_last_elapsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry()
    bus = EventBus()
    state: dict[str, Any] = {}
    now = [20.0]
    monkeypatch.setattr("orcha_agent.tui.statusline.monotonic", lambda: now[0])
    statusbar.register(_api(registry, bus, state))
    ctx = _ctx(tmp_path, state=state)

    await bus.emit(TurnStart("thread", "hello"))
    now[0] = 22.25
    assert time_segment(ctx).text == "2.2s"
    await bus.emit(TurnEnd("thread"))
    now[0] = 50.0
    assert time_segment(ctx).text == "2.2s"


@pytest.mark.asyncio
async def test_status_command_prints_effective_segments_without_markup(tmp_path: Path) -> None:
    registry = Registry()
    bus = EventBus()
    state = {"input_tokens": 10, "output_tokens": 2, "last_input_tokens": 10}
    statusbar.register(_api(registry, bus, state))
    cfg = _cfg(tmp_path, left=("model", "mode"), right=("tokens",))
    ctx = _ctx(tmp_path, registry=registry, state=state, cfg=cfg)

    await registry.commands["status"].handler(ctx, "")

    assert [line.partition(":")[0] for line in ctx.console.output] == ["model", "mode", "tokens"]
    assert all("<style" not in line and "class:" not in line for line in ctx.console.output)
    assert [name for name, _segment in visible_segments(ctx)] == ["model", "mode", "tokens"]


def test_disabled_statusbar_registers_nothing_and_renderer_is_empty(tmp_path: Path) -> None:
    registry = Registry()
    bus = EventBus()
    statusbar.register(_api(registry, bus, {}, enabled=False))
    assert registry.status_segments == []
    assert "status" not in registry.commands
    assert bus.handlers == []
    ctx = _ctx(tmp_path, registry=registry, cfg=_cfg(tmp_path, statusbar=False))
    assert render_statusline(ctx, _Theme(), width=80) == []
