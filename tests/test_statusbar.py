from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.formatted_text import HTML, to_formatted_text

from orcha_agent.builtin import statusbar
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.tui.app import _bottom_toolbar


class _FakeConsole:
    def __init__(self, *, width: int = 100, encoding: str = "utf-8") -> None:
        self.width = width
        self.encoding = encoding
        self.console = self
        self.output: list[tuple[object, ...]] = []

    def print(self, *objects: object, **_kwargs: Any) -> None:
        self.output.append(objects)

    def error(self, message: str) -> None:
        self.output.append((message,))

    def warning(self, message: str) -> None:
        self.output.append((message,))


class _FakeGit:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(result, BaseException):
            raise result
        return result


def _cfg(
    tmp_path: Path,
    *,
    model: str | list[str] = "codex:gpt-5.6-sol",
    mode: str = "ask",
    icons: bool = True,
    statusbar_enabled: bool = True,
    pricing: dict[str, dict[str, float]] | None = None,
    providers: dict[str, dict[str, Any]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        subagent_model=None,
        mode=mode,
        cwd=tmp_path,
        models={},
        providers=providers or {},
        icons=icons,
        statusbar=statusbar_enabled,
        pricing=pricing or {},
    )


def _ctx(
    tmp_path: Path,
    *,
    cfg: SimpleNamespace | None = None,
    state: dict[str, Any] | None = None,
    registry: Registry | None = None,
    width: int = 100,
    encoding: str = "utf-8",
) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=cfg or _cfg(tmp_path),
        registry=registry or Registry(),
        plugin_states={"statusbar": {} if state is None else state},
        console=_FakeConsole(width=width, encoding=encoding),
    )


def _api(
    registry: Registry,
    bus: EventBus,
    state: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> PluginAPI:
    return PluginAPI(
        name="statusbar",
        registry=registry,
        bus=bus,
        config=config or {},
        state=state,
        request_rebuild=lambda: None,
    )


def _add_provider(
    registry: Registry,
    prefix: str,
    *,
    models: tuple[str, ...],
    max_context: int | None,
) -> None:
    _api(registry, EventBus(), {}).add_provider(
        prefix,
        lambda model, config: (model, config),
        models=models,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=max_context,
        ),
    )


def _fragments(markup: str) -> list[tuple[str, str]]:
    return list(to_formatted_text(HTML(markup)))


def _plain(markup: str) -> str:
    return "".join(text for _style, text in _fragments(markup))


def _style_for(markup: str, needle: str) -> str:
    return " ".join(style for style, text in _fragments(markup) if needle in text)


def _completed(stdout: str, *, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _printed_text(value: object) -> str:
    if isinstance(value, str):
        return _plain(value)
    return "".join(text for _style, text in to_formatted_text(value))


def test_model_segment_uses_provider_display_name_and_fallback_count(tmp_path: Path) -> None:
    registry = Registry()
    _add_provider(
        registry,
        "codex",
        models=("gpt-5.6-sol", "gpt-5.6-luna"),
        max_context=None,
    )
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(
            tmp_path,
            model=[
                "codex:gpt-5.6-sol",
                "anthropic:claude-sonnet-5",
                "codex:gpt-5.6-luna",
            ],
        ),
        registry=registry,
    )

    rendered = statusbar.model_segment(ctx)

    assert _plain(rendered) == "󰚩 GPT-5.6 Sol +2"
    assert "ansicyan" in _style_for(rendered, "GPT-5.6 Sol")


def test_model_segment_uses_raw_spec_when_provider_does_not_list_it(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(tmp_path, model="private:raw-model", icons=False),
    )

    assert _plain(statusbar.model_segment(ctx)) == "model: private:raw-model"


@pytest.mark.parametrize(
    ("provider_config", "expected"),
    [
        pytest.param({"reasoning_effort": "high"}, "󰪣 high", id="reasoning-effort"),
        pytest.param({"thinking": "extended"}, "󰪣 extended", id="thinking"),
    ],
)
def test_effort_segment_shows_configured_reasoning_or_thinking(
    tmp_path: Path,
    provider_config: dict[str, str],
    expected: str,
) -> None:
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(tmp_path, providers={"codex": provider_config}),
    )

    assert _plain(statusbar.effort_segment(ctx)) == expected


def test_effort_segment_is_hidden_without_reasoning_or_thinking(tmp_path: Path) -> None:
    assert statusbar.effort_segment(_ctx(tmp_path)) is None


@pytest.mark.parametrize(
    ("mode", "color"),
    [pytest.param("ask", "ansiyellow", id="ask"), pytest.param("yolo", "ansired", id="yolo")],
)
def test_mode_segment_uses_safety_color(tmp_path: Path, mode: str, color: str) -> None:
    rendered = statusbar.mode_segment(_ctx(tmp_path, cfg=_cfg(tmp_path, mode=mode)))

    assert _plain(rendered).endswith(mode)
    assert color in _style_for(rendered, mode)


@pytest.mark.parametrize(
    ("width", "suffix"),
    [pytest.param(119, "project", id="narrow"), pytest.param(120, "parent/project", id="wide")],
)
def test_cwd_segment_expands_at_120_columns(
    tmp_path: Path,
    width: int,
    suffix: str,
) -> None:
    cwd = tmp_path / "parent" / "project"
    ctx = _ctx(tmp_path, cfg=_cfg(cwd), width=width)

    assert _plain(statusbar.cwd_segment(ctx)).endswith(suffix)


def test_git_segment_parses_porcelain_and_caches_for_two_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git = _FakeGit(
        _completed(
            "## feat/pi-agent...origin/feat/pi-agent [ahead 1]\n"
            "?? first.txt\n"
            "?? second.txt\n"
            " M modified.py\n"
            "A  staged.py\n"
        ),
        _completed("## fix/cache\n M changed.py\n"),
    )
    now = [100.0]
    monkeypatch.setattr(statusbar.subprocess, "run", git.run)
    monkeypatch.setattr(statusbar, "monotonic", lambda: now[0])
    ctx = _ctx(tmp_path)

    assert _plain(statusbar.git_segment(ctx)).endswith("feat/pi-agent ?2 +2")
    now[0] = 101.999
    assert _plain(statusbar.git_segment(ctx)).endswith("feat/pi-agent ?2 +2")
    assert len(git.calls) == 1

    now[0] = 102.0
    assert _plain(statusbar.git_segment(ctx)).endswith("fix/cache +1")
    assert len(git.calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FileNotFoundError("git"), id="git-not-installed"),
        pytest.param(_completed("", returncode=128), id="not-a-repository"),
    ],
)
def test_git_segment_hides_git_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: object,
) -> None:
    fake = _FakeGit(failure)
    monkeypatch.setattr(statusbar.subprocess, "run", fake.run)
    monkeypatch.setattr(statusbar, "monotonic", lambda: 500.0)

    assert statusbar.git_segment(_ctx(tmp_path)) is None


@pytest.mark.parametrize(
    ("model", "last_input", "suffix", "color"),
    [
        pytest.param(
            "codex:gpt-5.6-sol",
            136_000,
            "50.0%/272k",
            "ansigreen",
            id="codex-5-6",
        ),
        pytest.param(
            "codex:gpt-5.5",
            163_200,
            "60.0%/272k",
            "ansiyellow",
            id="codex-5-5",
        ),
        pytest.param(
            "codex:gpt-5.4",
            27_200,
            "10.0%/272k",
            "ansigreen",
            id="codex-5-4",
        ),
        pytest.param(
            "codex:gpt-5.3-codex-spark",
            108_800,
            "85.0%/128k",
            "ansired",
            id="codex-spark",
        ),
        pytest.param(
            "anthropic:claude-sonnet-5",
            590_000,
            "59.0%/1M",
            "ansigreen",
            id="anthropic-sonnet",
        ),
        pytest.param(
            "anthropic:claude-opus-5",
            600_000,
            "60.0%/1M",
            "ansiyellow",
            id="anthropic-opus",
        ),
        pytest.param(
            "anthropic:claude-haiku-4-5",
            170_000,
            "85.0%/200k",
            "ansired",
            id="anthropic-haiku",
        ),
    ],
)
def test_context_segment_uses_model_windows_and_thresholds(
    tmp_path: Path,
    model: str,
    last_input: int,
    suffix: str,
    color: str,
) -> None:
    rendered = statusbar.context_segment(
        _ctx(
            tmp_path,
            cfg=_cfg(tmp_path, model=model),
            state={"last_input_tokens": last_input},
        )
    )

    assert _plain(rendered).endswith(suffix)
    assert color in _style_for(rendered, suffix)

def test_context_segment_falls_back_to_provider_capability(tmp_path: Path) -> None:
    registry = Registry()
    _add_provider(registry, "private", models=("small",), max_context=64_000)

    rendered = statusbar.context_segment(
        _ctx(
            tmp_path,
            cfg=_cfg(tmp_path, model="private:small"),
            state={"last_input_tokens": 16_000},
            registry=registry,
        )
    )

    assert _plain(rendered).endswith("25.0%/64k")


def test_tokens_segment_formats_cumulative_session_usage(tmp_path: Path) -> None:
    rendered = statusbar.tokens_segment(
        _ctx(
            tmp_path,
            state={"input_tokens": 12_400, "output_tokens": 3_100},
        )
    )

    assert _plain(rendered) == "󰁨 12.4k↑ 3.1k↓"


@pytest.mark.parametrize(
    ("pricing", "expected"),
    [
        pytest.param({}, "󰙺 $10.25", id="shipped-codex-default"),
        pytest.param(
            {"codex:gpt-5.6-sol": {"input": 2, "output": 8, "cache_read": 0.2}},
            "󰙺 $3.10",
            id="config-override",
        ),
    ],
)
def test_cost_segment_uses_uncached_input_cache_read_and_output_rates(
    tmp_path: Path,
    pricing: dict[str, dict[str, float]],
    expected: str,
) -> None:
    # Hand calculation for defaults: .5M*5 + .5M*.5 + .25M*30 = $10.25.
    # Override: .5M*2 + .5M*.2 + .25M*8 = $3.10.
    rendered = statusbar.cost_segment(
        _ctx(
            tmp_path,
            cfg=_cfg(tmp_path, pricing=pricing),
            state={
                "input_tokens": 1_000_000,
                "output_tokens": 250_000,
                "cache_read_tokens": 500_000,
            },
        )
    )

    assert _plain(rendered) == expected


def test_cost_segment_is_hidden_without_a_price_table_entry(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(tmp_path, model="private:unpriced"),
        state={"input_tokens": 100, "output_tokens": 20},
    )

    assert statusbar.cost_segment(ctx) is None


@pytest.mark.parametrize(
    ("icons", "encoding"),
    [
        pytest.param(False, "utf-8", id="icons-disabled"),
        pytest.param(True, "ascii", id="non-utf8"),
    ],
)
def test_model_segment_uses_ascii_label_when_icons_are_unavailable(
    tmp_path: Path,
    icons: bool,
    encoding: str,
) -> None:
    registry = Registry()
    _add_provider(registry, "codex", models=("gpt-5.6-sol",), max_context=None)
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(tmp_path, icons=icons),
        encoding=encoding,
        registry=registry,
    )

    assert _plain(statusbar.model_segment(ctx)) == "model: GPT-5.6 Sol"


@pytest.mark.asyncio
async def test_status_command_prints_the_eight_segments_one_per_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    _add_provider(registry, "codex", models=("gpt-5.6-sol",), max_context=None)
    bus = EventBus()
    state = {
        "last_input_tokens": 136_000,
        "input_tokens": 136_000,
        "output_tokens": 10_000,
    }
    statusbar.register(_api(registry, bus, state))
    git = _FakeGit(_completed("## main\n?? new.txt\n M changed.py\n"))
    monkeypatch.setattr(statusbar.subprocess, "run", git.run)
    monkeypatch.setattr(statusbar, "monotonic", lambda: 900.0)
    cwd = tmp_path / "deepagent"
    ctx = _ctx(
        tmp_path,
        cfg=_cfg(
            cwd,
            icons=False,
            providers={"codex": {"reasoning_effort": "high"}},
        ),
        state=state,
        registry=registry,
    )

    await registry.commands["status"].handler(ctx, "")

    lines: list[str] = []
    for call in ctx.console.output:
        for value in call:
            lines.extend(_printed_text(value).splitlines())
    assert [line.partition(":")[0] for line in lines] == [
        "model",
        "effort",
        "mode",
        "cwd",
        "git",
        "ctx",
        "tokens",
        "cost",
    ]
    assert all(" · " not in line for line in lines)
    assert all(
        "<style" not in str(value)
        for call in ctx.console.output
        for value in call
    )


def test_disabled_statusbar_registers_nothing_and_toolbar_is_empty(tmp_path: Path) -> None:
    registry = Registry()
    bus = EventBus()
    statusbar.register(
        _api(registry, bus, {}, config={"statusbar": False, "icons": True, "pricing": {}})
    )

    assert getattr(registry, "status_segments", []) == []
    assert "status" not in registry.commands
    assert bus.handlers == []

    enabled_api = _api(registry, bus, {})
    enabled_api.add_status_segment("sentinel", lambda _ctx: "must not render")
    ctx = _ctx(tmp_path, cfg=_cfg(tmp_path, statusbar_enabled=False), registry=registry)
    assert _bottom_toolbar(ctx) in (None, "", [])
