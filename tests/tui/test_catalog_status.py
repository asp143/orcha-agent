from __future__ import annotations

from types import SimpleNamespace

import pytest

from orcha_agent.core.compaction import CompactionConfig
from orcha_agent.core.registry import Registry
from orcha_agent.tui.statusline import (
    _gauge,
    compaction_segment,
    context_segment,
    cost_segment,
    usage_segment,
)


def context(*, used: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            model="custom:large",
            compaction=CompactionConfig(threshold_tokens=50_000),
        ),
        registry=Registry(),
        plugin_states={
            "statusbar": {
                "_model_spec": "custom:large",
                "_model_window": 100_000,
                "last_input_tokens": used,
            }
        },
    )


def test_gauge_uses_cached_catalog_window_and_configured_threshold(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("paint must not query files")

    monkeypatch.setattr("orcha_agent.core.catalog.get_model", forbidden)
    ctx = context(used=46_000)
    segment = context_segment(ctx)
    assert segment.text == "46.0%/100k"
    assert segment.token == "warning"
    theme = SimpleNamespace(colors={"warning": "#ffaa00", "statusLineSep": "#333333"})
    fragments = _gauge(segment, 24, theme, transparent=True, ascii_mode=True)
    assert "".join(text for _, text in fragments) == "#########----------- 46%"
    assert "class:warning" in fragments[0][0]
    ctx.plugin_states["statusbar"]["last_input_tokens"] = 51_000
    assert context_segment(ctx).token == "error"


def test_usage_snapshot_survives_model_switch_without_repricing():
    ctx = context()
    ctx.plugin_states["usage"] = {"session_cost": 1.23, "today_cost": 4.56}
    assert usage_segment(ctx).text == "$1.23 session · $4.56 today"
    assert cost_segment(ctx).text == "$1.23"
    ctx.cfg.model = "custom:cheap"
    assert usage_segment(ctx).text == "$1.23 session · $4.56 today"
    assert cost_segment(ctx).text == "$1.23"


def test_compaction_status_is_visible_only_while_present():
    ctx = context()
    assert compaction_segment(ctx) is None
    ctx.compaction_status = "summarizing"
    assert compaction_segment(ctx).text == "/compact summarizing"
    assert compaction_segment(ctx).token == "warning"


def test_pricing_uses_catalog_cache_rates_and_explicit_override(monkeypatch):
    from orcha_agent.core.usage import usage_cost

    monkeypatch.setattr(
        "orcha_agent.core.catalog.get_model",
        lambda *args: SimpleNamespace(
            cost={"input": 2, "output": 10, "cache_read": 0.2, "cache_write": 3}
        ),
    )
    usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_token_details": {"cache_read": 300, "cache_creation": 100},
    }
    assert usage_cost("custom:large", usage, {}) == pytest.approx(0.00256)
    assert usage_cost("custom:large", usage, {"custom:large": {"output": 20}}) == pytest.approx(
        0.00356
    )


@pytest.mark.asyncio
async def test_snapshot_resolves_role_and_user_override_outside_paint(tmp_path, monkeypatch):
    from orcha_agent.core.config import load_config
    from orcha_agent.tui.statusline import refresh_model_snapshot

    config_path = tmp_path / "config.toml"
    config_path.write_text('[core]\nmodel="@smol"\n[model_roles]\nsmol="openai:custom"\n')
    (tmp_path / "models.yml").write_text(
        "providers:\n  openai:\n    models:\n      custom:\n"
        "        context_window: 64000\n        cost: {input: 2, output: 4}\n"
    )
    cfg = load_config([], env={"HOME": str(tmp_path)}, user_config_path=config_path)
    ctx = context(used=16_000)
    ctx.cfg = cfg
    await refresh_model_snapshot(ctx)

    def forbidden(*args, **kwargs):
        raise AssertionError("paint must consume the snapshot")

    monkeypatch.setattr("orcha_agent.core.catalog.get_model", forbidden)
    assert context_segment(ctx).text == "25.0%/64k"
    ctx.plugin_states["statusbar"].update(input_tokens=1_000_000, output_tokens=0)
    assert cost_segment(ctx).text == "$2.00"


def test_stream_labels_keep_resolved_catalog_provider():
    from langchain_core.messages import AIMessage

    from orcha_agent.tui.turn import _model_name

    assert (
        _model_name(
            AIMessage(content="answer"),
            {
                "orcha_model": "codex:gpt-5.6-sol",
                "ls_provider": "openai",
                "ls_model_name": "gpt-5.6-sol",
            },
        )
        == "codex:gpt-5.6-sol"
    )


@pytest.mark.asyncio
async def test_compaction_state_markers_do_not_render_as_assistant_output():
    from langchain_core.messages import AIMessageChunk, HumanMessage

    from orcha_agent.core.events import EventBus, ModelChunk
    from orcha_agent.tui.turn import _message_event, _ModelLabelBuffer, _ToolCallBuffer

    bus = EventBus()
    seen = []

    async def collect(event):
        seen.append(event.chunk.content)

    bus.on(ModelChunk, collect, plugin="test")
    ctx = SimpleNamespace(bus=bus)
    calls, labels = _ToolCallBuffer(), _ModelLabelBuffer()
    await _message_event(
        ctx,
        (
            HumanMessage(
                content="INTERNAL_SUMMARY", additional_kwargs={"lc_source": "summarization"}
            ),
            {},
        ),
        calls,
        labels,
    )
    await _message_event(ctx, (AIMessageChunk(content="visible answer"), {}), calls, labels)
    assert seen == ["visible answer"]
