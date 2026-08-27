"""Pluggable omp-style terminal status bar."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from orcha_agent.core.events import ModelChunk
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="statusbar", version="1.0.0")

WINDOWS = {
    "codex:gpt-5.6-sol": 272_000,
    "codex:gpt-5.6-luna": 272_000,
    "codex:gpt-5.6-terra": 272_000,
    "codex:gpt-5.5": 272_000,
    "codex:gpt-5.4": 272_000,
    "codex:gpt-5.4-mini": 272_000,
    "codex:gpt-5.3-codex-spark": 128_000,
}

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "codex:gpt-5.6-sol": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-luna": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-terra": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.5": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4-mini": {"input": 1.5, "output": 6, "cache_read": 0.15},
    "codex:gpt-5.3-codex-spark": {"input": 1.5, "output": 6, "cache_read": 0.15},
    "anthropic:claude-opus-5": {"input": 15, "output": 75, "cache_read": 1.5},
    "anthropic:claude-sonnet-5": {"input": 3, "output": 15, "cache_read": 0.3},
    "anthropic:claude-haiku-4-5": {"input": 0.8, "output": 4, "cache_read": 0.08},
}

ICONS = {
    "model": "󰚩",
    "effort": "󰪣",
    "mode": "󰘧",
    "cwd": "",
    "git": "",
    "context": "󰍛",
    "tokens": "󰁨",
    "cost": "󰙺",
}


def _state(ctx: Any) -> dict[str, Any]:
    return ctx.plugin_states.setdefault("statusbar", {})


def _icons(ctx: Any) -> bool:
    if not bool(getattr(ctx.cfg, "icons", True)):
        return False
    encoding = str(getattr(ctx.console, "encoding", "") or "").lower()
    if not encoding:
        encoding = str(
            getattr(getattr(ctx.console, "console", None), "encoding", "utf-8")
            or ""
        ).lower()
    return "utf" in encoding


def _label(ctx: Any, name: str, value: str) -> str:
    prefix = f"{ICONS[name]} " if _icons(ctx) else f"{name if name != 'context' else 'ctx'}: "
    return f"{prefix}{value}"


def _styled(ctx: Any, name: str, value: str, color: str) -> str:
    return f'<style fg="{color}">{html.escape(_label(ctx, name, value))}</style>'


def _spec(ctx: Any) -> tuple[str, int]:
    configured = ctx.cfg.model
    specs = configured if isinstance(configured, list) else [configured]
    return str(specs[0]), max(0, len(specs) - 1)


def _display_model(ctx: Any, spec: str) -> str:
    prefix, separator, model_name = spec.partition(":")
    if not separator:
        return spec
    provider = ctx.registry.providers.get(prefix)
    if provider is None or model_name not in provider.models:
        return spec
    if model_name.startswith("gpt-"):
        parts = model_name.split("-")
        return " ".join(
            [f"GPT-{parts[1]}", *(part.capitalize() for part in parts[2:])]
        )
    return " ".join(part.capitalize() for part in model_name.split("-"))


def model_segment(ctx: Any) -> str:
    spec, fallbacks = _spec(ctx)
    value = _display_model(ctx, spec)
    if fallbacks:
        value = f"{value} +{fallbacks}"
    return _styled(ctx, "model", value, "ansicyan")


def effort_segment(ctx: Any) -> str | None:
    spec, _ = _spec(ctx)
    prefix = spec.partition(":")[0]
    provider_config = getattr(ctx.cfg, "providers", {}).get(prefix, {})
    effort = provider_config.get("reasoning_effort") or provider_config.get("thinking")
    if isinstance(effort, Mapping):
        effort = effort.get("effort") or effort.get("type")
    if not effort:
        return None
    return _styled(ctx, "effort", str(effort), "ansimagenta")

def mode_segment(ctx: Any) -> str:
    mode = str(ctx.cfg.mode)
    return _styled(ctx, "mode", mode, "ansired" if mode == "yolo" else "ansiyellow")


def cwd_segment(ctx: Any) -> str:
    cwd = Path(ctx.cfg.cwd)
    width = int(getattr(ctx.console, "width", getattr(ctx.console.console, "width", 80)))
    value = f"{cwd.parent.name}/{cwd.name}" if width >= 120 else cwd.name
    return _styled(ctx, "cwd", value, "ansiblue")


def _parse_git(stdout: str) -> str | None:
    lines = stdout.splitlines()
    if not lines or not lines[0].startswith("## "):
        return None
    branch = lines[0][3:].split("...", 1)[0].split(" ", 1)[0]
    untracked = sum(line.startswith("??") for line in lines[1:])
    modified = sum(not line.startswith("??") for line in lines[1:] if line)
    suffix = ""
    if untracked:
        suffix += f" ?{untracked}"
    if modified:
        suffix += f" +{modified}"
    return f"{branch}{suffix}"


def git_segment(ctx: Any) -> str | None:
    state = _state(ctx)
    now = monotonic()
    cached_at = state.get("_git_at")
    if isinstance(cached_at, (int, float)) and now - cached_at < 2:
        return state.get("_git_value")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-b"],
            cwd=ctx.cfg.cwd,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        value = _parse_git(result.stdout) if result.returncode == 0 else None
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        value = None
    rendered = _styled(ctx, "git", value, "ansigreen") if value else None
    state["_git_at"] = now
    state["_git_value"] = rendered
    return rendered


def _window(ctx: Any, spec: str) -> int | None:
    if spec in WINDOWS:
        return WINDOWS[spec]
    prefix, _, model_name = spec.partition(":")
    lowered = model_name.lower()
    if prefix == "codex":
        if lowered.startswith("gpt-5.6-") or lowered.startswith("gpt-5.5") or lowered.startswith("gpt-5.4"):
            return 272_000
        if "spark" in lowered:
            return 128_000
    if prefix == "anthropic":
        if "opus" in lowered or "sonnet" in lowered:
            return 1_000_000
        if "haiku" in lowered:
            return 200_000
    provider = ctx.registry.providers.get(prefix)
    return None if provider is None else provider.capabilities.max_context


def _quantity(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def context_segment(ctx: Any) -> str | None:
    spec, _ = _spec(ctx)
    window = _window(ctx, spec)
    if not window:
        return None
    used = int(_state(ctx).get("last_input_tokens", 0))
    percent = used / window * 100
    color = "ansigreen" if percent < 60 else "ansiyellow" if percent < 85 else "ansired"
    return _styled(ctx, "context", f"{percent:.1f}%/{_quantity(window)}", color)


def tokens_segment(ctx: Any) -> str:
    state = _state(ctx)
    value = f"{_quantity(float(state.get('input_tokens', 0)))}↑ {_quantity(float(state.get('output_tokens', 0)))}↓"
    return _styled(ctx, "tokens", value, "ansiwhite")


def cost_segment(ctx: Any) -> str | None:
    spec, _ = _spec(ctx)
    price = {**DEFAULT_PRICING.get(spec, {}), **getattr(ctx.cfg, "pricing", {}).get(spec, {})}
    if not price:
        return None
    state = _state(ctx)
    inputs = float(state.get("input_tokens", 0))
    outputs = float(state.get("output_tokens", 0))
    cache = float(state.get("cache_read_tokens", 0))
    cost = (
        max(0.0, inputs - cache) * float(price.get("input", 0))
        + cache * float(price.get("cache_read", price.get("input", 0)))
        + outputs * float(price.get("output", 0))
    ) / 1_000_000
    return _styled(ctx, "cost", f"${cost:.2f}", "ansigreen")


def _usage(event: ModelChunk, state: dict[str, Any]) -> None:
    usage = getattr(event.chunk, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        return
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    details = usage.get("input_token_details", {})
    cache_read = int(details.get("cache_read", 0)) if isinstance(details, Mapping) else 0
    state["input_tokens"] = int(state.get("input_tokens", 0)) + input_tokens
    state["output_tokens"] = int(state.get("output_tokens", 0)) + output_tokens
    state["cache_read_tokens"] = int(state.get("cache_read_tokens", 0)) + cache_read
    state["last_input_tokens"] = input_tokens


SEGMENTS = (
    ("model", model_segment),
    ("effort", effort_segment),
    ("mode", mode_segment),
    ("cwd", cwd_segment),
    ("git", git_segment),
    ("context", context_segment),
    ("tokens", tokens_segment),
    ("cost", cost_segment),
)


def register(api: PluginAPI) -> None:
    if not bool(api.config.get("statusbar", True)):
        return
    for priority, (name, render) in enumerate(SEGMENTS, start=1):
        api.add_status_segment(name, render, priority=priority * 10)

    async def track(event: ModelChunk) -> None:
        _usage(event, api.state)

    async def show(ctx: Any, _args: str) -> None:
        for name, render in SEGMENTS:
            try:
                value = render(ctx)
            except Exception:
                value = f"!{name}"
            if value:
                ctx.console.print(value)

    api.on(ModelChunk, track, priority=10)
    api.add_command("status", show, help="Show status bar segments")
