"""Produce the first committed welcome block."""

from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from orcha_agent.core.events import AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="banner", version="1.0.0")

_LOGO = (
    " ██████╗ ██████╗  ██████╗██╗  ██╗ █████╗ ",
    "██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗",
    "██║   ██║██████╔╝██║     ███████║███████║",
    "██║   ██║██╔══██╗██║     ██╔══██║██╔══██║",
    "╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║",
)
_ASCII_LOGO = (
    " OOO  RRR   CCC H H  AAA ",
    "O   O R  R C    H H A   A",
    "O   O RRR  C    HHH AAAAA",
    "O   O R R  C    H H A   A",
    " OOO  R  R  CCC H H A   A",
)
_GRADIENT = ((137, 180, 250), (203, 166, 247), (245, 194, 231))


def _short_cwd(path: str | Path) -> str:
    cwd = Path(path)
    try:
        home = Path.home()
        relative = cwd.relative_to(home)
    except (ValueError, RuntimeError):
        return str(cwd)
    return "~" if not relative.parts else f"~/{relative.as_posix()}"


def _ascii_output(ctx: Any) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return True
    cfg = getattr(ctx, "cfg", None)
    if getattr(cfg, "symbols", "unicode") == "ascii":
        return True
    console = getattr(ctx, "console", None)
    target = getattr(console, "console", console)
    if int(getattr(target, "width", 80)) < 76:
        return True
    encoding_value = getattr(target, "encoding", "utf-8")
    try:
        encoding = encoding_value() if callable(encoding_value) else encoding_value
    except Exception:
        encoding = ""
    return bool(encoding and "utf" not in str(encoding).casefold())


def _interpolate(start: tuple[int, int, int], end: tuple[int, int, int], value: float) -> str:
    rgb = tuple(round(left + (right - left) * value) for left, right in zip(start, end))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _gradient_styles(lines: Iterable[str]) -> list[list[str]]:
    materialized = list(lines)
    denominator = max(1, len(materialized) + max(map(len, materialized), default=1) - 2)
    styles: list[list[str]] = []
    for row, line in enumerate(materialized):
        row_styles: list[str] = []
        for column, _character in enumerate(line):
            position = (row + column) / denominator
            if position <= 0.5:
                color = _interpolate(_GRADIENT[0], _GRADIENT[1], position * 2)
            else:
                color = _interpolate(_GRADIENT[1], _GRADIENT[2], (position - 0.5) * 2)
            row_styles.append(f"bold {color}")
        styles.append(row_styles)
    return styles


def _tip_lines() -> list[str]:
    try:
        text = resources.files("orcha_agent.tui").joinpath("tips.txt").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def choose_tip(tips: Iterable[str] | None = None, *, rng: Any = random) -> str:
    """Choose one packaged tip, giving ``[NEW]`` entries four times the weight."""

    weighted: list[str] = []
    for raw in _tip_lines() if tips is None else tips:
        tip = str(raw).strip()
        if not tip:
            continue
        is_new = tip.startswith("[NEW]")
        display = tip.removeprefix("[NEW]").strip()
        weighted.extend([display] * (4 if is_new else 1))
    return rng.choice(weighted) if weighted else "Type /help to see available commands."


def _age(created: str, now: datetime) -> str:
    try:
        timestamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        seconds = max(0, int((now - timestamp).total_seconds()))
    except (TypeError, ValueError):
        return "recently"
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _recent_sessions(ctx: Any, *, ascii_only: bool, now: datetime) -> list[str]:
    store = getattr(ctx, "session", None)
    try:
        sessions = list(store.list()) if store is not None else []
    except Exception:
        sessions = []
    bullet = "*" if ascii_only else "•"
    current = getattr(ctx, "session_id", None)
    recent = [session for session in sessions if getattr(session, "thread_id", None) != current]
    slots = [
        f"{bullet} {getattr(session, 'title', None) or 'Untitled'} ({_age(getattr(session, 'created', ''), now)})"
        for session in recent[:4]
    ]
    return [*slots, *([""] * (4 - len(slots)))]


def _provider_prefix(ctx: Any, model: str) -> str:
    spec: object = model.split(",", 1)[0]
    aliases = getattr(getattr(ctx, "cfg", None), "models", {})
    seen: set[str] = set()
    while isinstance(spec, str) and isinstance(aliases, Mapping):
        target = aliases.get(spec)
        if target is None or spec in seen:
            break
        seen.add(spec)
        spec = target[0] if isinstance(target, list) and target else target
    return spec.split(":", 1)[0] if isinstance(spec, str) and ":" in spec else "default"


def _provider_ready(ctx: Any, prefix: str) -> bool:
    registry = getattr(ctx, "registry", None)
    providers = getattr(registry, "providers", {})
    provider = providers.get(prefix)
    if provider is None:
        return False
    try:
        if provider.available() is not None:
            return False
    except Exception:
        return False
    env_keys = getattr(provider, "env_keys", ())
    if env_keys and not any(os.environ.get(key) for key in env_keys):
        return False
    auth = getattr(registry, "auth", {}).get(prefix)
    return auth is None or auth.flow.status() != "not logged in"


def _key_hint(ctx: Any) -> tuple[tuple[str, ...], str] | str:
    effective = getattr(getattr(ctx, "ui", None), "effective_keys", {})
    value = effective.get("expand_tools", ()) if isinstance(effective, Mapping) else ()
    if isinstance(value, str):
        bindings = (value,)
    elif isinstance(value, Iterable):
        bindings = tuple(str(binding) for binding in value)
    else:
        bindings = ()
    return (bindings, "expand tool output") if bindings else ""


def _hints(ctx: Any, model: str) -> list[Any]:
    cfg = getattr(ctx, "cfg", None)
    trusted = bool(getattr(cfg, "trust_cwd", False))
    trust = "✓ Trusted folder" if trusted else "Untrusted folder · project config skipped"
    plugins = getattr(ctx, "plugins", ()) or ()
    loaded = sum(1 for plugin in plugins if getattr(plugin, "error", None) is None)
    plugin_hint = f"{loaded} plugin{'s' if loaded != 1 else ''} loaded"
    provider = _provider_prefix(ctx, model)
    provider_hint = f"{provider} provider {'ready' if _provider_ready(ctx, provider) else 'unavailable'}"
    return [trust, plugin_hint, provider_hint, _key_hint(ctx)]


def build_welcome(
    ctx: Any,
    *,
    rng: Any = random,
    tips: Iterable[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build deterministic renderer data for the committed welcome block."""

    cfg = getattr(ctx, "cfg", None)
    raw_model = getattr(cfg, "model", "")
    model = ", ".join(map(str, raw_model)) if isinstance(raw_model, list) else str(raw_model)
    ascii_only = _ascii_output(ctx)
    logo = _ASCII_LOGO if ascii_only else _LOGO
    return {
        "logo": list(logo),
        "logo_styles": [[None] * len(line) for line in logo] if ascii_only else _gradient_styles(logo),
        "model": model or "not selected",
        "mode": str(getattr(cfg, "mode", "ask")),
        "cwd": _short_cwd(getattr(cfg, "cwd", Path.cwd())),
        "sessions": _recent_sessions(ctx, ascii_only=ascii_only, now=now or datetime.now(timezone.utc)),
        "hints": _hints(ctx, model),
        "tip": choose_tip(tips, rng=rng),
        "ascii": ascii_only,
    }


def _enabled(ctx: Any) -> bool:
    return bool(getattr(getattr(ctx, "cfg", None), "banner", True)) and os.environ.get("ORCHA_NO_BANNER") != "1"


def register(api: PluginAPI) -> None:
    async def show(event: AppStart) -> None:
        ctx = event.ctx
        if not _enabled(ctx):
            return
        transcript = getattr(ctx, "transcript", None)
        if transcript is not None and hasattr(transcript, "append_welcome"):
            transcript.append_welcome(build_welcome(ctx), immediate=True)

    api.on(AppStart, show, priority=100)


__all__ = ["PLUGIN", "build_welcome", "choose_tip", "register"]
