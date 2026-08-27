"""Startup banner rendered through the plugin event API."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

from orcha_agent.core.events import AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="banner", version="1.0.0")

WIDE_ART = (
    "  ██████╗ ██████╗  ██████╗██╗  ██╗ █████╗",
    " ██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗",
    " ██║   ██║██████╔╝██║     ███████║███████║",
    " ██║   ██║██╔══██╗██║     ██╔══██║██╔══██║",
    " ╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║",
    "  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝",
)


def _short_cwd(path: str | Path) -> str:
    cwd = Path(path)
    home = Path.home()
    try:
        relative = cwd.relative_to(home)
    except ValueError:
        return str(cwd)
    return "~" if not relative.parts else f"~/{relative.as_posix()}"


def _model_spec(value: Any, *, plain: bool) -> str:
    if value is None or value == "" or value == []:
        return "(none) - /model or /login codex" if plain else "(none) — /model or /login codex"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _plain_terminal(console: Any) -> bool:
    width = int(getattr(console, "width", 80))
    encoding = str(getattr(console, "encoding", "utf-8") or "").lower()
    return (
        width < 50
        or bool(os.environ.get("NO_COLOR"))
        or "utf" not in encoding
    )


def _codex_recovery(registry: Any) -> str:
    auth = getattr(registry, "auth", {}).get("codex")
    provider = getattr(registry, "providers", {}).get("codex")
    if (
        auth is not None
        and auth.flow.status() != "not logged in"
        and provider is not None
        and provider.default_model
    ):
        return f"/model codex:{provider.default_model}"
    return "/login codex"


def _model_availability(ctx: Any, model: Any, *, plain: bool) -> str:
    if not isinstance(model, str) or ":" not in model:
        return ""
    prefix = model.split(":", 1)[0]
    registry = getattr(ctx, "registry", None)
    providers = getattr(registry, "providers", {}) if registry is not None else {}
    provider = providers.get(prefix)
    if provider is None:
        return ""
    unavailable = provider.available()
    separator = " - " if plain else " — "
    recovery = _codex_recovery(registry)
    if unavailable:
        return f" (not configured{separator}{unavailable}, {recovery}, or /model)"
    missing_keys = [key for key in provider.env_keys if not os.environ.get(key)]
    if missing_keys:
        return (
            f" (not configured{separator}set {', '.join(missing_keys)}, "
            f"{recovery}, or /model)"
        )
    auth_entries = getattr(registry, "auth", {})
    auth = auth_entries.get(prefix)
    if auth is not None and auth.flow.status() == "not logged in":
        return f" (not configured{separator}/login {prefix}, or /model)"
    return ""


def _render(ctx: Any) -> None:
    cfg = getattr(ctx, "cfg", None)
    output = getattr(ctx, "console", None)
    if cfg is None or output is None:
        return
    if not bool(getattr(cfg, "banner", True)):
        return
    if os.environ.get("ORCHA_NO_BANNER") == "1":
        return
    console = output.console
    plain = _plain_terminal(console)
    version = importlib.metadata.version("orcha-agent")
    configured_model = getattr(cfg, "model", None)
    model = _model_spec(configured_model, plain=plain)
    model += _model_availability(ctx, configured_model, plain=plain)
    mode = str(getattr(cfg, "mode", "ask"))
    cwd = _short_cwd(getattr(cfg, "cwd", Path.cwd()))
    if plain:
        lines = [
            "ORCHA",
            f"pluggable coding agent - v{version}",
            f"model: {model}",
            f"mode: {mode}",
            f"cwd: {cwd}",
            "/help for commands",
        ]
    else:
        lines = [
            *WIDE_ART,
            f"        pluggable coding agent · v{version}",
            f"  model: {model}   mode: {mode}   cwd: {cwd}",
            "  /help for commands",
        ]
    output.print("\n".join(lines))

def register(api: PluginAPI) -> None:
    async def show(event: AppStart) -> None:
        _render(event.ctx)

    api.on(AppStart, show, priority=100)
