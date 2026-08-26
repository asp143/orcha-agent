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


def _render(ctx: Any) -> None:
    if not bool(getattr(ctx.cfg, "banner", True)):
        return
    if os.environ.get("ORCHA_NO_BANNER") == "1":
        return
    console = ctx.console.console
    plain = _plain_terminal(console)
    version = importlib.metadata.version("orcha-agent")
    model = _model_spec(getattr(ctx.cfg, "model", None), plain=plain)
    mode = str(getattr(ctx.cfg, "mode", "ask"))
    cwd = _short_cwd(getattr(ctx.cfg, "cwd", Path.cwd()))
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
    ctx.console.print("\n".join(lines))


def register(api: PluginAPI) -> None:
    async def show(event: AppStart) -> None:
        _render(event.ctx)

    api.on(AppStart, show, priority=100)
