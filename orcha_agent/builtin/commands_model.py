"""Model and operating-mode slash commands."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from orcha_agent.core.config import normalize_model_spec, save_core_value
from orcha_agent.core.events import AppStart, ModelSwitch
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="commands_model", version="1.0.0")


async def _model(ctx: Any, args: str) -> None:
    if not args.strip():
        cfg = ctx.cfg
        current = (
            cfg.model if isinstance(cfg.model, str) else ",".join(cfg.model)
        )
        subagent_value = cfg.subagent_model or cfg.model
        summarizer_value = cfg.summarizer_model or cfg.model
        subagent = (
            subagent_value
            if isinstance(subagent_value, str)
            else ",".join(subagent_value)
        )
        summarizer = (
            summarizer_value
            if isinstance(summarizer_value, str)
            else ",".join(summarizer_value)
        )
        ctx.console.print(f"Current model: {current}")
        ctx.console.print(
            f"Subagent model: {subagent} "
            f"({'inherited' if cfg.subagent_model is None else 'explicit'})"
        )
        ctx.console.print(
            f"Summarizer model: {summarizer} "
            f"({'inherited' if cfg.summarizer_model is None else 'explicit'})"
        )
        ctx.console.print("Usage: /model <provider:model>[,<provider:model>...]")
        return
    try:
        spec = normalize_model_spec(args.strip())
    except ValueError:
        ctx.console.error("Usage: /model <provider:model>[,<provider:model>...]")
        return
    specs = [spec] if isinstance(spec, str) else spec
    if any(any(character.isspace() for character in item) for item in specs):
        ctx.console.error("Usage: /model <provider:model>[,<provider:model>...]")
        return
    try:
        await ctx.switch_model(spec)
    except Exception as exc:
        reporter = getattr(ctx, "report_provider_error", None)
        if reporter is not None:
            reporter(exc)
        else:
            ctx.console.error(f"{type(exc).__name__}: {exc}")


async def _mode(ctx: Any, args: str) -> None:
    name = args.strip()
    if not name or any(character.isspace() for character in name):
        ctx.console.error("Usage: /mode <name>")
        return
    await ctx.switch_mode(name)


def register(api: PluginAPI) -> None:
    app: dict[str, Any] = {}

    async def remember_app(event: AppStart) -> None:
        app["ctx"] = event.ctx

    async def remember_model(_event: ModelSwitch) -> None:
        ctx = app.get("ctx")
        cfg = getattr(ctx, "cfg", None)
        path = getattr(cfg, "user_config_path", None)
        if not isinstance(path, Path):
            return
        try:
            save_core_value(path, "model", cfg.model)
        except OSError as exc:
            ctx.console.warning(f"Could not remember model in {path}: {exc}")

    api.on(AppStart, remember_app)
    api.on(ModelSwitch, remember_model)
    api.add_command("model", _model, help="Switch models: /model <provider:model>")
    api.add_command("mode", _mode, help="Switch operating modes: /mode <name>")
