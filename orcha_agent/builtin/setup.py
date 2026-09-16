"""Small, skippable first-run setup using the ordinary overlay extension point."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="setup", version="1.0.0")


def config_path(ctx: Any) -> Path:
    return ctx.cfg.user_config_path or Path.home() / ".config/orcha-agent/config.toml"


def should_setup(ctx: Any, *, interactive: bool | None = None) -> bool:
    """Do not mistake an installed generic adapter for configured credentials."""
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        return False
    if ctx.cfg.command == "setup":
        return True
    if ctx.cfg.resume or config_path(ctx).exists():
        return False
    if (ctx.cfg.cwd / ".orcha-agent/config.toml").exists():
        return False
    from orcha_agent.builtin.commands_core import _provider_usable

    for prefix, provider in ctx.registry.providers.items():
        if prefix == "langchain":
            continue
        # A credential-free adapter with no known models is not a configured server.
        if not provider.env_keys and prefix not in ctx.registry.auth and not provider.models:
            continue
        try:
            if _provider_usable(ctx, prefix):
                return False
        except Exception:
            continue
    return True


def setup_picker(_ctx: Any, *, title: str, items: tuple[str, ...]) -> Any:
    from orcha_agent.tui.overlays.select import SelectList

    return SelectList(f"Setup · {title} · Esc to skip", items)


async def run_setup(ctx: Any, _args: str = "") -> None:
    """Collect preferences before saving; cancellation never writes partial setup."""
    from orcha_agent.tui.overlays.settings import persist_setting

    async def pick(title: str, items: tuple[str, ...]) -> str | None:
        result = await ctx.ui.show("setup", title=title, items=items)
        return result if isinstance(result, str) and result in items else None

    themes = tuple(getattr(ctx.ui, "themes", {})) or ("dark", "light")
    theme = await pick("1/4 Theme", themes)
    if theme is None:
        return
    composer = await pick("2/4 Composer", ("box", "claude", "borderless", "band", "rail"))
    if composer is None:
        return
    prefix = await pick(
        "3/4 Provider",
        tuple(sorted(name for name in ctx.registry.providers if name != "langchain")),
    )
    if prefix is None:
        return
    provider = ctx.registry.providers[prefix]
    auth = ctx.registry.auth.get(prefix)
    if auth is not None:
        action = await pick("Provider sign-in", ("Sign in now", "Configure later"))
        if action is None:
            return
        if action == "Sign in now":
            try:
                await auth.flow.login(ctx, "auto")
            except Exception:
                ctx.console.warning("Sign-in did not complete. Use /login to retry.")
    elif provider.env_keys:
        hint = "Set " + " or ".join(provider.env_keys) + " in your shell, then restart orcha."
        if await pick(hint, ("Continue (API keys are never saved)",)) is None:
            return
    else:
        if (
            await pick("Configure your local or custom provider outside the wizard", ("Continue",))
            is None
        ):
            return
    models = tuple(f"{prefix}:{name}" for name in provider.models)
    if not models and provider.default_model:
        models = (f"{prefix}:{provider.default_model}",)
    current = ctx.cfg.model
    if isinstance(current, str) and current.startswith(prefix + ":") and current not in models:
        models = (*models, current)
    model = await pick("4/4 Model", (*models, "Enter model name", "Choose later with /model"))
    if model == "Enter model name":
        answer = await ctx.ui.ask(
            [
                {
                    "id": "model",
                    "question": f"Model name for {prefix} (use Custom input)",
                    "options": [],
                }
            ]
        )
        results = answer.get("results", []) if isinstance(answer, dict) else []
        raw = results[0].get("customInput", "").strip() if results else ""
        if not raw or any(char.isspace() for char in raw):
            return
        model = raw if raw.startswith(prefix + ":") else f"{prefix}:{raw}"
    if model is None:
        return
    path = config_path(ctx)
    persist_setting(path, "tui", "theme", theme)
    persist_setting(path, "tui", "composer", composer)
    if model != "Choose later with /model":
        persist_setting(path, "core", "model", model)
        try:
            await ctx.switch_model(model)
        except RuntimeError:
            # Keep the active session and its ledger consistent when credentials
            # or an optional adapter are not installed yet. The saved default is
            # applied on the next launch, once the provider has been configured.
            ctx.console.warning(
                "Model saved for the next launch. Configure the provider, then use /model to switch this session."
            )
    ctx.cfg = replace(ctx.cfg, theme=theme, composer=composer)
    ctx.ui.set_theme(theme)
    ctx.ui.apply_settings(ctx.cfg)
    ctx.console.print(f"Setup saved to {path}. Use /setup to run it again.")


async def startup_setup(ctx: Any) -> None:
    if not should_setup(ctx):
        return
    try:
        await run_setup(ctx)
    except (OSError, ValueError) as exc:
        ctx.console.warning(f"Setup could not save preferences: {exc}")


def register(api: PluginAPI) -> None:
    api.add_overlay("setup", setup_picker)
    api.add_command("setup", run_setup, "Run the theme, composer, provider and model setup wizard")
