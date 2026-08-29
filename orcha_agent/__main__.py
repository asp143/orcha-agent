"""Command-line entry point for orcha."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from dotenv import load_dotenv

from .core.events import EventBus
from .core.loader import load_plugins
from .core.registry import Registry
from .tui.console import ConsoleOutput
from .core.config import load_config
from .tui.app import run_app
from .tui.gallery import run_gallery


async def _run_login(cfg: object) -> int:
    registry = Registry()
    bus = EventBus()
    load_plugins(registry, bus, cfg)
    prefix = getattr(cfg, "login_prefix", None)
    registration = registry.auth.get(prefix)
    console = ConsoleOutput()
    if registration is None:
        console.error(f"Unknown auth prefix: {prefix}")
        return 1
    ctx = SimpleNamespace(
        cfg=cfg,
        registry=registry,
        bus=bus,
        console=console,
    )
    try:
        await registration.flow.login(
            ctx,
            getattr(cfg, "login_mode", "auto"),
        )
    except Exception as exc:
        console.error(str(exc))
        return 1
    return 0


def main() -> None:
    """Load configuration and run the terminal application."""

    cfg = load_config()
    if cfg.command == "gallery":
        raise SystemExit(run_gallery(cfg))
    if cfg.trust_cwd:
        load_dotenv(cfg.cwd / ".env", override=False)
    if cfg.command == "login":
        raise SystemExit(asyncio.run(_run_login(cfg)))
    raise SystemExit(asyncio.run(run_app(cfg)))


if __name__ == "__main__":
    main()
