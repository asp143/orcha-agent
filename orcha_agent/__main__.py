"""Command-line entry point for orcha."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from .core.config import load_config


def _run_sync(cfg: object) -> int:
    """Synchronize the configured Turso replica without starting the TUI."""

    from .core.persistence import open_session_store
    from .tui.console import ConsoleOutput

    console = ConsoleOutput()
    try:
        store = open_session_store(
            cfg,
            initial_sync=True,
            sync_on_close=False,
        )
        with store:
            if not bool(getattr(store, "supports_sync", False)):
                console.error("Sync is only available with the Turso persistence backend.")
                return 1
    except Exception as exc:
        console.error(str(exc))
        return 1
    console.print("Turso sync complete (sessions and structured memories).")
    return 0


async def _run_login(cfg: object) -> int:
    from .core.events import EventBus
    from .core.loader import load_plugins
    from .core.registry import Registry
    from .tui.console import ConsoleOutput

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
        from .tui.gallery import run_gallery

        raise SystemExit(run_gallery(cfg))
    if cfg.command == "stats":
        from .core.persistence import open_session_store
        from .core.usage_store import UsageStore, usage_table
        from .tui.console import ConsoleOutput

        console = ConsoleOutput()
        with open_session_store(cfg) as store:
            session_id = cfg.stats_session
            if session_id:
                try:
                    session_id = store.resolve_session(session_id).thread_id
                except LookupError as exc:
                    console.error(str(exc))
                    raise SystemExit(2) from None
            if cfg.stats_period == "session" and not session_id:
                console.error("Use orcha stats session --session SESSION")
                raise SystemExit(2)
            console.print(
                usage_table(
                    UsageStore(store).report(cfg.stats_period, session_id), cfg.stats_period
                )
            )
        raise SystemExit(0)
    if cfg.trust_cwd:
        from dotenv import load_dotenv

        load_dotenv(cfg.cwd / ".env", override=False)
    if cfg.command == "login":
        raise SystemExit(asyncio.run(_run_login(cfg)))
    if cfg.command == "sync":
        raise SystemExit(_run_sync(cfg))
    from .tui.app import run_app
    from .core.models import role_fallback_notices
    from .tui.console import ConsoleOutput
    from rich.text import Text

    for notice in role_fallback_notices(cfg.model, cfg):
        ConsoleOutput().print(Text(notice))
    raise SystemExit(asyncio.run(run_app(cfg)))


if __name__ == "__main__":
    main()
