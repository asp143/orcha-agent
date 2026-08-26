"""Command-line entry point for orcha."""

from __future__ import annotations

import asyncio
from dotenv import load_dotenv

from .core.config import load_config
from .tui.app import run_app


def main() -> None:
    """Load configuration and run the terminal application."""

    cfg = load_config()
    load_dotenv(cfg.cwd / ".env", override=False)
    raise SystemExit(asyncio.run(run_app(cfg)))


if __name__ == "__main__":
    main()
