"""Command-line entry point for orcha."""

from __future__ import annotations

import asyncio

from .core.config import load_config
from .tui.app import run_app


def main() -> None:
    """Load configuration and run the terminal application."""

    raise SystemExit(asyncio.run(run_app(load_config())))


if __name__ == "__main__":
    main()
