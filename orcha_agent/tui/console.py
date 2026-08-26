"""Rich console output helpers used by the terminal UI and plugins."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel


class ConsoleOutput:
    """Small, injectable facade around :class:`rich.console.Console`."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self.console.print(*objects, **kwargs)

    def error(self, message: str) -> None:
        self.console.print(Panel(message, title="Error", border_style="red"))

    def warning(self, message: str) -> None:
        self.console.print(Panel(message, title="Warning", border_style="yellow"))


__all__ = ["ConsoleOutput"]
