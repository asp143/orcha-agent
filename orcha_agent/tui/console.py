"""Rich console output helpers used by the terminal UI and plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from .transcript import Transcript


class ConsoleOutput:
    """Small, injectable facade around :class:`rich.console.Console`."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        transcript: Transcript | None = None,
    ) -> None:
        self.console = console or Console()
        self.transcript = transcript

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if self.transcript is not None:
            self.transcript.print(*objects, **kwargs)
            return
        self.console.print(*objects, **kwargs)

    def error(self, message: str) -> None:
        if self.transcript is not None:
            self.transcript.append_banner(message, level="error")
            return
        self.console.print(Panel(Text(message), title="Error", border_style="red"))

    def warning(self, message: str) -> None:
        if self.transcript is not None:
            self.transcript.append_banner(message, level="warning")
            return
        self.console.print(Panel(Text(message), title="Warning", border_style="yellow"))


__all__ = ["ConsoleOutput"]
