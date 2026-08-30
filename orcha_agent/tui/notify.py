"""Idle-aware desktop notification delivery."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import run_in_terminal

_IDLE_SECONDS = 5.0


def _spawn(command: list[str]) -> None:
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _safe_osc_text(value: str) -> str:
    return " ".join(str(value).replace("\x1b", "").replace("\x07", "").splitlines()).strip()


class DesktopNotifier:
    """Deliver notifications only after deterministic keyboard inactivity."""

    def __init__(
        self,
        *,
        enabled: bool,
        output: Any,
        clock: Callable[[], float] = time.monotonic,
        which: Callable[[str], str | None] = shutil.which,
        spawn: Callable[[list[str]], Any] = _spawn,
        run_terminal: Callable[[Callable[[], Any]], Any] = run_in_terminal,
    ) -> None:
        self.enabled = enabled
        self.output = output
        self._clock = clock
        self._which = which
        self._spawn = spawn
        self._run_terminal = run_terminal
        self._last_keypress = clock()

    @property
    def last_keypress(self) -> float:
        return self._last_keypress

    def record_keypress(self) -> None:
        self._last_keypress = self._clock()

    @property
    def idle_seconds(self) -> float:
        return max(0.0, self._clock() - self._last_keypress)

    async def notify(self, title: str, message: str) -> bool:
        """Return whether a backend was invoked; backend failures are contained."""

        if not self.enabled or self.idle_seconds <= _IDLE_SECONDS:
            return False
        try:
            executable = self._which("notify-send")
        except (Exception, KeyboardInterrupt, asyncio.CancelledError):
            executable = None
        if executable:
            try:
                self._spawn([executable, str(title), str(message)])
            except (Exception, KeyboardInterrupt, asyncio.CancelledError):
                pass
            else:
                return True

        payload = _safe_osc_text(message)
        if not payload:
            return False

        def emit() -> None:
            self.output.write_raw(f"\x1b]9;{payload}\x07")
            self.output.flush()

        try:
            result = self._run_terminal(emit)
            if inspect.isawaitable(result):
                await result
        except (Exception, KeyboardInterrupt, asyncio.CancelledError):
            return False
        return True



__all__ = ["DesktopNotifier"]
