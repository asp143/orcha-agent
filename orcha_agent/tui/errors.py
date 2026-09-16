"""Concise provider failures suitable for a terminal banner."""

from __future__ import annotations

import math
import re
import time


def humanize_error(exc: BaseException, *, now: float | None = None) -> str:
    message = str(exc).strip() or "An unexpected error occurred"
    lowered = message.lower()
    name = type(exc).__name__.lower()
    if "usage limit" in lowered:
        plan = re.search(r"plan:\s*([\w-]+)", message, re.I)
        reset = re.search(r"resets at:\s*(\d+(?:\.\d+)?)", message, re.I)
        result = "Codex usage limit reached"
        if plan:
            result += f" ({plan[1].title()} plan)"
        if reset:
            minutes = max(
                0, math.ceil((float(reset[1]) - (time.time() if now is None else now)) / 60)
            )
            hours, minutes = divmod(minutes, 60)
            result += f" · resets in {hours}h {minutes}m" if hours else f" · resets in {minutes}m"
        return result
    if any(
        value in lowered
        for value in (
            "token expired",
            "session expired",
            "auth expired",
            "refresh token",
            "refresh_token",
        )
    ):
        return "Session expired · run /login codex"
    if "authentication" in name:
        return "Authentication failed · check provider credentials"
    if "ratelimit" in name or "rate limit" in lowered or getattr(exc, "status_code", None) == 429:
        delay = getattr(exc, "retry_after", None)
        match = re.search(r"retry(?:ing)?(?: after| in|[-_ ]after:?)\s*(\d+)", lowered)
        if delay is None and match:
            delay = match[1]
        return (
            f"Rate limited · retrying in {delay}s"
            if delay is not None
            else "Rate limited · try again shortly"
        )
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(
        value in name for value in ("connection", "network", "timeout")
    ):
        return "Network error · check connection"
    return message
