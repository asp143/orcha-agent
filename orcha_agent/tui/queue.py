"""Ordered prompt queue and batch-submission parsing."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

_ARROW = re.compile(r"^\s*(?:->|=>)\s?(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def split_submission(text: str) -> list[str]:
    """Split an explicitly batched submission, otherwise preserve it verbatim."""

    lines = text.splitlines()
    if not lines:
        return [text]
    arrow_matches = [_ARROW.match(line) for line in lines]
    if all(match is not None for match in arrow_matches):
        return [match.group(1).strip() for match in arrow_matches if match.group(1).strip()]

    first = _NUMBERED.match(lines[0])
    if first is None:
        return [text]
    items: list[list[str]] = []
    expected = int(first.group(1))
    for line in lines:
        match = _NUMBERED.match(line)
        if match is not None:
            number = int(match.group(1))
            if number != expected:
                return [text]
            expected += 1
            items.append([match.group(2).strip()])
        elif items and line.strip():
            items[-1].append(line.strip())
        elif line.strip():
            return [text]
    return ["\n".join(item) for item in items if any(item)]


class PromptQueue:
    """Small FIFO queue with explicit recovery operations."""

    def __init__(self) -> None:
        self._items: deque[str] = deque()

    @property
    def items(self) -> tuple[str, ...]:
        return tuple(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, prompt: str) -> None:
        prompt = prompt.strip()
        if prompt:
            self._items.append(prompt)

    def extend(self, prompts: Iterable[str]) -> None:
        for prompt in prompts:
            self.append(prompt)

    def pop(self) -> str | None:
        return self._items.popleft() if self._items else None

    def pop_last(self) -> str | None:
        return self._items.pop() if self._items else None

    def clear(self) -> None:
        self._items.clear()

    def restore_text(self) -> str:
        items = tuple(self._items)
        self._items.clear()
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return "\n".join(f"-> {item}" for item in items)


__all__ = ["PromptQueue", "split_submission"]
