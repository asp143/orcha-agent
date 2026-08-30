"""Ordered prompt queue and batch-submission parsing."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

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


QueueMode = Literal["steer", "follow_up"]


@dataclass(frozen=True, slots=True)
class QueuedPrompt:
    text: str
    mode: QueueMode


class PromptQueue:
    """Small FIFO queue with explicit recovery operations."""

    def __init__(self) -> None:
        self._items: deque[QueuedPrompt] = deque()
        self._steering_open = False

    @property
    def items(self) -> tuple[str, ...]:
        return tuple(item.text for item in self._items)

    @property
    def entries(self) -> tuple[QueuedPrompt, ...]:
        return tuple(self._items)

    @property
    def steering_open(self) -> bool:
        return self._steering_open

    def open_steering(self) -> None:
        self._steering_open = True

    def close_steering(self, *, promote_pending: bool = False) -> None:
        self._steering_open = False
        if promote_pending:
            self._items = deque(
                QueuedPrompt(item.text, "follow_up")
                if item.mode == "steer"
                else item
                for item in self._items
            )

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, prompt: str, *, mode: QueueMode = "follow_up") -> None:
        prompt = prompt.strip()
        if prompt:
            self._items.append(QueuedPrompt(prompt, mode))

    def extend(
        self,
        prompts: Iterable[str],
        *,
        mode: QueueMode = "follow_up",
    ) -> None:
        for prompt in prompts:
            self.append(prompt, mode=mode)

    def pop(self, *, mode: QueueMode | None = None) -> str | None:
        if mode is None:
            return self._items.popleft().text if self._items else None
        for index, item in enumerate(self._items):
            if item.mode == mode:
                del self._items[index]
                return item.text
        return None

    def pop_last(self) -> str | None:
        return self._items.pop().text if self._items else None

    def clear(self) -> None:
        self._items.clear()

    def dump(self) -> list[dict[str, str]]:
        return [
            {"text": item.text, "mode": item.mode}
            for item in self._items
        ]

    def restore(self, values: Iterable[object]) -> bool:
        restored: list[QueuedPrompt] = []
        for value in values:
            if isinstance(value, str):
                prompt = QueuedPrompt(value.strip(), "follow_up")
            elif isinstance(value, Mapping):
                text = value.get("text")
                mode = value.get("mode")
                if not isinstance(text, str) or mode not in {"steer", "follow_up"}:
                    return False
                prompt = QueuedPrompt(text.strip(), mode)
            else:
                return False
            if prompt.text:
                restored.append(prompt)
        self._items.extend(restored)
        return True

    def restore_text(self) -> str:
        items = tuple(item.text for item in self._items)
        self._items.clear()
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return "\n".join(f"-> {item}" for item in items)


__all__ = ["PromptQueue", "QueuedPrompt", "QueueMode", "split_submission"]
