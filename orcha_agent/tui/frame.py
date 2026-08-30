"""Transcript block lifecycle and frame scheduling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any


class BlockState(str, Enum):
    ACTIVE = "active"
    SETTLED = "settled"
    COMMITTED = "committed"


@dataclass(slots=True)
class Block:
    id: str
    kind: str
    state: BlockState = BlockState.ACTIVE
    revision: int = 0
    source_id: str | None = None
    created: float = field(default_factory=time.monotonic)
    data: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        data: Mapping[str, Any] | None = None,
        /,
        **changes: Any,
    ) -> None:
        if self.state is not BlockState.ACTIVE:
            raise RuntimeError(f"cannot update {self.state.value} block {self.id}")
        if data:
            self.data.update(data)
        if changes:
            self.data.update(changes)
        self.revision += 1

    def settle(self) -> None:
        if self.state is BlockState.COMMITTED:
            raise RuntimeError(f"cannot settle committed block {self.id}")
        if self.state is BlockState.ACTIVE:
            self.state = BlockState.SETTLED
            self.revision += 1


@dataclass(frozen=True, slots=True)
class ViewportItem:
    block: Block
    rows: int


class Frame:
    """Ordered transcript blocks with commit and viewport invariants."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._ids = count(1)

    def add(
        self,
        kind: str,
        data: Mapping[str, Any] | None = None,
        *,
        source_id: str | None = None,
        state: BlockState = BlockState.ACTIVE,
        block_id: str | None = None,
    ) -> Block:
        block = Block(
            id=block_id or f"block-{next(self._ids)}",
            kind=kind,
            state=state,
            source_id=source_id,
            data=dict(data or {}),
        )
        self.blocks.append(block)
        return block

    add_block = add

    def settle(self, block: Block | str) -> Block:
        found = self.get(block) if isinstance(block, str) else block
        found.settle()
        return found

    def get(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise KeyError(block_id)

    def commit_ready(self) -> list[Block]:
        ready: list[Block] = []
        for block in self.blocks:
            if block.state is BlockState.COMMITTED:
                continue
            if block.state is BlockState.ACTIVE:
                break
            block.state = BlockState.COMMITTED
            block.revision += 1
            ready.append(block)
        return ready

    def prune_committed(self, blocks: list[Block]) -> None:
        """Release committed blocks after their scrollback write succeeds."""

        committed = {
            block.id
            for block in blocks
            if block.state is BlockState.COMMITTED
        }
        if committed:
            self.blocks[:] = [
                block
                for block in self.blocks
                if block.id not in committed
            ]

    @staticmethod
    def row_budget(
        *,
        terminal_rows: int,
        composer_rows: int,
        status_rows: int,
    ) -> int:
        return max(0, terminal_rows - composer_rows - status_rows)

    @staticmethod
    def tool_rows(available_rows: int) -> int:
        return min(3, max(0, available_rows))

    def viewport_plan(
        self,
        budget_rows: int,
        *,
        width: int | None = None,
        measure: Callable[[Block, int], int] | None = None,
    ) -> list[ViewportItem]:
        """Allocate measured visual rows newest-first in transcript order."""
        if width is not None:
            width = max(1, width)
        budget = max(0, budget_rows)
        if budget == 0:
            return []
        candidates = [
            block for block in self.blocks if block.state is not BlockState.COMMITTED
        ]
        selected = candidates[-budget:]
        allocations = {block.id: 1 for block in selected}
        remaining = budget - len(selected)

        def desired(block: Block) -> int:
            if block.kind == "tool":
                return 3
            if width is not None and measure is not None:
                return max(1, measure(block, width))
            content = str(block.data.get("text", block.data.get("message", "")))
            if width is None:
                return max(1, content.count("\n") + 1)
            return sum(
                max(1, (len(line) + width - 1) // width)
                for line in content.split("\n")
            )

        # Keep every active block observable. Non-tool prose gets surplus rows
        # before tool cards, which then degrade deterministically to 2/1 rows.
        for group in (
            [block for block in reversed(selected) if block.kind != "tool"],
            [block for block in reversed(selected) if block.kind == "tool"],
        ):
            for block in group:
                if remaining == 0:
                    break
                extra = min(remaining, max(0, desired(block) - 1))
                allocations[block.id] += extra
                remaining -= extra
        return [ViewportItem(block, allocations[block.id]) for block in selected]


class FrameScheduler:
    """Coalesce scrollback commits, invalidations, and spinner updates."""

    COMMIT_INTERVAL = 0.050
    INVALIDATE_INTERVAL = 1 / 30
    SPINNER_INTERVAL = 0.080

    def __init__(
        self,
        frame: Frame,
        *,
        commit: Callable[[list[Block]], None],
        invalidate: Callable[[], None],
        spinning: Callable[[], bool] | None = None,
        on_spinner_tick: Callable[[int], None] | None = None,
    ) -> None:
        self.frame = frame
        self._commit = commit
        self._invalidate = invalidate
        self._commit_task: asyncio.Task[None] | None = None
        self._invalidate_task: asyncio.Task[None] | None = None
        self._spinner_task: asyncio.Task[None] | None = None
        self._last_invalidation = 0.0
        self._spinning = spinning
        self._on_spinner_tick = on_spinner_tick
        self._spinner_frame = 0

    def commit_now(self) -> None:
        if self._commit_task is not None and not self._commit_task.done():
            self._commit_task.cancel()
        ready = self.frame.commit_ready()
        if ready:
            self._commit(ready)

    def request_commit(self) -> None:
        if self._commit_task is None or self._commit_task.done():
            self._commit_task = asyncio.create_task(self._flush_later())

    async def _flush_later(self) -> None:
        await asyncio.sleep(self.COMMIT_INTERVAL)
        ready = self.frame.commit_ready()
        if ready:
            self._commit(ready)

    def request_invalidate(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_invalidation
        if elapsed >= self.INVALIDATE_INTERVAL:
            self._last_invalidation = now
            self._invalidate()
            return
        if self._invalidate_task is None or self._invalidate_task.done():
            self._invalidate_task = asyncio.create_task(
                self._invalidate_later(self.INVALIDATE_INTERVAL - elapsed)
            )

    async def _invalidate_later(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._last_invalidation = time.monotonic()
        self._invalidate()

    def render_now(self) -> None:
        self._last_invalidation = time.monotonic()
        self._invalidate()

    def start_spinner(self) -> asyncio.Task[None]:
        if self._spinner_task is None or self._spinner_task.done():
            self._spinner_task = asyncio.create_task(self._tick_spinners())
        return self._spinner_task

    def _has_spinners(self) -> bool:
        if self._spinning is not None and self._spinning():
            return True
        return any(
            block.state is BlockState.ACTIVE
            and block.kind in {"thinking", "tool", "subagents"}
            for block in self.frame.blocks
        )

    def tick_spinners(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        for block in self.frame.blocks:
            if (
                block.state is not BlockState.ACTIVE
                or block.kind not in {"thinking", "tool", "subagents"}
            ):
                continue
            changes: dict[str, Any] = {
                "spinner_frame": (int(block.data.get("spinner_frame", 0)) + 1) % 8,
            }
            elapsed = max(0.0, current - block.created)
            if block.kind == "tool":
                changes["elapsed"] = elapsed
            elif block.kind == "thinking":
                tokens = int(block.data.get("reasoning_tokens", 0))
                changes["tokens_per_second"] = tokens / elapsed if elapsed else 0.0
            block.update(changes)


    async def _tick_spinners(self) -> None:
        while self._has_spinners():
            await asyncio.sleep(self.SPINNER_INTERVAL)
            if not self._has_spinners():
                break
            self.tick_spinners()
            self._spinner_frame = (self._spinner_frame + 1) % 8
            if self._on_spinner_tick is not None:
                self._on_spinner_tick(self._spinner_frame)
            self.request_invalidate()

    async def aclose(self) -> None:
        tasks = (
            self._commit_task,
            self._invalidate_task,
            self._spinner_task,
        )
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in tasks if task is not None),
            return_exceptions=True,
        )


__all__ = [
    "Block",
    "BlockState",
    "Frame",
    "FrameScheduler",
    "ViewportItem",
]
