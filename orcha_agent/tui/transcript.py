"""Translate kernel events and console output into transcript blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orcha_agent.core.events import (
    ModelChunk,
    ThreadSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.registry import Registry

from .frame import Block, BlockState, Frame, FrameScheduler


def _matches(match: Any, event: object) -> bool:
    if isinstance(match, type):
        return isinstance(event, match)
    if callable(match):
        return bool(match(event))
    return match == type(event).__name__ or match == getattr(event, "name", None)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(_text(item) for item in value)
    return str(value)


def _thinking_text(part: Mapping[str, Any]) -> str:
    if part.get("type") == "reasoning":
        return _text(part.get("summary"))
    if part.get("type") == "thinking":
        return _text(part.get("thinking"))
    return ""


class Transcript:
    """Event sink that owns source-aware accumulated transcript state."""

    def __init__(
        self,
        frame: Frame | None = None,
        *,
        registry: Registry | Any | None = None,
        scheduler: FrameScheduler | None = None,
    ) -> None:
        self.frame = frame or Frame()
        self.registry = registry
        self.scheduler = scheduler
        self._source_blocks: dict[tuple[str, str], Block] = {}
        self._tools: dict[str, Block] = {}

    def _commit(self, block: Block, *, immediate: bool = False) -> None:
        block.settle()
        if self.scheduler is None:
            self.frame.commit_ready()
        elif immediate:
            self.scheduler.commit_now()
        else:
            self.scheduler.request_commit()

    def append_raw(
        self,
        renderable: Any,
        *,
        immediate: bool = False,
        **options: Any,
    ) -> Block:
        block = self.frame.add(
            "raw",
            {"renderable": renderable, "options": options, "level": "raw"},
        )
        self._commit(block, immediate=immediate)
        return block

    def append_banner(
        self,
        message: str,
        *,
        level: str = "error",
        immediate: bool = False,
    ) -> Block:
        lines = message.splitlines()
        if level == "error" and len(lines) > 8:
            lines = [*lines[:7], "…"]
        block = self.frame.add("banner", {"message": "\n".join(lines), "level": level})
        self._commit(block, immediate=immediate)
        return block

    def _legacy(self, event: object) -> bool:
        if self.registry is None:
            return False
        for registration in self.registry.renderers:
            if not _matches(registration.match, event):
                continue
            rendered = registration.render(event)
            if rendered is None:
                continue
            self.append_raw(
                rendered,
                immediate=False,
                end="" if isinstance(event, ModelChunk) else "\n",
            )
            return True
        return False

    async def handle(self, event: object) -> None:
        if isinstance(event, TurnStart):
            for prior in self.frame.blocks:
                if prior.state is BlockState.ACTIVE:
                    prior.settle()
            self._source_blocks.clear()
            self._tools.clear()
            if self._legacy(event):
                return
            block = self.frame.add(
                "user",
                {"text": event.text, "thread_id": event.thread_id},
                source_id=event.thread_id,
            )
            self._commit(block, immediate=True)
            if self.scheduler is not None:
                self.scheduler.render_now()
            return
        if self._legacy(event):
            return
        if isinstance(event, ModelChunk):
            self._model_chunk(event)
            return
        if isinstance(event, ToolCallStart):
            block = self.frame.add(
                "tool",
                {"name": event.name, "args": event.args, "id": event.id},
                source_id=event.source_id,
            )
            self._tools[event.id] = block
            if self.scheduler is not None:
                self.scheduler.start_spinner()
                self.scheduler.request_invalidate()
            return
        if isinstance(event, ToolCallEnd):
            block = self._tools.get(event.id)
            if block is None:
                block = self.frame.add(
                    "tool",
                    {"name": event.name, "args": {}, "id": event.id},
                )
                self._tools[event.id] = block
            block.update(result=event.result)
            block.settle()
            if self.scheduler is not None:
                self.scheduler.request_commit()
                self.scheduler.request_invalidate()
            return
        if isinstance(event, ThreadSwitch):
            labels = {
                "compact": "⊟ compacted",
                "clear": "⊠ cleared",
                "branch": f"⎇ branched to {event.new}",
            }
            block = self.frame.add(
                "marker",
                {
                    "text": labels.get(event.reason, event.reason),
                    "old": event.old,
                    "new": event.new,
                    "session_id": event.session_id,
                },
            )
            self._commit(block)
            return
        if isinstance(event, TurnEnd):
            for block in self.frame.blocks:
                if block.state is BlockState.ACTIVE:
                    block.settle()
            if self.scheduler is not None:
                self.scheduler.request_commit()
                self.scheduler.request_invalidate()
            return
        if isinstance(event, BaseException):
            self.append_banner(f"{type(event).__name__}: {event}")

    def _source_block(self, source_id: str, kind: str, role: str) -> Block:
        key = (source_id, kind)
        block = self._source_blocks.get(key)
        if block is not None:
            return block
        data: dict[str, Any] = {"text": ""}
        if kind == "assistant":
            data.update(role=role, subagent=role == "subagent" or role.startswith("subagent:"))
        block = self.frame.add(kind, data, source_id=source_id)
        if kind == "thinking":
            assistant = self._source_blocks.get((source_id, "assistant"))
            if assistant is not None:
                self.frame.blocks.remove(block)
                self.frame.blocks.insert(self.frame.blocks.index(assistant), block)
        self._source_blocks[key] = block
        return block

    def _model_chunk(self, event: ModelChunk) -> None:
        source_id = str(event.source_id or event.role)
        value = getattr(event.chunk, "content", event.chunk)
        parts = value if isinstance(value, (list, tuple)) else (value,)
        for part in parts:
            if isinstance(part, Mapping) and part.get("type") in {"reasoning", "thinking"}:
                content = _thinking_text(part)
                kind = "thinking"
            else:
                content = _text(part)
                kind = "assistant"
            if not content:
                continue
            block = self._source_block(source_id, kind, event.role)
            block.update(text=f"{block.data['text']}{content}")
        if self.scheduler is not None:
            self.scheduler.request_invalidate()

    def print(self, *objects: Any, **kwargs: Any) -> Block:
        block = self.frame.add(
            "raw",
            {
                "objects": tuple(objects),
                "options": dict(kwargs),
                "level": "raw",
            },
        )
        self._commit(block)
        return block

    def clear(self) -> None:
        self.frame.blocks.clear()
        self._source_blocks.clear()
        self._tools.clear()


__all__ = ["Transcript"]
