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


def _thinking_fragments(
    part: Mapping[str, Any],
) -> list[tuple[tuple[str, Any] | None, str]]:
    if part.get("type") == "reasoning":
        summary = part.get("summary")
        if isinstance(summary, (list, tuple)):
            fragments: list[tuple[tuple[str, Any] | None, str]] = []
            for position, item in enumerate(summary):
                part_key: tuple[str, Any] | None = ("position", position)
                if isinstance(item, Mapping) and "index" in item:
                    part_key = ("index", item["index"])
                fragments.append((part_key, _text(item)))
            return fragments
        return [(None, _text(summary))]
    if part.get("type") == "thinking":
        return [(None, _text(part.get("thinking")))]
    return []


def _reasoning_run_key(part: Mapping[str, Any]) -> tuple[str, Any] | None:
    if "index" in part:
        return ("index", part["index"])
    if part.get("id") is not None:
        return ("id", part["id"])
    return None


def _reasoning_tokens(chunk: Any) -> int | None:
    usage = getattr(chunk, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("output_token_details")
    sources = (details, usage)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("reasoning", "reasoning_tokens"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                return max(0, int(value))
    return None


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
        self._source_tails: dict[str, Block] = {}
        self._tools: dict[str, Block] = {}
        self._read_groups: dict[str, Block] = {}

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

    def append_welcome(
        self,
        data: Mapping[str, Any],
        *,
        immediate: bool = True,
    ) -> Block:
        block = self.frame.add("welcome", data)
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
            self._source_tails.clear()
            self._tools.clear()
            self._read_groups.clear()
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
            self._tool_start(event)
            return
        if isinstance(event, ToolCallEnd):
            self._tool_end(event)
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

    def _tool_start(self, event: ToolCallStart) -> None:
        source = str(event.source_id or "main")
        self._source_tails.pop(source, None)
        block = self._read_groups.get(source) if event.name == "read_file" else None
        can_group = (
            block is not None
            and block.state is BlockState.ACTIVE
            and bool(self.frame.blocks)
            and self.frame.blocks[-1] is block
        )
        if can_group:
            calls = block.data.get("calls")
            if not isinstance(calls, list):
                calls = [
                    {
                        "id": block.data["id"],
                        "args": block.data.get("args", {}),
                    }
                ]
            block.update(
                calls=[
                    *calls,
                    {"id": event.id, "args": event.args},
                ]
            )
        else:
            block = self.frame.add(
                "tool",
                {"name": event.name, "args": event.args, "id": event.id},
                source_id=event.source_id,
            )
            if event.name == "read_file":
                self._read_groups[source] = block
        self._tools[event.id] = block
        if self.scheduler is not None:
            self.scheduler.start_spinner()
            self.scheduler.request_invalidate()

    def _tool_end(self, event: ToolCallEnd) -> None:
        block = self._tools.get(event.id)
        if block is None:
            block = self.frame.add(
                "tool",
                {"name": event.name, "args": {}, "id": event.id},
            )
            self._tools[event.id] = block
        calls = block.data.get("calls")
        if isinstance(calls, list):
            updated = [
                {**call, "result": event.result}
                if call.get("id") == event.id
                else call
                for call in calls
            ]
            block.update(calls=updated)
            complete = all("result" in call for call in updated)
        else:
            block.update(result=event.result)
            complete = True
        if complete:
            block.settle()
            if self.scheduler is not None:
                self.scheduler.request_commit()
        if self.scheduler is not None:
            self.scheduler.request_invalidate()

    def _source_block(
        self,
        source_id: str,
        kind: str,
        role: str,
        *,
        run_key: tuple[str, Any] | None = None,
    ) -> Block:
        key = (source_id, kind)
        block = self._source_blocks.get(key)
        if (
            block is not None
            and self._source_tails.get(source_id) is block
            and (
                kind != "thinking"
                or run_key is None
                or block.data.get("run_key") == run_key
            )
        ):
            return block
        data: dict[str, Any] = {"text": "", "role": role}
        if kind == "assistant":
            data.update(
                role=role,
                subagent=role == "subagent" or role.startswith("subagent:"),
            )
        else:
            data.update(run_key=run_key, summary_part=None)
        block = self.frame.add(kind, data, source_id=source_id)
        self._source_blocks[key] = block
        self._source_tails[source_id] = block
        return block

    def _model_chunk(self, event: ModelChunk) -> None:
        source_id = str(event.source_id or event.role)
        value = getattr(event.chunk, "content", event.chunk)
        parts = value if isinstance(value, (list, tuple)) else (value,)
        thinking_seen = False
        usage_tokens = _reasoning_tokens(event.chunk)
        for part in parts:
            if (
                isinstance(part, Mapping)
                and part.get("type") in {"reasoning", "thinking"}
            ):
                run_key = _reasoning_run_key(part)
                for summary_part, content in _thinking_fragments(part):
                    if not content:
                        continue
                    block = self._source_block(
                        source_id,
                        "thinking",
                        event.role,
                        run_key=run_key,
                    )
                    prior = str(block.data["text"])
                    previous_part = block.data.get("summary_part")
                    separator = (
                        "\n"
                        if prior
                        and summary_part is not None
                        and previous_part is not None
                        and summary_part != previous_part
                        else ""
                    )
                    accumulated = f"{prior}{separator}{content}"
                    changes: dict[str, Any] = {
                        "text": accumulated,
                        "summary_part": summary_part,
                        "reasoning_tokens": (
                            usage_tokens
                            if usage_tokens is not None
                            else max(1, (len(accumulated) + 3) // 4)
                        ),
                    }
                    block.update(changes)
                    thinking_seen = True
                continue

            content = _text(part)
            if not content:
                continue
            block = self._source_block(source_id, "assistant", event.role)
            block.update(text=f"{block.data['text']}{content}")
        if usage_tokens is not None:
            thinking = self._source_blocks.get((source_id, "thinking"))
            if (
                thinking is not None
                and thinking.data.get("reasoning_tokens") != usage_tokens
            ):
                thinking.update(reasoning_tokens=usage_tokens)
            thinking_seen = thinking is not None
        if self.scheduler is not None:
            if thinking_seen:
                self.scheduler.start_spinner()
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


    def release_committed(self, blocks: list[Block]) -> None:
        """Release accumulator references after scrollback owns the blocks."""

        committed = {
            block.id
            for block in blocks
            if block.state is BlockState.COMMITTED
        }
        if not committed:
            return
        self._source_blocks = {
            key: block
            for key, block in self._source_blocks.items()
            if block.id not in committed
        }
        self._source_tails = {
            key: block
            for key, block in self._source_tails.items()
            if block.id not in committed
        }
        self._tools = {
            key: block
            for key, block in self._tools.items()
            if block.id not in committed
        }
        self._read_groups = {
            key: block
            for key, block in self._read_groups.items()
            if block.id not in committed
        }
    def clear(self) -> None:
        self.frame.blocks.clear()
        self._source_blocks.clear()
        self._source_tails.clear()
        self._tools.clear()
        self._read_groups.clear()


__all__ = ["Transcript"]
