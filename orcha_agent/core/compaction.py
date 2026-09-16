"""Compaction policy and model-boundary control, independent of terminal painting."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command


from .compaction_config import CompactionConfig


def _estimated_tokens(messages: list[BaseMessage]) -> int:
    """Conservative provider-independent estimate, including tool arguments."""
    return sum(
        (len(str(message.content)) + len(str(getattr(message, "tool_calls", ""))) + 3) // 4 + 8
        for message in messages
    )


def estimate_tokens(messages: list[BaseMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict) and not message.additional_kwargs.get("compaction_usage_stale"):
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            if total:
                return total + _estimated_tokens(messages[index + 1 :])
    return _estimated_tokens(messages)


def is_context_overflow(exc: Exception) -> bool:
    value = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in value
        for marker in (
            "contextoverflow",
            "context_length_exceeded",
            "context window",
            "maximum context length",
            "prompt is too long",
            "input is too long",
            "too many input tokens",
            "exceeds the maximum number of tokens",
            "exceed context length",
        )
    )


def prune_results(messages: list[BaseMessage], policy: CompactionConfig) -> list[BaseMessage]:
    """Replace superseded payloads, retaining every tool call/result pair."""
    successful = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.status != "error"
    }
    reads: dict[str, str] = {}
    latest: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] in {"read", "read_file"}:
                    args = call["args"]
                    path = str(args.get("path", args.get("file_path", "")))
                    if path and call["id"] is not None:
                        # Include range: a later partial read must not erase another range.
                        key = json.dumps([path, args.get("offset"), args.get("limit")])
                        reads[call["id"]] = key
                        if call["id"] in successful:
                            latest[key] = call["id"]
    result = []
    for message in messages:
        if isinstance(message, ToolMessage):
            old_read = (
                policy.supersede_reads
                and message.tool_call_id in reads
                and latest.get(reads[message.tool_call_id], message.tool_call_id)
                != message.tool_call_id
            )
            useless = policy.drop_useless and message.additional_kwargs.get("superseded") is True
            if old_read or useless:
                message = message.model_copy(update={"content": "[Superseded tool result]"})
        result.append(message)
    return result


@dataclass(slots=True)
class CompactionResult:
    messages: list[BaseMessage]
    summary: str
    short_summary: str
    tokens_before: int
    method: str


class Compactor:
    def __init__(
        self, model: Any, policy: CompactionConfig, window: int = 128_000, bus: Any = None
    ) -> None:
        self.model = model
        self.policy = policy
        self.window = window
        self.bus = bus
        self._speculation: asyncio.Task[CompactionResult] | None = None
        self._speculation_key: str | None = None
        if callable(getattr(bus, "on", None)):
            import weakref
            from .events import AgentBuildBefore, AppExit, ModelSwitch, SessionSwitch, ThreadSwitch

            reference = weakref.ref(self)

            async def close_compaction(event: Any) -> None:
                owner = reference()
                if owner is not None:
                    await owner.close(event)

            for event in (AgentBuildBefore, AppExit, ModelSwitch, SessionSwitch, ThreadSwitch):
                bus.on(event, close_compaction, plugin="compaction")

    async def close(self, event: Any = None) -> None:
        task, self._speculation = self._speculation, None
        self._speculation_key = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @property
    def threshold(self) -> int:
        return self.policy.threshold(self.window)

    def needed(self, messages: list[BaseMessage], trigger: str) -> bool:
        if trigger == "manual":
            return True
        if not self.policy.enabled:
            return False
        return trigger in {"overflow", "length"} or estimate_tokens(messages) >= self.threshold

    def speculate(self, messages: list[BaseMessage]) -> None:
        if (
            not self.policy.enabled
            or not self.policy.speculative
            or estimate_tokens(messages) < self.threshold * 0.75
        ):
            return
        prefix, _ = self._partition(prune_results(messages, self.policy))
        key = self._key(prefix)
        if self._speculation_key == key:
            return
        if self._speculation is not None:
            self._speculation.cancel()
        self._speculation_key = key
        self._speculation = asyncio.create_task(self._compact(messages, ""))
        # Retrieve failures even when no later request consumes this speculation.
        self._speculation.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )

    @staticmethod
    def _key(messages: list[BaseMessage]) -> str:
        return hashlib.sha256(
            repr([message.model_dump() for message in messages]).encode()
        ).hexdigest()

    async def compact(
        self, messages: list[BaseMessage], instructions: str = ""
    ) -> CompactionResult:
        prefix, tail = self._partition(prune_results(messages, self.policy))
        if (
            not instructions
            and self._speculation is not None
            and self._speculation_key == self._key(prefix)
        ):
            task, self._speculation = self._speculation, None
            try:
                result = await task
                if result.method != "shake":
                    tail = [
                        message.model_copy(
                            update={
                                "additional_kwargs": {
                                    **message.additional_kwargs,
                                    "compaction_usage_stale": True,
                                }
                            }
                        )
                        if getattr(message, "usage_metadata", None)
                        else message
                        for message in tail
                    ]
                    tokens_before = estimate_tokens(messages)
                    marker = result.messages[0].model_copy(
                        update={
                            "additional_kwargs": {
                                **result.messages[0].additional_kwargs,
                                "compaction": {
                                    "short_summary": result.short_summary,
                                    "tokens_before": tokens_before,
                                    "method": result.method,
                                },
                            }
                        }
                    )
                    return CompactionResult(
                        [marker, *tail],
                        result.summary,
                        result.short_summary,
                        tokens_before,
                        result.method,
                    )
            except Exception:
                pass
        return await self._compact(messages, instructions)

    def _partition(self, cleaned: list[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage]]:
        # Retain complete user turns; never start the tail on an orphan tool result.
        cutoff = len(cleaned)
        recent = 0
        for index in range(len(cleaned) - 1, -1, -1):
            recent += _estimated_tokens([cleaned[index]])
            if recent > self.policy.keep_recent_tokens:
                break
            if index > 0 and isinstance(cleaned[index], HumanMessage):
                cutoff = index
        # A large latest turn is retained whole when earlier history exists.
        latest_user = next(
            (i for i in range(len(cleaned) - 1, 0, -1) if isinstance(cleaned[i], HumanMessage)),
            None,
        )
        if latest_user is not None:
            cutoff = min(cutoff, latest_user)
        return cleaned[:cutoff], cleaned[cutoff:]

    async def _compact(self, messages: list[BaseMessage], instructions: str) -> CompactionResult:
        from .events import CompactionStatus

        if self.bus is not None:
            await self.bus.emit(CompactionStatus(active=True))
        try:
            return await self._compact_impl(messages, instructions)
        finally:
            if self.bus is not None:
                await self.bus.emit(CompactionStatus(active=False))

    async def _compact_impl(
        self, messages: list[BaseMessage], instructions: str
    ) -> CompactionResult:
        tokens_before = estimate_tokens(messages)
        cleaned = prune_results(messages, self.policy)
        prefix, tail = self._partition(cleaned)
        last_error: Exception | None = None
        for method in self.policy.method_order:
            if method == "shake":
                if cleaned == messages:
                    continue
                if cleaned:
                    cleaned[-1] = cleaned[-1].model_copy(
                        update={
                            "additional_kwargs": {
                                **cleaned[-1].additional_kwargs,
                                "compaction_shake": {
                                    "tokens_before": tokens_before,
                                    "short_summary": "Removed superseded results",
                                    "method": "shake",
                                },
                            }
                        }
                    )
                return CompactionResult(
                    cleaned,
                    "Superseded tool results removed.",
                    "Removed superseded results",
                    tokens_before,
                    method,
                )
            prompt = "Summarize the conversation for continuation. Preserve decisions, current files, constraints, failures, and remaining work."
            if method == "handoff":
                prompt += " Write a structured handoff with headings: Goal, Constraints, Completed, Files, Failures, Next steps."
            if instructions:
                prompt += "\nAdditional instructions: " + instructions
            try:
                from .models import filter_foreign_blocks
                from langchain_core.runnables import Runnable
                from langgraph.constants import TAG_NOSTREAM

                # Summaries share accounting callbacks, but are not assistant output.
                invocation: dict[str, Any] = (
                    {"config": {"tags": [TAG_NOSTREAM]}} if isinstance(self.model, Runnable) else {}
                )
                response = await self.model.ainvoke(
                    [
                        *filter_foreign_blocks(prefix, {"reasoning", "thinking"}),
                        HumanMessage(content=prompt),
                    ],
                    **invocation,
                )
                summary = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                if not summary.strip():
                    raise ValueError("Compaction model returned an empty summary")
                short = summary.strip().splitlines()[0][:160]
                marker = HumanMessage(
                    content="Here is a summary of the conversation to date:\n\n" + summary,
                    id=str(uuid4()),
                    additional_kwargs={
                        "lc_source": "summarization",
                        "compaction": {
                            "short_summary": short,
                            "tokens_before": tokens_before,
                            "method": method,
                        },
                    },
                )
                tail = [
                    message.model_copy(
                        update={
                            "additional_kwargs": {
                                **message.additional_kwargs,
                                "compaction_usage_stale": True,
                            }
                        }
                    )
                    if getattr(message, "usage_metadata", None)
                    else message
                    for message in tail
                ]
                return CompactionResult([marker, *tail], summary, short, tokens_before, method)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return CompactionResult(messages, "", "Nothing to remove", tokens_before, "shake")


def _update(messages: list[BaseMessage]) -> dict[str, Any]:
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}


def _sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    context = copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, asyncio.run, coro).result()


class CompactionMiddleware(AgentMiddleware):
    @property
    def name(self) -> str:
        return "SummarizationMiddleware"

    def __init__(self, compactor: Compactor) -> None:
        self.compactor = compactor
        self.model = compactor.model

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        async def async_handler(updated: Any) -> Any:
            return handler(updated)

        async def run() -> Any:
            try:
                return await self.awrap_model_call(request, async_handler)
            finally:
                await self.compactor.close()

        return _sync(run())

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        messages = request.messages
        changed = False
        if self.compactor.needed(messages, "mid_turn"):
            result = await self.compactor.compact(messages)
            messages = result.messages
            changed = True
        try:
            response = await handler(request.override(messages=messages) if changed else request)
        except Exception as exc:
            if not is_context_overflow(exc) or not self.compactor.needed(messages, "overflow"):
                raise
            result = await self.compactor.compact(messages)
            messages = result.messages
            changed = True
            # Retry once; failures here escape this exception handler.
            response = await handler(request.override(messages=messages))
        combined = [*messages, *response.result]
        last = response.result[-1] if response.result else None
        if not getattr(last, "tool_calls", None):
            metadata = getattr(last, "response_metadata", {})
            reason = metadata.get("stop_reason", metadata.get("finish_reason"))
            trigger = "length" if reason in {"length", "max_tokens"} else "post_turn"
            if self.compactor.needed(combined, trigger):
                result = await self.compactor.compact(combined)
                combined = result.messages
                changed = True
            else:
                self.compactor.speculate(combined)
        if changed:
            return ExtendedModelResponse(
                model_response=response, command=Command(update=_update(combined))
            )
        return response
