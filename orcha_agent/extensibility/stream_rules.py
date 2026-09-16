"""Stream interception and rule matching, independent of TUI paint internals."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from contextvars import ContextVar
from time import monotonic
import json
import logging
import re
from pathlib import Path

import regex
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackManager
from langgraph.types import Command

from orcha_agent.core.events import Event, ModelChunk
from orcha_agent.core.plugin import Handled

from .rules import MAX_CONDITIONS, MAX_PATTERN_LENGTH, MAX_RULES, Rule, matches_paths
from .skill_globs import SkillGlobsMiddleware


@dataclass(slots=True)
class StreamInspect(Event):
    item: Any
    host: Any


@dataclass(frozen=True, slots=True)
class StreamRetry(Handled):
    messages: list[BaseMessage]


class StreamInterrupt(Exception):
    """A model-node failure carrying the authoritative retry state."""

    def __init__(self, retry: StreamRetry, usage: list[AIMessage] | None = None) -> None:
        super().__init__("Stream rule interrupted the model")
        self.retry = retry
        self.usage = usage or []


@dataclass(slots=True)
class StreamAborted(Event):
    """Mark the visible partial attempt before a model retry."""

    source_id: str = "main"


_stream_host: ContextVar[Any] = ContextVar("rules_stream_host", default=None)
_model_monitor: ContextVar[ModelStreamMonitor | None] = ContextVar(
    "rules_model_monitor", default=None
)


class ModelStreamMonitor:
    def __init__(self, host: Any) -> None:
        self.host = host
        self.pending: StreamInterrupt | None = None
        self.usage: list[AIMessage] = []

    async def inspect(self, message: AIMessage) -> None:
        if message.usage_metadata:
            # Keep provider-reported partial usage, even if this very chunk aborts.
            self.usage.append(message)
        decision = await self.host.bus.emit(
            StreamInspect(item=((), "messages", (message, {})), host=self.host)
        )
        if isinstance(decision, StreamRetry):
            self.pending = StreamInterrupt(decision, self.usage)
            raise self.pending


class RuleStreamCallback(AsyncCallbackHandler):
    raise_error = True
    run_inline = True

    async def on_llm_new_token(self, token: str, *, chunk: Any = None, **kwargs: Any) -> None:
        monitor = _model_monitor.get()
        if monitor is not None and chunk is not None and isinstance(chunk.message, AIMessage):
            await monitor.inspect(chunk.message)


def install_model_callback(model: Any) -> None:
    """Install an inert, context-local dispatcher without cloning model state.
class _RuleInterruptLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # LangChain logs before honoring raise_error. Match its exact structured
        # record, preserving genuine callback failures and all other warnings.
        return not (
            record.msg == "Error in %s.%s callback: %s"
            and record.args
            == (
                "RuleStreamCallback",
                "on_llm_new_token",
                "StreamInterrupt('Stream rule interrupted the model')",
            )
        )



    Cloning resets stateful fake/custom models. A single permanent dispatcher also
    avoids restoring shared callback lists in competing model-call finalizers.
    """
    callbacks = model.callbacks
    handlers = callbacks.handlers if isinstance(callbacks, BaseCallbackManager) else callbacks or []
    callback_logger = logging.getLogger("langchain_core.callbacks.manager")
    if not any(isinstance(item, _RuleInterruptLogFilter) for item in callback_logger.filters):
        callback_logger.addFilter(_RuleInterruptLogFilter())
    if any(isinstance(callback, RuleStreamCallback) for callback in handlers):
        return
    if isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(RuleStreamCallback(), inherit=False)
    else:
        callbacks = [*handlers, RuleStreamCallback()]
    model.callbacks = callbacks


async def intercepted_stream(host: Any, value: Any, **kwargs: Any) -> AsyncIterator[Any]:
    """Retry only authoritative failures raised from inside the model node.

    Closing an output iterator cannot roll back a successful LangGraph task: its
    done callback may already have committed writes. A StreamInterrupt instead
    leaves an ERROR task marker, so retry never restores rejected tool calls.
    """
    subscribed = any(
        issubclass(StreamInspect, registration.event_type)
        for registration in getattr(getattr(host, "bus", None), "handlers", ())
    )
    if not subscribed:
        async for item in host.agent.astream(value, **kwargs):
            yield item
        return
    token = _stream_host.set(host)
    try:
        while True:
            retry = None
            delivered_usage: set[int] = set()
            stream = host.agent.astream(value, **kwargs)
            try:
                async for item in stream:
                    mode, data = item[-2:]
                    if mode == "messages" and isinstance(data, tuple):
                        delivered_usage.add(id(data[0]))
                    yield item
            except StreamInterrupt as exc:
                retry = exc.retry
                await host.bus.emit(StreamAborted(source_id=getattr(host, "source_id", "main")))
                for chunk in exc.usage:
                    if id(chunk) not in delivered_usage:
                        await host.bus.emit(
                            ModelChunk(
                                chunk=AIMessageChunk(
                                    content="", id=chunk.id, usage_metadata=chunk.usage_metadata
                                ),
                                role="main",
                                source_id="main",
                                request_id=chunk.id,
                            )
                        )
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
            if retry is None:
                await host.bus.emit(StreamInspect(item=None, host=host))
                return
            value = Command(update={"messages": retry.messages})
    finally:
        _stream_host.reset(token)


class StreamRules:
    """Per-session condition buffer and completed-turn based repeat policy."""

    def __init__(
        self, rules: Mapping[str, Rule], settings: Mapping[str, Any], *, cwd: Path | None = None
    ) -> None:
        self.rules = rules
        self.compiled: dict[str, tuple[Any, ...]] = {}
        for rule in list(rules.values())[:MAX_RULES]:
            self.compiled[rule.name] = tuple(
                pattern
                if isinstance(pattern, regex.Pattern)
                else regex.compile(pattern.pattern, flags=pattern.flags)
                for pattern in rule.conditions[:MAX_CONDITIONS]
                if rule.trusted and len(pattern.pattern) <= MAX_PATTERN_LENGTH
            )
        self.paths = SkillGlobsMiddleware({})
        if cwd is not None:
            self.paths.cwd = cwd
        self.settings = settings
        self.buffers: dict[str, str] = {}
        self.fired: dict[str, int] = {}
        self.tool_buffers: dict[str, dict[str, str]] = {}
        self.match_sources: dict[str, str] = {}
        self.turn = 0
        self.current: set[str] = set()

    def start(self) -> None:
        self.buffers.clear()
        self.tool_buffers.clear()
        self.current.clear()

    def finish(self) -> None:
        self.turn += 1

    def inspect(self, item: Any) -> tuple[list[Rule], str]:
        if not self.settings.get("enabled", True):
            return [], ""
        deadline = monotonic() + 0.05
        if len(item) == 3:
            namespace, mode, data = item
        else:
            mode, data = item
            namespace = ()
        # Nested graphs must be interrupted at their own boundary, never by
        # replaying a parent's task tool. This interceptor monitors the main agent.
        if namespace or mode != "messages" or not isinstance(data, tuple):
            return [], ""
        message, metadata = data
        if not isinstance(message, AIMessage):
            return [], ""
        request = str(message.id or metadata.get("langgraph_step", "main"))
        candidates: list[tuple[str, str, str, list[str]]] = []
        if message.text:
            key = request + ":text"
            self.buffers[key] = (self.buffers.get(key, "") + message.text)[-262144:]
            candidates.append(("text", "", self.buffers[key], []))
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") in {"thinking", "reasoning"}:
                    key = request + ":thinking"
                    delta = str(block.get("thinking", block.get("text", "")))
                    self.buffers[key] = (self.buffers.get(key, "") + delta)[-262144:]
                    candidates.append(("thinking", "", self.buffers[key], []))
        chunks = getattr(message, "tool_call_chunks", ())
        for chunk in chunks:
            key = f"{request}:tool:{chunk.get('index', 0)}"
            pending = self.tool_buffers.setdefault(key, {"name": "", "args": ""})
            pending["name"] += chunk.get("name") or ""
            pending["args"] = (pending["args"] + (chunk.get("args") or ""))[-262144:]
            try:
                args = json.loads(pending["args"])
            except (ValueError, TypeError):
                args = {}
            paths = self.paths._paths(args) if isinstance(args, dict) else []
            candidates.append(("tool", pending["name"], pending["args"], paths))
        if not chunks:
            for call in message.tool_calls:
                candidates.append(
                    (
                        "tool",
                        call["name"],
                        json.dumps(call["args"]),
                        self.paths._paths(call["args"]),
                    )
                )
        matches: list[Rule] = []
        for source, tool, text, paths in candidates:
            for rule in list(self.rules.values())[:MAX_RULES]:
                if monotonic() >= deadline:
                    return matches, self.buffers.get(request + ":text", "")
                if rule.name in self.current or not _allows(rule, source, tool, paths):
                    continue
                if rule.name in self.fired:
                    if self.settings.get("repeatMode", "once") == "once":
                        continue
                    if self.turn - self.fired[rule.name] < int(self.settings.get("repeatGap", 10)):
                        continue
                for pattern in self.compiled.get(rule.name, ()):
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return matches, self.buffers.get(request + ":text", "")
                    try:
                        matched = pattern.search(text, timeout=min(0.01, remaining))
                    except TimeoutError:
                        continue
                    if matched:
                        self.fired[rule.name] = self.turn
                        self.current.add(rule.name)
                        self.match_sources[rule.name] = source
                        matches.append(rule)
                        break
        return matches, self.buffers.get(request + ":text", "")


def _allows(rule: Rule, source: str, tool: str, paths: list[str]) -> bool:
    if rule.globs and not matches_paths(rule, paths):
        return False
    for scope in rule.scope:
        if scope == source or (scope == "toolcall" and source == "tool"):
            return True
        if source == "tool":
            match = re.fullmatch(r"(?:tool:)?([\w-]+)(?:\((.+)\))?", scope)
            if match and match[1] == tool:
                if match[2] is None or matches_paths(Rule("scope", "", globs=(match[2],)), paths):
                    return True
    return False
