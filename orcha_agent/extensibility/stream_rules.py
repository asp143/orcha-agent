"""Stream interception and rule matching, independent of TUI paint internals."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
import json
import re
from pathlib import Path

import regex
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command

from orcha_agent.core.events import Event
from orcha_agent.core.plugin import Handled

from .rules import Rule, matches_paths
from .skill_globs import SkillGlobsMiddleware


@dataclass(slots=True)
class StreamInspect(Event):
    item: Any
    host: Any


@dataclass(frozen=True, slots=True)
class StreamRetry(Handled):
    messages: list[BaseMessage]


async def intercepted_stream(host: Any, value: Any, **kwargs: Any) -> AsyncIterator[Any]:
    """Close interrupted graph streams before resuming their pending model checkpoint.

    The graph commits complete nodes, so closing during a model node preserves
    the pre-model checkpoint, including completed tools. Never replay user input
    or completed tool calls on a retry.
    """
    subscribed = any(
        issubclass(StreamInspect, registration.event_type)
        for registration in getattr(host.bus, "handlers", ())
    )
    if not subscribed:
        async for item in host.agent.astream(value, **kwargs):
            yield item
        return
    while True:
        retry = None
        stream = host.agent.astream(value, **kwargs)
        try:
            async for item in stream:
                decision = await host.bus.emit(StreamInspect(item=item, host=host))
                if isinstance(decision, StreamRetry):
                    retry = decision
                    break
                yield item
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        if retry is None:
            await host.bus.emit(StreamInspect(item=None, host=host))
            return
        value = Command(update={"messages": retry.messages})


class StreamRules:
    """Per-session condition buffer and completed-turn based repeat policy."""

    def __init__(
        self, rules: Mapping[str, Rule], settings: Mapping[str, Any], *, cwd: Path | None = None
    ) -> None:
        self.rules = rules
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
            for rule in self.rules.values():
                if rule.name in self.current or not _allows(rule, source, tool, paths):
                    continue
                if rule.name in self.fired:
                    if self.settings.get("repeatMode", "once") == "once":
                        continue
                    if self.turn - self.fired[rule.name] < int(self.settings.get("repeatGap", 10)):
                        continue
                for pattern in rule.conditions:
                    try:
                        matched = regex.search(
                            pattern.pattern, text, flags=pattern.flags, timeout=0.01
                        )
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
