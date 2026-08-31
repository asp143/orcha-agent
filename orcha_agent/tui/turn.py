"""Model turn and stream processing."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

from orcha_agent.core.events import (
    InterruptRaised,
    ModelChunk,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.plugin import Handled, Resolved

from .context import AppContext
from .queue import PromptQueue

from .transcript import _matches
async def _render(ctx: AppContext, event: object, *, emit: bool = True) -> bool:
    transcript = getattr(ctx, "transcript", None)
    if emit:
        bus = getattr(ctx, "_bus", ctx.bus)
        handled = await bus.emit(event)
        if isinstance(handled, Handled):
            return True
        if transcript is not None:
            return True
    elif transcript is not None:
        await transcript.handle(event)
        return True
    for registration in ctx.registry.renderers:
        if not _matches(registration.match, event):
            continue
        rendered = registration.render(event)
        if rendered is None:
            continue
        if isinstance(event, ModelChunk):
            ctx.console.print(rendered, end="")
        else:
            ctx.console.print(rendered)
        return True
    return False


def _messages(value: Any) -> list[BaseMessage]:
    found: list[BaseMessage] = []
    if isinstance(value, BaseMessage):
        return [value]
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_messages(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_messages(item))
    return found


def _todos(value: Any) -> list[Any] | None:
    if isinstance(value, Mapping):
        direct = value.get("todos")
        if isinstance(direct, list):
            return direct
        for item in value.values():
            found = _todos(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _todos(item)
            if found is not None:
                return found
    return None


def _model_name(message: BaseMessage, metadata: Any) -> str | None:
    sources = (
        metadata,
        getattr(message, "response_metadata", None),
        getattr(message, "additional_kwargs", None),
    )
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        provider = next(
            (
                source[key]
                for key in ("ls_provider", "model_provider", "provider")
                if isinstance(source.get(key), str) and source[key]
            ),
            None,
        )
        model = next(
            (
                source[key]
                for key in ("ls_model_name", "model_name", "model")
                if isinstance(source.get(key), str) and source[key]
            ),
            None,
        )
        if model is None:
            continue
        if provider is not None and ":" not in model:
            return f"{provider}:{model}"
        return model
    return None


class _ModelLabelBuffer:
    """Return a model label only for the first chunk of each response."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def take(self, message: BaseMessage, metadata: Any) -> str | None:
        model_name = _model_name(message, metadata)
        if model_name is None:
            return None
        identifier = getattr(message, "id", None)
        if not identifier and isinstance(metadata, Mapping):
            identifier = metadata.get("run_id")
        key = str(identifier or model_name)
        if key in self._seen:
            return None
        self._seen.add(key)
        return model_name


@dataclass(slots=True)
class _PendingToolCall:
    name: str = ""
    args: str = ""
    id: str = ""


class _FileDiffCapture:
    """Capture local file state around Deepagents write/edit tool calls."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._pending: dict[str, tuple[str, str, Path]] = {}

    def _path(self, value: object) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = self.root / value.lstrip("/")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def start(self, event: ToolCallStart) -> None:
        if event.name not in {"edit_file", "write_file"}:
            return
        display_path = event.args.get("file_path", event.args.get("path"))
        path = self._path(display_path)
        if path is None:
            return
        try:
            before = path.read_text() if path.is_file() else ""
        except (OSError, UnicodeError):
            return
        self._pending[event.id] = (str(display_path), before, path)

    def finish(self, message: ToolMessage) -> ToolMessage:
        pending = self._pending.pop(message.tool_call_id, None)
        if pending is None or message.status == "error":
            return message
        display_path, before, path = pending
        try:
            after = path.read_text()
        except (OSError, UnicodeError):
            return message
        if before == after:
            return message
        artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
        existing_data = artifact.get("data")
        data = dict(existing_data) if isinstance(existing_data, Mapping) else {}
        data.update(path=display_path, before=before, after=after)
        return message.model_copy(update={"artifact": {**artifact, "data": data}})


def _start_file_diff_capture(
    ctx: Any,
    event: ToolCallStart,
    capture: _FileDiffCapture | None,
) -> _FileDiffCapture | None:
    if event.name not in {"edit_file", "write_file"}:
        return capture
    if capture is None:
        cfg = getattr(ctx, "cfg", None)
        cwd = getattr(cfg, "cwd", None)
        if cwd is None:
            return None
        capture = _FileDiffCapture(Path(cwd))
    capture.start(event)
    return capture


class _ToolCallBuffer:
    """Assemble provider tool-call chunks before rendering their arguments."""

    def __init__(self) -> None:
        self._pending: dict[tuple[str, int | str], _PendingToolCall] = {}
        self._emitted: set[tuple[str, str]] = set()

    def add(
        self,
        message: AIMessage,
        *,
        source_id: str = "main",
    ) -> list[ToolCallStart]:
        chunks = getattr(message, "tool_call_chunks", ())
        if not chunks:
            events: list[ToolCallStart] = []
            for call in message.tool_calls:
                identifier = call.get("id") or f"{call['name']}:{len(self._emitted)}"
                emitted_key = (source_id, identifier)
                if emitted_key in self._emitted:
                    continue
                self._emitted.add(emitted_key)
                events.append(
                    ToolCallStart(
                        name=call["name"],
                        args=call.get("args", {}),
                        id=identifier,
                        source_id=source_id,
                    )
                )
            return events

        completed: list[ToolCallStart] = []
        for chunk in chunks:
            chunk_key = chunk.get("index")
            if chunk_key is None:
                chunk_key = chunk.get("id") or len(self._pending)
            key = (source_id, chunk_key)
            pending = self._pending.setdefault(key, _PendingToolCall())
            name = chunk.get("name")
            if name:
                pending.name += name
            args = chunk.get("args")
            if args:
                pending.args += args
            identifier = chunk.get("id")
            if identifier:
                pending.id = identifier
            emitted_key = (source_id, pending.id)
            if not pending.name or not pending.id or emitted_key in self._emitted:
                continue
            try:
                parsed_args = json.loads(pending.args)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed_args, dict):
                continue
            self._emitted.add(emitted_key)
            completed.append(
                ToolCallStart(
                    name=pending.name,
                    args=parsed_args,
                    id=pending.id,
                    source_id=source_id,
                )
            )
            self._pending.pop(key, None)
        return completed


async def _message_event(
    ctx: AppContext,
    data: Any,
    tool_calls: _ToolCallBuffer,
    model_labels: _ModelLabelBuffer,
    namespace: tuple[str, ...] = (),
    *,
    file_diffs: _FileDiffCapture | None = None,
) -> _FileDiffCapture | None:
    if not isinstance(data, tuple) or len(data) != 2:
        return file_diffs
    message, metadata = data
    if not isinstance(message, BaseMessage):
        return file_diffs
    if isinstance(message, ToolMessage):
        return file_diffs
    source_id = "/".join(str(part) for part in namespace) if namespace else "main"
    if isinstance(message, AIMessage):
        for event in tool_calls.add(message, source_id=source_id):
            file_diffs = _start_file_diff_capture(ctx, event, file_diffs)
            await _render(ctx, event)
    node = metadata.get("langgraph_node", "") if isinstance(metadata, Mapping) else ""
    agent_type = metadata.get("ls_agent_type") if isinstance(metadata, Mapping) else None
    role = (
        "subagent"
        if namespace or agent_type == "subagent" or "subagent" in str(node)
        else "main"
    )
    if getattr(message, "content", None) or getattr(message, "usage_metadata", None):
        await _render(
            ctx,
            ModelChunk(
                chunk=message,
                role=role,
                model_name=model_labels.take(message, metadata),
                source_id=source_id,
            ),
        )
    return file_diffs


async def _updates_event(
    ctx: AppContext,
    data: Any,
    seen_results: set[str],
    seen_interrupts: set[str],
    *,
    file_diffs: _FileDiffCapture | None = None,
) -> Resolved | None:
    if not isinstance(data, dict):
        return None
    todos = _todos(data)
    ui = getattr(ctx, "ui", None)
    if todos is not None and ui is not None and hasattr(ui, "set_todos"):
        ui.set_todos(todos)
    interrupts = data.get("__interrupt__", ())
    if interrupts:
        interrupt = interrupts[0]
        interrupt_id = getattr(interrupt, "id", None)
        if isinstance(interrupt_id, str):
            if interrupt_id in seen_interrupts:
                return None
            seen_interrupts.add(interrupt_id)
        raw_payload = getattr(interrupt, "value", None)
        payload = (
            raw_payload
            if isinstance(raw_payload, dict)
            else {"action_requests": [], "value": raw_payload}
        )
        warning = "Approval unresolved; rejecting pending actions."
        try:
            resolution = await ctx._bus.emit(InterruptRaised(payload=payload))
        except Exception as exc:
            warning = (
                f"Approval handler failed ({type(exc).__name__}); "
                "rejecting pending actions."
            )
            resolution = None
        if isinstance(resolution, Resolved):
            return resolution
        actions = payload.get("action_requests", ())
        ctx.console.warning(warning)
        return Resolved(
            resume_value={
                "decisions": [{"type": "reject"} for _ in actions],
            }
        )
    for message in _messages(data):
        if not isinstance(message, ToolMessage) or message.tool_call_id in seen_results:
            continue
        seen_results.add(message.tool_call_id)
        result = file_diffs.finish(message) if file_diffs is not None else message
        await _render(
            ctx,
            ToolCallEnd(name=message.name or "tool", id=message.tool_call_id, result=result),
        )
    return None


def _pop_steering(ctx: AppContext) -> str | None:
    queue = getattr(ctx, "queue", None)
    if not isinstance(queue, PromptQueue):
        return None
    return queue.pop(mode="steer")

def _open_steering(ctx: AppContext) -> None:
    queue = getattr(ctx, "queue", None)
    if isinstance(queue, PromptQueue):
        queue.open_steering()


def _close_steering(ctx: AppContext, *, promote_pending: bool = False) -> None:
    queue = getattr(ctx, "queue", None)
    if isinstance(queue, PromptQueue):
        queue.close_steering(promote_pending=promote_pending)


def _user_input(text: str) -> dict[str, list[dict[str, str]]]:
    return {"messages": [{"role": "user", "content": text}]}


async def _run_turn(ctx: AppContext, text: str) -> None:
    if not await ctx.ensure_agent():
        return
    session_info = ctx.session.get(ctx.session_id)
    if session_info is not None and session_info.title is None:
        title = " ".join(text.split())[:80]
        if title:
            ctx.session.set_title(ctx.session_id, title)
    await ctx._bus.emit(TurnStart(thread_id=ctx.thread_id, text=text))
    _open_steering(ctx)
    next_input: Any = _user_input(text)
    tool_calls = _ToolCallBuffer()
    model_labels = _ModelLabelBuffer()
    seen_results: set[str] = set()
    seen_interrupts: set[str] = set()
    file_diffs: _FileDiffCapture | None = None
    cancelled = False
    stream_kwargs: dict[str, Any] = {
        "config": ctx.thread_config,
        "stream_mode": ["messages", "updates"],
        "subgraphs": True,
    }
    graph_nodes = getattr(ctx.agent, "nodes", None)
    if graph_nodes is None or "tools" in graph_nodes:
        stream_kwargs["interrupt_after"] = ["tools"]
    try:
        while True:
            resolution: Resolved | None = None
            static_tool_boundary = False
            async for stream_item in ctx.agent.astream(next_input, **stream_kwargs):
                if len(stream_item) == 3:
                    namespace, mode, data = stream_item
                else:
                    mode, data = stream_item
                    namespace = ()
                if (
                    not namespace
                    and mode == "updates"
                    and isinstance(data, dict)
                    and data.get("__interrupt__") == ()
                ):
                    static_tool_boundary = True
                if mode == "messages":
                    file_diffs = await _message_event(
                        ctx,
                        data,
                        tool_calls,
                        model_labels,
                        namespace,
                        file_diffs=file_diffs,
                    )
                elif mode == "updates":
                    candidate = await _updates_event(
                        ctx,
                        data,
                        seen_results,
                        seen_interrupts,
                        file_diffs=file_diffs,
                    )
                    if candidate is not None:
                        resolution = candidate
            if resolution is not None:
                next_input = Command(resume=resolution.resume_value)
                continue
            if static_tool_boundary:
                steering = _pop_steering(ctx)
                next_input = (
                    Command(update=_user_input(steering))
                    if steering is not None
                    else None
                )
                continue
            _close_steering(ctx)
            steering = _pop_steering(ctx)
            if steering is not None:
                _open_steering(ctx)
                next_input = _user_input(steering)
                continue
            break
    except asyncio.CancelledError:
        _close_steering(ctx, promote_pending=True)
        cancelled = True
        ctx.console.warning("interrupted")
    except Exception as exc:
        _close_steering(ctx, promote_pending=True)
        if not await _render(ctx, exc, emit=False):
            ctx.console.error(f"{type(exc).__name__}: {exc}")
    finally:
        _close_steering(ctx)
        try:
            ctx.capture_turn()
            if cancelled:
                ctx.record_exit("signal")
        finally:
            ctx.console.print()
            await ctx._bus.emit(TurnEnd(thread_id=ctx.thread_id))
            if ctx.rebuild_requested:
                await ctx.rebuild()


async def _run_cancellable_turn(ctx: AppContext, text: str) -> None:
    task = asyncio.create_task(_run_turn(ctx, text))
    loop = asyncio.get_running_loop()
    signal_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
        signal_installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        pass
    try:
        await task
    finally:
        if signal_installed:
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.default_int_handler)
