"""Prompt-toolkit input loop and graph stream dispatch."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config
from orcha_agent.core.events import (
    AppExit,
    AppStart,
    InterruptRaised,
    ModelChunk,
    ModelSwitch,
    SessionSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.loader import PluginRecord, load_plugins
from orcha_agent.core.models import ModelResolver, strip_foreign_blocks
from orcha_agent.core.plugin import Resolved
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore

from .console import ConsoleOutput


def _history_path() -> Path:
    return Path.home() / ".local/share/orcha-agent/history"


def _bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _accept(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    return bindings


def _matches(match: Any, event: object) -> bool:
    if callable(match):
        return bool(match(event))
    return match == type(event).__name__ or match == getattr(event, "name", None)


def _stored_model(value: str | list[str]) -> str | list[str]:
    if isinstance(value, list):
        return list(value)
    models = [model for model in value.split(",") if model]
    return models[0] if len(models) == 1 else models


def _model_specs(
    spec: str | list[str],
    aliases: Mapping[str, str | list[str]],
    seen: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    if isinstance(spec, list):
        return tuple(
            expanded
            for model in spec
            for expanded in _model_specs(model, aliases, seen)
        )
    target = aliases.get(spec)
    if target is None or spec in seen:
        return (spec,)
    return _model_specs(target, aliases, seen | {spec})


def _foreign_block_types(registry: Registry, cfg: Config) -> set[str]:
    foreign: set[str] = set()
    for spec in _model_specs(cfg.model, cfg.models):
        prefix, separator, _ = spec.partition(":")
        if not separator:
            continue
        provider = registry.providers.get(prefix)
        if provider is not None:
            foreign.update(provider.foreign_block_types)
    return foreign


async def dispatch_command(registry: Registry, ctx: Any, text: str) -> bool:
    """Dispatch slash commands without invoking the model."""

    if not text.startswith("/"):
        return False
    command_text = text[1:]
    name, separator, args = command_text.partition(" ")
    registration = registry.commands.get(name)
    if registration is None:
        ctx.console.error(f"Unknown command: /{name}")
        return True
    await registration.handler(ctx, args if separator else "")
    return True


@dataclass(slots=True)
class AppContext:
    cfg: Config
    registry: Registry
    bus: Any
    session: SessionStore
    plugins: list[PluginRecord]
    plugin_states: dict[str, dict[str, Any]]
    console: ConsoleOutput
    thread_id: str
    agent: Any = None
    summarizer: Any = None
    exit_requested: bool = False
    rebuild_requested: bool = False
    _title_written: bool = False

    def request_rebuild(self) -> None:
        self.rebuild_requested = True

    def _always_allowed(self) -> set[str]:
        allowed: set[str] = set()
        for state in self.plugin_states.values():
            value = state.get("always_allowed", ())
            if isinstance(value, list):
                allowed.update(item for item in value if isinstance(item, str))
        return allowed

    def persist_plugin_states(self) -> None:
        for plugin, state in self.plugin_states.items():
            self.session.set_plugin_state(self.thread_id, plugin, state)

    def _resolve_summarizer(self, cfg: Config) -> Any:
        if not self.registry.providers:
            return self.summarizer
        return ModelResolver(self.registry, cfg).resolve(
            cfg.summarizer_model,
            "summarizer",
        )

    async def rebuild(self) -> None:
        self.persist_plugin_states()
        candidate_agent = await build_agent(
            self.registry,
            self.cfg,
            self.session,
            self.bus,
            always_allowed=self._always_allowed(),
        )
        candidate_summarizer = self._resolve_summarizer(self.cfg)
        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.rebuild_requested = False

    async def clear(self) -> None:
        old = self.thread_id
        created = self.session.create(self.cfg.cwd, self.cfg.model)
        self.thread_id = created.thread_id
        for state in self.plugin_states.values():
            state.clear()
        await self.rebuild()
        self.console.console.clear()
        await self.bus.emit(SessionSwitch(old=old, new=self.thread_id))

    async def resume(self, thread_id: str) -> None:
        saved_session = self.session.get(thread_id)
        if saved_session is None:
            self.console.error(f"Unknown session: {thread_id}")
            return

        old = self.thread_id
        old_cfg = self.cfg
        old_rebuild_requested = self.rebuild_requested
        old_states = deepcopy(self.plugin_states)
        self.persist_plugin_states()
        self.thread_id = thread_id
        saved_states = self.session.all_plugin_state(thread_id)
        for name, state in self.plugin_states.items():
            state.clear()
            state.update(saved_states.get(name, {}))
        candidate_cfg = replace(
            self.cfg,
            cwd=Path(saved_session.cwd),
            model=_stored_model(saved_session.model),
        )
        self.cfg = candidate_cfg
        try:
            candidate_agent = await build_agent(
                self.registry,
                self.cfg,
                self.session,
                self.bus,
                always_allowed=self._always_allowed(),
            )
            candidate_summarizer = self._resolve_summarizer(candidate_cfg)
        except Exception:
            self.cfg = old_cfg
            self.rebuild_requested = old_rebuild_requested
            self.thread_id = old
            for name, state in self.plugin_states.items():
                state.clear()
                state.update(old_states.get(name, {}))
            raise

        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.rebuild_requested = False
        await self.bus.emit(SessionSwitch(old=old, new=thread_id))

    async def switch_model(self, spec: str) -> None:
        old = self.cfg.model if isinstance(self.cfg.model, str) else ",".join(self.cfg.model)
        candidate_cfg = replace(self.cfg, model=spec)
        candidate_agent = await build_agent(
            self.registry,
            candidate_cfg,
            self.session,
            self.bus,
            always_allowed=self._always_allowed(),
        )
        candidate_summarizer = self._resolve_summarizer(candidate_cfg)
        foreign = _foreign_block_types(self.registry, self.cfg)
        if foreign:
            strip_foreign_blocks(
                self.agent,
                self.thread_config,
                foreign,
            )
        self.cfg = candidate_cfg
        self.summarizer = candidate_summarizer
        self.agent = candidate_agent
        self.rebuild_requested = False
        await self.bus.emit(ModelSwitch(old=old, new=spec))

    async def switch_mode(self, name: str) -> None:
        if name not in self.registry.modes:
            self.console.error(f"Unknown mode: {name}")
            return
        self.cfg = replace(self.cfg, mode=name)
        await self.rebuild()

    async def compact(self) -> None:
        instruction = (
            "Summarize the conversation for continuation. Preserve decisions, current files, "
            "constraints, failures, and remaining work."
        )
        state = self.agent.get_state(self.thread_config)
        messages = list(getattr(state, "values", {}).get("messages", ()))
        if self.summarizer is None:
            raise RuntimeError("summarizer model is unavailable")
        summary = self.summarizer.invoke(
            [*messages, HumanMessage(content=instruction)]
        )
        self.agent.update_state(
            self.thread_config,
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), summary]},
        )
        self.console.print("Conversation compacted.")

    @property
    def thread_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}


async def _render(ctx: AppContext, event: object, *, emit: bool = True) -> bool:
    if emit:
        await ctx.bus.emit(event)
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


class _ToolCallBuffer:
    """Assemble provider tool-call chunks before rendering their arguments."""

    def __init__(self) -> None:
        self._pending: dict[int | str, _PendingToolCall] = {}
        self._emitted: set[str] = set()

    def add(self, message: AIMessage) -> list[ToolCallStart]:
        chunks = getattr(message, "tool_call_chunks", ())
        if not chunks:
            events: list[ToolCallStart] = []
            for call in message.tool_calls:
                identifier = call.get("id") or f"{call['name']}:{len(self._emitted)}"
                if identifier in self._emitted:
                    continue
                self._emitted.add(identifier)
                events.append(
                    ToolCallStart(
                        name=call["name"],
                        args=call.get("args", {}),
                        id=identifier,
                    )
                )
            return events

        completed: list[ToolCallStart] = []
        for chunk in chunks:
            key = chunk.get("index")
            if key is None:
                key = chunk.get("id") or len(self._pending)
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
            if not pending.name or not pending.id or pending.id in self._emitted:
                continue
            try:
                parsed_args = json.loads(pending.args)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed_args, dict):
                continue
            self._emitted.add(pending.id)
            completed.append(
                ToolCallStart(
                    name=pending.name,
                    args=parsed_args,
                    id=pending.id,
                )
            )
            self._pending.pop(key, None)
        return completed


async def _message_event(
    ctx: AppContext,
    data: Any,
    tool_calls: _ToolCallBuffer,
    model_labels: _ModelLabelBuffer,
) -> None:
    if not isinstance(data, tuple) or len(data) != 2:
        return
    message, metadata = data
    if not isinstance(message, BaseMessage):
        return
    if isinstance(message, ToolMessage):
        return
    if isinstance(message, AIMessage):
        for event in tool_calls.add(message):
            await _render(ctx, event)
    node = metadata.get("langgraph_node", "") if isinstance(metadata, Mapping) else ""
    role = "subagent" if "subagent" in str(node) else "main"
    if getattr(message, "content", None):
        await _render(
            ctx,
            ModelChunk(
                chunk=message,
                role=role,
                model_name=model_labels.take(message, metadata),
            ),
        )


async def _updates_event(ctx: AppContext, data: Any, seen_results: set[str]) -> Resolved | None:
    if not isinstance(data, dict):
        return None
    interrupts = data.get("__interrupt__", ())
    if interrupts:
        payload = interrupts[0].value
        resolution = await ctx.bus.emit(InterruptRaised(payload=payload))
        return resolution if isinstance(resolution, Resolved) else None
    for message in _messages(data):
        if not isinstance(message, ToolMessage) or message.tool_call_id in seen_results:
            continue
        seen_results.add(message.tool_call_id)
        await _render(
            ctx,
            ToolCallEnd(name=message.name or "tool", id=message.tool_call_id, result=message),
        )
    return None


async def _run_turn(ctx: AppContext, text: str) -> None:
    session_info = ctx.session.get(ctx.thread_id)
    if session_info is not None and session_info.title is None:
        title = " ".join(text.split())[:80]
        if title:
            ctx.session.set_title(ctx.thread_id, title)
    await ctx.bus.emit(TurnStart(thread_id=ctx.thread_id, text=text))
    next_input: Any = {"messages": [{"role": "user", "content": text}]}
    tool_calls = _ToolCallBuffer()
    model_labels = _ModelLabelBuffer()
    seen_results: set[str] = set()
    try:
        while True:
            resolution: Resolved | None = None
            async for mode, data in ctx.agent.astream(
                next_input,
                config=ctx.thread_config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    await _message_event(ctx, data, tool_calls, model_labels)
                elif mode == "updates":
                    candidate = await _updates_event(ctx, data, seen_results)
                    if candidate is not None:
                        resolution = candidate
            if resolution is None:
                break
            next_input = Command(resume=resolution.resume_value)
            if ctx.rebuild_requested:
                await ctx.rebuild()
    except Exception as exc:
        if not await _render(ctx, exc, emit=False):
            ctx.console.error(f"{type(exc).__name__}: {exc}")
    finally:
        ctx.console.print()
        await ctx.bus.emit(TurnEnd(thread_id=ctx.thread_id))


async def run_app(cfg: Config) -> int:
    """Compose plugins and run the interactive terminal application."""

    try:
        store = SessionStore(cfg.db_path)
    except Exception as exc:
        ConsoleOutput().error(f"Cannot open session database {cfg.db_path}: {exc}")
        return 1
    with store:
        if cfg.list_sessions:
            console = ConsoleOutput()
            for session in store.list():
                console.print(f"{session.thread_id}  {session.cwd}  {session.title or ''}")
            return 0
        if cfg.resume:
            saved_session = store.get(cfg.resume)
            if saved_session is None:
                ConsoleOutput().error(f"Unknown session: {cfg.resume}")
                return 1
            cfg = replace(
                cfg,
                cwd=Path(saved_session.cwd),
                model=_stored_model(saved_session.model),
            )
            thread_id = saved_session.thread_id
        else:
            thread_id = store.create(cfg.cwd, cfg.model).thread_id

        registry = Registry()
        from orcha_agent.core.events import EventBus

        bus = EventBus()
        states = store.all_plugin_state(thread_id)
        holder: dict[str, AppContext] = {}

        def request_rebuild() -> None:
            if "ctx" in holder:
                holder["ctx"].request_rebuild()

        records = load_plugins(registry, bus, cfg, states, request_rebuild)
        ctx = AppContext(
            cfg=cfg,
            registry=registry,
            bus=bus,
            session=store,
            plugins=records,
            plugin_states=states,
            console=ConsoleOutput(),
            thread_id=thread_id,
        )
        holder["ctx"] = ctx
        try:
            await ctx.rebuild()
        except Exception as exc:
            ctx.console.error(f"{type(exc).__name__}: {exc}")
            return 1
        await bus.emit(AppStart(ctx=ctx))

        history_path = _history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        prompt: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            multiline=True,
            key_bindings=_bindings(),
        )
        while not ctx.exit_requested:
            try:
                text = (await prompt.prompt_async("> ")).strip()
                if not text:
                    continue
                if await dispatch_command(registry, ctx, text):
                    continue
                await _run_turn(ctx, text)
            except KeyboardInterrupt:
                ctx.console.warning("interrupted")
            except EOFError:
                break
            except Exception as exc:
                ctx.console.error(f"{type(exc).__name__}: {exc}")
        ctx.persist_plugin_states()
        await bus.emit(AppExit())
        return 0
