"""Prompt-toolkit input loop and graph stream dispatch."""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config, is_trusted_cwd
from orcha_agent.core.events import (
    AppExit,
    AppStart,
    InterruptRaised,
    ModelChunk,
    ModelSwitch,
    SessionSwitch,
    ThreadSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.ledger import (
    CompactionEntry,
    CustomEntry,
    Ledger,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
    ResetBoundaryEntry,
    build_context,
)
from orcha_agent.core.loader import PluginRecord, load_plugins
from orcha_agent.core.models import (
    ModelResolver,
    filter_foreign_blocks,
    strip_foreign_blocks,
)
from orcha_agent.core.plugin import Handled, Resolved
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore

from .console import ConsoleOutput


def _history_path() -> Path:
    return Path.home() / ".local/share/orcha-agent/history"


def _bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return bindings

def _matches(match: Any, event: object) -> bool:
    if isinstance(match, type):
        return isinstance(event, match)
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


def _primary_provider_prefix(
    spec: str | list[str],
    aliases: Mapping[str, str | list[str]],
) -> str | None:
    specs = _model_specs(spec, aliases)
    if not specs:
        return None
    prefix, separator, _ = specs[0].partition(":")
    return prefix if separator else None


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


def _reseed_foreign_block_types(registry: Registry, cfg: Config) -> set[str]:
    target_providers: set[str] = set()
    for spec in _model_specs(cfg.model, cfg.models):
        prefix, separator, _ = spec.partition(":")
        if separator:
            target_providers.add(prefix)
    foreign: set[str] = set()
    for prefix, provider in registry.providers.items():
        if prefix not in target_providers:
            foreign.update(provider.foreign_block_types)
    return foreign


def _session_resolution_error(prefix: str, exc: LookupError) -> str:
    _, marker, candidates = str(exc).partition(" is ambiguous: ")
    if marker:
        return f"Ambiguous session prefix {prefix}: {candidates}"
    return f"Unknown session: {prefix}"


def _bottom_toolbar(ctx: Any) -> Any:
    if not bool(getattr(ctx.cfg, "statusbar", True)):
        return ""
    values: list[str] = []
    for segment in ctx.registry.status_segments:
        try:
            value = segment.render(ctx)
        except Exception:
            value = f"!{segment.name}"
        if value:
            values.append(value)
    return HTML(" · ".join(values)) if values else ""


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


class RegistryView:
    """Live read-only view of plugin registrations."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._registry, name)
        if isinstance(value, dict):
            return MappingProxyType(value)
        if isinstance(value, list):
            return tuple(value)
        if callable(value) or name.startswith("_"):
            raise AttributeError(name)
        return value


class EventBusView:
    """Live read-only view of event registrations and observations."""

    def __init__(self, bus: Any) -> None:
        self._bus = bus

    async def emit(self, event: object) -> Handled | None:
        return await self._bus.emit(event)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._bus, name)
        if isinstance(value, dict):
            return MappingProxyType(value)
        if isinstance(value, list):
            return tuple(value)
        if callable(value) or name.startswith("_"):
            raise AttributeError(name)
        return value


@dataclass(slots=True)
class AppContext:
    cfg: Config
    registry: Registry | RegistryView
    bus: Any
    session: SessionStore
    plugins: list[PluginRecord]
    plugin_states: dict[str, dict[str, Any]]
    console: ConsoleOutput
    thread_id: str
    session_id: str = field(default="", kw_only=True)
    agent: Any = None
    summarizer: Any = None
    history_model: str | list[str] | None = None
    exit_requested: bool = False
    rebuild_requested: bool = False
    _title_written: bool = False
    _pending_reseed: bool = field(default=False, init=False, repr=False)
    _pending_switch_old_thread: str | None = field(
        default=None, init=False, repr=False
    )
    _registry: Registry = field(init=False, repr=False)
    _bus: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            session_info = self.session.get(self.thread_id)
            if session_info is not None and session_info.current_thread is not None:
                self.session_id = session_info.thread_id
                self.thread_id = session_info.current_thread
            else:
                thread_info = self.session.get_thread(self.thread_id)
                if thread_info is None:
                    raise ValueError(
                        "session_id is required when thread identity cannot be inferred"
                    )
                self.session_id = thread_info.session_id
        if isinstance(self.registry, Registry):
            self._registry = self.registry
            self.registry = RegistryView(self.registry)
        else:
            self._registry = self.registry._registry
        self._bus = self.bus._bus if isinstance(self.bus, EventBusView) else self.bus
        self.bus = EventBusView(self._bus)

    @property
    def ledger(self) -> Ledger:
        return Ledger(self.session)

    @property
    def thread_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}

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
            self.session.set_plugin_state(self.session_id, plugin, state)

    def _resolve_summarizer(self, cfg: Config) -> Any:
        if not self.registry.providers:
            return self.summarizer
        return ModelResolver(self.registry, cfg).resolve(
            cfg.summarizer_model or cfg.model,
            "summarizer",
        )

    def _clean_history_for_model(
        self,
        graph: Any,
        source_model: str | list[str] | None,
        target_model: str | list[str],
    ) -> None:
        if source_model is None:
            return
        if _primary_provider_prefix(
            source_model,
            self.cfg.models,
        ) == _primary_provider_prefix(target_model, self.cfg.models):
            return
        source_cfg = replace(self.cfg, model=source_model)
        foreign = _foreign_block_types(self.registry, source_cfg)
        if foreign:
            strip_foreign_blocks(graph, self.thread_config, foreign)

    def report_provider_error(self, exc: Exception) -> None:
        self.console.error(
            f"{type(exc).__name__}: {exc}\n"
            "Set the required provider environment variable, or `/login codex`, "
            "or `/model <prefix:model>`."
        )

    async def ensure_agent(self, *, seed_pending: bool = True) -> bool:
        if self.agent is not None:
            if self._pending_reseed and seed_pending:
                await self._seed_ready_thread("reseed")
            return True
        try:
            candidate_agent = await build_agent(
                self.registry,
                self.cfg,
                self.session,
                self._bus,
                always_allowed=self._always_allowed(),
            )
            candidate_summarizer = self._resolve_summarizer(self.cfg)
            if not self._pending_reseed:
                self._clean_history_for_model(
                    candidate_agent,
                    self.history_model,
                    self.cfg.model,
                )
        except Exception as exc:
            self.report_provider_error(exc)
            return False
        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.rebuild_requested = False
        if self._pending_reseed and seed_pending:
            await self._seed_ready_thread("reseed")
        self.history_model = None
        return True

    async def rebuild(self) -> None:
        if self.agent is None:
            self.rebuild_requested = False
            return
        self.persist_plugin_states()
        candidate_agent = await build_agent(
            self.registry,
            self.cfg,
            self.session,
            self._bus,
            always_allowed=self._always_allowed(),
        )
        candidate_summarizer = self._resolve_summarizer(self.cfg)
        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.rebuild_requested = False
        if self._pending_reseed:
            await self._seed_ready_thread("reseed")

    def _activate_seeded_thread(
        self,
        thread_id: str,
        *,
        seeded_from: str | None,
        captured: int,
    ) -> None:
        with self.session.saver.lock:
            connection = self.session._connection
            connection.execute("BEGIN")
            try:
                connection.execute(
                    """
                    INSERT INTO threads(thread_id, session_id, seeded_from, captured)
                    VALUES (?, ?, ?, ?)
                    """,
                    (thread_id, self.session_id, seeded_from, captured),
                )
                cursor = connection.execute(
                    """
                    UPDATE sessions
                    SET current_thread = ?
                    WHERE thread_id = ?
                    """,
                    (thread_id, self.session_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError(f"Unknown session: {self.session_id}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    async def _seed_ready_thread(self, reason: str) -> None:
        old_thread = self._pending_switch_old_thread or self.thread_id
        pending_thread = (
            self._pending_reseed
            and self.thread_id.startswith(f"{self.session_id}.")
            and self.session.get_thread(self.thread_id) is None
        )
        new_thread = (
            self.thread_id
            if pending_thread
            else self.session.next_thread_id(self.session_id)
        )
        if pending_thread:
            self.session.saver.delete_thread(new_thread)
        context = build_context(
            self.ledger.path(self.session_id),
            strip=_reseed_foreign_block_types(self.registry, self.cfg),
        )
        config = {"configurable": {"thread_id": new_thread}}
        activated = False
        try:
            await self.agent.aupdate_state(
                config,
                {
                    "messages": context.messages,
                    "todos": context.todos,
                    "files": context.files,
                },
            )
            self._activate_seeded_thread(
                new_thread,
                seeded_from=self.ledger.leaf(self.session_id),
                captured=len(context.messages),
            )
            activated = True
        except BaseException:
            if not activated:
                self.session.saver.delete_thread(new_thread)
            raise
        self.thread_id = new_thread
        self._pending_reseed = False
        self._pending_switch_old_thread = None
        await self._bus.emit(
            ThreadSwitch(
                session_id=self.session_id,
                old=old_thread,
                new=new_thread,
                reason=reason,
            )
        )

    def _restore_position(
        self,
        *,
        leaf_id: str | None,
        thread_id: str | None,
    ) -> None:
        self.ledger.restore_position(
            self.session_id,
            leaf_id=leaf_id,
            thread_id=thread_id,
        )

    def _persisted_current_thread(self) -> str | None:
        session = self.session.get(self.session_id)
        if session is None:
            raise LookupError(f"Unknown session: {self.session_id}")
        return session.current_thread

    async def seed_thread(self, reason: str) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        await self._seed_ready_thread(reason)

    async def branch(self, entry_id: str) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        prior_leaf = self.ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        self.ledger.branch_for_reseed(self.session_id, entry_id)
        try:
            await self._seed_ready_thread("branch")
        except BaseException:
            if self.thread_id == prior_thread:
                self._restore_position(
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                )
            raise

    async def fork(self) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        old_session = self.session_id
        old_thread = self.thread_id
        self.persist_plugin_states()
        created = self.session.create(
            self.cfg.cwd,
            self.cfg.model,
            mode=self.cfg.mode,
            parent_session=old_session,
        )
        try:
            self.ledger.fork(old_session, created.thread_id)
            self.session_id = created.thread_id
            self.thread_id = old_thread
            await self._seed_ready_thread("reseed")
        except BaseException:
            self.session_id = old_session
            self.thread_id = old_thread
            self.session.delete_session(created.thread_id)
            raise
        await self._bus.emit(SessionSwitch(old=old_session, new=self.session_id))

    async def new_session(self) -> None:
        old_session = self.session_id
        old_states = deepcopy(self.plugin_states)
        old_agent = self.agent
        self.persist_plugin_states()
        for state in self.plugin_states.values():
            state.clear()
        try:
            if old_agent is not None:
                candidate_agent = await build_agent(
                    self.registry,
                    self.cfg,
                    self.session,
                    self._bus,
                    always_allowed=self._always_allowed(),
                )
                candidate_summarizer = self._resolve_summarizer(self.cfg)
            else:
                candidate_agent = None
                candidate_summarizer = None
            created = self.session.create(
                self.cfg.cwd,
                self.cfg.model,
                mode=self.cfg.mode,
            )
            if created.current_thread is None:
                self.session.delete_session(created.thread_id)
                raise RuntimeError(f"Session {created.thread_id} has no graph thread")
        except BaseException:
            for name, state in self.plugin_states.items():
                state.clear()
                state.update(old_states.get(name, {}))
            raise
        self.session_id = created.thread_id
        self.thread_id = created.current_thread
        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.history_model = None
        self._pending_reseed = False
        self._pending_switch_old_thread = None
        self.rebuild_requested = False
        self.console.console.clear()
        await self._bus.emit(SessionSwitch(old=old_session, new=self.session_id))

    async def clear(self) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        prior_leaf = self.ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        self.ledger.append_for_reseed(self.session_id, ResetBoundaryEntry())
        try:
            await self._seed_ready_thread("clear")
        except BaseException:
            if self.thread_id == prior_thread:
                self._restore_position(
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                )
            raise
        self.console.console.clear()

    async def resume(self, prefix: str) -> None:
        try:
            saved_session = self.session.resolve_session(prefix)
        except LookupError as exc:
            self.console.error(_session_resolution_error(prefix, exc))
            return

        old_session = self.session_id
        old_thread = self.thread_id
        old_cfg = self.cfg
        old_agent = self.agent
        old_summarizer = self.summarizer
        old_history_model = self.history_model
        old_pending_reseed = self._pending_reseed
        old_pending_switch = self._pending_switch_old_thread
        old_rebuild_requested = self.rebuild_requested
        old_states = deepcopy(self.plugin_states)
        self.persist_plugin_states()
        saved_states = self.session.all_plugin_state(saved_session.thread_id)
        for name, state in self.plugin_states.items():
            state.clear()
            state.update(saved_states.get(name, {}))
        candidate_cfg = replace(
            self.cfg,
            cwd=Path(saved_session.cwd),
            model=(
                self.cfg.model
                if self.cfg.model_overridden
                else _stored_model(saved_session.model)
            ),
            mode=saved_session.mode,
            trust_cwd=is_trusted_cwd(
                saved_session.cwd,
                self.cfg.trusted_dirs,
                trust_all=self.cfg.trust_all_cwd,
            ),
        )
        stored_model = _stored_model(saved_session.model)
        live_thread = saved_session.current_thread
        checkpoint_live = (
            live_thread is not None
            and self.session.checkpoint_exists(live_thread)
        )
        target_position_changed = False
        try:
            if checkpoint_live:
                self.recover_checkpoint(saved_session.thread_id, live_thread)
            context = build_context(self.ledger.path(saved_session.thread_id))
            pending_interrupt = (
                checkpoint_live
                and self.session.checkpoint_has_pending_interrupt(live_thread)
            )
            needs_reseed = not checkpoint_live or (
                bool(context.dangling) and not pending_interrupt
            )
            target_thread = (
                self.session.next_thread_id(saved_session.thread_id)
                if needs_reseed
                else live_thread
            )
            if needs_reseed:
                self.ledger.restore_position(
                    saved_session.thread_id,
                    leaf_id=self.ledger.leaf(saved_session.thread_id),
                    thread_id=None,
                )
                target_position_changed = True
            self.session_id = saved_session.thread_id
            self.thread_id = target_thread
            self.cfg = candidate_cfg
            self.history_model = stored_model
            self._pending_reseed = needs_reseed
            self._pending_switch_old_thread = old_thread if needs_reseed else None
            if old_agent is None:
                self.summarizer = None
            else:
                candidate_agent = await build_agent(
                    self.registry,
                    candidate_cfg,
                    self.session,
                    self._bus,
                    always_allowed=self._always_allowed(),
                )
                candidate_summarizer = self._resolve_summarizer(candidate_cfg)
                if not needs_reseed:
                    self._clean_history_for_model(
                        candidate_agent,
                        stored_model,
                        candidate_cfg.model,
                    )
                self.agent = candidate_agent
                self.summarizer = candidate_summarizer
                if needs_reseed:
                    await self._seed_ready_thread("reseed")
                self.history_model = None
            self.rebuild_requested = False
        except BaseException:
            if target_position_changed:
                self.ledger.restore_position(
                    saved_session.thread_id,
                    leaf_id=self.ledger.leaf(saved_session.thread_id),
                    thread_id=live_thread,
                )
            self.session_id = old_session
            self.thread_id = old_thread
            self.cfg = old_cfg
            self.agent = old_agent
            self.summarizer = old_summarizer
            self.history_model = old_history_model
            self._pending_reseed = old_pending_reseed
            self._pending_switch_old_thread = old_pending_switch
            self.rebuild_requested = old_rebuild_requested
            for name, state in self.plugin_states.items():
                state.clear()
                state.update(old_states.get(name, {}))
            raise

        await self._bus.emit(
            SessionSwitch(old=old_session, new=saved_session.thread_id)
        )
        self._warn_interrupted_resume()

    async def switch_model(self, spec: str | list[str]) -> None:
        old_model = self.cfg.model
        old_label = old_model if isinstance(old_model, str) else ",".join(old_model)
        new_label = spec if isinstance(spec, str) else ",".join(spec)
        candidate_cfg = replace(self.cfg, model=spec)
        candidate_agent = await build_agent(
            self.registry,
            candidate_cfg,
            self.session,
            self._bus,
            always_allowed=self._always_allowed(),
        )
        candidate_summarizer = self._resolve_summarizer(candidate_cfg)
        provider_changed = _primary_provider_prefix(
            self.cfg.model,
            self.cfg.models,
        ) != _primary_provider_prefix(spec, self.cfg.models)
        foreign = (
            _foreign_block_types(self.registry, self.cfg)
            if provider_changed
            else set()
        )
        prior_leaf = self.ledger.leaf(self.session_id)
        prior_persisted_thread = self._persisted_current_thread()
        audit_appended = False
        self.session.set_model(self.session_id, spec)
        try:
            self.ledger.append(
                self.session_id,
                ModelChangeEntry(model=spec if isinstance(spec, str) else list(spec)),
            )
            audit_appended = True
            if foreign:
                strip_foreign_blocks(
                    self.agent or candidate_agent,
                    self.thread_config,
                    foreign,
                )
        except BaseException:
            self.session.set_model(self.session_id, old_model)
            if audit_appended:
                self._restore_position(
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                )
            raise
        self.cfg = candidate_cfg
        self.summarizer = candidate_summarizer
        self.agent = candidate_agent
        self.rebuild_requested = False
        await self._bus.emit(ModelSwitch(old=old_label, new=new_label))

    async def switch_mode(self, name: str) -> None:
        if name not in self.registry.modes:
            self.console.error(f"Unknown mode: {name}")
            return
        old_mode = self.cfg.mode
        candidate_cfg = replace(self.cfg, mode=name)
        if self.agent is None:
            candidate_agent = None
            candidate_summarizer = None
        else:
            candidate_agent = await build_agent(
                self.registry,
                candidate_cfg,
                self.session,
                self._bus,
                always_allowed=self._always_allowed(),
            )
            candidate_summarizer = self._resolve_summarizer(candidate_cfg)
        self.session.set_mode(self.session_id, name)
        try:
            self.ledger.append(self.session_id, ModeChangeEntry(mode=name))
        except BaseException:
            self.session.set_mode(self.session_id, old_mode)
            raise
        self.cfg = candidate_cfg
        if candidate_agent is not None:
            self.agent = candidate_agent
            self.summarizer = candidate_summarizer
        self.rebuild_requested = False

    async def compact(self) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        instruction = (
            "Summarize the conversation for continuation. Preserve decisions, current files, "
            "constraints, failures, and remaining work."
        )
        messages = filter_foreign_blocks(
            build_context(self.ledger.path(self.session_id)).messages,
            {"reasoning", "thinking"},
        )
        if self.summarizer is None:
            raise RuntimeError("summarizer model is unavailable")
        summary = await self.summarizer.ainvoke(
            [*messages, HumanMessage(content=instruction)]
        )
        summary_text = (
            summary.content
            if isinstance(summary.content, str)
            else str(summary.content)
        )
        prior_leaf = self.ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        self.ledger.append_for_reseed(
            self.session_id,
            CompactionEntry(
                summary=summary_text,
                first_kept_id=None,
                tokens_before=None,
            ),
        )
        try:
            await self._seed_ready_thread("compact")
        except BaseException:
            if self.thread_id == prior_thread:
                self._restore_position(
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                )
            raise
        self.console.print("Conversation compacted.")

    def _capture_values(
        self,
        session_id: str,
        thread_id: str,
        values: Mapping[str, Any],
        *,
        only_if_new: bool,
    ) -> bool:
        thread = self.session.get_thread(thread_id)
        if thread is None or thread.session_id != session_id:
            raise LookupError(f"Unknown graph thread: {thread_id}")
        new_messages = values.get("messages", ())[thread.captured :]
        if only_if_new and not new_messages:
            return False
        entries = [
            MessageEntry(message=message_to_dict(message))
            for message in new_messages
        ]
        entries.append(
            CustomEntry(
                custom_type="turn_state",
                data={
                    "todos": values.get("todos", []),
                    "files": values.get("files", {}),
                },
            )
        )
        try:
            self.ledger.capture(
                session_id,
                thread_id,
                entries,
                message_count=len(new_messages),
            )
        except Exception as exc:
            message = (
                f"Failed to capture session {session_id} "
                f"thread {thread_id}: {exc}"
            )
            self.console.error(message)
            raise RuntimeError(message) from exc
        return True

    def recover_checkpoint(self, session_id: str, thread_id: str) -> bool:
        values = self.session.checkpoint_values(thread_id)
        if values is None:
            return False
        return self._capture_values(
            session_id,
            thread_id,
            values,
            only_if_new=True,
        )

    def capture_turn(self) -> None:
        state = self.agent.get_state(self.thread_config)
        values = getattr(state, "values", {})
        self._capture_values(
            self.session_id,
            self.thread_id,
            values,
            only_if_new=False,
        )

    def record_exit(self, kind: str) -> None:
        path = self.ledger.path(self.session_id)
        context = build_context(path)
        if kind == "normal":
            has_assistant = any(
                isinstance(messages_from_dict([entry.message])[0], AIMessage)
                for entry in path
                if isinstance(entry, MessageEntry)
            )
            if not has_assistant:
                return
            pending = []
        else:
            pending = [
                {"id": reference.id, "name": reference.name}
                for reference in context.dangling
            ]
        self.ledger.append(
            self.session_id,
            CustomEntry(
                custom_type="session_exit",
                data={"kind": kind, "pending_tool_calls": pending},
            ),
        )

    def _warn_interrupted_resume(self) -> None:
        if self.session.checkpoint_has_pending_interrupt(self.thread_id):
            return
        path = self.ledger.path(self.session_id)
        last_message = next(
            (entry for entry in reversed(path) if isinstance(entry, MessageEntry)),
            None,
        )
        if last_message is None:
            return
        message = messages_from_dict([last_message.message])[0]
        if not isinstance(message, AIMessage):
            return
        call_ids = {
            call_id
            for call in message.tool_calls
            if isinstance(call, Mapping)
            and isinstance((call_id := call.get("id")), str)
        }
        pending = [
            reference
            for reference in build_context(path).dangling
            if reference.id in call_ids
        ]
        if pending:
            self.console.warning(
                "Previous turn was interrupted; "
                f"{len(pending)} pending tool call(s) dropped."
            )


async def _render(ctx: AppContext, event: object, *, emit: bool = True) -> bool:
    if emit:
        bus = getattr(ctx, "_bus", ctx.bus)
        handled = await bus.emit(event)
        if isinstance(handled, Handled):
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
) -> None:
    if not isinstance(data, tuple) or len(data) != 2:
        return
    message, metadata = data
    if not isinstance(message, BaseMessage):
        return
    if isinstance(message, ToolMessage):
        return
    source_id = "/".join(str(part) for part in namespace) if namespace else "main"
    if isinstance(message, AIMessage):
        for event in tool_calls.add(message, source_id=source_id):
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


async def _updates_event(
    ctx: AppContext,
    data: Any,
    seen_results: set[str],
    seen_interrupts: set[str],
) -> Resolved | None:
    if not isinstance(data, dict):
        return None
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
        await _render(
            ctx,
            ToolCallEnd(name=message.name or "tool", id=message.tool_call_id, result=message),
        )
    return None


async def _run_turn(ctx: AppContext, text: str) -> None:
    if not await ctx.ensure_agent():
        return
    session_info = ctx.session.get(getattr(ctx, "session_id", ctx.thread_id))
    if session_info is not None and session_info.title is None:
        title = " ".join(text.split())[:80]
        if title:
            ctx.session.set_title(getattr(ctx, "session_id", ctx.thread_id), title)
    await ctx._bus.emit(TurnStart(thread_id=ctx.thread_id, text=text))
    next_input: Any = {"messages": [{"role": "user", "content": text}]}
    tool_calls = _ToolCallBuffer()
    model_labels = _ModelLabelBuffer()
    seen_results: set[str] = set()
    seen_interrupts: set[str] = set()
    cancelled = False
    try:
        while True:
            resolution: Resolved | None = None
            async for stream_item in ctx.agent.astream(
                next_input,
                config=ctx.thread_config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                if len(stream_item) == 3:
                    namespace, mode, data = stream_item
                else:
                    mode, data = stream_item
                    namespace = ()
                if mode == "messages":
                    await _message_event(
                        ctx,
                        data,
                        tool_calls,
                        model_labels,
                        namespace,
                    )
                elif mode == "updates":
                    candidate = await _updates_event(
                        ctx,
                        data,
                        seen_results,
                        seen_interrupts,
                    )
                    if candidate is not None:
                        resolution = candidate
            if resolution is None:
                break
            next_input = Command(resume=resolution.resume_value)
    except asyncio.CancelledError:
        cancelled = True
        ctx.console.warning("interrupted")
    except Exception as exc:
        if not await _render(ctx, exc, emit=False):
            ctx.console.error(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            capture_turn = getattr(ctx, "capture_turn", None)
            if callable(capture_turn):
                capture_turn()
            if cancelled:
                record_exit = getattr(ctx, "record_exit", None)
                if callable(record_exit):
                    record_exit("signal")
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


async def run_app(cfg: Config) -> int:
    """Compose plugins and run the interactive terminal application."""

    try:
        store = SessionStore(cfg.db_path)
    except Exception as exc:
        ConsoleOutput().error(f"Cannot open session database {cfg.db_path}: {exc}")
        return 1
    with store:
        history_model: str | list[str] | None = None
        pending_reseed = False
        pending_switch_old_thread: str | None = None
        resume_live_thread: str | None = None
        if cfg.list_sessions:
            console = ConsoleOutput()
            for session in store.list():
                console.print(f"{session.thread_id}  {session.cwd}  {session.title or ''}")
            return 0
        if cfg.resume:
            try:
                saved_session = store.resolve_session(cfg.resume)
            except LookupError as exc:
                ConsoleOutput().error(_session_resolution_error(cfg.resume, exc))
                return 1
            history_model = _stored_model(saved_session.model)
            cfg = replace(
                cfg,
                cwd=Path(saved_session.cwd),
                model=(
                    cfg.model
                    if cfg.model_overridden
                    else history_model
                ),
                mode=saved_session.mode,
                trust_cwd=is_trusted_cwd(
                    saved_session.cwd,
                    cfg.trusted_dirs,
                    trust_all=cfg.trust_all_cwd,
                ),
            )
            session_id = saved_session.thread_id
            resume_live_thread = saved_session.current_thread
            checkpoint_live = (
                resume_live_thread is not None
                and store.checkpoint_exists(resume_live_thread)
            )
            pending_reseed = not checkpoint_live
            if pending_reseed:
                thread_id = store.next_thread_id(session_id)
                pending_switch_old_thread = resume_live_thread
                Ledger(store).restore_position(
                    session_id,
                    leaf_id=Ledger(store).leaf(session_id),
                    thread_id=None,
                )
            else:
                thread_id = resume_live_thread
        else:
            created = store.create(
                cfg.cwd,
                cfg.model,
                mode=cfg.mode,
            )
            if created.current_thread is None:
                raise RuntimeError(f"Session {created.thread_id} has no graph thread")
            session_id = created.thread_id
            thread_id = created.current_thread

        registry = Registry()
        from orcha_agent.core.events import EventBus

        bus = EventBus()
        states = store.all_plugin_state(session_id)
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
            session_id=session_id,
            history_model=history_model,
        )
        ctx._pending_reseed = pending_reseed
        ctx._pending_switch_old_thread = pending_switch_old_thread
        holder["ctx"] = ctx
        if cfg.resume and resume_live_thread is not None and store.checkpoint_exists(
            resume_live_thread
        ):
            ctx.recover_checkpoint(session_id, resume_live_thread)
            context = build_context(ctx.ledger.path(session_id))
            pending_interrupt = store.checkpoint_has_pending_interrupt(
                resume_live_thread
            )
            if context.dangling and not pending_interrupt:
                old_thread = ctx.thread_id
                ctx.ledger.restore_position(
                    session_id,
                    leaf_id=ctx.ledger.leaf(session_id),
                    thread_id=None,
                )
                ctx.thread_id = store.next_thread_id(session_id)
                ctx._pending_reseed = True
                ctx._pending_switch_old_thread = old_thread
        await bus.emit(AppStart(ctx=ctx))
        if ctx._pending_reseed and ctx.agent is not None:
            await ctx.ensure_agent()
        if ctx.rebuild_requested:
            await ctx.rebuild()
        if cfg.resume:
            ctx._warn_interrupted_resume()

        history_path = _history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        prompt: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_path)),
            multiline=True,
            bottom_toolbar=lambda: _bottom_toolbar(ctx),
            refresh_interval=0.5,
            key_bindings=_bindings(),
        )
        while not ctx.exit_requested:
            try:
                text = (await prompt.prompt_async("> ")).strip()
                if not text:
                    continue
                if not text.startswith("/"):
                    first_word = text.split(maxsplit=1)[0]
                    if first_word in registry.commands:
                        ctx.console.warning(f"Did you mean /{text}?")
                        continue
                if await dispatch_command(registry, ctx, text):
                    if ctx.rebuild_requested:
                        await ctx.rebuild()
                    continue
                await _run_cancellable_turn(ctx, text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                ctx.console.warning("interrupted")
            except EOFError:
                break
            except Exception as exc:
                ctx.console.error(f"{type(exc).__name__}: {exc}")
        ctx.persist_plugin_states()
        ctx.record_exit("normal")
        await bus.emit(AppExit())
        return 0
