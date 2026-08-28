"""Application context and session operations."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    message_to_dict,
    messages_from_dict,
)

from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config, is_trusted_cwd
from orcha_agent.core.events import ModelSwitch, SessionSwitch, ThreadSwitch
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
from orcha_agent.core.loader import PluginRecord
from orcha_agent.core.models import (
    ModelResolver,
    filter_foreign_blocks,
    strip_foreign_blocks,
)
from orcha_agent.core.plugin import Handled
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore

from .console import ConsoleOutput

def _compat(name: str, default: Any) -> Any:
    facade = sys.modules.get("orcha_agent.tui.app")
    return getattr(facade, name, default) if facade is not None else default


_SUMMARIZATION_PREFIX = "Here is a summary of the conversation to date:\n\n"

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


def _uncheckpointed_seed_target(
    store: SessionStore,
    session_id: str,
    current_thread: str | None,
) -> str:
    for candidate in (current_thread, f"{session_id}.0"):
        if candidate is None:
            continue
        thread = store.get_thread(candidate)
        if (
            thread is not None
            and thread.session_id == session_id
            and not store.checkpoint_exists(candidate)
        ):
            return candidate
    return store.next_thread_id(session_id)
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
    session_id: str = field(kw_only=True)
    thread_id: str
    agent: Any = None
    summarizer: Any = None
    history_model: str | list[str] | None = None
    exit_requested: bool = False
    rebuild_requested: bool = False
    ui: Any = None
    transcript: Any = None
    _title_written: bool = False
    _pending_switch_old_thread: str | None = field(
        default=None, init=False, repr=False
    )
    _registry: Registry = field(init=False, repr=False)
    _bus: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
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

    async def _clear_terminal(self) -> None:
        clear = getattr(self.ui, "clear", None)
        if callable(clear):
            await clear()
        else:
            self.console.console.clear()

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
            _compat("strip_foreign_blocks", strip_foreign_blocks)(graph, self.thread_config, foreign)

    def report_provider_error(self, exc: Exception) -> None:
        self.console.error(
            f"{type(exc).__name__}: {exc}\n"
            "Set the required provider environment variable, or `/login codex`, "
            "or `/model <prefix:model>`."
        )

    async def ensure_agent(self, *, seed_pending: bool = True) -> bool:
        reseed_pending = self._reseed_pending()
        if self.agent is not None:
            if reseed_pending and seed_pending:
                await self._seed_ready_thread("reseed")
            return True
        try:
            candidate_agent = await _compat("build_agent", build_agent)(self.registry,
            self.cfg,
            self.session,
            self._bus,
            always_allowed=self._always_allowed(),)
            candidate_summarizer = self._resolve_summarizer(self.cfg)
            if not reseed_pending:
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
        if reseed_pending and seed_pending:
            await self._seed_ready_thread("reseed")
        self.history_model = None
        return True

    async def rebuild(self) -> None:
        if self.agent is None:
            self.rebuild_requested = False
            return
        self.persist_plugin_states()
        candidate_agent = await _compat("build_agent", build_agent)(self.registry,
        self.cfg,
        self.session,
        self._bus,
        always_allowed=self._always_allowed(),)
        candidate_summarizer = self._resolve_summarizer(self.cfg)
        self.agent = candidate_agent
        self.summarizer = candidate_summarizer
        self.rebuild_requested = False
        if self._reseed_pending():
            await self._seed_ready_thread("reseed")

    def _persisted_current_thread(self) -> str | None:
        session = self.session.get(self.session_id)
        if session is None:
            raise LookupError(f"Unknown session: {self.session_id}")
        return session.current_thread

    def _reseed_pending(self) -> bool:
        return self._persisted_current_thread() != self.thread_id

    async def _seed_ready_thread(self, reason: str) -> None:
        if not self._reseed_pending():
            return
        new_thread = self.thread_id
        old_thread = self._pending_switch_old_thread or new_thread
        context = build_context(
            self.ledger.path(self.session_id),
            strip=_reseed_foreign_block_types(self.registry, self.cfg),
        )
        config = {"configurable": {"thread_id": new_thread}}
        self.session.saver.delete_thread(new_thread)
        try:
            await self.agent.aupdate_state(
                config,
                {
                    "messages": context.messages,
                    "todos": context.todos,
                    "files": context.files,
                },
                as_node="model",
            )
            seeded_state = self.agent.get_state(config)
            seeded_values = getattr(seeded_state, "values", {})
            seeded_messages = (
                seeded_values.get("messages", context.messages)
                if isinstance(seeded_values, Mapping)
                else context.messages
            )
            captured_message_ids = tuple(
                message.id
                for message in seeded_messages
                if isinstance(message.id, str)
            )
            self.session.activate_thread(
                self.session_id,
                new_thread,
                seeded_from=self.ledger.leaf(self.session_id),
                captured=len(seeded_messages),
                captured_message_ids=captured_message_ids,
            )
        except BaseException:
            self.session.saver.delete_thread(new_thread)
            raise
        self._pending_switch_old_thread = None
        await self._bus.emit(
            ThreadSwitch(
                session_id=self.session_id,
                old=old_thread,
                new=new_thread,
                reason=reason,
            )
        )

    async def seed_thread(self, reason: str) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        if not self._reseed_pending():
            old_thread = self.thread_id
            target_thread = _uncheckpointed_seed_target(
                self.session,
                self.session_id,
                old_thread,
            )
            self.ledger.set_position(
                self.session_id,
                leaf_id=self.ledger.leaf(self.session_id),
                thread_id=None,
            )
            self.thread_id = target_thread
            self._pending_switch_old_thread = old_thread
        await self._seed_ready_thread(reason)

    async def branch(self, entry_id: str) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        ledger = self.ledger
        prior_leaf = ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        prior_switch_old_thread = self._pending_switch_old_thread
        was_pending = prior_persisted_thread != prior_thread
        ledger.set_position(
            self.session_id,
            leaf_id=entry_id,
            thread_id=None,
        )
        if not was_pending:
            self.thread_id = self.session.next_thread_id(self.session_id)
            self._pending_switch_old_thread = prior_thread
        try:
            await self._seed_ready_thread("branch")
        except BaseException:
            if self._reseed_pending():
                ledger.set_position(
                    self.session_id,
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                )
                self.thread_id = prior_thread
                self._pending_switch_old_thread = prior_switch_old_thread
            raise

    async def fork(self) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        old_session = self.session_id
        old_thread = self.thread_id
        old_switch_thread = self._pending_switch_old_thread
        self.persist_plugin_states()
        created = self.session.create(
            self.cfg.cwd,
            self.cfg.model,
            mode=self.cfg.mode,
            parent_session=old_session,
        )
        try:
            self.ledger.fork(old_session, created.thread_id)
            self.session.copy_plugin_state(old_session, created.thread_id)
            if created.current_thread is None:
                raise RuntimeError(f"Session {created.thread_id} has no graph thread")
            self.ledger.set_position(
                created.thread_id,
                leaf_id=self.ledger.leaf(created.thread_id),
                thread_id=None,
            )
            self.session_id = created.thread_id
            self.thread_id = _uncheckpointed_seed_target(
                self.session,
                created.thread_id,
                created.current_thread,
            )
            self._pending_switch_old_thread = old_switch_thread or old_thread
            await self._seed_ready_thread("reseed")
        except BaseException:
            if self.session_id == created.thread_id and not self._reseed_pending():
                raise
            self.session_id = old_session
            self.thread_id = old_thread
            self._pending_switch_old_thread = old_switch_thread
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
                candidate_agent = await _compat("build_agent", build_agent)(self.registry,
                self.cfg,
                self.session,
                self._bus,
                always_allowed=self._always_allowed(),)
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
        self._pending_switch_old_thread = None
        self.rebuild_requested = False
        await self._clear_terminal()
        await self._bus.emit(SessionSwitch(old=old_session, new=self.session_id))

    async def clear(self) -> None:
        if self.agent is None and not await self.ensure_agent(seed_pending=False):
            return
        ledger = self.ledger
        prior_leaf = ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        prior_switch_old_thread = self._pending_switch_old_thread
        was_pending = prior_persisted_thread != prior_thread
        appended = ledger.append(
            self.session_id,
            ResetBoundaryEntry(),
            thread_id=None,
        )
        if not was_pending:
            self.thread_id = self.session.next_thread_id(self.session_id)
            self._pending_switch_old_thread = prior_thread
        try:
            await self._seed_ready_thread("clear")
        except BaseException:
            if self._reseed_pending():
                ledger.set_position(
                    self.session_id,
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                    discard_entry_id=appended.id,
                )
                self.thread_id = prior_thread
                self._pending_switch_old_thread = prior_switch_old_thread
            raise
        await self._clear_terminal()

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
        needs_reseed = False
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
            if needs_reseed:
                target_thread = (
                    _uncheckpointed_seed_target(
                        self.session,
                        saved_session.thread_id,
                        live_thread,
                    )
                    if not checkpoint_live
                    else self.session.next_thread_id(saved_session.thread_id)
                )
                self.ledger.set_position(
                    saved_session.thread_id,
                    leaf_id=self.ledger.leaf(saved_session.thread_id),
                    thread_id=None,
                )
                target_position_changed = True
            else:
                if live_thread is None:
                    raise RuntimeError(
                        f"Session {saved_session.thread_id} has no graph thread"
                    )
                target_thread = live_thread
            self.session_id = saved_session.thread_id
            self.thread_id = target_thread
            self.cfg = candidate_cfg
            self.history_model = stored_model
            self._pending_switch_old_thread = (
                old_pending_switch or old_thread
                if needs_reseed
                else None
            )
            if old_agent is None:
                self.summarizer = None
            else:
                candidate_agent = await _compat("build_agent", build_agent)(self.registry,
                candidate_cfg,
                self.session,
                self._bus,
                always_allowed=self._always_allowed(),)
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
            target_activated = (
                self.session_id == saved_session.thread_id
                and needs_reseed
                and not self._reseed_pending()
            )
            if target_activated:
                raise
            if target_position_changed:
                self.ledger.set_position(
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
        candidate_agent = await _compat("build_agent", build_agent)(self.registry,
        candidate_cfg,
        self.session,
        self._bus,
        always_allowed=self._always_allowed(),)
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
        audit: ModelChangeEntry | None = None
        self.session.set_model(self.session_id, spec)
        try:
            audit = self.ledger.append(
                self.session_id,
                ModelChangeEntry(model=spec if isinstance(spec, str) else list(spec)),
            )
            if foreign:
                _compat("strip_foreign_blocks", strip_foreign_blocks)(self.agent or candidate_agent,
                self.thread_config,
                foreign,)
        except BaseException:
            self.session.set_model(self.session_id, old_model)
            if audit is not None:
                self.ledger.set_position(
                    self.session_id,
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                    discard_entry_id=audit.id,
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
            candidate_agent = await _compat("build_agent", build_agent)(self.registry,
            candidate_cfg,
            self.session,
            self._bus,
            always_allowed=self._always_allowed(),)
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
        ledger = self.ledger
        prior_leaf = ledger.leaf(self.session_id)
        prior_thread = self.thread_id
        prior_persisted_thread = self._persisted_current_thread()
        prior_switch_old_thread = self._pending_switch_old_thread
        was_pending = prior_persisted_thread != prior_thread
        appended = ledger.append(
            self.session_id,
            CompactionEntry(
                summary=summary_text,
                first_kept_id=None,
                tokens_before=None,
            ),
            thread_id=None,
        )
        if not was_pending:
            self.thread_id = self.session.next_thread_id(self.session_id)
            self._pending_switch_old_thread = prior_thread
        try:
            await self._seed_ready_thread("compact")
        except BaseException:
            if self._reseed_pending():
                ledger.set_position(
                    self.session_id,
                    leaf_id=prior_leaf,
                    thread_id=prior_persisted_thread,
                    discard_entry_id=appended.id,
                )
                self.thread_id = prior_thread
                self._pending_switch_old_thread = prior_switch_old_thread
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
        messages = list(values.get("messages", ()))
        current_message_ids = tuple(
            message.id
            for message in messages
            if isinstance(message.id, str)
        )
        previous_message_ids = thread.captured_message_ids
        previous_id_set = set(previous_message_ids)
        current_id_set = set(current_message_ids)
        shrunk = (
            bool(previous_message_ids)
            and not previous_id_set.issubset(current_id_set)
        ) or (not previous_message_ids and len(messages) < thread.captured)
        entries: list[CompactionEntry | MessageEntry | CustomEntry] = []
        candidates = messages
        summary_index: int | None = None
        if shrunk:
            summary_index = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, HumanMessage)
                    and message.additional_kwargs.get("lc_source")
                    == "summarization"
                ),
                None,
            )
            if summary_index is not None:
                summary_message = messages[summary_index]
                summary = (
                    summary_message.content
                    if isinstance(summary_message.content, str)
                    else str(summary_message.content)
                )
                candidates = messages[summary_index + 1 :]
                first_retained_id = next(
                    (
                        message.id
                        for message in candidates
                        if isinstance(message.id, str)
                        and message.id in previous_id_set
                    ),
                    None,
                )
                first_kept_id = None
                if first_retained_id is not None:
                    path = self.ledger.path(session_id)
                    retained_at = next(
                        (
                            index
                            for index, entry in enumerate(path)
                            if isinstance(entry, MessageEntry)
                            and messages_from_dict([entry.message])[0].id
                            == first_retained_id
                        ),
                        None,
                    )
                    if retained_at is not None and retained_at > 0:
                        first_kept_id = path[retained_at - 1].id
                entries.append(
                    CompactionEntry(
                        summary=summary.removeprefix(_SUMMARIZATION_PREFIX),
                        first_kept_id=first_kept_id,
                    )
                )

        if previous_message_ids:
            for index, message in enumerate(candidates):
                message_id = message.id
                if isinstance(message_id, str):
                    unseen = message_id not in previous_id_set
                else:
                    absolute_index = (
                        index
                        if summary_index is None
                        else summary_index + 1 + index
                    )
                    unseen = absolute_index >= thread.captured
                if unseen:
                    entries.append(MessageEntry(message=message_to_dict(message)))
        elif summary_index is not None:
            entries.extend(
                MessageEntry(message=message_to_dict(message))
                for message in candidates
            )
        else:
            entries.extend(
                MessageEntry(message=message_to_dict(message))
                for message in messages[thread.captured :]
            )

        if only_if_new and not entries:
            return False
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
                captured=len(messages),
                captured_message_ids=current_message_ids,
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

