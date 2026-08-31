"""In-process agent runs and their application-scoped lifecycle registry."""

from __future__ import annotations

import asyncio
import json
import re
import reprlib
import secrets
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from langchain_core.messages import BaseMessage, message_to_dict

from orcha_agent.tui.turn import run_turn

from .agent import FILESYSTEM_TOOL_NAMES, build_agent
from .agent_types import AgentType
from .capture import capture_graph_values
from .config import Config, is_trusted_cwd
from .events import (
    AgentDelivered,
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    EventBus,
    ModelChunk,
    ToolCallEnd,
    ToolCallStart,
)
from .ledger import CustomEntry, Ledger
from .registry import Registry
from .session import SessionStore
from .usage import usage_cost

AgentStatusName = Literal[
    "running", "idle", "parked", "done", "failed", "aborted"
]
AbortReason = Literal["cancel", "timeout", "budget", "shutdown"]
_TERMINAL = frozenset({"done", "failed", "aborted"})
_WRAP_UP = "Wrap up and yield now."
_WAKE = object()
_MAX_TOOL_ARGS_CHARS = 4096
_TOOL_ARGS_REPR = reprlib.Repr()
_TOOL_ARGS_REPR.maxlevel = 4
_TOOL_ARGS_REPR.maxdict = 16
_TOOL_ARGS_REPR.maxlist = 16
_TOOL_ARGS_REPR.maxtuple = 16
_TOOL_ARGS_REPR.maxset = 16
_TOOL_ARGS_REPR.maxfrozenset = 16
_TOOL_ARGS_REPR.maxdeque = 16
_TOOL_ARGS_REPR.maxstring = 1024
_TOOL_ARGS_REPR.maxother = 1024
_MAX_PAYLOAD_BYTES = 256 * 1024
_TRUNCATION_MARKER = f"...[truncated at {_MAX_PAYLOAD_BYTES} bytes]"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def bound_text(value: str) -> str:
    if len(_json_bytes(value)) <= _MAX_PAYLOAD_BYTES:
        return value
    suffix = f"\n{_TRUNCATION_MARKER}"
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(_json_bytes(value[:middle] + suffix)) <= _MAX_PAYLOAD_BYTES:
            low = middle
        else:
            high = middle - 1
    return value[:low] + suffix


def bound_payload(value: Any) -> Any:
    if isinstance(value, str):
        return bound_text(value)
    try:
        encoded = _json_bytes(value)
    except (TypeError, ValueError):
        return bound_text(str(value))
    if len(encoded) <= _MAX_PAYLOAD_BYTES:
        return value
    return {
        "truncated": True,
        "original_bytes": len(encoded),
        "preview": (
            encoded.decode("utf-8", errors="ignore")[:4096]
            + f"\n{_TRUNCATION_MARKER}"
        ),
    }


def _model_label(value: str | list[str]) -> str:
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def _tool_args_text(args: Mapping[str, Any]) -> str:
    rendered = _TOOL_ARGS_REPR.repr(args)
    if len(rendered) <= _MAX_TOOL_ARGS_CHARS:
        return rendered
    return f"{rendered[: _MAX_TOOL_ARGS_CHARS - 1]}…"


def _restored_status(value: Any) -> AgentStatusName:
    if value == "done":
        return "done"
    if value == "failed":
        return "failed"
    if value == "aborted":
        return "aborted"
    return "parked"


class _RunEventBus:
    """Count one run's streamed activity before forwarding to the app bus."""

    def __init__(self, run: AgentRun, target: EventBus) -> None:
        self._run = run
        self._target = target

    async def emit(self, event: Any) -> Any:
        persist = False
        if isinstance(event, ToolCallStart) and event.source_id == self._run.id:
            self._run.tool_calls += 1
            self._run.last_tool = event.name
            self._run.last_tool_args = _tool_args_text(event.args)
            self._run.current_tool = event.name
            self._run.current_tool_args = self._run.last_tool_args
            self._run._active_tool_calls[event.id] = (
                event.name,
                self._run.last_tool_args,
            )
            persist = True
        elif isinstance(event, ToolCallEnd) and event.source_id == self._run.id:
            active = self._run._active_tool_calls
            removed = active.pop(event.id, None)
            if removed is not None and (
                self._run.current_tool,
                self._run.current_tool_args,
            ) == removed:
                if active:
                    self._run.current_tool, self._run.current_tool_args = next(
                        reversed(active.values())
                    )
                else:
                    self._run.current_tool = None
                    self._run.current_tool_args = None
                persist = True
        elif isinstance(event, ModelChunk) and event.source_id == self._run.id:
            if event.model_name and event.model_name != self._run.model_label:
                self._run.model_label = event.model_name
                persist = True
            usage = getattr(event.chunk, "usage_metadata", None)
            if isinstance(usage, dict):
                self._run.tokens_in += int(usage.get("input_tokens", 0) or 0)
                self._run.tokens_out += int(usage.get("output_tokens", 0) or 0)
                self._run.cost += usage_cost(
                    self._run.model_label,
                    usage,
                    self._run.cfg.pricing,
                )
                persist = True
        if persist:
            self._run.updated_at = datetime.now(UTC)
            self._run.owner._persist_job(self._run)
        if not getattr(self._run, "visible", True):
            return None
        is_active = getattr(self._run.owner, "_is_active", None)
        if callable(is_active) and not is_active(self._run):
            return None
        return await self._target.emit(event)


class AgentRun:
    """One independently checkpointed agent and its message-driven turn loop."""

    def __init__(
        self,
        owner: AgentRegistry,
        *,
        run_id: str,
        name: str,
        agent_type: AgentType,
        parent_id: str,
        parent_session: str,
        session_id: str,
        thread_id: str,
        cfg: Config,
        depth: int,
        output_schema: dict[str, Any] | None,
        schema_mode: str,
        blocking: bool,
        visible: bool = True,
        description: str = "",
        model_label: str = "",
    ) -> None:
        self.owner = owner
        self.id = run_id
        self.name = name
        self.agent_type = agent_type
        self.parent_id = parent_id
        self.parent_session = parent_session
        self.session_id = session_id
        self.thread_id = thread_id
        self.cfg = cfg
        self.depth = depth
        self.output_schema = output_schema
        self.schema_mode = schema_mode
        self.blocking = blocking
        self.visible = visible
        self.description = description
        self.model_label = model_label or _model_label(cfg.model)
        self.cwd = cfg.cwd
        self.source_id = run_id
        self.session = owner.session
        self.registry = owner.registry
        self.bus = _RunEventBus(self, owner.bus)
        self.console = None
        self.raise_turn_errors = True
        self.agent: Any = None
        self.task: asyncio.Task[None] | None = None
        self.inbox: asyncio.Queue[str | object | None] = asyncio.Queue()
        self.advice_outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.status: AgentStatusName = "running"
        self.abort_reason: AbortReason | None = None
        self.requests = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.last_tool: str | None = None
        self.last_tool_args: str | None = None
        self.current_tool: str | None = None
        self.current_tool_args: str | None = None
        self._active_tool_calls: dict[str, tuple[str, str]] = {}
        self.result: Any = None
        self.partial_findings: list[Any] = []
        self.schema_overridden = False
        self.validation_attempts = 0
        self.yield_count = 0
        self.last_yield: Any = None
        self.delivered = False
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.agent_ready = asyncio.Event()
        self._status_changed = asyncio.Condition()
        self._turn_task: asyncio.Task[None] | None = None
        self._completion: tuple[AgentStatusName, Any] | None = None
        self._abort_requested: AbortReason | None = None
        self._runtime_timer: asyncio.Task[None] | None = None
        self._job_entry_id: str | None = None
        self._attached = True
        self._runtime_deadline = (
            monotonic() + cfg.agents.max_runtime_s
            if cfg.agents.max_runtime_s > 0 and agent_type.name != "advisor"
            else None
        )

    @property
    def thread_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL


    def snapshot(
        self, *, delivered: bool | None = None, include_payload: bool = True
    ) -> dict[str, Any]:
        """Return this run's durable, JSON-serializable presentation state."""
        data = {
            "run_id": self.id,
            "name": self.name,
            "agent_type": self.agent_type.name,
            "tools": (
                sorted(self.agent_type.tools)
                if self.agent_type.tools is not None
                else None
            ),
            "description": self.description,
            "model_label": self.model_label,
            "parent_id": self.parent_id,
            "parent_session": self.parent_session,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "depth": self.depth,
            "output_schema": self.output_schema,
            "schema_mode": self.schema_mode,
            "blocking": self.blocking,
            "visible": self.visible,
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "last_tool": self.last_tool,
            "last_tool_args": self.last_tool_args,
            "current_tool": self.current_tool,
            "current_tool_args": self.current_tool_args,
            "validation_attempts": self.validation_attempts,
            "yield_count": self.yield_count,
            "status": self.status,
            "abort_reason": self.abort_reason,
            "schema_overridden": self.schema_overridden,
            "delivered": self.delivered if delivered is None else delivered,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_payload:
            data.update(
                partial_findings=bound_payload(self.partial_findings),
                last_yield=bound_payload(self.last_yield),
                result=bound_payload(self.result),
            )
        return data

    async def ensure_agent(self) -> bool:
        self.owner._refresh_run_mode(self)
        if self.agent is not None:
            return True
        try:
            extras = tuple(self.owner.extra_tools(self))
            scope = self._tool_scope(extras)
            self.agent = await build_agent(
                self.registry,
                self.cfg,
                self.session,
                self.bus,
                always_allowed=self.owner.always_allowed,
                extra_tools=extras,
                system_prompt=self.agent_type.system_prompt,
                exclude_general_purpose=True,
                tool_scope=scope,
            )
        finally:
            self.agent_ready.set()
        return True

    def _tool_scope(self, extras: Iterable[Any]) -> set[str] | None:
        scope = None if self.agent_type.tools is None else set(self.agent_type.tools)
        extra_names = {
            str(getattr(tool, "name", "")) for tool in extras
        }
        may_spawn = self.agent_type.spawns and self.depth < self.cfg.agents.max_depth
        if scope is not None:
            if self.agent_type.name != "advisor":
                scope.update(extra_names & {"yield", "hub"})
            if may_spawn and "task" in extra_names:
                scope.add("task")
        if may_spawn:
            return scope
        if scope is None:
            scope = {
                *self.registry.tools,
                *FILESYSTEM_TOOL_NAMES,
                *extra_names,
            }
        scope.discard("task")
        return scope

    def capture_turn(self) -> None:
        state = self.agent.get_state(self.thread_config)
        capture_graph_values(
            self.session,
            self.session_id,
            self.thread_id,
            getattr(state, "values", {}),
            only_if_new=False,
        )

    def record_exit(self, kind: str) -> None:
        Ledger(self.session).append(
            self.session_id,
            CustomEntry(
                custom_type="session_exit",
                data={"kind": kind, "pending_tool_calls": []},
            ),
        )

    async def wait_status(
        self, status: AgentStatusName, timeout_s: float = 2.0
    ) -> None:
        async with asyncio.timeout(timeout_s):
            async with self._status_changed:
                await self._status_changed.wait_for(lambda: self.status == status)

    async def complete(
        self,
        result: Any,
        *,
        status: Literal["done", "failed"] = "done",
    ) -> None:
        if self.terminal:
            return
        self._completion = (status, bound_payload(result))
        await self.inbox.put(_WAKE)

    async def request_abort(self, reason: AbortReason) -> None:
        cascade = getattr(self.owner, "_request_abort_tree", None)
        if callable(cascade):
            await cascade(self, reason)
            return
        await self._request_abort_self(reason)

    async def _request_abort_self(self, reason: AbortReason) -> None:
        if self.terminal:
            return
        if self._abort_requested is None:
            self._abort_requested = reason
        abort_reason = self._abort_requested
        current = self._turn_task
        if current is not None and not current.done():
            current.cancel()
        elif self.task is not None and not self.task.done():
            self.task.cancel()
        await self.inbox.put(_WAKE)
        if self.status == "parked" or self.task is None or self.task.done():
            await self._settle_abort(abort_reason)

    async def _set_status(
        self, status: AgentStatusName, reason: str | None = None
    ) -> None:
        if status != "running":
            self.current_tool = None
            self.current_tool_args = None
            self._active_tool_calls.clear()
        self.status = status
        self.updated_at = datetime.now(UTC)
        self.owner._persist_job(self, include_payload=status in _TERMINAL)
        await self.owner._notify()
        async with self._status_changed:
            self._status_changed.notify_all()
        if self.visible and self.owner._is_active(self):
            await self.owner.bus.emit(
                AgentStatus(
                    run_id=self.id,
                    parent_id=self.parent_id,
                    name=self.name,
                    agent_type=self.agent_type.name,
                    status=status,
                    reason=reason,
                )
            )

    async def _runtime_watchdog(self) -> None:
        if self._runtime_deadline is None:
            return
        await asyncio.sleep(max(0.0, self._runtime_deadline - monotonic()))
        await self.request_abort("timeout")

    def start(self, prompt: str | None) -> None:
        self.task = asyncio.create_task(self._run(prompt), name=f"agent:{self.id}")

    async def _run(self, prompt: str | None) -> None:
        if self._runtime_deadline is not None:
            self._runtime_timer = asyncio.create_task(self._runtime_watchdog())
        try:
            await self.ensure_agent()
            await self._loop(prompt)
        except asyncio.CancelledError:
            await self._settle_abort(self._abort_requested or "shutdown")
        except Exception as exc:
            await self._settle("failed", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            if self._runtime_timer is not None:
                self._runtime_timer.cancel()
                self._runtime_timer = None

    async def _loop(self, prompt: str | None) -> None:
        message = prompt
        while True:
            if self._abort_requested is not None:
                await self._settle_abort(self._abort_requested)
                return
            if self._completion is not None:
                status, result = self._completion
                await self._settle(status, result)
                return
            if message is not None:
                await self._set_status("running")
                async with self.owner.semaphore:
                    if self._abort_requested is not None:
                        await self._settle_abort(self._abort_requested)
                        return
                    self.requests += 1
                    self._turn_task = asyncio.create_task(run_turn(self, message))
                    try:
                        await self._turn_task
                    finally:
                        self._turn_task = None
                if self._abort_requested is not None:
                    await self._settle_abort(self._abort_requested)
                    return
                if self._completion is not None:
                    status, result = self._completion
                    await self._settle(status, result)
                    return
                if self.agent_type.name != "advisor":
                    budget = self.cfg.agents.soft_request_budget
                    if self.requests >= budget + 10:
                        abort_descendants = getattr(
                            self.owner, "_request_abort_descendants", None
                        )
                        if callable(abort_descendants):
                            await abort_descendants(self, "budget")
                        await self._settle_abort("budget")
                        return
                    if self.requests >= budget:
                        message = _WRAP_UP
                        continue
            await self._set_status("idle")
            if self.agent_type.name == "advisor":
                incoming = await self.inbox.get()
            else:
                try:
                    incoming = await asyncio.wait_for(
                        self.inbox.get(), timeout=self.cfg.agents.idle_ttl_s
                    )
                except TimeoutError:
                    await self._set_status("parked")
                    return
            if incoming is _WAKE:
                message = None
                continue
            if isinstance(incoming, str):
                message = incoming

    async def _settle_abort(self, reason: AbortReason) -> None:
        self.abort_reason = reason
        Ledger(self.session).append(
            self.session_id,
            CustomEntry(
                custom_type="session_exit",
                data={"kind": "aborted", "reason": reason},
            ),
        )
        await self._settle("aborted", {"error": reason})

    async def _settle(self, status: AgentStatusName, result: Any) -> None:
        if self.terminal:
            return
        result = bound_payload(result)
        self.result = result
        Ledger(self.session).append(
            self.session_id,
            CustomEntry(
                custom_type="agent_result",
                data={
                    "run_id": self.id,
                    "status": status,
                    "result": result,
                    "schema_overridden": self.schema_overridden,
                },
            ),
        )
        await self._set_status(status, self.abort_reason)
        if self.visible and self.owner._is_active(self):
            await self.owner.bus.emit(
                AgentFinished(
                    run_id=self.id,
                    parent_id=self.parent_id,
                    name=self.name,
                    agent_type=self.agent_type.name,
                    result=result,
                )
            )


class AgentRegistry:
    """Own every child run for one interactive application."""

    def __init__(
        self,
        registry: Registry,
        cfg: Config,
        session: SessionStore,
        bus: EventBus,
        parent_session_id: str,
        *,
        always_allowed: set[str] | None = None,
        extra_tools: Callable[[AgentRun], Iterable[Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.cfg = cfg
        self.session = session
        self.bus = bus
        self.parent_session_id = parent_session_id
        self.always_allowed = always_allowed if always_allowed is not None else set()
        self.extra_tools = extra_tools or (lambda _run: ())
        self.semaphore = asyncio.Semaphore(cfg.agents.max_concurrency)
        self._runs: dict[str, AgentRun] = {}
        self._order: list[str] = []
        self._mailboxes: dict[str, deque[dict[str, Any]]] = {}
        self._views: dict[
            str,
            tuple[
                dict[str, AgentRun],
                list[str],
                dict[str, deque[dict[str, Any]]],
            ],
        ] = {}
        self._changed = asyncio.Condition()
        self._activity_waiters: dict[tuple[str, str], int] = {}
        self._activity_reservations: dict[
            tuple[str, str, str], tuple[str, str]
        ] = {}
        self._delivery_lock = asyncio.Lock()
        self._hydrate_parent(parent_session_id, "main", 0, set())

    def _detach_job(self, run: AgentRun) -> None:
        if not run._attached:
            return
        run._attached = False
        if run.terminal or run._abort_requested is not None:
            return
        run._abort_requested = "cancel"
        asyncio.create_task(
            run.request_abort("cancel"),
            name=f"detach-agent:{run.id}",
        )

    def _job_on_active_path(self, run: AgentRun) -> bool:
        if not run._attached:
            return False
        if run._job_entry_id is None:
            return True
        if any(
            entry.id == run._job_entry_id
            for entry in Ledger(self.session).path(run.parent_session)
        ):
            return True
        self._detach_job(run)
        return False

    def _persist_job(
        self,
        run: AgentRun,
        *,
        delivered: bool | None = None,
        include_payload: bool = False,
    ) -> bool:
        if not self._job_on_active_path(run):
            return False
        entry = Ledger(self.session).append(
            run.parent_session,
            CustomEntry(
                custom_type="agent_job",
                data=run.snapshot(
                    delivered=delivered, include_payload=include_payload
                ),
            ),
        )
        run._job_entry_id = entry.id
        return True

    def _runs_for(self, run: AgentRun) -> dict[str, AgentRun] | None:
        return next(
            (
                runs
                for _view_id, runs, _order, _mailboxes in self._all_views()
                if runs.get(run.id) is run
            ),
            None,
        )

    async def _request_abort_descendants(
        self, run: AgentRun, reason: AbortReason
    ) -> None:
        runs = self._runs_for(run)
        if runs is None:
            return
        descendants: list[AgentRun] = []
        parents = [run.id]
        while parents:
            parent_id = parents.pop()
            children = [
                candidate
                for candidate in runs.values()
                if candidate.parent_id == parent_id
            ]
            descendants.extend(children)
            parents.extend(child.id for child in children)
        for descendant in reversed(descendants):
            await descendant._request_abort_self(reason)

    async def _request_abort_tree(
        self, run: AgentRun, reason: AbortReason
    ) -> None:
        reason = run._abort_requested or reason
        await self._request_abort_descendants(run, reason)
        await run._request_abort_self(reason)

    def _session_for_job(
        self, parent_session: str, run_id: str, data: Mapping[str, Any]
    ) -> Any:
        terminal = data.get("status") in _TERMINAL
        session_id = data.get("session_id")
        if isinstance(session_id, str):
            if info := self.session.get(session_id):
                if info.parent_session == parent_session or terminal:
                    return info
                return None

        candidates = self.session.children(parent_session)
        for info in candidates:
            if run_id in Ledger(self.session).latest_custom(
                info.thread_id, "agent_result", key="run_id"
            ):
                return info

        # Forked parent ledgers retain child job entries while the immutable child
        # session still points at the source parent session. Legacy job payloads
        # did not carry session_id, so fall back to the globally unique run id.
        if not terminal:
            return None
        direct_ids = {info.thread_id for info in candidates}
        for info in self.session.list(include_children=True):
            if info.thread_id in direct_ids or info.parent_session is None:
                continue
            if run_id in Ledger(self.session).latest_custom(
                info.thread_id, "agent_result", key="run_id"
            ):
                return info
        return None

    @staticmethod
    def _timestamp(value: Any, fallback: datetime) -> datetime:
        if not isinstance(value, str):
            return fallback
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def _hydrate_parent(
        self,
        parent_session: str,
        parent_id: str,
        depth: int,
        visited_sessions: set[str],
    ) -> None:
        if parent_session in visited_sessions:
            return
        visited_sessions.add(parent_session)
        jobs = Ledger(self.session).latest_custom(
            parent_session, "agent_job", key="run_id"
        )
        for run_id, entry in jobs.items():
            if run_id in self._runs or not isinstance(entry.data, Mapping):
                continue
            data = entry.data
            info = self._session_for_job(parent_session, run_id, data)
            if info is None or info.current_thread is None:
                continue
            foreign_snapshot = info.parent_session != parent_session
            type_name = data.get("agent_type")
            if not isinstance(type_name, str):
                type_name = "task"
            spec = self.registry.agent_types.get(type_name)
            if spec is None:
                spec = self.registry.agent_types.get("task")
            if spec is None:
                continue
            visible = bool(data.get("visible", spec.name != "advisor"))
            stored_tools = data.get("tools")
            if (
                spec.name == "advisor"
                and not visible
                and isinstance(stored_tools, list)
                and all(isinstance(tool, str) for tool in stored_tools)
            ):
                spec = replace(spec, tools=set(stored_tools))
            name = data.get("name")
            if not isinstance(name, str) or not name:
                name = info.title or spec.name
            output_schema = data.get("output_schema")
            if isinstance(output_schema, Mapping):
                output_schema = dict(output_schema)
            elif output_schema is not None:
                output_schema = None
            schema_mode = (
                "strict" if data.get("schema_mode") == "strict" else "permissive"
            )
            description = data.get("description")
            if not isinstance(description, str):
                description = name
            model_label = data.get("model_label")
            if not isinstance(model_label, str):
                model_label = _model_label(info.model)
            child_cwd = Path(info.cwd)
            hydrated_cfg = replace(
                self.cfg,
                model=info.model,
                cwd=child_cwd,
                mode=self.cfg.mode,
                trust_cwd=is_trusted_cwd(
                    child_cwd,
                    self.cfg.trusted_dirs,
                    trust_all=self.cfg.trust_all_cwd,
                ),
            )
            run = AgentRun(
                self,
                run_id=run_id,
                name=name,
                agent_type=spec,
                parent_id=parent_id,
                parent_session=parent_session,
                session_id=info.thread_id,
                thread_id=info.current_thread,
                cfg=hydrated_cfg,
                depth=depth,
                output_schema=output_schema,
                schema_mode=schema_mode,
                blocking=bool(data.get("blocking", spec.blocking)),
                visible=visible,
                description=description,
                model_label=model_label,
            )
            run.status = _restored_status(data.get("status"))
            run.result = data.get("result")
            run.schema_overridden = bool(data.get("schema_overridden", False))
            run.delivered = data.get("delivered") is True
            run._job_entry_id = entry.id
            child_result = (
                None
                if foreign_snapshot
                else Ledger(self.session)
                .latest_custom(run.session_id, "agent_result", key="run_id")
                .get(run.id)
            )
            if child_result is not None and isinstance(child_result.data, Mapping):
                child_status = child_result.data.get("status")
                if child_status in _TERMINAL:
                    run.status = child_status
                    run.result = child_result.data.get("result")
                    run.schema_overridden = bool(
                        child_result.data.get("schema_overridden", False)
                    )
            elif run.status not in _TERMINAL:
                for child_entry in reversed(Ledger(self.session).path(run.session_id)):
                    if (
                        isinstance(child_entry, CustomEntry)
                        and child_entry.custom_type == "session_exit"
                        and isinstance(child_entry.data, Mapping)
                        and child_entry.data.get("kind") == "aborted"
                    ):
                        reason = child_entry.data.get("reason")
                        run.status = "aborted"
                        run.result = {"error": reason or "aborted"}
                        if reason in {"cancel", "timeout", "budget", "shutdown"}:
                            run.abort_reason = reason
                        break
            for attribute in (
                "requests",
                "tool_calls",
                "tokens_in",
                "tokens_out",
                "validation_attempts",
                "yield_count",
            ):
                value = data.get(attribute)
                if isinstance(value, int) and not isinstance(value, bool):
                    setattr(run, attribute, value)
            cost = data.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                run.cost = float(cost)
            last_tool = data.get("last_tool")
            if isinstance(last_tool, str):
                run.last_tool = last_tool
            last_tool_args = data.get("last_tool_args")
            if isinstance(last_tool_args, str):
                run.last_tool_args = last_tool_args[:_MAX_TOOL_ARGS_CHARS]
            partial_findings = data.get("partial_findings")
            if isinstance(partial_findings, list):
                run.partial_findings = list(partial_findings)
            run.last_yield = data.get("last_yield")
            abort_reason = data.get("abort_reason")
            if abort_reason in {"cancel", "timeout", "budget", "shutdown"}:
                run.abort_reason = abort_reason
            run.created_at = self._timestamp(data.get("created_at"), run.created_at)
            run.updated_at = self._timestamp(
                data.get("updated_at"),
                self._timestamp(entry.ts, run.created_at),
            )
            if run._runtime_deadline is not None:
                elapsed = max(
                    0.0, (datetime.now(UTC) - run.created_at).total_seconds()
                )
                run._runtime_deadline = monotonic() + max(
                    0.0, run.cfg.agents.max_runtime_s - elapsed
                )
            self._runs[run.id] = run
            self._order.append(run.id)
            if not foreign_snapshot:
                self._hydrate_parent(
                    run.session_id, run.id, run.depth + 1, visited_sessions
                )

    def _refresh_run_mode(self, run: AgentRun) -> None:
        mode_changed = run.cfg.mode != self.cfg.mode
        if mode_changed:
            run.cfg = replace(run.cfg, mode=self.cfg.mode)
            if not run.terminal:
                run.agent = None
                run.agent_ready.clear()
            self.session.set_mode(run.session_id, self.cfg.mode)

    def retarget(
        self, parent_session_id: str, cfg: Config | None = None
    ) -> None:
        """Point the registry at one active parent ledger and hydrate its runs."""
        if cfg is not None:
            self.cfg = cfg
        for run in tuple(self._runs.values()):
            self._refresh_run_mode(run)
            self._job_on_active_path(run)
        if parent_session_id == self.parent_session_id:
            return
        self._views[self.parent_session_id] = (
            self._runs,
            self._order,
            self._mailboxes,
        )
        self.parent_session_id = parent_session_id
        restored = self._views.pop(parent_session_id, None)
        if restored is not None:
            self._runs, self._order, self._mailboxes = restored
            for run in tuple(self._runs.values()):
                self._refresh_run_mode(run)
                self._job_on_active_path(run)
            return
        self._runs = {}
        self._order = []
        self._mailboxes = {}
        self._hydrate_parent(parent_session_id, "main", 0, set())

    def _all_views(
        self,
    ) -> Iterable[
        tuple[
            str,
            dict[str, AgentRun],
            list[str],
            dict[str, deque[dict[str, Any]]],
        ]
    ]:
        runs = getattr(self, "_runs", {})
        order = getattr(self, "_order", list(runs))
        mailboxes = getattr(self, "_mailboxes", {})
        yield (
            getattr(self, "parent_session_id", ""),
            runs,
            order,
            mailboxes,
        )
        for view_id, (runs, order, mailboxes) in getattr(
            self, "_views", {}
        ).items():
            yield view_id, runs, order, mailboxes

    def _view_for_caller(
        self, caller: str | None
    ) -> tuple[
        str,
        dict[str, AgentRun],
        list[str],
        dict[str, deque[dict[str, Any]]],
    ]:
        if caller is None or caller == "main":
            return next(iter(self._all_views()))
        for view in self._all_views():
            if caller in view[1]:
                return view
        raise LookupError(f"Unknown agent: {caller}")

    def _view_for_ids(
        self, ids: Iterable[str], caller: str | None = None
    ) -> tuple[
        str,
        dict[str, AgentRun],
        list[str],
        dict[str, deque[dict[str, Any]]],
    ]:
        if caller is not None:
            return self._view_for_caller(caller)
        selected_view = None
        for run_id in ids:
            for view in self._all_views():
                if run_id not in view[1]:
                    continue
                if selected_view is not None and selected_view[0] != view[0]:
                    raise LookupError("Agents span registry views")
                selected_view = view
                break
        return selected_view or self._view_for_caller(None)

    def _is_active(self, run: AgentRun) -> bool:
        return run._attached and self._runs.get(run.id) is run

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    def get(self, run_id: str, *, caller: str | None = None) -> AgentRun | None:
        return self._view_for_caller(caller)[1].get(run_id)

    def list(
        self, status: str | None = None, *, caller: str | None = None
    ) -> list[AgentRun]:
        _view_id, runs, order, _mailboxes = self._view_for_caller(caller)
        return [
            runs[run_id]
            for run_id in order
            if getattr(runs[run_id], "_attached", True)
            and runs[run_id].visible
            and (status is None or runs[run_id].status == status)
        ]

    def advisor_run(self, parent_session: str) -> AgentRun | None:
        for _view_id, runs, order, _mailboxes in self._all_views():
            found = next(
                (
                    run
                    for run_id in reversed(order)
                    if (run := runs[run_id]).agent_type.name == "advisor"
                    and not run.visible
                    and run.parent_session == parent_session
                    and not run.terminal
                ),
                None,
            )
            if found is not None:
                return found
        return None

    def tree(self, *, caller: str | None = None) -> list[AgentRun]:
        _view_id, runs, _order, _mailboxes = self._view_for_caller(caller)
        children: dict[str, list[AgentRun]] = {}
        roots: list[AgentRun] = []
        for run in self.list(caller=caller):
            if run.parent_id in runs:
                children.setdefault(run.parent_id, []).append(run)
            else:
                roots.append(run)
        ordered: list[AgentRun] = []
        stack = list(reversed(roots))
        while stack:
            run = stack.pop()
            ordered.append(run)
            stack.extend(reversed(children.get(run.id, ())))
        return ordered

    def _parent(
        self, parent: str, caller: str | None = None
    ) -> tuple[
        str,
        str,
        int,
        str,
        dict[str, AgentRun],
        list[str],
        Config,
    ]:
        view_id, runs, order, _mailboxes = self._view_for_caller(
            caller if parent == "main" else parent
        )
        if parent == "main":
            base_cfg = (
                runs[caller].cfg
                if caller is not None and caller != "main" and caller in runs
                else self.cfg
            )
            return "main", view_id, 0, view_id, runs, order, base_cfg
        run = runs.get(parent)
        if run is None:
            raise LookupError(f"Unknown parent agent: {parent}")
        if not run._attached:
            raise RuntimeError(f"Parent agent {run.name} is detached")
        return (
            run.id,
            run.session_id,
            run.depth + 1,
            view_id,
            runs,
            order,
            run.cfg,
        )

    def _name(
        self,
        prompt: str,
        requested: str | None,
        parent_id: str,
        runs: Mapping[str, AgentRun],
    ) -> str:
        raw = requested or " ".join(re.findall(r"[A-Za-z0-9]+", prompt)[:3]) or "Agent"
        base = "".join(part[:1].upper() + part[1:] for part in re.findall(r"[A-Za-z0-9]+", raw))[:32]
        if not base:
            base = "Agent"
        sibling_names = {
            run.name for run in runs.values() if run.parent_id == parent_id
        }
        candidate = base
        suffix = 2
        while candidate in sibling_names:
            tail = str(suffix)
            candidate = f"{base[: 32 - len(tail)]}{tail}"
            suffix += 1
        return candidate

    @staticmethod
    def _model(spec: AgentType, cfg: Config) -> str | list[str]:
        configured = cfg.model_roles.get(spec.model_role)
        if configured is not None:
            return configured
        if ":" in spec.model_role:
            return spec.model_role
        if spec.model_role == "task" and cfg.subagent_model is not None:
            return cfg.subagent_model
        return cfg.model

    async def spawn(
        self,
        agent_type: str,
        prompt: str,
        *,
        name: str | None = None,
        parent: str,
        output_schema: dict[str, Any] | None = None,
        schema_mode: Literal["permissive", "strict"] = "permissive",
        blocking: bool = False,
        model: str | list[str] | None = None,
        tools: Iterable[str] | None = None,
        visible: bool = True,
        caller: str | None = None,
    ) -> AgentRun:
        spec = self.registry.agent_types.get(agent_type)
        if spec is None:
            available = ", ".join(sorted(self.registry.agent_types))
            raise LookupError(f"Unknown agent type {agent_type!r}; available: {available}")
        if tools is not None:
            scoped_tools = set(tools)
            if spec.name == "advisor":
                scoped_tools.add("advise")
            spec = replace(spec, tools=scoped_tools)
        if schema_mode not in {"permissive", "strict"}:
            raise ValueError("schema_mode must be permissive or strict")
        (
            parent_id,
            parent_session,
            depth,
            view_id,
            runs,
            order,
            base_cfg,
        ) = self._parent(parent, caller)
        if depth > base_cfg.agents.max_depth:
            raise ValueError(
                f"maximum agent depth {base_cfg.agents.max_depth} exceeded by depth {depth}"
            )
        live_runs = {
            run.id
            for _view_id, view_runs, _order, _mailboxes in self._all_views()
            for run in view_runs.values()
            if run._attached and not run.terminal
        }
        if len(live_runs) >= self.cfg.agents.max_live_runs:
            raise RuntimeError(
                f"live agent limit {self.cfg.agents.max_live_runs} reached"
            )
        run_name = self._name(prompt, name, parent_id, runs)
        model = self._model(spec, base_cfg) if model is None else model
        info = self.session.create(
            base_cfg.cwd,
            model,
            base_cfg.mode,
            title=run_name,
            parent_session=parent_session,
            kind="agent",
        )
        if info.current_thread is None:
            raise RuntimeError(f"Agent session {info.thread_id} has no thread")
        run = AgentRun(
            self,
            run_id=secrets.token_hex(4),
            name=run_name,
            agent_type=spec,
            parent_id=parent_id,
            parent_session=parent_session,
            session_id=info.thread_id,
            thread_id=info.current_thread,
            cfg=replace(base_cfg, model=model),
            depth=depth,
            output_schema=(
                spec.output_schema if output_schema is None else output_schema
            ),
            schema_mode=schema_mode,
            blocking=blocking or spec.blocking,
            visible=visible,
            description=prompt,
            model_label=_model_label(model),
        )
        self._persist_job(run)
        runs[run.id] = run
        order.append(run.id)
        if run.visible and view_id == self.parent_session_id:
            await self.bus.emit(
                AgentSpawned(
                    run_id=run.id,
                    parent_id=run.parent_id,
                    name=run.name,
                    agent_type=spec.name,
                )
            )
        run.start(prompt)
        await self._notify()
        return run

    async def send(
        self,
        run_id: str,
        text: str,
        *,
        interrupt: bool = False,
        caller: str | None = None,
    ) -> AgentRun:
        _view_id, runs, _order, _mailboxes = self._view_for_ids(
            (run_id,), caller
        )
        run = runs.get(run_id)
        if run is None:
            raise LookupError(f"Unknown agent: {run_id}")
        if run.terminal:
            raise RuntimeError(f"Agent {run.name} is {run.status}")
        if run.status == "parked":
            await self.revive(run.session_id, caller=run.id)
        if interrupt and run._turn_task is not None:
            run._turn_task.cancel()
        await run.inbox.put(bound_text(text))
        return run

    async def cancel(
        self,
        run_id: str,
        reason: AbortReason = "cancel",
        *,
        caller: str | None = None,
    ) -> AgentRun:
        _view_id, runs, _order, _mailboxes = self._view_for_ids(
            (run_id,), caller
        )
        run = runs.get(run_id)
        if run is None:
            raise LookupError(f"Unknown agent: {run_id}")
        await run.request_abort(reason)
        return run

    async def wait(
        self,
        ids: Iterable[str] | None = None,
        *,
        timeout_s: float = 300,
        caller: str | None = None,
    ) -> list[AgentRun]:
        selected = None if ids is None else set(ids)
        _view_id, runs, order, _mailboxes = (
            self._view_for_caller(caller)
            if selected is None
            else self._view_for_ids(selected, caller)
        )
        if selected is None:
            selected = set(runs)

        def settled() -> list[AgentRun]:
            return [
                runs[run_id]
                for run_id in order
                if run_id in selected
                and runs[run_id]._attached
                and runs[run_id].visible
                and runs[run_id].status in _TERMINAL
            ]

        if found := settled():
            return found
        if not selected:
            return []
        try:
            async with asyncio.timeout(timeout_s):
                async with self._changed:
                    await self._changed.wait_for(lambda: bool(settled()))
        except TimeoutError:
            return []
        return settled()

    async def revive(
        self, session_id: str, *, caller: str | None = None
    ) -> AgentRun:
        views = (
            (self._view_for_caller(caller),)
            if caller is not None
            else tuple(self._all_views())
        )
        run = next(
            (
                candidate
                for _view_id, runs, _order, _mailboxes in views
                for candidate in runs.values()
                if candidate.session_id == session_id
            ),
            None,
        )
        if run is None:
            raise LookupError(f"Unknown agent session: {session_id}")
        self._refresh_run_mode(run)
        if run.status != "parked":
            return run
        await run._set_status("idle")
        if run._attached:
            run.start(None)
        return run

    async def shutdown(self) -> None:
        runs = list(self._runs.values())
        for stored_runs, _order, _mailboxes in self._views.values():
            runs.extend(stored_runs.values())
        unique_runs = list({run.id: run for run in runs}.values())
        for run in unique_runs:
            if not run.terminal:
                await run.request_abort("shutdown")
        pending = [run.task for run in unique_runs if run.task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def resolve(
        self,
        selector: str,
        *,
        caller: str | None = None,
        visible_only: bool = True,
    ) -> str:
        """Resolve main, an exact run id, or an unambiguous run name."""
        _view_id, runs, _order, _mailboxes = self._view_for_caller(caller)
        if selector == "main":
            return selector
        exact = runs.get(selector)
        if exact is not None and (
            not visible_only or (exact._attached and exact.visible)
        ):
            return selector
        matches = [
            run.id
            for run in runs.values()
            if run.name == selector
            and (not visible_only or (run._attached and run.visible))
        ]
        if not matches:
            raise LookupError(f"Unknown agent: {selector}")
        if len(matches) > 1:
            raise LookupError(f"Ambiguous agent name: {selector}")
        return matches[0]

    async def post_message(
        self, sender: str, recipient: str, message: str
    ) -> bool:
        """Queue one serializable peer message and report a synchronous waiter."""
        view_id, _runs, _order, mailboxes = self._view_for_caller(sender)
        target = self.resolve(recipient, caller=sender)
        envelope = {
            "from": sender,
            "to": target,
            "message": bound_text(message),
            "created_at": datetime.now(UTC).isoformat(),
        }
        mailboxes.setdefault(target, deque()).append(envelope)
        waiting = self._activity_waiters.get((view_id, target), 0) > 0
        await self._notify()
        return waiting

    def drain_messages(
        self,
        recipient: str,
        *,
        sender: str | None = None,
        caller: str | None = None,
    ) -> list[dict[str, Any]]:
        """Drain all addressed messages, optionally from one sender."""
        actor = caller if caller is not None else (
            recipient if recipient != "main" else None
        )
        _view_id, _runs, _order, mailboxes = self._view_for_caller(actor)
        mailbox = mailboxes.get(recipient)
        if not mailbox:
            return []
        if sender is None:
            messages = list(mailbox)
            mailbox.clear()
            return messages
        messages: list[dict[str, Any]] = []
        retained: deque[dict[str, Any]] = deque()
        while mailbox:
            item = mailbox.popleft()
            (messages if item["from"] == sender else retained).append(item)
        mailboxes[recipient] = retained
        return messages

    def unread_count(
        self, recipient: str, *, caller: str | None = None
    ) -> int:
        actor = caller if caller is not None else (
            recipient if recipient != "main" else None
        )
        _view_id, runs, _order, mailboxes = self._view_for_caller(actor)
        return len(mailboxes.get(recipient, ())) + sum(
            1
            for run in runs.values()
            if run._attached
            and run.visible
            and run.parent_id == recipient
            and run.terminal
            and not run.delivered
        )

    def jobs(
        self,
        parent: str,
        ids: Iterable[str] | None = None,
        *,
        caller: str | None = None,
    ) -> list[AgentRun]:
        actor = caller if caller is not None else (
            parent if parent != "main" else None
        )
        _view_id, runs, order, _mailboxes = self._view_for_caller(actor)
        selected = None if ids is None else set(ids)
        return [
            runs[run_id]
            for run_id in order
            if runs[run_id]._attached
            and runs[run_id].visible
            and runs[run_id].parent_id == parent
            and (selected is None or run_id in selected)
        ]

    async def record_yield(self, run: AgentRun, payload: Any) -> None:
        run.yield_count += 1
        run.last_yield = bound_payload(payload)
        self._persist_job(run)
        await self._notify()

    async def record_advice(
        self, run: AgentRun, payload: dict[str, Any]
    ) -> None:
        if run.owner is not self or run.agent_type.name != "advisor":
            raise ValueError("advice can only be recorded by a registered advisor run")
        await run.advice_outbox.put(payload)

    async def deliver(
        self,
        parent: str,
        ids: Iterable[str] | None = None,
        *,
        caller: str | None = None,
    ) -> list[AgentRun]:
        actor = caller if caller is not None else (
            parent if parent != "main" else None
        )
        view_id, _runs, _order, _mailboxes = self._view_for_caller(actor)
        async with self._delivery_lock:
            delivered = [
                run
                for run in self.jobs(parent, ids, caller=actor)
                if run.terminal
                and not run.delivered
                and self._job_on_active_path(run)
            ]
            if not delivered:
                return []
            parent_sessions = {run.parent_session for run in delivered}
            if len(parent_sessions) != 1:
                raise RuntimeError("Agent jobs for one parent span multiple sessions")
            snapshots = tuple(run.snapshot(delivered=True) for run in delivered)
            entries = Ledger(self.session).append_many(
                parent_sessions.pop(),
                [
                    CustomEntry(
                        custom_type="agent_job",
                        data=snapshot,
                    )
                    for snapshot in snapshots
                ],
            )
            for run, entry in zip(delivered, entries, strict=True):
                run.delivered = True
                run._job_entry_id = entry.id
        if view_id == self.parent_session_id:
            await self.bus.emit(
                AgentDelivered(
                    parent_id=parent,
                    run_ids=tuple(run.id for run in delivered),
                    jobs=snapshots,
                )
            )
        await self._notify()
        return delivered

    async def wait_all(
        self,
        ids: Iterable[str],
        *,
        timeout_s: float = 300,
        caller: str | None = None,
    ) -> list[AgentRun]:
        selected = set(ids)
        _view_id, runs, order, _mailboxes = self._view_for_ids(selected, caller)

        def complete() -> bool:
            return all(
                (run := runs.get(run_id)) is not None
                and (run.terminal or not run._attached)
                for run_id in selected
            )

        if selected and not complete():
            try:
                async with asyncio.timeout(timeout_s):
                    async with self._changed:
                        await self._changed.wait_for(complete)
            except TimeoutError:
                pass
        return [
            runs[run_id]
            for run_id in order
            if run_id in selected
            and runs[run_id]._attached
            and runs[run_id].terminal
        ]

    def reserve_activity_waiter(self, caller: str) -> tuple[str, str, str]:
        view_id, _runs, _order, _mailboxes = self._view_for_caller(caller)
        key = (view_id, caller)
        token = (view_id, caller, secrets.token_hex(8))
        self._activity_reservations[token] = key
        self._activity_waiters[key] = self._activity_waiters.get(key, 0) + 1
        return token

    def release_activity_waiter(self, token: tuple[str, str, str]) -> None:
        key = self._activity_reservations.pop(token, None)
        if key is None:
            return
        remaining = self._activity_waiters[key] - 1
        if remaining:
            self._activity_waiters[key] = remaining
        else:
            del self._activity_waiters[key]

    async def wait_activity(
        self,
        caller: str,
        ids: Iterable[str] | None = None,
        *,
        timeout_s: float = 300,
        peer: str | None = None,
        after_yield: int | None = None,
        reserved: bool = False,
    ) -> bool:
        selected = None if ids is None else set(ids)
        view_id, runs, _order, mailboxes = self._view_for_caller(caller)

        def ready() -> bool:
            mailbox = mailboxes.get(caller, ())
            if peer is not None:
                if any(message.get("from") == peer for message in mailbox):
                    return True
                if peer == "main":
                    return False
                run = runs.get(peer)
                return run is not None and run._attached and (
                    run.yield_count > (after_yield or 0) or run.terminal
                )
            if mailbox:
                return True
            if any(
                run._attached
                and run.visible
                and run.parent_id == caller
                and (selected is None or run.id in selected)
                and run.terminal
                and not run.delivered
                for run in runs.values()
            ):
                return True
            return False

        if ready():
            return True
        key = (view_id, caller)
        if not reserved:
            self._activity_waiters[key] = self._activity_waiters.get(key, 0) + 1
        try:
            async with asyncio.timeout(timeout_s):
                async with self._changed:
                    await self._changed.wait_for(ready)
        except TimeoutError:
            return False
        finally:
            if not reserved:
                remaining = self._activity_waiters[key] - 1
                if remaining:
                    self._activity_waiters[key] = remaining
                else:
                    del self._activity_waiters[key]
        return True


__all__ = ["AgentRegistry", "AgentRun", "AgentStatusName", "AbortReason", "bound_payload", "bound_text"]
