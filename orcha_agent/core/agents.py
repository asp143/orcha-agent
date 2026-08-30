"""In-process agent runs and their application-scoped lifecycle registry."""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Literal

from langchain_core.messages import BaseMessage, message_to_dict

from orcha_agent.tui.turn import run_turn

from .agent import FILESYSTEM_TOOL_NAMES, build_agent
from .agent_types import AgentType
from .capture import capture_graph_values
from .config import Config
from .events import (
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    EventBus,
    ModelChunk,
    ToolCallStart,
)
from .ledger import CustomEntry, Ledger
from .registry import Registry
from .session import SessionStore

AgentStatusName = Literal[
    "running", "idle", "parked", "done", "failed", "aborted"
]
AbortReason = Literal["cancel", "timeout", "budget", "shutdown"]
_TERMINAL = frozenset({"done", "failed", "aborted"})
_WRAP_UP = "Wrap up and yield now."
_WAKE = object()


class _RunEventBus:
    """Count one run's streamed activity before forwarding to the app bus."""

    def __init__(self, run: AgentRun, target: EventBus) -> None:
        self._run = run
        self._target = target

    async def emit(self, event: Any) -> Any:
        if isinstance(event, ToolCallStart) and event.source_id == self._run.id:
            self._run.tool_calls += 1
            self._run.last_tool = event.name
        elif isinstance(event, ModelChunk) and event.source_id == self._run.id:
            usage = getattr(event.chunk, "usage_metadata", None)
            if isinstance(usage, dict):
                self._run.tokens_in += int(usage.get("input_tokens", 0) or 0)
                self._run.tokens_out += int(usage.get("output_tokens", 0) or 0)
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
        self.status: AgentStatusName = "running"
        self.abort_reason: AbortReason | None = None
        self.requests = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.last_tool: str | None = None
        self.result: Any = None
        self.partial_findings: list[Any] = []
        self.schema_overridden = False
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.agent_ready = asyncio.Event()
        self._status_changed = asyncio.Condition()
        self._turn_task: asyncio.Task[None] | None = None
        self._completion: tuple[AgentStatusName, Any] | None = None
        self._abort_requested: AbortReason | None = None
        self._runtime_timer: asyncio.Task[None] | None = None
        self._runtime_deadline = (
            monotonic() + cfg.agents.max_runtime_s
            if cfg.agents.max_runtime_s > 0
            else None
        )

    @property
    def thread_config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    async def ensure_agent(self) -> bool:
        if self.agent is not None:
            return True
        try:
            extras = tuple(self.owner.extra_tools(self))
            scope = self._tool_scope(extras)
            self.agent = await build_agent(
                self.registry,
                self.cfg,
                self.session,
                self.owner.bus,
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
        may_spawn = self.agent_type.spawns and self.depth < self.cfg.agents.max_depth
        if may_spawn:
            return scope
        if scope is None:
            scope = {
                *self.registry.tools,
                *FILESYSTEM_TOOL_NAMES,
                *(str(getattr(tool, "name", "")) for tool in extras),
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
        self._completion = (status, result)
        await self.inbox.put(_WAKE)

    async def request_abort(self, reason: AbortReason) -> None:
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
        self.status = status
        self.updated_at = datetime.now(UTC)
        async with self._status_changed:
            self._status_changed.notify_all()
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
        await self.owner._notify()

    async def _runtime_watchdog(self) -> None:
        if self._runtime_deadline is None:
            return
        await asyncio.sleep(max(0.0, self._runtime_deadline - monotonic()))
        await self.request_abort("timeout")

    def start(self, prompt: str | None) -> None:
        self.task = asyncio.create_task(self._run(prompt), name=f"agent:{self.id}")

    async def _run(self, prompt: str | None) -> None:
        if self.cfg.agents.max_runtime_s > 0:
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
                budget = self.cfg.agents.soft_request_budget
                if self.requests >= budget + 10:
                    await self._settle_abort("budget")
                    return
                if self.requests >= budget:
                    message = _WRAP_UP
                    continue
            await self._set_status("idle")
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
        self._changed = asyncio.Condition()

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def list(self, status: str | None = None) -> list[AgentRun]:
        return [
            self._runs[run_id]
            for run_id in self._order
            if status is None or self._runs[run_id].status == status
        ]

    def tree(self) -> list[AgentRun]:
        children: dict[str, list[AgentRun]] = {}
        roots: list[AgentRun] = []
        for run in self.list():
            if run.parent_id in self._runs:
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

    def _parent(self, parent: str) -> tuple[str, str, int]:
        if parent == "main":
            return "main", self.parent_session_id, 0
        run = self.get(parent)
        if run is None:
            raise LookupError(f"Unknown parent agent: {parent}")
        return run.id, run.session_id, run.depth + 1

    def _name(self, prompt: str, requested: str | None, parent_id: str) -> str:
        raw = requested or " ".join(re.findall(r"[A-Za-z0-9]+", prompt)[:3]) or "Agent"
        base = "".join(part[:1].upper() + part[1:] for part in re.findall(r"[A-Za-z0-9]+", raw))[:32]
        if not base:
            base = "Agent"
        sibling_names = {
            run.name for run in self._runs.values() if run.parent_id == parent_id
        }
        candidate = base
        suffix = 2
        while candidate in sibling_names:
            tail = str(suffix)
            candidate = f"{base[: 32 - len(tail)]}{tail}"
            suffix += 1
        return candidate

    def _model(self, spec: AgentType) -> str | list[str]:
        configured = self.cfg.model_roles.get(spec.model_role)
        if configured is not None:
            return configured
        if ":" in spec.model_role:
            return spec.model_role
        if spec.model_role == "task" and self.cfg.subagent_model is not None:
            return self.cfg.subagent_model
        return self.cfg.model

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
    ) -> AgentRun:
        spec = self.registry.agent_types.get(agent_type)
        if spec is None:
            available = ", ".join(sorted(self.registry.agent_types))
            raise LookupError(f"Unknown agent type {agent_type!r}; available: {available}")
        if schema_mode not in {"permissive", "strict"}:
            raise ValueError("schema_mode must be permissive or strict")
        parent_id, parent_session, depth = self._parent(parent)
        if depth > self.cfg.agents.max_depth:
            raise ValueError(
                f"maximum agent depth {self.cfg.agents.max_depth} exceeded by depth {depth}"
            )
        run_name = self._name(prompt, name, parent_id)
        model = self._model(spec)
        info = self.session.create(
            self.cfg.cwd,
            model,
            self.cfg.mode,
            title=run_name,
            parent_session=parent_session,
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
            cfg=replace(self.cfg, model=model),
            depth=depth,
            output_schema=output_schema or spec.output_schema,
            schema_mode=schema_mode,
            blocking=blocking or spec.blocking,
        )
        self._runs[run.id] = run
        self._order.append(run.id)
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

    async def send(self, run_id: str, text: str, *, interrupt: bool = False) -> AgentRun:
        run = self.get(run_id)
        if run is None:
            raise LookupError(f"Unknown agent: {run_id}")
        if run.terminal:
            raise RuntimeError(f"Agent {run.name} is {run.status}")
        if run.status == "parked":
            await self.revive(run.session_id)
        if interrupt and run._turn_task is not None:
            run._turn_task.cancel()
        await run.inbox.put(text)
        return run

    async def cancel(self, run_id: str, reason: AbortReason = "cancel") -> AgentRun:
        run = self.get(run_id)
        if run is None:
            raise LookupError(f"Unknown agent: {run_id}")
        await run.request_abort(reason)
        return run

    async def wait(
        self, ids: Iterable[str] | None = None, *, timeout_s: float = 300
    ) -> list[AgentRun]:
        selected = set(ids) if ids is not None else set(self._runs)

        def settled() -> list[AgentRun]:
            return [
                run
                for run in self.list()
                if run.id in selected and run.status in _TERMINAL
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

    async def revive(self, session_id: str) -> AgentRun:
        run = next(
            (candidate for candidate in self._runs.values() if candidate.session_id == session_id),
            None,
        )
        if run is None:
            raise LookupError(f"Unknown agent session: {session_id}")
        if run.status != "parked":
            return run
        await run._set_status("idle")
        run.start(None)
        return run

    async def shutdown(self) -> None:
        for run in self.list():
            if not run.terminal:
                await run.request_abort("shutdown")
        pending = [run.task for run in self.list() if run.task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


__all__ = ["AgentRegistry", "AgentRun", "AgentStatusName", "AbortReason"]
