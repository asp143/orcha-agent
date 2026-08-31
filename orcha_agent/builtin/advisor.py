"""Persistent transcript watchdog for interactive sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
)

from orcha_agent.core.agents import AgentRun
from orcha_agent.core.events import Advisory, TurnEnd
from orcha_agent.core.ledger import MessageEntry
from orcha_agent.core.models import filter_foreign_blocks
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="advisor", version="1.0.0")

_PRIVATE_BLOCK_TYPES = frozenset({"reasoning", "thinking", "thought"})


@dataclass(slots=True)
class _SessionState:
    cursor: str | None = None
    turns: int = 0
    last_interrupt_turn: int | None = None
    run: AgentRun | None = None
    spawn_task: asyncio.Task[AgentRun] | None = None
    watchdog: str | None = None
    watchdog_loaded: bool = False


def _configured_model(cfg: Any) -> str | list[str]:
    configured = cfg.advisor.model
    if not configured.startswith("@"):
        return configured
    role = configured[1:]
    selected = getattr(cfg, "model_roles", {}).get(role)
    if selected is not None:
        return selected
    if role == "advisor":
        return cfg.model
    raise ValueError(f"Unknown advisor model role: {configured}")


def _watchdog_path(
    cwd: Path,
    *,
    trust_cwd: bool = True,
    trusted_dirs: tuple[Path, ...] | None = None,
) -> Path | None:
    root = cwd.resolve()
    if trust_cwd:
        directories = (root, *root.parents)
        if trusted_dirs is not None:
            containing_roots = [
                trusted
                for configured in trusted_dirs
                if root == (trusted := Path(configured).resolve()) or root.is_relative_to(trusted)
            ]
            boundary = max(
                containing_roots,
                key=lambda path: len(path.parts),
                default=root,
            )
            directories = directories[: directories.index(boundary) + 1]
        for directory in directories:
            candidate = directory / "WATCHDOG.md"
            if candidate.is_file():
                return candidate
    fallback = Path.home() / ".config" / "orcha-agent" / "WATCHDOG.md"
    return fallback if fallback.is_file() else None


def _is_private_block(value: Any) -> bool:
    name = str(value).casefold()
    return name in _PRIVATE_BLOCK_TYPES or name.startswith(("reasoning_", "thinking_", "thought_"))


def _without_private_blocks(value: Any) -> Any:
    if isinstance(value, list):
        cleaned: list[Any] = []
        for item in value:
            filtered = _without_private_blocks(item)
            if filtered is not None:
                cleaned.append(filtered)
        return cleaned
    if isinstance(value, dict):
        if _is_private_block(value.get("type", "")):
            return None
        return {
            key: filtered
            for key, item in value.items()
            if not _is_private_block(key)
            and (filtered := _without_private_blocks(item)) is not None
        }
    return value


def _content(value: Any) -> Any:
    cleaned = _without_private_blocks(value)
    return "" if cleaned is None else cleaned


def _transcript_delta(entries: list[Any]) -> list[dict[str, Any]]:
    serialized = [entry.message for entry in entries if isinstance(entry, MessageEntry)]
    if not serialized:
        return []
    messages = filter_foreign_blocks(
        messages_from_dict(serialized),
        _PRIVATE_BLOCK_TYPES,
    )
    delta: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            delta.append({"role": "user", "content": _content(message.content)})
            continue
        if isinstance(message, AIMessage):
            content = _content(message.content)
            if content not in ("", []):
                delta.append({"role": "assistant", "content": content})
            for call in message.tool_calls:
                delta.append(
                    {
                        "role": "tool",
                        "name": call.get("name"),
                        "arguments": _content(call.get("args")),
                        "id": call.get("id"),
                    }
                )
            continue
        if isinstance(message, ToolMessage):
            delta.append(
                {
                    "role": "result",
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "content": _content(message.content),
                }
            )
    return delta


def _prompt(delta: list[dict[str, Any]], watchdog: str | None) -> str:
    transcript = escape(
        json.dumps(delta, ensure_ascii=False, separators=(",", ":")),
        quote=False,
    )
    parts = [
        "Review this transcript delta. Call advise exactly once with either "
        "note and severity (nit, concern, or blocker), or none=true.",
        "\n<transcript-delta>\n" + transcript + "\n</transcript-delta>",
    ]
    if watchdog:
        parts.append(
            "\n<watchdog-instructions>\n"
            + escape(watchdog, quote=False)
            + "\n</watchdog-instructions>"
        )
    return "\n".join(parts)


def _followup(note: str, severity: str) -> str:
    return (
        f'<advisory advisor="advisor" severity="{severity}" '
        'guidance="weigh, don\'t blindly obey">\n'
        f"{escape(note, quote=False)}\n"
        "</advisory>"
    )


class AdvisorService:
    """Schedule bounded looks while retaining one advisor run per session."""

    def __init__(self, ctx: Any, *, submit_followup: Any) -> None:
        self.ctx = ctx
        self._submit_followup = submit_followup
        self._states: dict[str, _SessionState] = {}
        self._look_tasks: dict[str, asyncio.Task[None]] = {}
        self._followup_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def _state(self, session_id: str) -> _SessionState:
        return self._states.setdefault(session_id, _SessionState())

    def _watchdog(self, state: _SessionState) -> str | None:
        if state.watchdog_loaded:
            return state.watchdog
        state.watchdog_loaded = True
        cfg = self.ctx.cfg
        path = _watchdog_path(
            Path(cfg.cwd),
            trust_cwd=bool(getattr(cfg, "trust_cwd", False)),
            trusted_dirs=tuple(getattr(cfg, "trusted_dirs", ()) or ()),
        )
        if path is not None:
            try:
                state.watchdog = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                state.watchdog = None
        return state.watchdog

    async def on_main_turn_end(self, event: TurnEnd) -> None:
        if self._closed or event.source_id != "main" or not self.ctx.cfg.advisor.enabled:
            return
        session_id = str(self.ctx.session_id)
        state = self._state(session_id)
        state.turns += 1
        active = self._look_tasks.get(session_id)
        if active is not None and not active.done():
            return

        task = asyncio.create_task(
            self._look_at_delta(session_id, state),
            name=f"advisor-look:{session_id}",
        )
        self._look_tasks[session_id] = task
        task.add_done_callback(lambda completed, sid=session_id: self._forget_task(sid, completed))

    async def _look_at_delta(self, session_id: str, state: _SessionState) -> None:
        run = state.run
        if run is not None and (run.terminal or not self._matches_config(run)):
            if run.terminal:
                state.run = None
            else:
                await self._reset_run(state)
            state.cursor = None
        path = self.ctx.ledger.path(session_id)
        start = 0
        if state.cursor is not None:
            matched = next(
                (index for index, entry in enumerate(path) if entry.id == state.cursor),
                None,
            )
            if matched is None:
                await self._reset_run(state)
                state.cursor = None
            else:
                start = matched + 1
        delta = _transcript_delta(path[start:])
        if not delta:
            return
        await self._look(
            session_id,
            state,
            _prompt(delta, self._watchdog(state)),
            cursor=path[-1].id,
        )

    async def _reset_run(self, state: _SessionState) -> None:
        run = state.run
        state.run = None
        if run is not None and not run.terminal:
            await run.request_abort("cancel")
            if run.task is not None:
                await asyncio.gather(run.task, return_exceptions=True)

    def _forget_task(self, session_id: str, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
        if self._look_tasks.get(session_id) is task:
            self._look_tasks.pop(session_id, None)

    def _matches_config(self, run: AgentRun) -> bool:
        expected_tools = {*self.ctx.cfg.advisor.tools, "advise"}
        actual_tools = getattr(run.agent_type, "tools", None)
        run_cfg = getattr(run, "cfg", None)
        return (
            actual_tools is not None
            and set(actual_tools) == expected_tools
            and getattr(run_cfg, "model", None) == _configured_model(self.ctx.cfg)
            and getattr(run_cfg, "mode", None) == self.ctx.cfg.mode
            and getattr(run_cfg, "backend", None) == self.ctx.cfg.backend
        )

    async def _ready_run(self, state: _SessionState, prompt: str) -> AgentRun:
        run = state.run
        if run is not None and not run.terminal and not self._matches_config(run):
            await self._reset_run(state)
            run = None
        if run is None:
            existing = getattr(self.ctx.agents, "advisor_run", None)
            if callable(existing):
                candidate = existing(str(self.ctx.session_id))
                if candidate is not None:
                    if not self._matches_config(candidate):
                        state.run = candidate
                        await self._reset_run(state)
                    else:
                        run = candidate
                        state.run = run
        if run is None or run.terminal:
            spawn_task = state.spawn_task
            owns_spawn_prompt = spawn_task is None
            if spawn_task is None:
                spawn_task = asyncio.create_task(
                    self.ctx.agents.spawn(
                        "advisor",
                        prompt,
                        name="Advisor",
                        parent="main",
                        model=_configured_model(self.ctx.cfg),
                        tools=self.ctx.cfg.advisor.tools,
                        visible=False,
                    ),
                    name=f"advisor-run:{self.ctx.session_id}",
                )
                state.spawn_task = spawn_task
                spawn_task.add_done_callback(
                    lambda completed, target=state: self._remember_run(target, completed)
                )
            run = await asyncio.shield(spawn_task)
            state.run = run
            if owns_spawn_prompt:
                return run

        if run.status == "running":
            await run.wait_status(
                "idle",
                timeout_s=self.ctx.cfg.advisor.timeout_s,
            )
        while not run.advice_outbox.empty():
            run.advice_outbox.get_nowait()
        if run.status == "parked":
            await self.ctx.agents.revive(run.session_id)
        await self.ctx.agents.send(run.id, prompt)
        return run

    @staticmethod
    async def _next_advice(run: AgentRun) -> tuple[bool, dict[str, Any] | None]:
        advice = asyncio.create_task(run.advice_outbox.get())
        try:
            run_task = run.task
            if run_task is None:
                return True, await advice
            done, _pending = await asyncio.wait(
                (advice, run_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if advice in done:
                return True, advice.result()
            return False, None
        finally:
            if not advice.done():
                advice.cancel()
                await asyncio.gather(advice, return_exceptions=True)

    @staticmethod
    def _remember_run(state: _SessionState, task: asyncio.Task[AgentRun]) -> None:
        if task.cancelled():
            state.spawn_task = None
            return
        state.spawn_task = None
        try:
            state.run = task.result()
        except Exception:
            pass

    async def _look(
        self,
        session_id: str,
        state: _SessionState,
        prompt: str,
        *,
        cursor: str | None = None,
    ) -> None:
        previous_cursor = state.cursor
        try:
            async with asyncio.timeout(self.ctx.cfg.advisor.timeout_s):
                run = await self._ready_run(state, prompt)
                if cursor is not None:
                    state.cursor = cursor
                received, payload = await self._next_advice(run)
                if not received or payload is None:
                    state.cursor = previous_cursor
                    return
        except (TimeoutError, asyncio.CancelledError):
            state.cursor = previous_cursor
            return
        except Exception:
            state.cursor = previous_cursor
            return
        if str(self.ctx.session_id) != session_id:
            state.cursor = previous_cursor
            return

        note = payload.get("note")
        severity = payload.get("severity")
        if payload.get("none") is True:
            note = None
            severity = "none"
        if not isinstance(note, str) or severity not in {
            "nit",
            "concern",
            "blocker",
        }:
            note = None
            severity = "none"

        interrupt = False
        if note is not None and severity in {"concern", "blocker"}:
            last = state.last_interrupt_turn
            if last is None or state.turns - last >= self.ctx.cfg.advisor.immune_turns:
                interrupt = True
                state.last_interrupt_turn = state.turns

        await self.ctx.bus.emit(
            Advisory(
                note=note,
                severity=severity,
                advisor_id=run.id,
                interrupt=interrupt,
            )
        )
        if interrupt and note is not None:
            followup = asyncio.create_task(
                self._submit_followup(session_id, _followup(note, severity)),
                name=f"advisor-followup:{session_id}",
            )
            self._followup_tasks.add(followup)
            followup.add_done_callback(self._forget_followup)

    def _forget_followup(self, task: asyncio.Task[None]) -> None:
        self._followup_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def before_user_prompt(self) -> None:
        """Detach a pending look so advisor latency never gates fresh input."""
        task = self._look_tasks.get(str(self.ctx.session_id))
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = tuple(self._look_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        followups = tuple(self._followup_tasks)
        if followups:
            await asyncio.gather(*followups, return_exceptions=True)
        spawn_tasks = [
            task
            for state in self._states.values()
            if (task := state.spawn_task) is not None and not task.done()
        ]
        for task in spawn_tasks:
            task.cancel()
        if spawn_tasks:
            await asyncio.gather(*spawn_tasks, return_exceptions=True)
        runs = [run for state in self._states.values() if (run := state.run) is not None]
        for run in runs:
            if not run.terminal:
                await run.request_abort("shutdown")
        pending = [run.task for run in runs if run.task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def register(api: PluginAPI) -> None:
    del api


__all__ = ["AdvisorService", "register"]
