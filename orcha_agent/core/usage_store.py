"""Persistent request accounting, independent of transcript rendering."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from rich.table import Table
from rich.text import Text

from .events import Event
from .usage import usage_cost

SCHEMA = """CREATE TABLE IF NOT EXISTS usage_requests (
    request_id TEXT PRIMARY KEY, ts REAL NOT NULL, session TEXT NOT NULL,
    model TEXT NOT NULL, provider TEXT NOT NULL, role TEXT NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
    cache_read INTEGER NOT NULL, cache_write INTEGER NOT NULL,
    cost REAL NOT NULL, ttft REAL, duration REAL NOT NULL, stop_reason TEXT NOT NULL
)"""


@dataclass(slots=True)
class UsageRecorded(Event):
    """A committed usage row is available for cached status refresh."""


@dataclass(frozen=True)
class UsageRequest:
    request_id: str
    ts: float
    session: str
    model: str
    provider: str
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    ttft: float | None = None
    duration: float = 0.0
    stop_reason: str = ""


class UsageStore:
    """Use the session store's connection and lock, including Turso replicas."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def record(self, request: UsageRequest) -> None:
        values = asdict(request)
        self.session._write(
            f"INSERT OR IGNORE INTO usage_requests ({','.join(values)}) "
            f"VALUES ({','.join('?' for _ in values)})",
            tuple(values.values()),
        )

    def report(
        self, period: str = "session", session_id: str | None = None, *, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        if period not in {"today", "week", "session", "all"}:
            raise ValueError("Usage period must be today, week, session, or all")
        where = "1=1"
        parameters: tuple[Any, ...] = ()
        if period == "session":
            if not session_id:
                raise ValueError("A session is required for session usage")
            # Child agents are part of their owning session's total.
            where = "session IN (WITH RECURSIVE family(id) AS (SELECT ? UNION ALL SELECT s.thread_id FROM sessions s JOIN family f ON s.parent_session=f.id) SELECT id FROM family)"
            parameters = (session_id,)
        elif period in {"today", "week"}:
            current = now or datetime.now().astimezone()
            start = current.replace(hour=0, minute=0, second=0, microsecond=0)
            if period == "week":
                start -= timedelta(days=start.weekday())
            where = "ts >= ? AND ts <= ?"
            parameters = (start.timestamp(), current.timestamp())
        with self.session.saver.lock:
            rows = self.session._connection.execute(
                f"SELECT model, provider, COUNT(*) requests, SUM(input_tokens) input_tokens, "
                f"SUM(output_tokens) output_tokens, SUM(cache_read) cache_read, "
                f"SUM(cache_write) cache_write, SUM(cost) cost, AVG(ttft) ttft, "
                f"AVG(duration) duration FROM usage_requests WHERE {where} "
                "GROUP BY model, provider ORDER BY cost DESC, model",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]


def usage_table(rows: list[dict[str, Any]], period: str) -> Table:
    table = Table(title=f"Usage · {period} (estimated API cost)")
    for name in (
        "Model",
        "Requests",
        "Input",
        "Output",
        "Cache read",
        "Cache write",
        "Cost",
        "TTFT",
        "Duration",
    ):
        table.add_column(name, justify="left" if name == "Model" else "right")
    for row in rows:
        table.add_row(
            Text(str(row["model"])),
            *(
                str(row[key])
                for key in (
                    "requests",
                    "input_tokens",
                    "output_tokens",
                    "cache_read",
                    "cache_write",
                )
            ),
            f"${row['cost']:.4f}",
            "—" if row["ttft"] is None else f"{row['ttft']:.2f}s",
            f"{row['duration']:.2f}s",
        )
    table.add_row(
        "Total",
        *(
            str(sum(row[key] for row in rows))
            for key in ("requests", "input_tokens", "output_tokens", "cache_read", "cache_write")
        ),
        f"${sum(row['cost'] for row in rows):.4f}",
        "",
        "",
    )
    return table


def _add_usage(total: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens"):
        total[key] = total.get(key, 0) + values.get(key, 0)
    details = total.setdefault("input_token_details", {})
    for key, value in values.get("input_token_details", {}).items():
        details[key] = details.get(key, 0) + (value or 0)


class UsageCallback(AsyncCallbackHandler):
    """One row per model run; callbacks are inherited by nested model requests."""

    def __init__(
        self,
        session: Any,
        cfg: Any,
        *,
        session_id: str | None = None,
        model: str | None = None,
        role: str = "main",
        bus: Any = None,
    ) -> None:
        self.bus = bus
        self.store = UsageStore(session)
        self.cfg = cfg
        self.session_id = session_id
        self.model = model
        self.role = role
        self.pending: dict[UUID, dict[str, Any]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        fallback = self.cfg.model
        if isinstance(fallback, list):
            fallback = fallback[0]
        model = str(meta.get("orcha_model") or self.model or meta.get("ls_model_name") or fallback)
        if ":" not in model:
            model = f"{meta.get('ls_provider', str(fallback).partition(':')[0])}:{model}"
        self.pending[run_id] = dict(
            start=time.monotonic(),
            ts=time.time(),
            ttft=None,
            model=model,
            role=str(
                meta.get("orcha_agent_type", "main")
                if meta.get("orcha_role", self.role) == "main"
                else meta.get("orcha_role") or self.role
            ),
            thread=str(meta.get("thread_id") or ""),
        )

    async def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        pending = self.pending.get(run_id)
        if pending is not None and pending["ttft"] is None and token:
            pending["ttft"] = time.monotonic() - pending["start"]

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        pending = self.pending.pop(run_id, None)
        if pending is None:
            return
        usage: dict[str, Any] = {}
        stop = ""
        for group in response.generations:
            for generation in group:
                message = getattr(generation, "message", None)
                values = getattr(message, "usage_metadata", None) or {}
                _add_usage(usage, values)
                metadata = getattr(message, "response_metadata", None) or {}
                stop = str(
                    metadata.get("stop_reason")
                    or metadata.get("finish_reason")
                    or (generation.generation_info or {}).get("finish_reason")
                    or stop
                )
        await self._record(run_id, pending, usage, stop)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        pending = self.pending.pop(run_id, None)
        if pending is not None:
            from orcha_agent.extensibility.stream_rules import StreamInterrupt

            usage: dict[str, Any] = {}
            if isinstance(error, StreamInterrupt):
                # The authoritative rule callback may abort on a chunk that
                # already reports billable usage. Keep those deltas on this
                # failed request; the retry receives its own run ID and row.
                for message in error.usage:
                    _add_usage(usage, message.usage_metadata or {})
            await self._record(run_id, pending, usage, "error")

    async def _record(
        self, run_id: UUID, pending: dict[str, Any], usage: dict[str, Any], stop: str
    ) -> None:
        def write() -> None:
            session_id = self.session_id
            if not session_id:
                with self.store.session.saver.lock:
                    thread = self.store.session.get_thread(pending["thread"])
                session_id = thread.session_id if thread else pending["thread"]
            details = usage.get("input_token_details", {})
            request = UsageRequest(
                str(run_id),
                pending["ts"],
                session_id or "unknown",
                pending["model"],
                pending["model"].partition(":")[0],
                pending["role"],
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
                int(details.get("cache_read", details.get("cache_read_input_tokens", 0)) or 0),
                int(
                    details.get(
                        "cache_creation",
                        details.get("cache_write", details.get("cache_creation_input_tokens", 0)),
                    )
                    or 0
                ),
                usage_cost(pending["model"], usage, self.cfg.pricing, config=self.cfg),
                pending["ttft"],
                time.monotonic() - pending["start"],
                stop,
            )
            self.store.record(request)

        await asyncio.to_thread(write)
        if self.bus is not None:
            await self.bus.emit(UsageRecorded())
