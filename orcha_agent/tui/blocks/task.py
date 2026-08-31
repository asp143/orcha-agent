"""Agent task and delivered-result transcript cards."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_spinner, theme_symbol, theme_value
from .tool import EXPAND_HINT, SPINNER_FRAMES, _frame, _result_text

_TERMINAL = frozenset(
    {
        "aborted",
        "cancelled",
        "canceled",
        "done",
        "error",
        "failed",
        "success",
        "succeeded",
    }
)
_SUCCESS = frozenset({"done", "success", "succeeded"})
_FAILURE = frozenset({"aborted", "cancelled", "canceled", "error", "failed"})


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _identity(agent: Mapping[str, Any], index: int) -> str:
    return str(agent.get("run_id") or agent.get("id") or agent.get("name") or index)


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if value is not None:
            target[key] = value


def _agents(block: Block) -> list[dict[str, Any]]:
    """Normalize task definitions and every supported job snapshot shape."""

    groups: list[list[Any]] = []
    supplied = _sequence(block.data.get("agents"))
    if supplied:
        groups.append(supplied)

    result = block.data.get("result")
    if isinstance(result, Mapping):
        for key in ("agents", "tasks", "spawned", "jobs", "results"):
            values = _sequence(result.get(key))
            if values:
                groups.append(values)
    elif _sequence(result):
        groups.append(_sequence(result))

    job = block.data.get("job")
    if isinstance(job, Mapping):
        groups.append([job])
    elif not groups and any(key in block.data for key in ("run_id", "name", "status")):
        groups.append([block.data])

    if not groups:
        args = block.data.get("args")
        tasks = _sequence(args.get("tasks")) if isinstance(args, Mapping) else []
        groups.append(
            [
                {
                    **(dict(item) if isinstance(item, Mapping) else {"task": str(item)}),
                    "status": "pending",
                }
                for item in tasks
            ]
        )

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for index, value in enumerate(group):
            item = dict(value) if isinstance(value, Mapping) else {"description": str(value)}
            identity = _identity(item, index)
            current = by_id.get(identity)
            if current is None:
                current = {}
                by_id[identity] = current
                ordered.append(current)
            _merge(current, item)
            current.setdefault("run_id", identity)
    return ordered


def _status(agent: Mapping[str, Any]) -> str:
    value = str(agent.get("status") or "running").casefold()
    if value == "completed":
        return "done"
    return value


def _glyph(block: Block, status: str, theme: Any) -> tuple[str, str]:
    if status == "running":
        return (
            theme_spinner(
                theme,
                "spinner.activity",
                int(block.data.get("spinner_frame", 0)),
                SPINNER_FRAMES,
            ),
            "accent",
        )
    if status in _SUCCESS:
        return str(theme_symbol(theme, "status.success", "✔")), "success"
    if status in {"error", "failed"}:
        return str(theme_symbol(theme, "status.error", "✘")), "error"
    if status in {"cancelled", "canceled", "aborted"}:
        return "⏹", "warning"
    if status == "parked":
        return "⏸", "muted"
    if status == "idle":
        return (
            theme_spinner(
                theme,
                "spinner.activity",
                int(block.data.get("spinner_frame", 0)),
                SPINNER_FRAMES,
            ),
            "accent",
        )
    return str(theme_symbol(theme, "status.pending", "○")), "accent"


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clip(value: Any, maximum: int = 40) -> str:
    text = _one_line(
        json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, Mapping) else value
    )
    return text if len(text) <= maximum else f"{text[: maximum - 1]}…"


def _seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        try:
            return max(0.0, float(value.removesuffix("s")))
        except ValueError:
            return None
    return None


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _elapsed(agent: Mapping[str, Any]) -> float:
    supplied = _seconds(agent.get("elapsed", agent.get("age_s")))
    if supplied is not None:
        return supplied
    created = _timestamp(agent.get("created_at"))
    if created is None:
        return 0.0
    updated = _timestamp(agent.get("updated_at"))
    if updated is None or _status(agent) not in _TERMINAL:
        updated = datetime.now(UTC)
    return max(0.0, (updated - created).total_seconds())


def _duration(value: float) -> str:
    seconds = max(0, int(round(value)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _tokens(agent: Mapping[str, Any]) -> int | None:
    supplied = agent.get("tokens")
    if isinstance(supplied, (int, float)):
        return int(supplied)
    values = (agent.get("tokens_in"), agent.get("tokens_out"))
    if any(isinstance(value, (int, float)) for value in values):
        return sum(int(value or 0) for value in values)
    return None


def _metrics(agent: Mapping[str, Any]) -> str:
    values: list[str] = []
    tokens = _tokens(agent)
    if tokens is not None:
        values.append(f"{tokens} tok")
    requests = agent.get("requests", agent.get("req"))
    if isinstance(requests, (int, float)):
        values.append(f"{int(requests)} req")
    cost = agent.get("cost")
    if isinstance(cost, (int, float)):
        values.append(f"${float(cost):.2f}")
    return "/".join(values)


def _agent_rows(
    block: Block,
    agent: Mapping[str, Any],
    theme: Any,
    *,
    expanded: bool,
) -> list[str]:
    status = _status(agent)
    glyph, _token = _glyph(block, status, theme)
    name = _one_line(agent.get("name") or agent.get("run_id") or agent.get("id") or "agent")
    description = _one_line(agent.get("description") or agent.get("task") or agent.get("prompt"))
    repeat_description = bool(block.data.get("repeat_description"))
    label = (
        f"{name}: {description}"
        if description and (description != name or repeat_description)
        else name
    )
    metrics = _metrics(agent)

    row = f"{glyph} {label} ⟦{status}⟧"
    row += f" {metrics} · {_duration(_elapsed(agent))}"

    rows = [row]
    tool = agent.get("current_tool") or agent.get("last_tool") or agent.get("tool")
    if tool:
        if agent.get("current_tool"):
            args = agent.get("current_tool_args", "")
        elif agent.get("last_tool"):
            args = agent.get("last_tool_args", "")
        else:
            args = agent.get("args", "")
        clipped = _clip(args)
        detail = f": {clipped}" if clipped else ""
        rows.append(f"└ {_one_line(tool)}{detail}")
    if expanded:
        result = agent.get("result")
        if result is None:
            result = agent.get("partial_findings", agent.get("findings"))
        for line in _result_text(result).splitlines():
            rows.append(f"  {line}")
    return rows


def _footer(agents: Sequence[Mapping[str, Any]], block: Block) -> str:
    succeeded = sum(_status(agent) in _SUCCESS for agent in agents)
    failed = sum(_status(agent) in _FAILURE for agent in agents)
    requests = sum(int(agent.get("requests", agent.get("req", 0)) or 0) for agent in agents)
    elapsed = _seconds(block.data.get("elapsed"))
    result = block.data.get("result")
    if elapsed is None and isinstance(result, Mapping):
        elapsed = _seconds(result.get("elapsed"))
    if elapsed is None:
        elapsed = max((_elapsed(agent) for agent in agents), default=0.0)
    return f"⟦{succeeded} succeeded · {failed} failed · {requests} req · {_duration(elapsed)}⟧"


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    """Render one aggregate task invocation or compact registry snapshot."""

    if budget_rows <= 0:
        return None
    agents = _agents(block)
    compact = bool(block.data.get("compact"))
    separator = str(theme_symbol(theme, "sep.thin", "·"))
    if compact:
        running = int(block.data.get("running", sum(_status(item) == "running" for item in agents)))
        idle = int(block.data.get("idle", sum(_status(item) == "idle" for item in agents)))
        queued = int(block.data.get("queued", sum(_status(item) == "queued" for item in agents)))
        header = f"Subagents {separator} {running} running {separator} {idle} idle"
        if queued:
            header += f" {separator} {queued} queued"
    else:
        task_glyph = "Task" if separator.isascii() else "⇶ Task"
        header = f"{task_glyph} {separator} {len(agents)} agents"

    if budget_rows == 1:
        return Text(header, style=str(theme_value(theme, "toolTitle")))

    visible = agents if expanded else agents[-4:]
    rows: list[str | Text] = []
    if not expanded and len(agents) > len(visible):
        rows.append(f"… {len(agents) - len(visible)} earlier agents {EXPAND_HINT}")
    for agent in visible:
        rows.extend(_agent_rows(block, agent, theme, expanded=expanded))
    if not compact:
        rows.append(_footer(agents, block))
    statuses = {_status(agent) for agent in agents}
    if statuses & {"running", "queued", "starting", "idle"}:
        frame_state = "running"
    elif statuses & _FAILURE:
        frame_state = "error"
    else:
        frame_state = "success"
    return _frame(
        header,
        rows,
        width=width,
        budget_rows=budget_rows,
        theme=theme,
        border_token="borderMuted",
        state=frame_state,
    )


def render_delivery(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text | None:
    """Render a settled agent result as a system delivery card."""

    if budget_rows <= 0:
        return None
    job = block.data.get("job")
    snapshot = dict(job) if isinstance(job, Mapping) else dict(block.data)
    name = _one_line(
        snapshot.get("name") or snapshot.get("run_id") or snapshot.get("id") or "Agent"
    )
    status = _status(snapshot)
    result = snapshot.get("result", block.data.get("result"))
    if result is None:
        result = snapshot.get("partial_findings", snapshot.get("findings"))
    lines = _result_text(result).splitlines() or ["No result was returned."]
    if not expanded and len(lines) > 4:
        lines = [*lines[:4], f"… {len(lines) - 4} more lines {EXPAND_HINT}"]
    border = (
        "error"
        if status in {"error", "failed"}
        else ("warning" if status in _FAILURE else "borderMuted")
    )
    return _frame(
        f"↩ {name} finished",
        lines,
        width=width,
        budget_rows=budget_rows,
        theme=theme,
        border_token=border,
        state=("error" if status in {"error", "failed"} or status in _FAILURE else "success"),
    )


render_task = render

__all__ = ["render", "render_delivery", "render_task"]
