"""Context-bound tools for spawning and coordinating in-process agents."""

from __future__ import annotations

import asyncio
import html
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import StructuredTool

from orcha_agent.core.agents import bound_payload
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="tools-agents", version="1.0.0")

_AGENT_PROMPT = (
    "Use task for genuinely independent work and write each assignment with Target, "
    "Change, and Acceptance sections. Task results arrive asynchronously; use hub "
    "wait or jobs instead of polling. Spawned runs must finish with yield."
)

_AGENT_MESSAGE_MAX_BYTES = 256 * 1024
_AGENT_MESSAGE_TRUNCATED = "...[truncated at 262144 bytes]"


def _agent_message(sender_name: str, role: str, message: str) -> str:
    sender = html.escape(sender_name, quote=True)
    header = f'<agent-message from="{sender}" role="{role}">'
    footer = "</agent-message>"

    def framed(length: int, *, truncated: bool) -> str:
        content = html.escape(message[:length])
        if truncated:
            content += html.escape(f"\n{_AGENT_MESSAGE_TRUNCATED}")
        return f"{header}{content}{footer}"

    complete = framed(len(message), truncated=False)
    if len(complete.encode("utf-8")) <= _AGENT_MESSAGE_MAX_BYTES:
        return complete
    low, high = 0, len(message)
    while low < high:
        middle = (low + high + 1) // 2
        if len(framed(middle, truncated=True).encode("utf-8")) <= _AGENT_MESSAGE_MAX_BYTES:
            low = middle
        else:
            high = middle - 1
    return framed(low, truncated=True)

_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "context": {"type": "string"},
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "agent": {"type": "string", "default": "task"},
                    "task": {"type": "string"},
                    "output_schema": {"type": "object"},
                    "schema_mode": {
                        "type": "string",
                        "enum": ["permissive", "strict"],
                        "default": "permissive",
                    },
                    "blocking": {"type": "boolean", "default": False},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tasks"],
    "additionalProperties": False,
}
_YIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"type": {"type": "string"}, "data": {}, "error": {"type": "string"}},
    "required": ["type"],
    "additionalProperties": False,
}
_HUB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["list", "send", "wait", "inbox", "cancel", "jobs"]},
        "status": {"type": "string"},
        "to": {"type": "string"},
        "message": {"type": "string"},
        "interrupt": {"type": "boolean", "default": False},
        "await": {"type": "boolean", "default": False},
        "ids": {"type": "array", "items": {"type": "string"}},
        "timeout_s": {"type": "number", "minimum": 0, "default": 300},
        "reason": {"type": "string", "enum": ["cancel", "timeout", "budget", "shutdown"], "default": "cancel"},
    },
    "required": ["op"],
    "additionalProperties": False,
}
_ADVISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "minLength": 1},
        "severity": {"type": "string", "enum": ["nit", "concern", "blocker"]},
        "none": {"type": "boolean", "const": True},
    },
    "oneOf": [
        {"required": ["note", "severity"], "not": {"required": ["none"]}},
        {
            "required": ["none"],
            "not": {"anyOf": [{"required": ["note"]}, {"required": ["severity"]}]},
        },
    ],
    "additionalProperties": False,
}


def _registry(host: Any) -> Any:
    value = getattr(host, "agents", None) or getattr(host, "owner", None)
    if value is None:
        raise RuntimeError("agent tools require a host bound to an AgentRegistry")
    return value


def _summary(run: Any) -> dict[str, Any]:
    return {"id": run.id, "name": run.name, "type": run.agent_type.name, "status": run.status, "blocking": run.blocking}


def _job(run: Any) -> dict[str, Any]:
    return {
        "id": run.id, "name": run.name, "type": run.agent_type.name,
        "status": run.status, "result": run.result,
        "schema_overridden": run.schema_overridden,
        "findings": list(run.partial_findings), "delivered": run.delivered,
    }


def _roster(registry: Any, run: Any, caller: str) -> dict[str, Any]:
    age = max(0.0, (datetime.now(UTC) - run.created_at).total_seconds())
    return {
        "id": run.id, "name": run.name, "type": run.agent_type.name,
        "status": run.status, "age_s": round(age, 3), "last_tool": run.last_tool,
        "tokens": run.tokens_in + run.tokens_out, "cost": run.cost,
        "unread": registry.unread_count(run.id, caller=caller),
    }


def _timeout(registry: Any) -> float:
    configured = float(registry.cfg.agents.max_runtime_s)
    return configured if configured > 0 else 300.0


def _matches(value: Any, expected: str) -> bool:
    if expected == "null": return value is None
    if expected == "boolean": return isinstance(value, bool)
    if expected == "string": return isinstance(value, str)
    if expected == "integer": return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object": return isinstance(value, dict)
    if expected == "array": return isinstance(value, list)
    return False

_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "string", "integer", "number", "object", "array"}
)
_SCHEMA_KEYWORDS = frozenset({"type", "enum", "required", "properties", "items"})


def _validate_schema(schema: Any, path: str = "$") -> str | None:
    if not isinstance(schema, dict):
        return f"{path}: schema must be an object"
    for keyword in schema:
        if keyword not in _SCHEMA_KEYWORDS:
            return f"{path}: unsupported output schema keyword {keyword!r}"
    if "type" in schema:
        expected = schema["type"]
        choices = expected if isinstance(expected, list) else [expected]
        if not choices or not all(isinstance(item, str) for item in choices):
            return f"{path}: schema type must be a string or list of strings"
        unsupported = [item for item in choices if item not in _SCHEMA_TYPES]
        if unsupported:
            return f"{path}: unsupported schema type {unsupported[0]!r}"
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            return f"{path}: schema enum must be a non-empty array"
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(
            isinstance(key, str) for key in required
        ):
            return f"{path}: schema required must be an array of strings"
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict) or not all(
            isinstance(key, str) for key in properties
        ):
            return f"{path}: schema properties must be an object with string keys"
        for key, child in properties.items():
            if error := _validate_schema(child, f"{path}.properties.{key}"):
                return error
    if "items" in schema:
        if error := _validate_schema(schema["items"], f"{path}.items"):
            return error
    return None


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> str | None:
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_matches(value, item) for item in choices):
            return f"{path}: expected {' or '.join(choices)}"
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{path}: value is not one of {enum!r}"
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"{path}: missing required property {key!r}"
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value and (
                error := _validate_value(value[key], child, f"{path}.{key}")
            ):
                return error
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            if error := _validate_value(item, schema["items"], f"{path}[{index}]"):
                return error
    return None


def _validate(value: Any, schema: Any, path: str = "$") -> str | None:
    if error := _validate_schema(schema):
        return f"invalid output schema: {error}"
    return _validate_value(value, schema, path)


def agent_tools(host: Any) -> tuple[StructuredTool, ...]:
    """Return orchestration tools allowed for one main or child host."""
    registry = _registry(host)
    caller = str(getattr(host, "source_id", "main"))
    is_main = caller == "main"

    async def task_call(tasks: list[dict[str, Any]], context: str | None = None) -> dict[str, Any]:
        if len(tasks) > 16:
            raise ValueError("task accepts at most 16 items")
        shared = context.strip() if context else ""

        async def spawn(item: dict[str, Any]) -> Any:
            prompt = "\n\n".join(part for part in (shared, str(item["task"]).strip()) if part)
            output_schema = item.get("output_schema")
            if output_schema is not None and (
                error := _validate_schema(output_schema)
            ):
                raise ValueError(f"invalid output schema: {error}")
            return await registry.spawn(
                str(item.get("agent") or "task"), prompt, name=item.get("name"), parent=caller,
                output_schema=output_schema,
                schema_mode=str(item.get("schema_mode") or "permissive"),
                blocking=bool(item.get("blocking", False)),
            )

        outcomes = await asyncio.gather(*(spawn(item) for item in tasks), return_exceptions=True)
        runs, errors = [], []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                errors.append({"index": index, "error": f"{type(outcome).__name__}: {outcome}"})
            else:
                runs.append(outcome)
        blocking = [run for run in runs if run.blocking]
        if blocking:
            await registry.wait_all((run.id for run in blocking), timeout_s=_timeout(registry), caller=caller)
        completed = [run for run in blocking if run.terminal]
        if completed:
            await registry.deliver(caller, (run.id for run in completed))
        return {
            "spawned": [_summary(run) for run in runs],
            "results": [_job(run) for run in completed],
            "timed_out": [run.id for run in blocking if not run.terminal],
            "errors": errors,
        }

    async def yield_call(type: str, data: Any = None, error: str | None = None) -> dict[str, Any]:
        run = host
        if type == "findings":
            finding = bound_payload(
                data if error is None else {"data": data, "error": error}
            )
            accumulated = bound_payload([*run.partial_findings, finding])
            run.partial_findings = (
                accumulated if isinstance(accumulated, list) else [accumulated]
            )
            await registry.record_yield(run, {"type": type, "data": finding})
            return {"accepted": True, "terminal": False, "findings": len(run.partial_findings)}
        payload, status = bound_payload(data), "done"
        if type == "error" or error is not None:
            payload, status = bound_payload(
                {"error": error or data or "agent reported an error"}
            ), "failed"
        validation_error = _validate(payload, run.output_schema) if status == "done" and run.output_schema is not None else None
        if validation_error is not None:
            run.validation_attempts += 1
            attempt = run.validation_attempts
            message = f"output schema validation failed: {validation_error}"
            if attempt < 3:
                await registry.record_yield(run, {"type": type, "accepted": False, "error": message})
                return {"accepted": False, "terminal": False, "attempt": attempt, "error": message}
            if run.schema_mode == "strict":
                await registry.record_yield(run, {"type": type, "accepted": False, "error": message})
                await run.complete({"error": message}, status="failed")
                return {"accepted": False, "terminal": True, "status": "failed", "attempt": attempt, "error": message}
            run.schema_overridden = True
        await registry.record_yield(run, {"type": type, "data": payload})
        await run.complete(payload, status=status)
        return {"accepted": True, "terminal": True, "status": status, "schema_overridden": run.schema_overridden}

    async def advise_call(
        note: str | None = None,
        severity: str | None = None,
        none: bool | None = None,
    ) -> dict[str, Any]:
        if none is True and note is None and severity is None:
            payload: dict[str, Any] = {"none": True}
        elif none is None and isinstance(note, str) and note.strip() and severity in {"nit", "concern", "blocker"}:
            payload = {"note": note.strip(), "severity": severity}
        else:
            raise ValueError("advise requires exactly note and severity, or none=true")
        await registry.record_advice(host, payload)
        return {"accepted": True, "terminal": False}

    async def hub_call(**kwargs: Any) -> dict[str, Any]:
        op = str(kwargs["op"])
        if op == "list":
            return {"agents": [_roster(registry, run, caller) for run in registry.list(status=kwargs.get("status"), caller=caller)]}
        if op == "send":
            target, message = kwargs.get("to"), kwargs.get("message")
            if not isinstance(target, str) or not isinstance(message, str):
                return {"error": "hub send requires string to and message"}
            await_reply = bool(kwargs.get("await", False))
            reservation = None
            try:
                try:
                    target_id = registry.resolve(target, caller=caller, visible_only=True)
                    peer = registry.get(target_id, caller=caller) if target_id != "main" else None
                    before = 0 if peer is None else peer.yield_count
                    if await_reply:
                        reservation = registry.reserve_activity_waiter(caller)
                    sender_name = "main" if is_main else str(getattr(host, "name", caller))
                    role = (
                        "parent"
                        if peer is not None and getattr(peer, "parent_id", None) == caller
                        else "peer"
                    )
                    has_waiter = await registry.post_message(
                        caller,
                        target_id,
                        message,
                        mailbox_only_if_waiting=peer is not None,
                    )
                    if peer is not None and not has_waiter:
                        await registry.send(
                            peer.id,
                            _agent_message(sender_name, role, message),
                            interrupt=bool(kwargs.get("interrupt", False)),
                            caller=caller,
                        )
                except (LookupError, RuntimeError, ValueError) as exc:
                    return {"error": str(exc), "op": op}
                response: dict[str, Any] = {"sent": {"to": target_id, "name": "main" if peer is None else peer.name, "status": "running" if peer is None else peer.status}}
                if not await_reply:
                    return response
                await registry.wait_activity(caller, timeout_s=_timeout(registry), peer=target_id, after_yield=before, reserved=True)
                replies = registry.drain_messages(caller, sender=target_id, caller=caller)
                if replies: response["event"] = {"kind": "message", "messages": replies}
                elif peer is not None and peer.yield_count > before: response["event"] = {"kind": "yield", "from": peer.id, "payload": peer.last_yield}
                else: response["event"] = {"kind": "timeout"}
                return response
            finally:
                if reservation is not None:
                    registry.release_activity_waiter(reservation)
        if op == "inbox": return {"messages": registry.drain_messages(caller, caller=caller)}
        if op == "wait":
            ids, timeout_s = kwargs.get("ids"), float(kwargs.get("timeout_s", 300))
            await registry.wait_activity(caller, ids=ids, timeout_s=timeout_s)
            messages = registry.drain_messages(caller, caller=caller)
            settled = [run for run in registry.jobs(caller, ids=ids, caller=caller) if run.terminal and not run.delivered]
            claimed = (
                await registry.deliver(caller, (run.id for run in settled))
                if settled
                else []
            )
            return {"messages": messages, "jobs": [_job(run) for run in claimed], "timed_out": not messages and not claimed}
        if op == "jobs":
            jobs = registry.jobs(caller, caller=caller)
            pending = [run for run in jobs if run.terminal and not run.delivered]
            if pending: await registry.deliver(caller, (run.id for run in pending))
            return {"jobs": [_job(run) for run in jobs]}
        if op == "cancel":
            ids = kwargs.get("ids")
            if not isinstance(ids, list) or not ids: return {"error": "hub cancel requires a non-empty ids list"}
            cancelled, errors = [], []
            for selector in ids:
                try:
                    run_id = registry.resolve(str(selector), caller=caller, visible_only=True)
                    if run_id == "main": raise ValueError("main cannot be cancelled through hub")
                    cancelled.append(_summary(await registry.cancel(run_id, reason=str(kwargs.get("reason") or "cancel"), caller=caller)))
                except (LookupError, RuntimeError, ValueError) as exc:
                    errors.append({"id": str(selector), "error": str(exc)})
            return {"cancelled": cancelled, "errors": errors}
        return {"error": f"unknown hub operation: {op}", "op": op}

    task_tool = StructuredTool.from_function(coroutine=task_call, name="task", description="Spawn one concurrent batch. Give each self-contained task Target, Change, and Acceptance sections. Results arrive asynchronously through hub unless blocking; blocking is runtime-bounded.", args_schema=_TASK_SCHEMA)
    yield_tool = StructuredTool.from_function(coroutine=yield_call, name="yield", description="Submit incremental findings or a terminal schema-validated result. Validation retries up to three times; permissive overrides and strict failures then settle.", args_schema=_YIELD_SCHEMA)
    hub_tool = StructuredTool.from_function(coroutine=hub_call, name="hub", description="List, message, wait for, cancel, and collect serializable snapshots of registered agents.", args_schema=_HUB_SCHEMA)
    advise_tool = StructuredTool.from_function(coroutine=advise_call, name="advise", description="Submit one nonterminal advisor assessment: either a note with nit, concern, or blocker severity, or none=true.", args_schema=_ADVISE_SCHEMA)
    agent_type = getattr(host, "agent_type", None)
    is_advisor = not is_main and str(getattr(agent_type, "name", "")) == "advisor"
    may_spawn = is_main or (bool(getattr(agent_type, "spawns", False)) and int(getattr(host, "depth", 0)) < int(registry.cfg.agents.max_depth))
    return tuple([
        *([task_tool] if may_spawn else []),
        *([yield_tool] if not is_main else []),
        hub_tool,
        *([advise_tool] if is_advisor else []),
    ])


def register(api: PluginAPI) -> None:
    api.system_prompt_fragment(_AGENT_PROMPT, priority=70)


__all__ = ["agent_tools", "register"]
