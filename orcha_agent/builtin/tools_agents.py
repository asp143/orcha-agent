"""Context-bound tools for spawning and coordinating in-process agents."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import StructuredTool

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="tools-agents", version="1.0.0")

_AGENT_PROMPT = (
    "Use task for genuinely independent work and write each assignment with Target, "
    "Change, and Acceptance sections. Task results arrive asynchronously; use hub "
    "wait or jobs instead of polling. Spawned runs must finish with yield."
)

_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "context": {"type": "string"},
        "tasks": {
            "type": "array",
            "minItems": 1,
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


def _roster(registry: Any, run: Any) -> dict[str, Any]:
    age = max(0.0, (datetime.now(UTC) - run.created_at).total_seconds())
    return {
        "id": run.id, "name": run.name, "type": run.agent_type.name,
        "status": run.status, "age_s": round(age, 3), "last_tool": run.last_tool,
        "tokens": run.tokens_in + run.tokens_out, "cost": run.cost,
        "unread": registry.unread_count(run.id),
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


def _validate(value: Any, schema: Any, path: str = "$") -> str | None:
    if not isinstance(schema, dict):
        return f"{path}: schema must be an object"
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not choices or not all(isinstance(item, str) for item in choices):
            return f"{path}: schema type must be a string or list of strings"
        if not any(_matches(value, item) for item in choices):
            return f"{path}: expected {' or '.join(choices)}"
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{path}: value is not one of {enum!r}"
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list): return f"{path}: schema required must be an array"
        for key in required:
            if key not in value: return f"{path}: missing required property {key!r}"
        properties = schema.get("properties", {})
        if not isinstance(properties, dict): return f"{path}: schema properties must be an object"
        for key, child in properties.items():
            if key in value and (error := _validate(value[key], child, f"{path}.{key}")):
                return error
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            if error := _validate(item, schema["items"], f"{path}[{index}]"):
                return error
    return None


def agent_tools(host: Any) -> tuple[StructuredTool, ...]:
    """Return orchestration tools allowed for one main or child host."""
    registry = _registry(host)
    caller = str(getattr(host, "source_id", "main"))
    is_main = caller == "main"

    async def task_call(tasks: list[dict[str, Any]], context: str | None = None) -> dict[str, Any]:
        shared = context.strip() if context else ""

        async def spawn(item: dict[str, Any]) -> Any:
            prompt = "\n\n".join(part for part in (shared, str(item["task"]).strip()) if part)
            return await registry.spawn(
                str(item.get("agent") or "task"), prompt, name=item.get("name"), parent=caller,
                output_schema=item.get("output_schema"),
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
            await registry.wait_all((run.id for run in blocking), timeout_s=_timeout(registry))
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
            finding = data if error is None else {"data": data, "error": error}
            run.partial_findings.append(finding)
            await registry.record_yield(run, {"type": type, "data": finding})
            return {"accepted": True, "terminal": False, "findings": len(run.partial_findings)}
        payload, status = data, "done"
        if type == "error" or error is not None:
            payload, status = {"error": error or data or "agent reported an error"}, "failed"
        validation_error = _validate(data, run.output_schema) if status == "done" and run.output_schema is not None else None
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

    async def hub_call(**kwargs: Any) -> dict[str, Any]:
        op = str(kwargs["op"])
        if op == "list":
            return {"agents": [_roster(registry, run) for run in registry.list(status=kwargs.get("status"))]}
        if op == "send":
            target, message = kwargs.get("to"), kwargs.get("message")
            if not isinstance(target, str) or not isinstance(message, str):
                return {"error": "hub send requires string to and message"}
            try:
                target_id = registry.resolve(target)
                peer = registry.get(target_id) if target_id != "main" else None
                before = 0 if peer is None else peer.yield_count
                await registry.post_message(caller, target_id, message)
                if peer is not None:
                    await registry.send(peer.id, message, interrupt=bool(kwargs.get("interrupt", False)))
            except (LookupError, RuntimeError, ValueError) as exc:
                return {"error": str(exc), "op": op}
            response: dict[str, Any] = {"sent": {"to": target_id, "name": "main" if peer is None else peer.name, "status": "running" if peer is None else peer.status}}
            if not bool(kwargs.get("await", False)): return response
            await registry.wait_activity(caller, timeout_s=_timeout(registry), peer=target_id, after_yield=before)
            replies = registry.drain_messages(caller, sender=target_id)
            if replies: response["event"] = {"kind": "message", "messages": replies}
            elif peer is not None and peer.yield_count > before: response["event"] = {"kind": "yield", "from": peer.id, "payload": peer.last_yield}
            else: response["event"] = {"kind": "timeout"}
            return response
        if op == "inbox": return {"messages": registry.drain_messages(caller)}
        if op == "wait":
            ids, timeout_s = kwargs.get("ids"), float(kwargs.get("timeout_s", 300))
            await registry.wait_activity(caller, ids=ids, timeout_s=timeout_s)
            messages = registry.drain_messages(caller)
            settled = [run for run in registry.jobs(caller, ids=ids) if run.terminal and not run.delivered]
            if settled: await registry.deliver(caller, (run.id for run in settled))
            return {"messages": messages, "jobs": [_job(run) for run in settled], "timed_out": not messages and not settled}
        if op == "jobs":
            jobs = registry.jobs(caller)
            pending = [run for run in jobs if run.terminal and not run.delivered]
            if pending: await registry.deliver(caller, (run.id for run in pending))
            return {"jobs": [_job(run) for run in jobs]}
        if op == "cancel":
            ids = kwargs.get("ids")
            if not isinstance(ids, list) or not ids: return {"error": "hub cancel requires a non-empty ids list"}
            cancelled, errors = [], []
            for selector in ids:
                try:
                    run_id = registry.resolve(str(selector))
                    if run_id == "main": raise ValueError("main cannot be cancelled through hub")
                    cancelled.append(_summary(await registry.cancel(run_id, reason=str(kwargs.get("reason") or "cancel"))))
                except (LookupError, RuntimeError, ValueError) as exc:
                    errors.append({"id": str(selector), "error": str(exc)})
            return {"cancelled": cancelled, "errors": errors}
        return {"error": f"unknown hub operation: {op}", "op": op}

    task_tool = StructuredTool.from_function(coroutine=task_call, name="task", description="Spawn one concurrent batch. Give each self-contained task Target, Change, and Acceptance sections. Results arrive asynchronously through hub unless blocking; blocking is runtime-bounded.", args_schema=_TASK_SCHEMA)
    yield_tool = StructuredTool.from_function(coroutine=yield_call, name="yield", description="Submit incremental findings or a terminal schema-validated result. Validation retries up to three times; permissive overrides and strict failures then settle.", args_schema=_YIELD_SCHEMA)
    hub_tool = StructuredTool.from_function(coroutine=hub_call, name="hub", description="List, message, wait for, cancel, and collect serializable snapshots of registered agents.", args_schema=_HUB_SCHEMA)
    may_spawn = is_main or (bool(getattr(getattr(host, "agent_type", None), "spawns", False)) and int(getattr(host, "depth", 0)) < int(registry.cfg.agents.max_depth))
    return tuple([*( [task_tool] if may_spawn else []), *( [yield_tool] if not is_main else []), hub_tool])


def register(api: PluginAPI) -> None:
    api.system_prompt_fragment(_AGENT_PROMPT, priority=70)


__all__ = ["agent_tools", "register"]
