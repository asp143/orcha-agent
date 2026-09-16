"""Bounded declarative hook execution and real tool execution interception."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import signal
import sys
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from orcha_agent.core.config import HookConfig
from orcha_agent.core.events import Event, ToolCallAfter, ToolCallBefore

WRITE_TOOLS = frozenset({"write", "write_file", "edit", "edit_file", "delete", "apply_patch"})
_PYTHON_RUNNER = """import asyncio, importlib, inspect, json, sys
module, name = sys.argv[1].split(":", 1)
fn = getattr(importlib.import_module(module), name)
result = fn(json.load(sys.stdin))
if inspect.isawaitable(result): result = asyncio.run(result)
if result is not None: print(json.dumps(result))
"""


@dataclass(slots=True)
class HookResult:
    code: int
    output: str = ""
    error: str = ""


def matches(hook: HookConfig, payload: dict[str, Any]) -> bool:
    import regex

    if hook.matcher == "*":
        return True
    if hook.event.startswith("tool_call_") and not hook.matcher.startswith("regex:"):
        return fnmatch.fnmatchcase(str(payload.get("name", "")), hook.matcher)
    text = payload.get("text", json.dumps(payload, default=str))
    try:
        return (
            regex.search(hook.matcher.removeprefix("regex:"), str(text), timeout=0.05) is not None
        )
    except TimeoutError:
        return False


async def _exchange(process: asyncio.subprocess.Process, data: bytes) -> tuple[bytes, bytes]:
    async def read(stream: asyncio.StreamReader | None) -> bytes:
        assert stream is not None
        output = bytearray()
        while chunk := await stream.read(65536):
            output.extend(chunk)
            if len(output) > 1024 * 1024:
                raise ValueError("Hook output exceeded 1 MiB")
        return bytes(output)

    async def write() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(data)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    readers = [asyncio.create_task(read(process.stdout)), asyncio.create_task(read(process.stderr))]
    writer = asyncio.create_task(write())
    try:
        stdout, stderr = await asyncio.gather(*readers)
        await writer
        await process.wait()
        return stdout, stderr
    finally:
        for task in [*readers, writer]:
            if not task.done():
                task.cancel()
        pending: list[asyncio.Task[Any]] = [*readers, writer]
        await asyncio.gather(*pending, return_exceptions=True)


async def run_hook(hook: HookConfig, payload: dict[str, Any], cwd: Path) -> HookResult:
    options: dict[str, Any] = dict(
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )
    if hook.command is not None:
        process = await asyncio.create_subprocess_shell(hook.command, **options)
    else:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _PYTHON_RUNNER, str(hook.python), **options
        )
    try:
        stdout, stderr = await asyncio.wait_for(
            _exchange(process, json.dumps(payload, default=str).encode()),
            hook.timeout,
        )
    except (TimeoutError, ValueError, asyncio.CancelledError) as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return HookResult(
            1,
            error=(
                str(exc)
                if isinstance(exc, ValueError)
                else f"Hook timed out after {hook.timeout:g}s"
            ),
        )
    return HookResult(
        process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    )


class HooksMiddleware(AgentMiddleware):
    def __init__(self, emit: Callable[[Event], Awaitable[Any]]) -> None:
        self.emit = emit

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = request.tool_call
        before = ToolCallBefore(call["name"], dict(call["args"]), call["id"])
        await self.emit(before)
        if before.block_message:
            result = ToolMessage(
                content=before.block_message,
                tool_call_id=before.id,
                name=before.name,
                status="error",
            )
        else:
            updated = {**call, "args": before.args} if before.name in WRITE_TOOLS else call
            result = await handler(request.override(tool_call=updated))
        await self.emit(ToolCallAfter(before.name, before.id, result))
        return result
