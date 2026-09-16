"""Bounded declarative hook execution and real tool execution interception."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import ToolMessage

from orcha_agent.core.config import HookConfig
from orcha_agent.core.events import Compaction, Event, ToolCallAfter, ToolCallBefore
from orcha_agent.core.tools.shell import OUTPUT_BYTES, session_environment

WRITE_TOOLS = frozenset({"write", "write_file", "edit", "edit_file", "delete", "apply_patch"})
_PYTHON_RUNNER = """import asyncio, importlib, inspect, json, sys
sys.path.insert(0, sys.argv[2])
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


def hook_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bound tool output independently of the command's original representation."""
    if "result" not in payload:
        return payload
    result = payload["result"]
    content = result.content if isinstance(result, ToolMessage) else result
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {
        **payload,
        "result": text.encode("utf-8")[:OUTPUT_BYTES].decode("utf-8", errors="ignore"),
    }


async def run_hook(
    hook: HookConfig,
    payload: dict[str, Any],
    cwd: Path,
    *,
    user_config_dir: Path | None = None,
    shell_env_passthrough: Sequence[str] = (),
    provider_env_keys: Sequence[str] = (),
) -> HookResult:
    config_dir = user_config_dir or Path.home() / ".config" / "orcha-agent"
    environment = session_environment((*shell_env_passthrough, *hook.env_passthrough))
    denied = {name.upper() for name in provider_env_keys}
    environment = {
        key: value
        for key, value in environment.items()
        if key.upper() not in denied
        and not re.search(r"(?:_API_KEY$|_TOKEN$|SECRET|PASSWORD)", key, re.I)
    }
    options: dict[str, Any] = dict(
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=config_dir if hook.scope == "user" else cwd,
        env=environment,
        start_new_session=True,
    )
    if hook.command is not None:
        process = await asyncio.create_subprocess_shell(hook.command, **options)
    else:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _PYTHON_RUNNER,
            str(hook.python),
            str(config_dir / "hooks"),
            **options,
        )
    try:
        stdout, stderr = await asyncio.wait_for(
            _exchange(process, json.dumps(hook_payload(payload), default=str).encode()),
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


class HooksState(AgentState):
    _orcha_hook_summary_before: Annotated[NotRequired[str], PrivateStateAttr]


def _summary_signature(event: Any) -> str:
    if not isinstance(event, dict):
        return ""
    message = event.get("summary_message")
    text = getattr(message, "text", "")
    return hashlib.sha256(
        json.dumps([event.get("cutoff_index"), event.get("file_path"), text], default=str).encode()
    ).hexdigest()


class HooksMiddleware(AgentMiddleware):
    state_schema = HooksState

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        # Baseline is graph state, so parallel threads/subgraphs cannot share it.
        # Establish it on every model call, including resumed checkpoints.
        return {"_orcha_hook_summary_before": _summary_signature(state.get("_summarization_event"))}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        event = state.get("_summarization_event")
        signature = _summary_signature(event)
        if not signature or signature == state.get("_orcha_hook_summary_before", ""):
            return None
        message = event.get("summary_message")
        thread_id = getattr(getattr(runtime, "execution_info", None), "thread_id", None)
        if thread_id is None:
            from langgraph.config import get_config

            try:
                thread_id = get_config().get("configurable", {}).get("thread_id", "")
            except RuntimeError:
                thread_id = ""
        await self.emit(Compaction(str(thread_id), str(getattr(message, "text", ""))))
        return {"_orcha_hook_summary_before": signature}

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
