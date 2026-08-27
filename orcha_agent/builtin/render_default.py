"""Default Rich renderers for streamed model and tool events."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from typing import Any

from rich.panel import Panel
from rich.text import Text

from orcha_agent.core.events import ModelChunk, ToolCallEnd, ToolCallStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="render_default", version="1.0.0")
_FILE_TOOLS = frozenset({"edit_file", "write_file"})


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            content = value.get(key)
            if content is not None:
                return _text(content)
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(_text(part) for part in value)
    return str(value)


def _render_model_chunk(event: ModelChunk) -> Text | None:
    content = _text(getattr(event.chunk, "content", event.chunk))
    if not content:
        return None
    subagent = event.role == "subagent" or event.role.startswith("subagent:")
    prefix = f"[{event.model_name}] " if event.model_name else ""
    return Text(f"{prefix}{content}", style="dim" if subagent else "")


def _limited_arguments(args: Mapping[str, Any]) -> Text:
    try:
        rendered = json.dumps(args, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = repr(args)
    lines = rendered.splitlines()
    if len(lines) > 20:
        lines = [*lines[:19], f"… ({len(lines) - 19} more lines)"]
    return Text("\n".join(lines))


def _render_tool_start(event: ToolCallStart) -> Panel:
    return Panel(
        _limited_arguments(event.args),
        title=f"⚙ {event.name}",
        border_style="cyan",
    )


def _candidate_mappings(result: Any) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(result, Mapping):
        candidates.append(result)
    artifact = getattr(result, "artifact", None)
    if isinstance(artifact, Mapping):
        candidates.append(artifact)
    for candidate in tuple(candidates):
        data = candidate.get("data")
        if isinstance(data, Mapping):
            candidates.append(data)
    return candidates


def _unified_diff(result: Any) -> str | None:
    for data in _candidate_mappings(result):
        supplied = data.get("diff")
        if isinstance(supplied, str):
            return supplied
        before = data.get("before")
        after = data.get("after")
        if isinstance(before, str) and isinstance(after, str):
            path = str(data.get("file_path") or data.get("path") or "file")
            return "\n".join(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=path,
                    tofile=path,
                    lineterm="",
                )
            )

    content = getattr(result, "content", result)
    rendered = _text(content)
    if rendered.startswith("--- ") or "\n@@ " in rendered:
        return rendered
    return None


def _status(result: Any) -> str | None:
    status = getattr(result, "status", None)
    if isinstance(status, str):
        return status
    for data in _candidate_mappings(result):
        value = data.get("status")
        if isinstance(value, str):
            return value
    return None


def _error_text(result: Any) -> str | None:
    if _status(result) == "error":
        return _text(getattr(result, "content", result)) or "Tool failed"
    for data in _candidate_mappings(result):
        error = data.get("error")
        if error:
            return _text(error)
    content = _text(getattr(result, "content", result))
    if content.lstrip().lower().startswith("error:"):
        return content
    return None


def _execute_panel(event: ToolCallEnd) -> Panel:
    result = event.result
    content = _text(getattr(result, "content", result))
    exit_code: Any = getattr(result, "exit_code", None)
    for data in _candidate_mappings(result):
        stdout = _text(data.get("stdout"))
        stderr = _text(data.get("stderr"))
        if stdout or stderr:
            content = "\n".join(part for part in (stdout, stderr) if part)
        if data.get("exit_code") is not None:
            exit_code = data["exit_code"]
    if exit_code is not None:
        content = (
            f"{content}\n\nExit code: {exit_code}"
            if content
            else f"Exit code: {exit_code}"
        )
    return Panel(Text(content), title=f"⚙ {event.name}", border_style="cyan")


def _render_tool_end(event: ToolCallEnd) -> Panel | Text | None:
    error = _error_text(event.result)
    if error is not None:
        return Panel(Text(error), title=f"⚠ {event.name}", border_style="yellow")
    if (
        event.name in _FILE_TOOLS
        and (diff := _unified_diff(event.result)) is not None
    ):
        return Text(diff)
    if event.name == "execute":
        return _execute_panel(event)
    return None


def _is_error(event: object) -> bool:
    return isinstance(event, BaseException) or type(event).__name__ in {
        "Error",
        "StreamError",
    }


def _render_error(event: object) -> Panel:
    message = str(event)
    if not message:
        message = type(event).__name__
    return Panel(Text(message), title="Error", border_style="red")


async def _thinking_command(api: PluginAPI, ctx: Any, args: str) -> None:
    value = args.strip()
    modes = {"on": "summary", "off": "off"}
    if value not in modes:
        ctx.console.error("Usage: /thinking on|off")
        return

    mode = modes[value]
    api.state["thinking"] = mode
    ctx.plugin_states.setdefault("provider_anthropic", {})["thinking"] = mode
    ctx.persist_plugin_states()
    await ctx.rebuild()
    ctx.console.print(f"Thinking: {value}")


def register(api: PluginAPI) -> None:
    api.add_command(
        "thinking",
        lambda ctx, args: _thinking_command(api, ctx, args),
        help="Toggle thinking display: /thinking on|off",
    )
    api.add_renderer("ModelChunk", _render_model_chunk)
    api.add_renderer("ToolCallStart", _render_tool_start)
    api.add_renderer("ToolCallEnd", _render_tool_end)
    api.add_renderer(_is_error, _render_error)
