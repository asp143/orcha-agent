"""Default Rich renderers for streamed model and tool events."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from orcha_agent.core.events import ModelChunk, ToolCallEnd, ToolCallStart, TurnStart
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


def _is_subagent(role: str) -> bool:
    return role == "subagent" or role.startswith("subagent:")


def _thinking_text(block: Mapping[str, Any]) -> str:
    if block.get("type") == "reasoning":
        return _text(block.get("summary"))
    if block.get("type") == "thinking":
        return _text(block.get("thinking"))
    return ""


class _StreamRenderer:
    def __init__(self, api: PluginAPI) -> None:
        self._api = api
        self._configured_thinking = str(api.config.get("thinking", "summary"))
        self._icons = bool(api.config.get("icons", True))
        self._seen_blocks: set[tuple[str, object, str, object]] = set()
        self._pending_model_prefix: dict[str, str] = {}
        self._needs_answer_gap: set[str] = set()
        self._output_open = False
        self._last_source_id: str | None = None

    def _thinking_mode(self) -> str:
        return str(self._api.state.get("thinking", self._configured_thinking))

    def _show_thinking(self, role: str) -> bool:
        mode = self._thinking_mode()
        return mode != "off" and (not _is_subagent(role) or mode == "all")

    @staticmethod
    def _source_id(event: ModelChunk) -> str:
        return str(event.source_id or event.role)

    @classmethod
    def _block_key(
        cls,
        event: ModelChunk,
        block: Mapping[str, Any],
    ) -> tuple[str, object, str, object]:
        chunk_id = getattr(event.chunk, "id", None)
        index = block.get("index", block.get("id"))
        return (cls._source_id(event), chunk_id, str(block.get("type")), index)

    async def reset(self, _event: TurnStart) -> None:
        self._seen_blocks.clear()
        self._pending_model_prefix.clear()
        self._needs_answer_gap.clear()
        self._output_open = False
        self._last_source_id = None

    def model_chunk(self, event: ModelChunk) -> Text | None:
        source_id = self._source_id(event)
        if event.model_name:
            self._pending_model_prefix[source_id] = f"[{event.model_name}] "

        value = getattr(event.chunk, "content", event.chunk)
        parts = value if isinstance(value, (list, tuple)) else (value,)
        rendered = Text()
        for part in parts:
            if isinstance(part, Mapping) and part.get("type") in {"reasoning", "thinking"}:
                content = _thinking_text(part)
                if not content or not self._show_thinking(event.role):
                    continue
                key = self._block_key(event, part)
                if key not in self._seen_blocks:
                    if self._output_open:
                        rendered.append("\n", style="dim italic")
                    prefix = self._pending_model_prefix.pop(source_id, "")
                    header = "󰟶 thinking" if self._icons else "[thinking]"
                    rendered.append(f"{prefix}{header}\n", style="dim italic")
                    self._seen_blocks.add(key)
                rendered.append(content, style="dim italic")
                self._needs_answer_gap.add(source_id)
                self._output_open = not content.endswith("\n")
                self._last_source_id = source_id
                continue

            content = _text(part)
            if not content:
                continue
            if source_id in self._needs_answer_gap:
                rendered.append("\n\n" if self._output_open else "\n")
                self._needs_answer_gap.remove(source_id)
            elif self._output_open and self._last_source_id != source_id:
                rendered.append("\n")
            rendered.append(self._pending_model_prefix.pop(source_id, ""))
            rendered.append(content, style="dim" if _is_subagent(event.role) else "")
            self._output_open = not content.endswith("\n")
            self._last_source_id = source_id

        return rendered if rendered.plain else None

    def tool_start(self, event: ToolCallStart) -> Panel | Group:
        panel = _render_tool_start(event)
        source_id = event.source_id or "main"
        if source_id in self._needs_answer_gap:
            separator = "\n\n" if self._output_open else "\n"
            self._needs_answer_gap.remove(source_id)
        elif self._output_open:
            separator = "\n"
        else:
            return panel
        self._output_open = False
        self._last_source_id = source_id
        return Group(Text(separator[:-1]), panel)


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
    stream = _StreamRenderer(api)
    api.add_command(
        "thinking",
        lambda ctx, args: _thinking_command(api, ctx, args),
        help="Toggle thinking display: /thinking on|off",
    )
    api.on(TurnStart, stream.reset)
    api.add_renderer("ModelChunk", stream.model_chunk)
    api.add_renderer("ToolCallStart", stream.tool_start)
    api.add_renderer("ToolCallEnd", _render_tool_end)
    api.add_renderer(_is_error, _render_error)
