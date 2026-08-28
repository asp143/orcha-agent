"""Tool card renderer and row-budget degradation."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from orcha_agent.tui.frame import Block, BlockState

from . import theme_symbol, theme_value
from .diff import render as render_diff
from .thinking import SPINNER_FRAMES

_PREVIEW_LINES = 20
_PREVIEW_COLUMNS = 4000
_FILE_TOOLS = frozenset({"edit_file", "write_file"})


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        found.append(value)
    artifact = getattr(value, "artifact", None)
    if isinstance(artifact, Mapping):
        found.append(artifact)
    for item in tuple(found):
        data = item.get("data")
        if isinstance(data, Mapping):
            found.append(data)
    return found


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("stdout", "stderr", "error", "content", "text", "output"):
            if key in value and value[key] is not None:
                if key == "stderr" and value.get("stdout"):
                    continue
                return _text(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    content = getattr(value, "content", None)
    return _text(content) if content is not None else str(value)


def _result_text(value: Any) -> str:
    parts: list[str] = []
    for item in _mappings(value):
        stdout = _text(item.get("stdout"))
        stderr = _text(item.get("stderr"))
        if stdout or stderr:
            parts = [part for part in (stdout, stderr) if part]
            break
    if not parts:
        content = _text(value)
        if content:
            parts.append(content)
    return "\n".join(parts)


def _exit_code(value: Any) -> Any:
    direct = getattr(value, "exit_code", None)
    if direct is not None:
        return direct
    for item in _mappings(value):
        if item.get("exit_code") is not None:
            return item["exit_code"]
    return None


def _error(value: Any) -> bool:
    if str(getattr(value, "status", "")).lower() == "error":
        return True
    exit_code = _exit_code(value)
    if exit_code is not None and exit_code != 0:
        return True
    for item in _mappings(value):
        if str(item.get("status", "")).lower() == "error" or item.get("error"):
            return True
    return _text(value).lstrip().lower().startswith("error:")


def _diff(value: Any) -> str | None:
    for item in _mappings(value):
        supplied = item.get("diff")
        if isinstance(supplied, str):
            return supplied
        before = item.get("before")
        after = item.get("after")
        if isinstance(before, str) and isinstance(after, str):
            path = str(item.get("path") or item.get("file_path") or "file")
            return "\n".join(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=path,
                    tofile=path,
                    lineterm="",
                )
            )
    content = _text(value)
    if content.startswith("--- ") or "\n@@ " in content or content.startswith("@@ "):
        return content
    return None


def _detail(args: Mapping[str, Any]) -> str:
    for key in ("path", "file_path", "filename"):
        value = args.get(key)
        if value:
            return str(value)
    command = args.get("command", args.get("cmd", ""))
    return str(command)[:40]


def _bounded(
    content: Text,
    *,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Text:
    logical_lines = list(content.split("\n", allow_blank=True))
    if not expanded:
        logical_lines = [
            line[:_PREVIEW_COLUMNS]
            for line in logical_lines[:_PREVIEW_LINES]
        ]
    content_width = max(1, width - 4)
    available_rows = max(1, budget_rows - 2)
    console = Console(
        width=content_width,
        height=available_rows,
        force_terminal=False,
        color_system=None,
    )
    visual_lines: list[Text] = []
    for line in logical_lines:
        wrapped = list(
            line.wrap(
                console,
                content_width,
                overflow="fold",
                no_wrap=False,
            )
        )
        visual_lines.extend(wrapped or [Text()])
        if len(visual_lines) >= available_rows:
            break
    return Text("\n").join(visual_lines[:available_rows])


def _group_content(calls: list[Any]) -> Text:
    output = Text()
    for index, call in enumerate(calls):
        if index:
            output.append("\n")
        args = call.get("args", {}) if isinstance(call, Mapping) else {}
        result = call.get("result") if isinstance(call, Mapping) else None
        output.append(f"{_detail(args)}", style="bold")
        content = _result_text(result)
        if content:
            output.append(f"\n{content}")
    return output


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel | Text | None:
    if budget_rows <= 0:
        return None
    name = str(block.data.get("name", "tool"))
    args = block.data.get("args", {})
    if not isinstance(args, Mapping):
        args = {}
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        name_label = f"{name} ×{len(calls)}"
        detail = ""
        pending = any(isinstance(call, Mapping) and call.get("result") is None for call in calls)
        error = any(isinstance(call, Mapping) and _error(call.get("result")) for call in calls if call.get("result") is not None)
    else:
        name_label = name
        detail = _detail(args)
        pending = block.state is BlockState.ACTIVE and "result" not in block.data
        error = not pending and _error(block.data.get("result"))
    execute_code = (
        _exit_code(block.data.get("result"))
        if name == "execute" and not pending
        else None
    )
    title = (
        f"{name_label}"
        f"{f' · {detail}' if detail else ''}"
        f"{f' · exit {execute_code}' if execute_code is not None else ''}"
    )
    elapsed = float(block.data.get("elapsed", 0.0))
    glyph = (
        "✘"
        if error
        else (
            SPINNER_FRAMES[
                int(block.data.get("spinner_frame", 0)) % len(SPINNER_FRAMES)
            ]
            if pending
            else "✔"
        )
    )

    if budget_rows == 1:
        return Text(
            f"{glyph} {title} · {elapsed:.1f}s",
            style=str(theme_value(theme, "toolTitle")),
        )
    if budget_rows == 2:
        corner = getattr(theme_symbol(theme, "boxRound", box.ROUNDED), "top_left", "╭")
        return Text(
            f"{corner}─ {title}",
            style=f"bold {theme_value(theme, 'toolTitle')}",
        )

    if isinstance(calls, list) and calls:
        body = _group_content(calls)
        has_output = any(
            isinstance(call, Mapping) and call.get("result") is not None
            for call in calls
        )
    else:
        result = block.data.get("result")
        diff = _diff(result) if name in _FILE_TOOLS and result is not None else None
        if diff is not None:
            diff_block = replace(block, kind="diff", data={"text": diff})
            body = render_diff(diff_block, theme, width, budget_rows, expanded)
            has_output = True
        else:
            content = _result_text(result) if result is not None else json.dumps(args, indent=2, ensure_ascii=False, default=str)
            body = Text(content, style=str(theme_value(theme, "toolOutput")))
            has_output = result is not None
    body = _bounded(
        body,
        width=width,
        budget_rows=budget_rows,
        expanded=expanded,
    )

    background = "toolPendingBg" if pending else ("toolErrorBg" if error else "toolSuccessBg")
    return Panel(
        body,
        title=Text(f"{glyph} {title}", style=f"bold {theme_value(theme, 'toolTitle')}"),
        title_align="left",
        border_style=str(theme_value(theme, "error" if error else "accent")),
        style=f"on {theme_value(theme, background)}",
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        subtitle=(
            Text("[Ctrl+O] expand", style="dim")
            if not expanded and has_output
            else None
        ),
        subtitle_align="right",
        padding=(0, 1),
    )
