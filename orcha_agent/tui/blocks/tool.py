"""oh-my-pi compatible tool cards and inline result renderers."""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from rich.cells import cell_len, set_cell_size
from rich.text import Text

from orcha_agent.tui.frame import Block, BlockState

from . import theme_spinner, theme_symbol, theme_value
from .diff import render as render_diff

SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
EXPAND_HINT = "⟦Ctrl+O: Expand⟧"
_READ = frozenset({"read", "read_file"})
_WRITE = frozenset({"write", "write_file"})
_EDIT = frozenset({"edit", "edit_file", "apply_patch"})
_BASH = frozenset({"execute", "bash", "shell"})
_INLINE = frozenset({"grep", "glob", "web_search"})
_MUTED_FRAME = frozenset({"edit", "edit_file", "apply_patch", "task", "ask", "todo"})


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, Mapping):
        for key in ("stdout", "content", "text", "output", "error", "stderr"):
            if key in value and value[key] is not None:
                return _text(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    content = getattr(value, "content", None)
    return _text(content) if content is not None else str(value)


def _result_text(value: Any) -> str:
    for item in _mappings(value):
        stdout = _text(item.get("stdout"))
        stderr = _text(item.get("stderr"))
        if stdout or stderr:
            return "\n".join(part for part in (stdout, stderr) if part)
    return _text(value)


def _value(value: Any, key: str, default: Any = None) -> Any:
    direct = getattr(value, key, None)
    if direct is not None:
        return direct
    for item in _mappings(value):
        if item.get(key) is not None:
            return item[key]
    return default


def _exit_code(value: Any) -> Any:
    return _value(value, "exit_code", _value(value, "returncode"))


def _state(block: Block) -> str:
    explicit = str(block.data.get("status", "")).casefold()
    result = block.data.get("result")
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        if any(isinstance(call, Mapping) and "result" not in call for call in calls):
            return "running"
        if any(isinstance(call, Mapping) and _exit_code(call.get("result")) not in (None, 0) for call in calls):
            return "error"
        return "done"
    statuses = {explicit, str(getattr(result, "status", "")).casefold()}
    statuses.update(str(item.get("status", "")).casefold() for item in _mappings(result))
    if statuses & {"aborted", "cancelled", "canceled"}:
        return "aborted"
    if "running" in statuses or "pending" in statuses:
        return "running"
    if "warning" in statuses:
        return "warning"
    if "error" in statuses or any(item.get("error") for item in _mappings(result)):
        return "error"
    code = _exit_code(result)
    content = _result_text(result).casefold()
    if code == 124 or "timed out" in content or _value(result, "timed_out", False):
        return "warning"
    if code not in (None, 0):
        return "error"
    if block.state is BlockState.ACTIVE and "result" not in block.data:
        return "running"
    return "done"


def _glyph(block: Block, state: str, theme: Any = None) -> str:
    if state == "running":
        return theme_spinner(theme, "spinner.activity", int(block.data.get("spinner_frame", 0)), SPINNER_FRAMES)
    defaults = {"done": "✔", "error": "✘", "warning": "⚠", "info": "ⓘ", "pending": "⏳", "aborted": "⏹"}
    keys = {"done": "status.success", "error": "status.error", "warning": "status.warning", "pending": "status.pending"}
    return str(theme_symbol(theme, keys.get(state, ""), defaults.get(state, "•")))


def _path(args: Mapping[str, Any]) -> str:
    for key in ("path", "file_path", "filename"):
        if args.get(key):
            return str(args[key])
    return ""


def _selection(args: Mapping[str, Any], line_count: int | None = None) -> str:
    start = args.get("offset", args.get("start", args.get("start_line")))
    limit = args.get("limit")
    if isinstance(start, int):
        end = start + ((line_count if line_count is not None else limit) or 1) - 1
        return f":{start}-{end}"
    if isinstance(args.get("line_start"), int):
        end = args.get("line_end", args["line_start"])
        return f":{args['line_start']}-{end}"
    return ""


def _detail(name: str, args: Mapping[str, Any]) -> str:
    if name in _BASH:
        return str(args.get("command", args.get("cmd", "")))[:40]
    return _path(args) or str(args.get("pattern", args.get("query", "")))[:40]


def _label(name: str) -> str:
    return {
        "execute": "Bash", "bash": "Bash", "shell": "Bash",
        "read": "Read", "read_file": "Read", "write": "Write", "write_file": "Write",
        "edit": "Edit", "edit_file": "Edit", "apply_patch": "Edit",
        "web_search": "Web Search", "grep": "Grep", "glob": "Glob",
        "task": "Task", "todo": "Todo", "ask": "Ask",
    }.get(name, name.replace("_", " ").title())


def _append_line(target: Text, line: str | Text) -> None:
    if target:
        target.append("\n")
    target.append(line if isinstance(line, Text) else Text(line))


def _box_char(theme: Any, name: str, default: str, attr: str) -> str:
    supplied = theme_symbol(theme, f"boxRound.{name}", None)
    if supplied is not None:
        return str(supplied)
    box_value = theme_symbol(theme, "boxRound", None)
    return str(getattr(box_value, attr, default))


def _frame(
    header: str,
    rows: list[str | Text],
    *,
    width: int,
    budget_rows: int,
    theme: Any,
    border_token: str,
    sections: Mapping[int, str] | None = None,
    edit: bool = False,
) -> Text:
    width = max(4, width)
    border = str(theme_value(theme, border_token, theme_value(theme, "muted")))
    tl = _box_char(theme, "topLeft", "╭", "top_left")
    tr = _box_char(theme, "topRight", "╮", "top_right")
    bl = _box_char(theme, "bottomLeft", "╰", "bottom_left")
    br = _box_char(theme, "bottomRight", "╯", "bottom_right")
    h = _box_char(theme, "horizontal", "─", "top")
    v = _box_char(theme, "vertical", "│", "mid_left")
    tee_right = _box_char(theme, "teeRight", "├", "row_left")
    tee_left = _box_char(theme, "teeLeft", "┤", "row_right")
    output = Text()
    header_text = f" {header} " if header else ""
    top_fill = max(0, width - 6 - cell_len(header_text))
    _append_line(output, Text(f"{tl}{h * 3}{header_text}{h * top_fill}{h}{tr}", style=border))
    capacity = max(0, budget_rows - 2)
    for index, row in enumerate(rows[:capacity]):
        if sections and index in sections:
            label = f" {sections[index]} "
            fill = max(0, width - 6 - cell_len(label))
            _append_line(output, Text(f"{tee_right}{h * 3}{label}{h * fill}{h}{tee_left}", style=border))
            continue
        content_width = width - (2 if edit else 4)
        value = row.copy() if isinstance(row, Text) else Text(str(row))
        value.truncate(content_width, overflow="ellipsis", pad=True)
        framed = Text(v, style=border)
        if not edit:
            framed.append(" ")
        framed.append(value)
        if not edit:
            framed.append(" ")
        framed.append(v, style=border)
        _append_line(output, framed)
    _append_line(output, Text(f"{bl}{h * (width - 2)}{br}", style=border))
    return output


def _border_token(name: str, state: str) -> str:
    if name in _MUTED_FRAME:
        return "borderMuted"
    if state == "error":
        return "error"
    if state == "warning":
        return "warning"
    if state in {"running", "pending"}:
        return "accent"
    return "dim"


def _limited(lines: list[str], maximum: int, expanded: bool) -> list[str]:
    if expanded or len(lines) <= maximum:
        return lines
    return [*lines[:maximum], f"… {len(lines) - maximum} more lines {EXPAND_HINT}"]


def _diff(value: Any) -> str | None:
    for item in _mappings(value):
        supplied = item.get("diff")
        if isinstance(supplied, str):
            return supplied
        before, after = item.get("before"), item.get("after")
        if isinstance(before, str) and isinstance(after, str):
            path = str(item.get("path") or item.get("file_path") or "file")
            return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=path, tofile=path, lineterm=""))
    text = _text(value)
    return text if text.startswith(("@@ ", "--- ")) or "\n@@ " in text else None


def _read_rows(block: Block, args: Mapping[str, Any], *, expanded: bool) -> tuple[str, list[str | Text]]:
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        rows: list[str | Text] = []
        for index, call in enumerate(calls):
            call_args = call.get("args", {}) if isinstance(call, Mapping) else {}
            result = call.get("result") if isinstance(call, Mapping) else None
            content = _result_text(result).splitlines()
            branch = "└─" if index == len(calls) - 1 else "├─"
            rows.append(f"{branch} {_path(call_args)}{_selection(call_args, len(content))}")
            preview = content if expanded else content[:3]
            rows.extend(f"   {line[:4000]}" for line in preview)
        return f"• Read ({len(calls)})", rows
    content = _result_text(block.data.get("result")).splitlines()
    path = _path(args)
    if _state(block) == "running":
        return f"⏳ Read: {path}{_selection(args)}", []
    start = args.get("offset", 1)
    start = start if isinstance(start, int) else 1
    digits = max(2, len(str(start + max(0, len(content) - 1))))
    rows = [f"{start + index:>{digits}}│{line[:4000]}" for index, line in enumerate(content)]
    return f"• Read {path}{_selection(args, len(content))}", _limited(rows, 12, expanded)


def _write_rows(block: Block, args: Mapping[str, Any]) -> tuple[str, list[str]]:
    lines = str(args.get("content", "")).splitlines()
    path = _path(args)
    if _state(block) == "running":
        rows = lines[-12:]
        if len(lines) > 12:
            rows.insert(0, "… (content above)")
        rows.append(f"{SPINNER_FRAMES[int(block.data.get('spinner_frame', 0)) % 8]} (streaming)")
        return f"Write: {path}", rows
    return f"✎ Write: {path} ({len(lines)} lines)", lines[:6]


def _edit_rows(block: Block, args: Mapping[str, Any], theme: Any, width: int, expanded: bool) -> tuple[str, list[Text]]:
    diff = _diff(block.data.get("result")) or _diff(args) or ""
    first = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)", diff)
    line = f":{first.group(1)}" if first else ""
    added = sum(1 for value in diff.splitlines() if value.startswith("+") and not value.startswith("+++"))
    removed = sum(1 for value in diff.splitlines() if value.startswith("-") and not value.startswith("---"))
    header = f"{_glyph(block, _state(block), theme)} Edit: {_path(args)}{line} ⟦+{added}/-{removed}⟧"
    rendered = render_diff(replace(block, kind="diff", data={"text": diff}), theme, width, 10_000, expanded)
    rows = list(rendered.split("\n", allow_blank=True))
    if block.state is BlockState.ACTIVE:
        hidden = max(0, len(rows) - 12)
        rows = rows[-12:]
        if hidden:
            rows.insert(0, Text("… (content above)", style="dim"))
        rows.append(Text(f"{SPINNER_FRAMES[int(block.data.get('spinner_frame', 0)) % 8]} (preview)", style="dim"))
    return header, rows


def _bash_rows(block: Block, args: Mapping[str, Any], expanded: bool) -> tuple[list[str | Text], dict[int, str]]:
    command_lines = str(args.get("command", args.get("cmd", ""))).splitlines() or [""]
    if len(command_lines) > 6 and not expanded:
        command_lines = [f"… {len(command_lines) - 6} earlier lines {EXPAND_HINT}", *command_lines[-6:]]
    output = _result_text(block.data.get("result")).splitlines()
    if len(output) > 10 and not expanded:
        total = len(output)
        output = [f"… ({total - 10} earlier lines, showing 10 of {total}) (ctrl+o to expand)", *output[-10:]]
    rows: list[str | Text] = [*(f"$ {line}" for line in command_lines)]
    section_index = len(rows)
    rows.append("")
    rows.extend(Text.from_ansi(line) for line in output)
    result = block.data.get("result")
    wall = _value(result, "wall_time", block.data.get("elapsed", 0.0))
    footer = f"⟦Wall: {float(wall):.1f}s | Exit: {_exit_code(result) if _exit_code(result) is not None else '—'}"
    if args.get("timeout") is not None:
        footer += f" | Timeout: {args['timeout']}s"
    rows.append(f"{footer}⟧")
    return rows, {section_index: "Output"}


def _items_from_result(result: Any, *keys: str) -> list[Any]:
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return list(result)
    for item in _mappings(result):
        for key in keys:
            value = item.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
    return _result_text(result).splitlines()


def _inline_rows(block: Block, name: str, args: Mapping[str, Any], expanded: bool, theme: Any) -> Text:
    state = _state(block)
    result = block.data.get("result")
    if state == "error":
        return Text(f"✘ Error: {_result_text(result)[:110]}", style=str(theme_value(theme, "error")))
    items = _items_from_result(result, "matches", "items", "results")
    if not items:
        return Text("⚠ No matches found", style=str(theme_value(theme, "warning")))
    limit = 24 if expanded else (8 if name == "glob" else 6)
    pattern = str(args.get("pattern", args.get("query", "")))
    if name == "grep":
        files = {str(item).split(":", 1)[0] for item in items}
        header = f"🔍 Grep: {pattern}  {len(items)} matches · {len(files)} files · in {args.get('path', args.get('cwd', '.'))}"
    elif name == "glob":
        header = f"🔍 Glob: {pattern}  {len(items)} items"
    else:
        header = f"🔍 Web Search: {pattern}  {len(items)} results"
    visible = items[:limit]
    rows = [header]
    for index, item in enumerate(visible):
        branch = "└─" if index == len(visible) - 1 and len(visible) == len(items) else "├─"
        rows.append(f"  {branch} {str(item)}")
    if len(items) > limit:
        rows.append(f"  … {len(items) - limit} more {EXPAND_HINT}")
    return Text("\n".join(rows))


def _task_rows(result: Any, theme: Any) -> tuple[str, list[str]]:
    agents = _items_from_result(result, "agents", "tasks")
    rows: list[str] = []
    succeeded = failed = requests = 0
    for agent in agents[-4:]:
        item = agent if isinstance(agent, Mapping) else {"description": str(agent)}
        status = str(item.get("status", "running")).casefold()
        glyph = "✔" if status in {"success", "succeeded", "done"} else ("✘" if status in {"error", "failed"} else "⣾")
        succeeded += glyph == "✔"
        failed += glyph == "✘"
        requests += int(item.get("requests", item.get("req", 0)) or 0)
        metrics = []
        if item.get("tokens") is not None: metrics.append(str(item["tokens"]))
        if item.get("requests") is not None: metrics.append(f"{item['requests']} req")
        if item.get("cost") is not None: metrics.append(f"${float(item['cost']):.2f}")
        rows.append(f"{glyph} {item.get('id', item.get('name', 'agent'))}: {item.get('description', item.get('task', ''))} ⟦{status}⟧ {'/'.join(metrics)} · {item.get('elapsed', 0)}s")
        if item.get("tool"):
            rows.append(f"└ {item['tool']}: {str(item.get('args', ''))[:40]}")
    elapsed = 0
    for item in _mappings(result):
        requests = int(item.get("requests", requests) or requests)
        elapsed = item.get("elapsed", 0)
        break
    rows.append(f"⟦{succeeded} succeeded · {failed} failed · {requests} req · {elapsed}s⟧")
    separator = theme_symbol(theme, "sep.thin", "·")
    task_glyph = "Task" if str(separator).isascii() else "⇶ Task"
    return f"{task_glyph} {separator} {len(agents)} agents", rows


def _todo_rows(args: Mapping[str, Any], result: Any, theme: Any) -> tuple[str, list[Text]]:
    items = _items_from_result(args.get("items", result), "items", "todos", "tasks")
    rows: list[Text] = []
    for item in items:
        value = item if isinstance(item, Mapping) else {"text": str(item)}
        label = str(value.get("text", value.get("content", value.get("title", ""))))
        done = bool(value.get("done") or value.get("status") in {"done", "completed"})
        glyph = theme_symbol(
            theme,
            "status.success" if done else "status.pending",
            "☑" if done else "☐",
        )
        rows.append(Text(f"{glyph} {label}", style=f"{theme_value(theme, 'success')} strike" if done else str(theme_value(theme, "accent"))))
    header_glyph = theme_symbol(theme, "status.success", "☑")
    separator = theme_symbol(theme, "sep.thin", "·")
    return f"{header_glyph} Todo {separator} {len(items)} tasks", rows


def _generic_rows(args: Mapping[str, Any], result: Any, expanded: bool) -> list[str]:
    arg_text = " ".join(f"{key}={value}" for key, value in args.items())
    output = _result_text(result).splitlines()
    return [f"└─ {arg_text}" if arg_text else "└─", *_limited(output, 12 if expanded else 4, expanded)]


def _render_impl(block: Block, theme: Any, width: int, budget_rows: int, expanded: bool) -> Text | None:
    if budget_rows <= 0:
        return None
    name = str(block.data.get("name", "tool"))
    args = block.data.get("args", {})
    if not isinstance(args, Mapping):
        args = {}
    state = _state(block)
    label, detail = _label(name), _detail(name, args)
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        label = f"{label} ({len(calls)})"
    if name == "task":
        agents = _items_from_result(block.data.get("result", {}), "agents", "tasks")
        detail = f"{len(agents)} agents"
    separator = theme_symbol(theme, "sep.thin", "·")
    compact = f"{label}{f' {separator} {detail}' if detail else ''} {separator} {float(block.data.get('elapsed', 0.0)):.1f}s"
    if budget_rows == 1:
        return Text(f"{_glyph(block, state, theme)} {compact}", style=str(theme_value(theme, "toolTitle")))
    if budget_rows == 2:
        top_left = _box_char(theme, "topLeft", "╭", "top_left")
        bottom_left = _box_char(theme, "bottomLeft", "╰", "bottom_left")
        horizontal = _box_char(theme, "horizontal", "─", "top")
        return Text(f"{top_left}{horizontal} {compact}\n{bottom_left}", style=str(theme_value(theme, _border_token(name, state))))
    if name in _INLINE:
        return _inline_rows(block, name, args, expanded, theme)
    if state == "error" and name not in _BASH:
        header = f"✘ {label}{f' {detail}' if detail else ''}"
        return _frame(
            header,
            _result_text(block.data.get("result")).splitlines() or ["Unknown error"],
            width=width,
            budget_rows=budget_rows,
            theme=theme,
            border_token="error",
        )
    sections: dict[int, str] | None = None
    edit = False
    if name in _READ:
        header, rows = _read_rows(block, args, expanded=expanded)
    elif name in _WRITE:
        header, rows = _write_rows(block, args)
    elif name in _EDIT:
        header, rows = _edit_rows(block, args, theme, width, expanded); edit = True
    elif name in _BASH:
        header = ""; rows, sections = _bash_rows(block, args, expanded)
    elif name == "task":
        header, rows = _task_rows(block.data.get("result", block.data.get("progress", {})), theme)
    elif name == "todo":
        header, rows = _todo_rows(args, block.data.get("result"), theme)
    else:
        header = f"{_glyph(block, state, theme)} {label}{f': {detail}' if detail else ''}"
        rows = _generic_rows(args, block.data.get("result"), expanded)
    return _frame(header, rows, width=width, budget_rows=budget_rows, theme=theme, border_token=_border_token(name, state), sections=sections, edit=edit)


def render(block: Block, theme: Any, width: int, budget_rows: int, expanded: bool) -> Text | None:
    try:
        return _render_impl(block, theme, width, budget_rows, expanded)
    except Exception:
        name = _label(str(block.data.get("name", "tool")))
        args = block.data.get("args", {})
        path = _path(args) if isinstance(args, Mapping) else ""
        raw = _result_text(block.data.get("result")) or _text(block.data)
        return Text(f"✘ {name}{f' {path}' if path else ''}\n{raw}", style=str(theme_value(theme, "error")))


__all__ = ["EXPAND_HINT", "SPINNER_FRAMES", "render"]
