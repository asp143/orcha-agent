"""oh-my-pi compatible tool cards and inline result renderers."""

from __future__ import annotations

import ast
import difflib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.cells import cell_len, set_cell_size, split_graphemes
from rich.console import Group
from rich.text import Text
from rich.style import Style

from orcha_agent.tui.frame import Block, BlockState

from . import theme_spinner, theme_symbol, theme_value, with_leading_spacer
from .diff import render as render_diff
from .syntax import highlight, language_from_path
from .terminal import terminal_rows
from .image import has_image, render as render_image
from .limits import BASH_LINES, GROUP_READ_LINES, LIST_LINES, READ_LINES, SEARCH_LINES

SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
EXPAND_HINT = "⟦Ctrl+O: Expand⟧"
_READ = frozenset({"read", "read_file"})
_WRITE = frozenset({"write", "write_file"})
_EDIT = frozenset({"edit", "edit_file", "apply_patch"})
_BASH = frozenset({"execute", "bash", "shell"})
_INLINE = frozenset({"grep", "glob", "ls", "web_search"})
_MUTED_FRAME = frozenset({"edit", "edit_file", "apply_patch", "ask", "todo"})


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
        if any(
            isinstance(call, Mapping)
            and (
                _exit_code(call.get("result")) not in (None, 0)
                or getattr(call.get("result"), "status", "") == "error"
            )
            for call in calls
        ):
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
    if block.state is BlockState.ACTIVE and (
        "result" not in block.data or (str(block.data.get("name")) in _BASH and code is None)
    ):
        return "running"
    return "done"


def _glyph(block: Block, state: str, theme: Any = None) -> str:
    if state == "running":
        return theme_spinner(
            theme, "spinner.activity", int(block.data.get("spinner_frame", 0)), SPINNER_FRAMES
        )
    defaults = {
        "done": "✔",
        "error": "✘",
        "warning": "⚠",
        "info": "ⓘ",
        "pending": "⏳",
        "aborted": "⏹",
    }
    keys = {
        "done": "status.success",
        "error": "status.error",
        "warning": "status.warning",
        "pending": "status.pending",
    }
    return str(theme_symbol(theme, keys.get(state, ""), defaults.get(state, "•")))


def _one_line(value: Any) -> str:
    return re.sub(r"[\r\n]+", " ", str(value)).strip()


def _path(args: Mapping[str, Any], cwd: Any = None) -> str:
    if not any(args.get(key) for key in ("path", "file_path", "filename")):
        patch = args.get("patch")
        if isinstance(patch, str) and (header := re.match(r"\[([^\n]+)#[0-9a-fA-F]+\]", patch)):
            return _path({"path": header[1]}, cwd)
    for key in ("path", "file_path", "filename"):
        if not args.get(key):
            continue
        value = re.sub(r":\d+(?:-\d+)?$", "", _one_line(args[key]))
        if not value or "://" in value:
            return value
        candidate = Path(value)
        if candidate.is_absolute() and cwd:
            try:
                relative = candidate.relative_to(Path(str(cwd)))
            except ValueError:
                pass
            else:
                return relative.as_posix() or "."
        if candidate.is_absolute():
            try:
                relative = candidate.relative_to(Path.home())
            except ValueError:
                pass
            else:
                return f"~/{relative.as_posix()}" if relative.parts else "~"
        return value
    return ""


def _selection(args: Mapping[str, Any], line_count: int | None = None) -> str:
    selector = re.search(r":(\d+)(?:-(\d+))?$", str(args.get("path", args.get("file_path", ""))))
    if selector:
        return selector.group(0)
    offset = args.get("offset")
    if isinstance(offset, int):
        start = max(0, offset) + 1
    else:
        start = args.get("start", args.get("start_line"))
    limit = args.get("limit")
    if isinstance(start, int):
        end = start + ((line_count if line_count is not None else limit) or 1) - 1
        return f":{start}-{end}"
    if isinstance(args.get("line_start"), int):
        end = args.get("line_end", args["line_start"])
        return f":{args['line_start']}-{end}"
    return ""


def _detail(name: str, args: Mapping[str, Any], cwd: Any = None) -> str:
    if name in _BASH:
        return _one_line(args.get("command", args.get("cmd", "")))[:40]
    return _path(args, cwd) or _one_line(args.get("pattern", args.get("query", "")))[:40]


def _label(name: str) -> str:
    return {
        "execute": "Bash",
        "bash": "Bash",
        "shell": "Bash",
        "read": "Read",
        "read_file": "Read",
        "write": "Write",
        "write_file": "Write",
        "edit": "Edit",
        "edit_file": "Edit",
        "apply_patch": "Edit",
        "web_search": "Web Search",
        "grep": "Grep",
        "glob": "Glob",
        "task": "Task",
        "todo": "Todo",
        "ask": "Ask",
    }.get(name, name.replace("_", " ").title())


def _append_line(target: Text, line: str | Text) -> None:
    if target:
        target.append("\n")
    target.append(line if isinstance(line, Text) else Text(line))


def _cell_suffix(value: str, width: int) -> str:
    spans, _ = split_graphemes(value)
    start = len(value)
    used = 0
    for span_start, _span_end, size in reversed(spans):
        if used + size > width:
            break
        start = span_start
        used += size
    return value[start:]


def _middle_ellipsis(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width == 1:
        return "…"
    available = width - 1
    prefix_width = max(1, available // 3)
    suffix_width = available - prefix_width
    return f"{set_cell_size(value, prefix_width)}…{_cell_suffix(value, suffix_width)}"


def _box_char(theme: Any, name: str, default: str, attr: str) -> str:
    supplied = theme_symbol(theme, f"boxRound.{name}", None)
    if supplied is not None:
        return str(supplied)
    box_value = theme_symbol(theme, "boxRound", None)
    return str(getattr(box_value, attr, default))


def _format_seconds(value: float) -> str:
    rounded = round(value, 1)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"


def _timing_label(block: Block, state: str) -> str:
    key = "elapsed" if state == "running" else "duration"
    supplied = block.data.get(key)
    if not isinstance(supplied, (int, float)) or isinstance(supplied, bool):
        return ""
    value = float(supplied)
    if not math.isfinite(value) or value < 0:
        return ""
    prefix = "Elapsed" if state == "running" else "Took"
    return f"{prefix} {_format_seconds(value)}s"


def _header_with_timing(header: str | Text, block: Block, state: str, theme: Any) -> Text:
    value = header.copy() if isinstance(header, Text) else Text(_one_line(header))
    timing = _timing_label(block, state)
    if timing:
        muted = str(theme_value(theme, "dim", theme_value(theme, "muted")))
        value.append(f" {theme_symbol(theme, 'sep.thin', '·')} ", style=muted)
        value.append(timing, style=muted)
    return value


def _card_background_token(state: str) -> str:
    if state == "error":
        return "toolErrorBg"
    if state in {"running", "pending", "warning"}:
        return "toolPendingBg"
    return "toolSuccessBg"


def _frame(
    header: str | Text,
    rows: list[str | Text],
    *,
    width: int,
    budget_rows: int,
    theme: Any,
    border_token: str,
    state: str,
    sections: Mapping[int, str] | None = None,
    edit: bool = False,
    expanded: bool = False,
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
    header_value = header.copy() if isinstance(header, Text) else Text(_one_line(header))
    if "\n" in header_value.plain or "\r" in header_value.plain:
        header_value = Text(_one_line(header_value.plain))
    max_header_width = max(0, width - 8)
    if header_value.cell_len > max_header_width:
        header_value = Text(_middle_ellipsis(header_value.plain, max_header_width))
    header_width = header_value.cell_len
    header_text_width = header_width + 2 if header_width else 0
    top_fill = max(0, width - 6 - header_text_width)
    top = Text(f"{tl}{h * 3}", style=border)
    if header_width:
        top.append(" ")
        top.append(header_value)
        top.append(" ")
    top.append(f"{h * top_fill}{h}{tr}", style=border)
    _append_line(output, top)
    capacity = max(0, budget_rows - 2)
    for index, row in enumerate(rows[:capacity]):
        if sections and index in sections:
            section = _middle_ellipsis(_one_line(sections[index]), max(0, width - 8))
            label = f" {section} " if section else ""
            fill = max(0, width - 6 - cell_len(label))
            _append_line(
                output, Text(f"{tee_right}{h * 3}{label}{h * fill}{h}{tee_left}", style=border)
            )
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
    footer = Text(bl, style=border)
    hint = ""
    if expanded or len(rows) > max(0, budget_rows - 2):
        action = "Collapse" if expanded else "Expand"
        hint = (
            f"[Ctrl+O: {action}]"
            if theme_symbol(theme, "preset", "") == "ascii"
            else f"⟦Ctrl+O: {action}⟧"
        )
    if hint and cell_len(hint) + 4 <= width:
        footer.append(h * (width - cell_len(hint) - 4), style=border)
        footer.append(f" {hint} ", style=f"dim {theme_value(theme, 'dim')}")
        footer.append(br, style=border)
    else:
        footer = Text(f"{bl}{h * (width - 2)}{br}", style=border)
    _append_line(output, footer)
    background = theme_value(theme, _card_background_token(state), None)
    if background not in (None, "", "default"):
        output.stylize_before(f"on {background}")
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
        before = item.get("before", item.get("old_string"))
        after = item.get("after", item.get("new_string"))
        if isinstance(before, str) and isinstance(after, str):
            path = str(item.get("path") or item.get("file_path") or "file")
            return "\n".join(
                difflib.unified_diff(
                    before.splitlines(), after.splitlines(), fromfile=path, tofile=path, lineterm=""
                )
            )
    text = _text(value)
    return text if text.startswith(("@@ ", "--- ")) or "\n@@ " in text else None


def _read_source_rows(
    value: Any, args: Mapping[str, Any]
) -> tuple[list[tuple[str, str]], int | None, int | None]:
    lines = _result_text(value).split("\n")
    if lines and not lines[-1]:
        lines.pop()
    parsed: list[tuple[str, str]] = []
    real_numbers: list[int] = []
    numbered = True
    hashline = any(re.fullmatch(r"\[.+#[0-9a-fA-F]{4}\]", line) for line in lines)
    for line in lines:
        match = (
            re.match(r"^(\d+):(.*)$", line)
            if hashline
            else re.match(r"^\s*(\d+(?:\.\d+)?)  (.*)$", line)
        )
        if match:
            marker, source = match.groups()
            parsed.append((marker, source))
            real_numbers.append(int(marker.partition(".")[0]))
        else:
            parsed.append(("", line))
            if line:
                numbered = False
    if real_numbers and (
        hashline
        or numbered
        or all(marker or not source or source.startswith("[notice]") for marker, source in parsed)
    ):
        return parsed, min(real_numbers), max(real_numbers)

    offset = args.get("offset")
    if isinstance(offset, int):
        start = max(0, offset) + 1
    else:
        selector = re.search(r":(\d+)(?:-\d+)?$", str(args.get("path", args.get("file_path", ""))))
        supplied = args.get(
            "start_line", args.get("start", int(selector.group(1)) if selector else 1)
        )
        start = supplied if isinstance(supplied, int) else 1
    generated = [(str(start + index), line) for index, line in enumerate(lines)]
    end = start + len(generated) - 1 if generated else None
    return generated, start if generated else None, end


def _read_display_rows(
    source_rows: list[tuple[str, str]], *, expanded: bool, theme: Any, path: str = ""
) -> list[Text]:
    visible = source_rows if expanded else source_rows[:READ_LINES]
    gutter_width = max(2, max((len(marker) for marker, _ in source_rows), default=0))
    rows: list[Text] = []
    gutter_style = str(theme_value(theme, "dim", theme_value(theme, "muted")))
    highlighted = highlight(
        "\n".join(source[:4000] for _, source in visible), language_from_path(path), theme
    ).split("\n")
    for index, (marker, source) in enumerate(visible):
        row = Text(f"{marker:>{gutter_width}}│", style=gutter_style)
        row.append(highlighted[index] if index < len(highlighted) else source[:4000])
        rows.append(row)
    hidden = len(source_rows) - len(visible)
    if hidden:
        rows.append(
            Text(
                f"{' ' * (gutter_width + 1)}… {hidden} more lines {EXPAND_HINT}", style=gutter_style
            )
        )
    return rows


def _read_rows(
    block: Block, args: Mapping[str, Any], *, expanded: bool, theme: Any
) -> tuple[str, list[str | Text]]:
    cwd = block.data.get("cwd")
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        rows: list[str | Text] = []
        for index, call in enumerate(calls):
            call_args = call.get("args", {}) if isinstance(call, Mapping) else {}
            result = call.get("result") if isinstance(call, Mapping) else None
            source_rows, first, last = _read_source_rows(result, call_args)
            branch = "└─" if index == len(calls) - 1 else "├─"
            selection = f":{first}-{last}" if first is not None and last is not None else ""
            path = _path(call_args, cwd)
            if str(block.data.get("name")) == "read" and first is not None:
                path = re.sub(r"(?::(?:raw|[-\d,+]+))+$", "", path)
            rows.append(f"{branch} {path}{selection}")
            preview = source_rows if expanded else source_rows[:GROUP_READ_LINES]
            colored = highlight(
                "\n".join(source[:4000] for _marker, source in preview),
                language_from_path(path),
                theme,
            ).split("\n")
            for line in colored if preview else []:
                row = Text("   ")
                row.append(line)
                rows.append(row)
        return f"• Read ({len(calls)})", rows
    path = _path(args, cwd)
    if _state(block) == "running":
        return f"⏳ Read: {path}{_selection(args)}", []
    source_rows, first, last = _read_source_rows(block.data.get("result"), args)
    if str(block.data.get("name")) == "read" and first is not None:
        path = re.sub(r"(?::(?:raw|[-\d,+]+))+$", "", path)
    selection = f":{first}-{last}" if first is not None and last is not None else ""
    return f"• Read {path}{selection}", _read_display_rows(
        source_rows, expanded=expanded, theme=theme, path=path
    )


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


def _edit_rows(
    block: Block, args: Mapping[str, Any], theme: Any, width: int, expanded: bool
) -> tuple[str, list[Text]]:
    diff = _diff(block.data.get("result")) or _diff(args) or ""
    first = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)", diff)
    line = f":{first.group(1)}" if first else ""
    added = sum(
        1 for value in diff.splitlines() if value.startswith("+") and not value.startswith("+++")
    )
    removed = sum(
        1 for value in diff.splitlines() if value.startswith("-") and not value.startswith("---")
    )
    header = (
        f"{_glyph(block, _state(block), theme)} Edit: {_path(args)}{line} ⟦+{added}/-{removed}⟧"
    )
    rendered = render_diff(
        replace(block, kind="diff", data={"text": diff, "path": _path(args)}),
        theme,
        width,
        10_000,
        expanded,
    )
    rows = list(rendered.split("\n", allow_blank=True))
    if block.state is BlockState.ACTIVE:
        hidden = max(0, len(rows) - 12)
        rows = rows[-12:]
        if hidden:
            rows.insert(0, Text("… (content above)", style="dim"))
        rows.append(
            Text(
                f"{SPINNER_FRAMES[int(block.data.get('spinner_frame', 0)) % 8]} (preview)",
                style="dim",
            )
        )
    return header, rows


def _bash_jobs_rows(
    block: Block, args: Mapping[str, Any], expanded: bool, width: int
) -> tuple[list[str | Text], dict[int, str]]:
    result = block.data.get("result")
    action = str(args.get("action", "list"))
    if action == "read":
        return _bash_rows(
            block, {"command": f"job {args.get('job_id', args.get('id', ''))}"}, expanded, width
        )
    if action == "kill":
        job = args.get("job_id", args.get("id", ""))
        return [f"Job {job}", _result_text(result) or "Stopped"], {}
    jobs = _value(result, "jobs", result if isinstance(result, list) else [])
    rows: list[str | Text] = []
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, Mapping):
                identifier = job.get("job_id", job.get("id", "—"))
                rows.append(
                    f"{identifier} · {job.get('status', 'running')} · {job.get('command', '')}"
                )
    if not rows:
        rows = [
            _result_text(result) if result and not isinstance(jobs, list) else "No background jobs"
        ]
    if not expanded and len(rows) > BASH_LINES:
        rows = [*rows[:BASH_LINES], f"… {len(rows) - BASH_LINES} more jobs"]
    return rows, {}


def _bash_rows(
    block: Block, args: Mapping[str, Any], expanded: bool, width: int = 80
) -> tuple[list[str | Text], dict[int, str]]:
    command_lines = str(args.get("command", args.get("cmd", ""))).splitlines() or [""]
    if len(command_lines) > 6 and not expanded:
        command_lines = [
            f"… {len(command_lines) - 6} earlier lines {EXPAND_HINT}",
            *command_lines[-6:],
        ]
    result = block.data.get("result")
    source = str(_value(result, "raw_output", _result_text(result)))
    output = terminal_rows(
        block.id,
        source,
        width - 4,
        streaming=block.state is BlockState.ACTIVE,
    )
    if _value(result, "raw_output") is not None:
        # Notices belong outside the terminal replay: cursor movement in raw
        # shell output must not erase truncation or timeout diagnostics.
        output = [
            *output,
            *(
                Text(line)
                for line in _text(result).splitlines()
                if line.startswith(("[notice]", "[Output limited:", "Command timed out"))
            ),
        ]
    if len(output) > BASH_LINES and not expanded:
        total = len(output)
        output = [
            Text(
                f"… ({total - BASH_LINES} earlier lines, showing {BASH_LINES} of {total}) (ctrl+o to expand)"
            ),
            *output[-BASH_LINES:],
        ]
    rows: list[str | Text] = [*(f"$ {line}" for line in command_lines)]
    section_index = len(rows)
    rows.append("")
    rows.extend(output)
    result = block.data.get("result")
    wall = _value(
        result,
        "wall_time",
        block.data.get("elapsed")
        if _state(block) == "running"
        else block.data.get("duration", 0.0),
    )
    wall = wall if isinstance(wall, (int, float)) and math.isfinite(wall) else 0.0
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
    text = _result_text(result).strip()
    if not text:
        return []
    if text.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
    return text.splitlines()


def _numeric_value(value: Any, *keys: str) -> int | None:
    for key in keys:
        supplied = _value(value, key)
        if isinstance(supplied, (int, float)) and not isinstance(supplied, bool):
            return max(0, int(supplied))
    return None


def _path_item(item: Any) -> str:
    if not isinstance(item, Mapping):
        return str(item)
    value = str(item.get("path", item.get("name", item.get("file", ""))))
    kind = str(item.get("type", item.get("kind", ""))).casefold()
    is_dir = bool(item.get("is_dir") or item.get("directory") or kind in {"dir", "directory"})
    return f"{value}/" if is_dir and value and not value.endswith("/") else value


def _grep_items(result: Any, output_mode: str) -> tuple[list[str], int, int]:
    structured = _items_from_result(result, "matches")
    has_structured_matches = any(
        isinstance(item.get("matches"), Sequence)
        and not isinstance(item.get("matches"), (str, bytes))
        for item in _mappings(result)
    )
    if has_structured_matches:
        rows: list[str] = []
        paths: set[str] = set()
        for item in structured:
            if not isinstance(item, Mapping):
                row = str(item)
                rows.append(row)
                paths.add(row.split(":", 1)[0])
                continue
            path = str(item.get("path", item.get("file", "")))
            line = item.get("line", item.get("line_number"))
            text = str(item.get("text", item.get("content", "")))
            paths.add(path)
            rows.append(f"{path}:{line}:{text}" if line is not None else f"{path}:{text}")
        return rows, len(rows), len(paths - {""})

    raw_lines = _result_text(result).splitlines()
    if output_mode == "content" and any(re.match(r"^\s+\d+[:-] ", line) for line in raw_lines):
        rows: list[str] = []
        paths: set[str] = set()
        current_path = ""
        total = 0
        for line in raw_lines:
            if line.startswith("[notice]"):
                try:
                    rows.append(json.loads(line.removeprefix("[notice] "))["message"])
                except (ValueError, KeyError):
                    rows.append(line)
            elif line and not line[0].isspace() and line.endswith(":"):
                current_path = line[:-1]
                paths.add(current_path)
            elif match := re.match(r"^\s+(\d+)([:-]) (.*)$", line):
                number, marker, source = match.groups()
                total += marker == ":"
                rows.append(f"{current_path}:{number}{marker}{source}")
            elif line.strip():
                rows.append(line)
        return rows, total, len(paths)
    if output_mode == "files_with_matches":
        rows = [line for line in raw_lines if line]
        return rows, len(rows), len(set(rows))
    rows = []
    paths: set[str] = set()
    current_path: str | None = None
    count_total = 0
    count_mode = output_mode == "count"
    for line in raw_lines:
        count_match = re.match(r"^(.+): (\d+)$", line) if count_mode else None
        if count_match and not line.startswith((" ", "\t")):
            path, count = count_match.groups()
            paths.add(path)
            count_total += int(count)
            rows.append(line)
            continue
        count_mode = False
        if line and not line.startswith((" ", "\t")) and line.endswith(":"):
            current_path = line[:-1]
            paths.add(current_path)
            continue
        match = re.match(r"^\s+(\d+):\s?(.*)$", line)
        if match and current_path is not None:
            number, text = match.groups()
            rows.append(f"{current_path}:{number}:{text}")
            continue
        if line.strip():
            rows.append(line.strip())
            paths.add(line.split(":", 1)[0])
    if count_mode and rows:
        return rows, count_total, len(paths)
    return rows, len(rows), len(paths - {""})


def _timed_inline_header(text: str, timing: str, theme: Any) -> Text:
    header = Text(text)
    if timing:
        muted = str(theme_value(theme, "dim", theme_value(theme, "muted")))
        header.append(f" {theme_symbol(theme, 'sep.thin', '·')} ", style=muted)
        header.append(timing, style=muted)
    return header


def _tree_output(
    header: str | Text, items: list[str], *, total: int, limit: int, item_type: str
) -> Text:
    visible = items[:limit]
    hidden = max(0, total - len(visible))
    output = header.copy() if isinstance(header, Text) else Text(header)
    for index, item in enumerate(visible):
        branch = "└─" if index == len(visible) - 1 and hidden == 0 else "├─"
        output.append(f"\n  {branch} {item}")
    if hidden:
        noun = item_type if hidden == 1 else {"match": "matches"}.get(item_type, f"{item_type}s")
        output.append(f"\n  … {hidden} more {noun} {EXPAND_HINT}")
    return output


def _inline_rows(
    block: Block, name: str, args: Mapping[str, Any], expanded: bool, theme: Any
) -> Text:
    state = _state(block)
    result = block.data.get("result")
    pattern = str(args.get("pattern", args.get("query", "")))
    timing = _timing_label(block, state)
    if name == "ls":
        detail = _path(args) or "."
    elif name == "web_search":
        detail = pattern
    else:
        detail = pattern or _path(args)
    if state == "running":
        title = f"{_glyph(block, state, theme)} {_label(name)}{f': {detail}' if detail else ''}"
        return _timed_inline_header(title, timing, theme)
    if state == "error":
        title = f"✘ {_label(name)}{f': {detail}' if detail else ''}"
        output = _timed_inline_header(title, timing, theme)
        output.append("\n")
        output.append(
            _result_text(result)[:110] or "Unknown error", style=str(theme_value(theme, "error"))
        )
        return output
    if name == "grep":
        grep_text = _result_text(result)
        grep_result = (
            None if grep_text.strip() in {"No matches found", "No matches found."} else result
        )
        default_mode = (
            "content" if re.search(r"(?m)^\s+\d+[:-] ", grep_text) else "files_with_matches"
        )
        output_mode = str(args.get("output_mode", default_mode))
        items, parsed_total, parsed_files = _grep_items(grep_result, output_mode)
        supplied_total = _numeric_value(grep_result, "match_count", "total_matches", "count")
        supplied_files = _numeric_value(grep_result, "file_count", "files_with_matches")
        total = parsed_total if supplied_total is None else supplied_total
        files = parsed_files if supplied_files is None else supplied_files
        if not items and total == 0:
            return Text("⚠ No matches found", style=str(theme_value(theme, "warning")))
        header = _timed_inline_header(
            f"🔍 Grep: {pattern}  {total} matches · {files} files · in {args.get('path', args.get('cwd', '.'))}",
            timing,
            theme,
        )
        return _tree_output(
            header, items, total=total, limit=24 if expanded else SEARCH_LINES, item_type="match"
        )
    if name == "ls":
        items = [
            _path_item(item)
            for item in _items_from_result(result, "entries", "items", "files", "paths")
        ]
        items = [item for item in items if item and item != "No files found"]
        supplied_total = _numeric_value(result, "count", "total", "total_count")
        total = len(items) if supplied_total is None else supplied_total
        if not items and total == 0:
            return Text("⚠ Empty directory", style=str(theme_value(theme, "warning")))
        path = _path(args) or "."
        return _tree_output(
            _timed_inline_header(f"📂 Ls: {path}  {total} items", timing, theme),
            items,
            total=total,
            limit=24 if expanded else LIST_LINES,
            item_type="item",
        )

    items = [
        _path_item(item)
        for item in _items_from_result(result, "matches", "items", "results", "files", "paths")
    ]
    if name == "glob" and any(item.startswith("  ") for item in items):
        directory = ""
        paths = []
        for item in items:
            if item.startswith("  "):
                paths.append(f"{directory}{item.strip()}")
            elif item.endswith("/"):
                directory = "" if item == "./" else item
            else:
                paths.append(item)
        items = paths
    items = [
        item
        for item in items
        if item and item not in {"No files found", "No files found matching pattern"}
    ]
    supplied_total = _numeric_value(result, "count", "total", "total_count", "result_count")
    total = len(items) if supplied_total is None else supplied_total
    if not items and total == 0:
        empty = "No files found" if name == "glob" else "No results found"
        return Text(f"⚠ {empty}", style=str(theme_value(theme, "warning")))
    label = "Glob" if name == "glob" else "Web Search"
    noun = "items" if name == "glob" else "results"
    limit = 24 if expanded else 8 if name == "glob" else 6
    return _tree_output(
        _timed_inline_header(f"🔍 {label}: {pattern}  {total} {noun}", timing, theme),
        items,
        total=total,
        limit=limit,
        item_type=noun.removesuffix("s"),
    )


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
        row = Text(
            f"{glyph} {label}", style=str(theme_value(theme, "success" if done else "accent"))
        )
        if done:
            supplied = value.get("completion_progress", 1.0)
            progress = (
                min(1.0, max(0.0, float(supplied))) if isinstance(supplied, (int, float)) else 1.0
            )
            # Leave the status symbol legible; sweep the strike through the label.
            if progress >= 1:
                row.stylize("strike")
            else:
                row.stylize(
                    "strike", len(str(glyph)) + 1, len(str(glyph)) + 1 + int(len(label) * progress)
                )
        rows.append(row)
    header_glyph = theme_symbol(theme, "status.success", "☑")
    separator = theme_symbol(theme, "sep.thin", "·")
    return f"{header_glyph} Todo {separator} {len(items)} tasks", rows


def _generic_rows(args: Mapping[str, Any], result: Any, expanded: bool) -> list[str]:
    arg_text = " ".join(f"{key}={value}" for key, value in args.items())
    output = _result_text(result).splitlines()
    return [
        f"└─ {arg_text}" if arg_text else "└─",
        *_limited(output, 12 if expanded else 4, expanded),
    ]


def _render_impl(
    block: Block, theme: Any, width: int, budget_rows: int, expanded: bool
) -> Text | None:
    if budget_rows <= 0:
        return None
    name = str(block.data.get("name", "tool"))
    if name == "task":
        from .task import render_task

        task = replace(block, kind="task")
        return render_task(task, theme, width, budget_rows, expanded)
    args = block.data.get("args", {})
    if not isinstance(args, Mapping):
        args = {}
    state = _state(block)
    label, detail = _label(name), _detail(name, args, block.data.get("cwd"))
    calls = block.data.get("calls")
    if isinstance(calls, list) and calls:
        label = f"{label} ({len(calls)})"
    separator = theme_symbol(theme, "sep.thin", "·")
    timing = _timing_label(block, state)
    compact = f"{label}{f' {separator} {detail}' if detail else ''}{f' {separator} {timing}' if timing else ''}"
    if budget_rows == 1:
        return Text(
            f"{_glyph(block, state, theme)} {compact}", style=str(theme_value(theme, "toolTitle"))
        )
    if budget_rows == 2:
        top_left = _box_char(theme, "topLeft", "╭", "top_left")
        bottom_left = _box_char(theme, "bottomLeft", "╰", "bottom_left")
        horizontal = _box_char(theme, "horizontal", "─", "top")
        return Text(
            f"{top_left}{horizontal} {compact}\n{bottom_left}",
            style=str(theme_value(theme, _border_token(name, state))),
        )
    if name in _INLINE:
        inline = _inline_rows(block, name, args, expanded, theme)
        if bool(theme_value(theme, "hyperlinks", False)) and name in {"grep", "glob", "ls"}:
            for match in re.finditer(r"(?m)^  [├└]─ (.+)$", inline.plain):
                content = match.group(1)
                location = re.match(r"^(.*?):(\d+):", content) if name == "grep" else None
                path = location.group(1) if location else content
                if not path or path.startswith("…"):
                    continue
                target = Path(path).expanduser()
                if not target.is_absolute():
                    root = Path(str(block.data.get("cwd", Path.cwd())))
                    if name == "ls":
                        root = root / str(args.get("path", "."))
                    target = root / target
                uri = target.absolute().as_uri() + (f"#L{location.group(2)}" if location else "")
                inline.stylize(Style(link=uri), match.start(1), match.start(1) + len(path))
        return inline
    if state == "error" and name not in _BASH:
        header = _header_with_timing(
            f"✘ {label}{f' {detail}' if detail else ''}", block, state, theme
        )
        return _frame(
            header,
            _result_text(block.data.get("result")).splitlines() or ["Unknown error"],
            width=width,
            budget_rows=budget_rows,
            theme=theme,
            border_token="error",
            state=state,
            expanded=expanded,
        )
    if has_image(block):
        return render_image(block, theme, width, budget_rows, expanded)
    sections: dict[int, str] | None = None
    edit = False
    if name in _READ:
        header, rows = _read_rows(block, args, expanded=expanded, theme=theme)
    elif name in _WRITE:
        header, rows = _write_rows(block, args)
    elif name in _EDIT:
        header, rows = _edit_rows(block, args, theme, width, expanded)
        edit = True
    elif name == "bash_jobs":
        header = f"{_glyph(block, state, theme)} Bash jobs · {args.get('action', 'list')}"
        rows, sections = _bash_jobs_rows(block, args, expanded, width)
    elif name in _BASH:
        header = f"{_glyph(block, state, theme)} Bash"
        rows, sections = _bash_rows(block, args, expanded, width)
    elif name == "todo":
        header, rows = _todo_rows(args, block.data.get("result"), theme)
    else:
        header = f"{_glyph(block, state, theme)} {label}{f': {detail}' if detail else ''}"
        rows = _generic_rows(args, block.data.get("result"), expanded)
    timed_header = _header_with_timing(header, block, state, theme)
    if bool(theme_value(theme, "hyperlinks", False)) and name in _READ | _WRITE | _EDIT:
        raw_path = str(args.get("path", args.get("file_path", "")))
        display_path = _path(args, block.data.get("cwd"))
        if raw_path and display_path in timed_header.plain:
            selector = re.search(r":(\d+)(?:-\d+)?$", raw_path)
            clean_path = raw_path[: selector.start()] if selector else raw_path
            target = Path(clean_path).expanduser()
            if not target.is_absolute():
                target = Path(str(block.data.get("cwd", Path.cwd()))) / target
            uri = target.absolute().as_uri() + (f"#L{selector.group(1)}" if selector else "")
            start = timed_header.plain.index(display_path)
            timed_header.stylize(Style(link=uri), start, start + len(display_path))
    return _frame(
        timed_header,
        rows,
        width=width,
        budget_rows=budget_rows,
        theme=theme,
        border_token=_border_token(name, state),
        state=state,
        sections=sections,
        edit=edit,
        expanded=expanded,
    )


def render(
    block: Block, theme: Any, width: int, budget_rows: int, expanded: bool
) -> Group | Text | None:
    try:
        content = _render_impl(block, theme, width, budget_rows, expanded)
    except Exception:
        name = _label(str(block.data.get("name", "tool")))
        args = block.data.get("args", {})
        path = f" {_path(args)}" if isinstance(args, Mapping) and _path(args) else ""
        raw = _result_text(block.data.get("result")) or _text(block.data)
        content = Text(f"✘ {name}{path}\n{raw}", style=str(theme_value(theme, "error")))
    if content is None or block.data.get("leading_spacer") is False:
        return content
    return with_leading_spacer(content)


__all__ = ["EXPAND_HINT", "SPINNER_FRAMES", "render"]
