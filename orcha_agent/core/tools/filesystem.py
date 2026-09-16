from __future__ import annotations

from datetime import datetime
from difflib import unified_diff
import hashlib
from pathlib import Path
import re
import shlex

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool, ToolException

from .common import atomic_write, notice, resolve_path, split_text_lines
from .fuzzy import find_matches, replace

_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff", ".avif"}


def parse_selector(path: str, total: int) -> tuple[str, list[tuple[int, int]], bool]:
    raw = False
    chunks = path.split(":")
    selectors: list[str] = []
    while len(chunks) > 1 and (chunks[-1] == "raw" or re.fullmatch(r"[-\d,+]+", chunks[-1])):
        selectors.insert(0, chunks.pop())
    if "raw" in selectors:
        raw = True
        selectors.remove("raw")
    if len(selectors) > 1:
        raise ValueError("Invalid selector; use file:N-M, file:N+K, file:-N or file:raw")
    ranges: list[tuple[int, int]] = []
    for part in selectors[0].split(",") if selectors else []:
        if re.fullmatch(r"-[1-9]\d*", part):
            ranges.append((max(1, total - int(part[1:]) + 1), total))
            continue
        match = re.fullmatch(r"([1-9]\d*)(?:([-+])(\d*))?", part)
        if not match:
            raise ValueError(f"Invalid line selector: {part}")
        start = int(match[1])
        operator, amount = match[2], match[3]
        end = (start + 199) if operator is None else (total if not amount else int(amount))
        if operator == "+":
            if not amount or int(amount) < 1:
                raise ValueError("Line count must be positive")
            end = start + int(amount) - 1
        if end < start:
            raise ValueError("Range end must be at least its start")
        ranges.append((start, min(end, total)))
    return ":".join(chunks), ranges or [(1, total if raw else min(total, 200))], raw


def _diff(path: str, old: str, new: str) -> str:
    lines = list(
        unified_diff(
            split_text_lines(old), split_text_lines(new), fromfile=path, tofile=path, lineterm=""
        )
    )
    result = "\n".join(lines[:200])
    if len(lines) > 200:
        result += "\n" + notice(
            f"Elided {len(lines) - 200} diff lines; use read on the changed path to inspect.",
            omitted_lines=len(lines) - 200,
        )
    return result


def _bounded_read(
    selected: list[tuple[int, str]], content: str, target: Path, bare: str, raw: bool
) -> str:
    """Clip source text without clipping metadata or confusing gutters with file bytes."""
    head, tail = selected[:60], selected[max(60, len(selected) - 25) :]
    kept = head + tail
    source_lines = split_text_lines(content, keepends=True)
    offsets = [0]
    for line in source_lines:
        offsets.append(offsets[-1] + len(line.encode()))
    omitted_ranges = [
        (offsets[number - 1], offsets[number])
        for number, _ in selected[60 : max(60, len(selected) - 25)]
    ]
    rows: list[str] = []
    budget = 20_000
    clipped = False
    for number, text in kept:
        prefix = "" if raw else f"{number:6}  "
        separator = int(bool(rows))
        available = max(0, budget - separator - len(prefix.encode()))
        encoded = text.encode()
        shown = (
            encoded[:available].decode("utf-8", errors="ignore")
            if budget >= separator + len(prefix.encode())
            else ""
        )
        shown_bytes = len(shown.encode())
        if shown_bytes < len(encoded) or budget < separator + len(prefix.encode()):
            clipped = True
            omitted_ranges.append((offsets[number - 1] + shown_bytes, offsets[number]))
        if budget >= separator + len(prefix.encode()):
            row = prefix + shown
            rows.append(row)
            budget -= separator + len(row.encode())
        # Once a partial line consumed the budget, never render a later tiny
        # row inside the same source line's missing suffix.
        if shown_bytes < len(encoded):
            budget = 0
    result = "\n".join(rows)
    middle_count = len(selected) - len(kept)
    result += "\n" + notice(
        f"Elided {middle_count} middle lines; use {bare}:{head[-1][0] + 1}-{tail[0][0] - 1} to inspect them."
        if middle_count
        else "Output exceeds the byte budget; recover omitted source bytes below.",
        omitted_lines=middle_count,
        requested_ranges=[(head[0][0], head[-1][0])]
        + ([(tail[0][0], tail[-1][0])] if tail else []),
    )
    if clipped:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(set(omitted_ranges)):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            elif end > start:
                merged.append((start, end))
        omitted = sum(end - start for start, end in merged)
        script = (
            f"import sys; f=open({str(target)!r}, 'rb'); "
            f"[(f.seek(a), sys.stdout.buffer.write(f.read(b-a))) for a,b in {merged!r}]"
        )
        command = "python -c " + shlex.quote(script)
        result += "\n" + notice(
            f"Elided {omitted} original file bytes in zero-based, end-exclusive ranges {merged}. "
            f"Recover them exactly with bash(command={command!r}).",
            omitted_bytes=omitted,
            byte_ranges=merged,
            recovery_command=command,
            file_path=str(target),
        )
    return result


class FilesystemTools:
    def __init__(self, cwd: Path):
        self.cwd = cwd
        self.reads: dict[tuple[str, Path], str] = {}
        self.repeats: dict[tuple[str, str, str], tuple[str, int]] = {}

    def read(self, path: str, *, thread: str = "default", turn: str = "default") -> str:
        try:
            bare, _, _ = parse_selector(path, 0)
            target = resolve_path(self.cwd, bare)
            if target.is_dir():
                return self.ls(bare)
            if target.suffix.lower() in _IMAGE:
                return f"Image file: {bare} ({target.stat().st_size} bytes). Image bytes are not included."
            data = target.read_bytes()
            content = data.decode("utf-8")
            if "\x00" in content:
                return f"Binary file: {bare} ({len(data)} bytes); contents are not displayed."
            digest = hashlib.sha256(data).hexdigest()
            lines = split_text_lines(content)
            _, ranges, raw = parse_selector(path, len(lines))
            key = (thread, turn, str(target) + repr(ranges) + str(raw))
            previous, count = self.repeats.get(key, ("", 0))
            if previous == digest:
                self.repeats[key] = (digest, count + 1)
                warning = (
                    " Re-reading this range will not change the output; choose a different range or act on the content already read."
                    if count >= 2
                    else ""
                )
                return notice("Unchanged since last read of this range in this turn." + warning)
            # Retain only this thread's current turn to keep long sessions bounded.
            self.repeats = {k: v for k, v in self.repeats.items() if k[0] != thread or k[1] == turn}
            self.repeats[key] = (digest, 1)
            self.reads[(thread, target)] = digest
            selected = [(i, lines[i - 1]) for start, end in ranges for i in range(start, end + 1)]
            if not selected and lines:
                return notice(
                    f"Range is outside this file ({len(lines)} lines); use {bare}:1.",
                    total_lines=len(lines),
                )

            def render(rows: list[tuple[int, str]]) -> str:
                return "\n".join(text if raw else f"{i:6}  {text}" for i, text in rows)

            if path == bare and 500 <= len(lines) <= 20_000 and len(data) <= 2 * 1024 * 1024:
                outline = _outline(target, content)
                if outline:
                    return (
                        render(outline)
                        + "\n"
                        + notice(
                            f"Declaration outline of {len(lines)} lines; bodies elided. Read {bare}:N-M to inspect a declaration body.",
                            total_lines=len(lines),
                            omitted_lines=len(lines) - len(outline),
                        )
                    )
            result = render(selected)
            elided = len(result.encode()) > 20_000
            if elided:
                result = _bounded_read(selected, content, target, bare, raw)
            selected_count = len(set(i for i, _ in selected))
            if selected_count < len(lines):
                last = max((i for i, _ in selected), default=0)
                continuation = f"{bare}:{last + 1}" if last < len(lines) else f"{bare}:1"
                first = min((i for i, _ in selected), default=0)
                result += "\n" + notice(
                    f"{'Requested' if elided else 'Showing'} lines {first}-{last} of {len(lines)}; use {continuation} to continue.",
                    shown_ranges=ranges,
                    total_lines=len(lines),
                    next_path=continuation,
                )
            return result or "(empty file)"
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def edit(
        self,
        path: str,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
        edits: list[dict[str, str]] | None = None,
        *,
        thread: str = "default",
    ) -> str:
        try:
            target = resolve_path(self.cwd, path)
            data = target.read_bytes()
            if self.reads.get((thread, target)) != hashlib.sha256(data).hexdigest():
                return "Error: Read the current file before editing; it has not been read or changed since your last read."
            original = data.decode("utf-8")
            content = original
            if edits is not None and (old_string is not None or new_string is not None):
                raise ValueError("Use either edits or old_string/new_string, not both")
            changes = edits if edits is not None else [{"old": old_string, "new": new_string}]
            if not changes:
                raise ValueError("edits must contain at least one replacement")
            for change in changes:
                old, new = change.get("old"), change.get("new")
                if old is None or new is None:
                    raise ValueError("Each edit requires old and new strings")
                content = _replace_eol(content, old, new, replace_all)
            updated = content
            atomic_write(target, updated)
            self.reads[(thread, target)] = hashlib.sha256(updated.encode()).hexdigest()
            return f"Edited {path} ({len(changes)} replacement requests).\n" + _diff(
                path, original, updated
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def write(self, path: str, content: str) -> str:
        try:
            target = resolve_path(self.cwd, path)
            original = target.read_text() if target.exists() else ""
            atomic_write(target, content)
            return (
                f"Wrote {path}: {len(content.encode())} bytes, {len(split_text_lines(content))} lines.\n"
                + _diff(path, original, content)
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def ls(self, path: str = ".", limit: int = 200) -> str:
        try:
            target = resolve_path(self.cwd, path)
            if not target.is_dir():
                return f"Error: Not a directory: {path}"
            if limit < 1:
                return "Error: limit must be positive"
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            rows: list[str] = []
            for entry in entries[:limit]:
                info = entry.stat()
                rows.append(
                    f"{entry.name}{'/' if entry.is_dir() else ''}\t{info.st_size} bytes\t{datetime.fromtimestamp(info.st_mtime).isoformat(timespec='seconds')}"
                )
            if len(entries) > limit:
                rows.append(
                    notice(
                        f"Showing {limit} of {len(entries)} entries; use limit={len(entries)} to see all.",
                        omitted_entries=len(entries) - limit,
                    )
                )
            return "\n".join(rows) or "(empty directory)"
        except (OSError, ValueError) as exc:
            return f"Error: {exc}"


def _replace_eol(content: str, old: str, new: str, replace_all: bool) -> str:
    normalized = content.replace("\r\n", "\n")
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")
    # Validate ambiguity before touching bytes, then map normalized offsets back.
    replace(normalized, old, new, replace_all)
    offsets = [
        index
        for index, char in enumerate(content)
        if not (char == "\n" and index > 0 and content[index - 1] == "\r")
    ]
    offsets.append(len(content))
    for start, end in reversed(find_matches(normalized, old)):
        original_start, original_end = offsets[start], offsets[end]
        matched = content[original_start:original_end]
        eol = "\r\n" if "\r\n" in matched else "\n"
        if "\n" not in matched:
            next_newline = content.find("\n", original_start)
            if next_newline > 0 and content[next_newline - 1] == "\r":
                eol = "\r\n"
        content = content[:original_start] + new.replace("\n", eol) + content[original_end:]
    return content


def _outline(path: Path, content: str) -> list[tuple[int, str]]:
    """Use Python's parser for a reliable declaration outline without native deps."""
    if path.suffix != ".py" or re.search(r"\r(?!\n)", content):
        return []
    import ast

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    lines = split_text_lines(content)
    declarations = sorted(
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    return [
        (line, f"{lines[line - 1]}  # body: {path.name}:{line}-{end}")
        for line, end in declarations[:200]
    ]


def _checked(result: str) -> str:
    if result.startswith("Error:"):
        raise ToolException(result)
    return result


def _identity(runtime: ToolRuntime) -> tuple[str, str]:
    thread = str(runtime.config.get("configurable", {}).get("thread_id", "default"))
    messages = runtime.state.get("messages", [])
    humans = [
        (index, message)
        for index, message in enumerate(messages)
        if getattr(message, "type", "") == "human"
    ]
    turn = str(getattr(humans[-1][1], "id", None) or humans[-1][0]) if humans else "default"
    return thread, turn


def create_filesystem_tools(cwd: Path, edit_format: str = "replace") -> list[BaseTool]:
    service = FilesystemTools(cwd)

    @tool
    def read(path: str, runtime: ToolRuntime) -> str:
        """Read text files with numbered lines, or list directories. Use file:N (page), file:N-M, file:N+K, file:-N (tail), file:5-16,960-973, or file:raw. Defaults to 200 lines / 20 KB. Read before editing; follow notices for omitted content. Images return a descriptive note."""
        thread, turn = _identity(runtime)
        return _checked(service.read(path, thread=thread, turn=turn))

    @tool
    def edit(
        path: str,
        runtime: ToolRuntime,
        old_string: str | None = None,
        new_string: str | None = None,
        replace_all: bool = False,
        edits: list[dict[str, str]] | None = None,
    ) -> str:
        """Precisely replace existing text in a file you have read. Supply unique old_string and new_string, or transactional edits=[{"old":"before","new":"after"}]. Matching tolerates typography and whitespace differences; ambiguous matches require more context or replace_all=true. Returns a unified diff."""
        thread, _ = _identity(runtime)
        return _checked(
            service.edit(path, old_string, new_string, replace_all, edits, thread=thread)
        )

    @tool
    def write(path: str, content: str) -> str:
        """Create or overwrite a UTF-8 file. Creates parent directories automatically. Prefer edit for precise changes to existing files. Returns byte/line counts and a diff."""
        return _checked(service.write(path, content))

    @tool
    def ls(path: str = ".", limit: int = 200) -> str:
        """List directory entries with file sizes and modification times; directories come first. Defaults to 200 entries. Use glob to locate files recursively."""
        return _checked(service.ls(path, limit))

    for native_tool in [read, edit, write, ls]:
        native_tool.handle_tool_error = True
    if edit_format == "hashline":
        from .hashline import create_hashline_tools

        return [*create_hashline_tools(service), write, ls]
    return [read, edit, write, ls]
