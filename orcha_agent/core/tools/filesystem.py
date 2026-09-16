from __future__ import annotations

from datetime import datetime
from difflib import unified_diff
import hashlib
import codecs
import os
import stat
from pathlib import Path
import re
import shlex

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool, ToolException

from .common import PathPolicy, atomic_write, notice, split_text_lines
from .fuzzy import find_matches, replace


def _image_magic(data: bytes) -> bool:
    return (
        data.startswith(
            (
                b"\x89PNG\r\n\x1a\n",
                b"\xff\xd8\xff",
                b"GIF87a",
                b"GIF89a",
                b"BM",
                b"\x00\x00\x01\x00",
                b"II*\x00",
                b"MM\x00*",
            )
        )
        or (data.startswith(b"RIFF") and data[8:12] == b"WEBP")
        or (data[4:8] == b"ftyp" and data[8:12] in (b"avif", b"avis"))
    )


def parse_selector(path: str, total: int) -> tuple[str, list[tuple[int, int]], bool]:
    raw = False
    chunks = path.split(":")
    selectors: list[str] = []
    while len(chunks) > 1 and (
        chunks[-1] in ("raw", "summary") or re.fullmatch(r"[-\d,+]+", chunks[-1])
    ):
        selectors.insert(0, chunks.pop())
    if "summary" in selectors:
        selectors.remove("summary")
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
    selected: list[tuple[int, str]],
    content: str | None,
    target: Path,
    bare: str,
    raw: bool,
    byte_offsets: dict[int, tuple[int, int]] | None = None,
) -> str:
    """Clip source text without clipping metadata or confusing gutters with file bytes."""
    head, tail = selected[:60], selected[max(60, len(selected) - 25) :]
    kept = head + tail
    if byte_offsets is None:
        byte_offsets = {}
        offset = 0
        for number, line in enumerate(split_text_lines(content or "", keepends=True), 1):
            end = offset + len(line.encode())
            byte_offsets[number] = (offset, end)
            offset = end
    omitted_ranges = [
        byte_offsets[number] for number, _ in selected[60 : max(60, len(selected) - 25)]
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
            omitted_ranges.append((byte_offsets[number][0] + shown_bytes, byte_offsets[number][1]))
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
    def __init__(
        self,
        cwd: Path,
        *,
        policy: PathPolicy | None = None,
        max_read_bytes: int = 64 * 1024 * 1024,
        read_summary: bool = False,
    ):
        if max_read_bytes < 1:
            raise ValueError("max_read_bytes must be positive")
        self.cwd = cwd
        self.policy = policy or PathPolicy(cwd)
        self.max_read_bytes = max_read_bytes
        self.read_summary = read_summary
        self.reads: dict[tuple[str, Path], str] = {}
        self.repeats: dict[tuple[str, str, str], tuple[str, int]] = {}

    def _repeat_hint(self, key: tuple[str, str, str], digest: str) -> str:
        previous, count = self.repeats.get(key, ("", 0))
        count = count + 1 if previous == digest else 1
        thread, turn, _ = key
        self.repeats = {k: v for k, v in self.repeats.items() if k[0] != thread or k[1] == turn}
        self.repeats[key] = (digest, count)
        return (
            (
                "\n"
                + notice(
                    "Unchanged since previous reads in this turn; repeated reads will not change the content. Choose a different range or use the content above."
                )
            )
            if count >= 3
            else ""
        )

    def read(self, path: str, *, thread: str = "default", turn: str = "default") -> str:
        try:
            bare, _, _ = parse_selector(path, 0)
            target = self.policy.resolve(bare)
            if stat.S_ISDIR(self.policy.stat(target).st_mode):
                return self.ls(bare)
            # Hash/count in fixed blocks, then retain only requested lines. The
            # same descriptor prevents a path replacement between both passes.
            with self.policy.open_read(target) as stream:
                info = os.fstat(stream.fileno())
                if info.st_size > self.max_read_bytes:
                    raise ValueError(
                        f"File exceeds max_read_bytes={self.max_read_bytes}; use a bounded shell command or raise tools.max_read_bytes explicitly."
                    )
                digest_builder = hashlib.sha256()
                decoder = codecs.getincrementaldecoder("utf-8")()
                total_bytes = total_lines = 0
                last = b""
                binary = False
                first = stream.read(16)
                if _image_magic(first):
                    return (
                        f"Image file: {bare} ({info.st_size} bytes). Image bytes are not included."
                    )
                stream.seek(0)
                while chunk := stream.read(64 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > self.max_read_bytes:
                        raise ValueError(
                            f"File exceeds max_read_bytes={self.max_read_bytes}; use a bounded shell command or raise tools.max_read_bytes explicitly."
                        )
                    binary |= b"\x00" in chunk
                    if not binary:
                        decoder.decode(chunk)
                    digest_builder.update(chunk)
                    total_lines += chunk.count(b"\n")
                    last = chunk[-1:]
                if binary:
                    return f"Binary file: {bare} ({total_bytes} bytes); contents are not displayed."
                decoder.decode(b"", final=True)
                total_lines += int(bool(last) and last != b"\n")
                digest = digest_builder.hexdigest()
                _, ranges, raw = parse_selector(path, total_lines)
                summary = path.endswith(":summary") or (path == bare and self.read_summary)
                if summary and 500 <= total_lines <= 20_000 and total_bytes <= 2 * 1024 * 1024:
                    stream.seek(0)
                    summary_data = stream.read(self.max_read_bytes + 1)
                    if hashlib.sha256(summary_data).hexdigest() != digest:
                        raise ValueError("File changed during read; read again before editing.")
                    content = summary_data.decode("utf-8")
                    outline = _outline(target, content)
                    if outline:
                        self.reads[(thread, target)] = digest
                        return (
                            "\n".join(f"{i:6}  {text}" for i, text in outline)
                            + "\n"
                            + notice(
                                f"Declaration outline of {total_lines} lines; bodies elided. Read {bare}:N-M to inspect a declaration body.",
                                total_lines=total_lines,
                                omitted_lines=total_lines - len(outline),
                            )
                            + self._repeat_hint((thread, turn, str(target) + ":summary"), digest)
                        )
                stream.seek(0)
                selected = []
                offsets = {}
                offset = 0
                second_digest = hashlib.sha256()
                for number, line in enumerate(
                    iter(lambda: stream.readline(self.max_read_bytes + 1), b""), 1
                ):
                    end = offset + len(line)
                    if end > self.max_read_bytes:
                        raise ValueError(
                            "File grew beyond max_read_bytes during read; retry with a bounded file."
                        )
                    second_digest.update(line)
                    if any(start <= number <= end_line for start, end_line in ranges):
                        text = line[:-1].removesuffix(b"\r") if line.endswith(b"\n") else line
                        selected.append((number, text.decode("utf-8")))
                        offsets[number] = (offset, end)
                    offset = end
                if second_digest.hexdigest() != digest:
                    raise ValueError("File changed during read; read again before editing.")
            key = (thread, turn, str(target) + repr(ranges) + str(raw))
            repeat_hint = self._repeat_hint(key, digest)
            self.reads[(thread, target)] = digest
            if not selected and total_lines:
                return notice(
                    f"Range is outside this file ({total_lines} lines); use {bare}:1.",
                    total_lines=total_lines,
                )
            result = "\n".join(text if raw else f"{i:6}  {text}" for i, text in selected)
            elided = len(result.encode()) > 20_000
            if elided:
                result = _bounded_read(selected, None, target, bare, raw, offsets)
            if len(selected) < total_lines:
                last_line = max((i for i, _ in selected), default=0)
                continuation = f"{bare}:{last_line + 1}" if last_line < total_lines else f"{bare}:1"
                first_line = min((i for i, _ in selected), default=0)
                result += "\n" + notice(
                    f"{'Requested' if elided else 'Showing'} lines {first_line}-{last_line} of {total_lines}; use {continuation} to continue.",
                    shown_ranges=ranges,
                    total_lines=total_lines,
                    next_path=continuation,
                )
            return (result or "(empty file)") + repeat_hint
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
            target = self.policy.resolve(path)
            data = self.policy.read_bytes(target, self.max_read_bytes)
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
            atomic_write(target, updated, policy=self.policy)
            self.reads[(thread, target)] = hashlib.sha256(updated.encode()).hexdigest()
            return f"Edited {path} ({len(changes)} replacement requests).\n" + _diff(
                path, original, updated
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def write(self, path: str, content: str, *, thread: str = "default") -> str:
        try:
            target = self.policy.resolve(path)
            original = (
                self.policy.read_bytes(target, self.max_read_bytes).decode("utf-8")
                if target.exists()
                else ""
            )
            atomic_write(target, content, policy=self.policy)
            self.reads[(thread, target)] = hashlib.sha256(content.encode()).hexdigest()
            return (
                f"Wrote {path}: {len(content.encode())} bytes, {len(split_text_lines(content))} lines.\n"
                + _diff(path, original, content)
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def ls(self, path: str = ".", limit: int = 200) -> str:
        try:
            target = self.policy.resolve(path)
            if not stat.S_ISDIR(self.policy.stat(target).st_mode):
                return f"Error: Not a directory: {path}"
            if limit < 1:
                return "Error: limit must be positive"
            entries = sorted(
                self.policy.listdir(target),
                key=lambda p: (not stat.S_ISDIR(self.policy.stat(p).st_mode), p.name.lower()),
            )
            rows: list[str] = []
            for entry in entries[:limit]:
                info = self.policy.stat(entry)
                rows.append(
                    f"{entry.name}{'/' if stat.S_ISDIR(info.st_mode) else ''}\t{info.st_size} bytes\t{datetime.fromtimestamp(info.st_mtime).isoformat(timespec='seconds')}"
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


def create_filesystem_tools(
    cwd: Path,
    edit_format: str = "replace",
    *,
    policy: PathPolicy | None = None,
    max_read_bytes: int = 64 * 1024 * 1024,
    read_summary: bool = False,
) -> list[BaseTool]:
    service = FilesystemTools(
        cwd, policy=policy, max_read_bytes=max_read_bytes, read_summary=read_summary
    )

    @tool
    def read(path: str, runtime: ToolRuntime) -> str:
        """Read text files with numbered lines, or list directories. Use file:N (page), file:N-M, file:N+K, file:-N (tail), file:5-16,960-973, file:raw, or file:summary for a declaration outline. Defaults to 200 lines / 20 KB. Read before editing; follow notices for omitted content. Images return a descriptive note."""
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
    def write(path: str, content: str, runtime: ToolRuntime) -> str:
        """Create or overwrite a UTF-8 file. Creates parent directories automatically. Prefer edit for precise changes to existing files. Returns byte/line counts and a diff."""
        thread, _ = _identity(runtime)
        return _checked(service.write(path, content, thread=thread))

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
