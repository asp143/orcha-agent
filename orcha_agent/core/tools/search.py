"""Bounded, paginated content and filename searches for local workspaces."""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fnmatch
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import time
from collections.abc import Iterator

import regex

from langchain_core.tools import BaseTool, ToolException, tool

from .common import PathPolicy, notice as output_notice, split_text_lines

MAX_FILE_BYTES = 4 * 1024 * 1024
INTERNAL_MATCH_CAP = 2000
GLOB_TIMEOUT = 5.0
REGEX_TIMEOUT = 0.25


def _matches(name: str, pattern: str) -> bool:
    """Match path globs, including zero-directory **/ and basename patterns."""
    if "/" not in pattern:
        return fnmatch.fnmatchcase(name.rsplit("/", 1)[-1], pattern)
    names, patterns = name.split("/"), pattern.removeprefix("./").split("/")

    @lru_cache(maxsize=None)
    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(names)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(names) and match(i + 1, j))
        return i < len(names) and fnmatch.fnmatchcase(names[i], patterns[j]) and match(i + 1, j + 1)

    return match(0, 0)


def _trim_ignore_rule(line: str) -> str:
    # Git discards trailing spaces unless escaped with an odd backslash run.
    while line.endswith(" "):
        prefix = line[:-1]
        backslashes = len(prefix) - len(prefix.rstrip("\\"))
        if backslashes % 2:
            break
        line = prefix
    return line


def _rules(directory: Path, policy: PathPolicy) -> list[tuple[Path, str]]:
    if not policy.permits(directory / ".gitignore"):
        return []
    try:
        ignore = directory / ".gitignore"
        if not stat.S_ISREG(policy.stat(ignore).st_mode):
            return []
        with policy.open_read(ignore) as handle:
            lines = split_text_lines(handle.read(MAX_FILE_BYTES).decode("utf-8"))
        return [
            (directory, trimmed)
            for line in lines
            if (trimmed := _trim_ignore_rule(line)) and not trimmed.startswith("#")
        ]
    except (OSError, UnicodeError, ValueError):
        return []


def _unescape_ignore_pattern(pattern: str) -> str:
    # fnmatch does not implement gitwildmatch's backslash quoting. Character
    # classes preserve escaped wildcard literals while other escapes lose '\'.
    def literal(match: re.Match[str]) -> str:
        char = match[1]
        return {"*": "[*]", "?": "[?]", "[": "[[]", "]": "[]]"}.get(char, char)

    return re.sub(r"\\(.)", literal, pattern)


def _ignored(path: Path, is_dir: bool, rules: list[tuple[Path, str]]) -> bool:
    ignored = False
    for base, rule in rules:
        negate = rule.startswith("!")
        pattern = _unescape_ignore_pattern(rule[1:] if negate else rule)
        directory_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        if directory_only and not is_dir:
            continue
        relative = path.relative_to(base).as_posix()
        matched = (
            _matches(relative, pattern) and (not anchored or "/" in pattern or "/" not in relative)
            if anchored or "/" in pattern
            else fnmatch.fnmatchcase(path.name, pattern)
        )
        if matched:
            ignored = not negate
    return ignored


def _files(root: Path, hidden: bool, deadline: float, policy: PathPolicy) -> Iterator[Path]:
    """Walk incrementally so a timed-out glob can retain discoveries."""
    inherited: list[tuple[Path, str]] = []
    for parent in reversed(root.parents):
        if policy.permits(parent):
            inherited.extend(_rules(parent, policy))

    def walk(directory: Path, rules: list[tuple[Path, str]]) -> Iterator[Path]:
        if time.monotonic() >= deadline:
            return
        rules = rules + _rules(directory, policy)
        try:
            with policy.directory_fd(directory) as descriptor, os.scandir(descriptor) as entries:
                for entry in entries:
                    if time.monotonic() >= deadline:
                        return
                    if entry.name == ".git" or (not hidden and entry.name.startswith(".")):
                        continue
                    child = directory / entry.name
                    if not policy.permits(child):
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if _ignored(child, is_dir, rules):
                            continue
                        if is_dir:
                            yield from walk(child, rules)
                        elif entry.is_file(follow_symlinks=False):
                            yield child
                    except OSError:
                        continue
        except OSError:
            return

    yield from walk(root, inherited)


def _label(file: Path, root: Path) -> str:
    if root.is_file():
        return file.name
    return file.relative_to(root).as_posix()


def _match_lines(
    pattern: str, expression: re.Pattern[str], text: str, case: bool, skip: int = 0
) -> list[int]:
    """Bound matching lines, including expansion of a single multiline match."""
    cap = INTERNAL_MATCH_CAP + skip
    # Build line offsets once rather than recounting the prefix for every match.
    newlines = [index for index, char in enumerate(text) if char == "\n"]
    matches: set[int] = set()
    bounded = regex.compile(pattern, regex.MULTILINE | (0 if case else regex.IGNORECASE))
    for match in bounded.finditer(text, timeout=REGEX_TIMEOUT, concurrent=True):
        start = bisect_right(newlines, match.start() - 1) + 1
        end = bisect_right(newlines, max(match.start(), match.end() - 1) - 1) + 1
        for number in range(start, end + 1):
            matches.add(number)
            if len(matches) >= cap:
                return sorted(matches)[skip:]
    return sorted(matches)[skip:]


def _rg_matches(
    rg: str, pattern: str, files: list[Path], case: bool, skip: int, policy: PathPolicy
) -> tuple[dict[Path, list[int]], set[Path]]:
    """Search the policy-filtered file set with one rg process, never falling back on errors."""
    args = [
        rg,
        "--json",
        "--color",
        "never",
        "--multiline",
        "--pcre2",
        "--max-count",
        str(INTERNAL_MATCH_CAP + skip),
        "--max-filesize",
        str(MAX_FILE_BYTES),
    ]
    if not case:
        args.append("--ignore-case")
    args.extend(["--regexp", pattern, "--"])
    targets: dict[str, Path] = {}
    try:
        # Hand rg pinned regular-file descriptors, not names it could follow after
        # validation. This also protects parent directories against rename races.
        with ExitStack() as opened:
            descriptors: list[int] = []
            fd_root = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            try:
                in_use = len(os.listdir("/proc/self/fd"))
            except OSError:
                in_use = 64
            # Reserve descriptors for safe parent traversal, subprocess pipes and
            # unrelated session activity; scan the remainder after closing these.
            budget = min(256, max(0, soft_limit - in_use - 32)) if soft_limit >= 0 else 256
            for file in files:
                if len(descriptors) >= budget:
                    break
                source = opened.enter_context(policy.open_read(file))
                if os.fstat(source.fileno()).st_size > MAX_FILE_BYTES:
                    source.close()
                    continue
                fd = source.fileno()
                descriptors.append(fd)
                targets[f"{fd_root}/{fd}"] = file
            args.extend(targets or ["-"])
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=10,
                pass_fds=tuple(descriptors),
                input="" if not targets else None,
            )
    except subprocess.TimeoutExpired as exc:
        raise ToolException(
            "Error: pattern timed out after 10 s; narrow path or simplify pattern."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ToolException(f"Error: rg could not run: {exc}") from exc
    if result.returncode not in (0, 1):
        raise ToolException(f"Error: rg rejected the search: {result.stderr.strip()}")
    found: dict[Path, list[int]] = {}
    for raw in result.stdout.splitlines():
        event = json.loads(raw)
        if event.get("type") != "match":
            continue
        data = event["data"]
        name = data["path"].get("text")
        if name is None:
            import base64

            name = os.fsdecode(base64.b64decode(data["path"]["bytes"]))
        file = targets[name]
        start = data["line_number"]
        count = max(1, len(split_text_lines(data["lines"].get("text", ""))))
        numbers = found.setdefault(file, [])
        numbers.extend(range(start, start + min(count, INTERNAL_MATCH_CAP + skip - len(numbers))))
    return found, set(files) - set(targets.values())


def create_search_tools(cwd: Path, *, policy: PathPolicy | None = None) -> list[BaseTool]:
    """Create tools bound to the agent workspace, independent of shell cwd."""
    cwd = cwd.resolve()
    active_policy = policy or PathPolicy(cwd)

    @tool
    def grep(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        case: bool = True,
        context: int = 0,
        limit: int | None = None,
        skip: int = 0,
    ) -> str:
        """Search file contents with regex; use instead of shell grep/rg.

        path is a file or directory (default workspace). glob filters filenames.
        Literal or escaped newline patterns search across lines. Results are grouped
        by file. case=true is case-sensitive; case=false ignores case. limit controls files/page (default/max 20); skip paginates matching files.
        For a single file, limit controls matching lines (default/max 200) and skip
        skips matching lines, so skip=200 continues after the first full page.
        Narrow path to one file for up to 200 matching lines instead of 20/file.
        Hidden, gitignored and binary files are excluded from directory searches.
        """
        if (limit is not None and limit < 1) or skip < 0 or context < 0:
            raise ToolException(
                "Error: limit must be positive; skip and context must be nonnegative."
            )
        context = min(context, 20)
        try:
            expression = re.compile(pattern, re.MULTILINE | (0 if case else re.IGNORECASE))
        except re.error as exc:
            raise ToolException(f"Error: invalid regex: {exc}") from exc
        try:
            root = active_policy.resolve(path or ".")
        except (OSError, ValueError) as exc:
            raise ToolException(f"Error: {exc}") from exc
        if not root.exists():
            raise ToolException(f"Error: path does not exist: {root}")
        single = root.is_file()
        page_size = min(limit or (200 if single else 20), 200 if single else 20)
        per_file = page_size if single else 20
        deadline = time.monotonic() + 30
        candidates = [root] if single else sorted(_files(root, False, deadline, active_policy))
        found: list[tuple[Path, list[str], list[int]]] = []
        notices: list[str] = []
        if time.monotonic() >= deadline:
            notices.append(
                "File discovery timed out after 30 s; unvisited paths omitted. Narrow path."
            )
        candidates = [file for file in candidates if not glob or _matches(_label(file, root), glob)]
        rg = shutil.which("rg")
        rg_matches: dict[Path, list[int]] = {}
        fallback_files: set[Path] = set()
        if rg:
            rg_matches, fallback_files = _rg_matches(
                rg, pattern, candidates, case, skip if single else 0, active_policy
            )
            if fallback_files:
                notices.append(
                    f"Scanning {len(fallback_files)} files with timed Python regex fallback "
                    "because of descriptor or file-size bounds; no files omitted by this fallback."
                )
        match_count = 0
        for file in candidates:
            if time.monotonic() >= deadline:
                notices.append(
                    "Search timed out after 30 s; unvisited files omitted. Narrow path/glob."
                )
                break
            if glob and not _matches(_label(file, root), glob):
                continue
            try:
                with active_policy.open_read(file) as handle:
                    data = handle.read(MAX_FILE_BYTES + 1)
            except (OSError, ValueError):
                continue
            if b"\x00" in data:
                continue
            if len(data) > MAX_FILE_BYTES:
                notices.append(
                    f"{_label(file, root)}: scanned first {MAX_FILE_BYTES} bytes; "
                    "remaining bytes omitted (count unknown if the file is changing). Use read selectors for the remaining content."
                )
            text = data[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
            lines = split_text_lines(text)
            try:
                if rg and file not in fallback_files and len(data) <= MAX_FILE_BYTES:
                    numbers = rg_matches.get(file, [])[skip if single else 0 :]
                else:
                    # regex releases the GIL and enforces its own deadline, unlike re.
                    # The worker is always joined after timeout; no abandoned threads.
                    with ThreadPoolExecutor(max_workers=1) as worker:
                        numbers = worker.submit(
                            _match_lines, pattern, expression, text, case, skip if single else 0
                        ).result()
                numbers = [number for number in numbers if number <= len(lines)]
            except TimeoutError:
                notices.append(
                    f"{_label(file, root)}: pattern timed out after {REGEX_TIMEOUT:g} s; "
                    "file matches omitted. Simplify pattern or narrow path."
                )
                continue
            if not numbers:
                continue
            remaining = INTERNAL_MATCH_CAP - match_count
            found.append((file, lines, numbers[:remaining]))
            match_count += len(numbers)
            if match_count >= INTERNAL_MATCH_CAP:
                notices.append(
                    f"Internal limit reached ({INTERNAL_MATCH_CAP} matching lines); "
                    "later matches/files not scanned. Narrow path/glob or pattern."
                )
                break
        output: list[str] = []
        page = found if single else found[skip : skip + page_size]
        for file, lines, numbers in page:
            output.append(f"{_label(file, root)}:")
            selected = numbers[:per_file]
            included: set[int] = set()
            for number in selected:
                included.update(
                    range(max(1, number - context), min(len(lines), number + context) + 1)
                )
            for number in sorted(included):
                marker = ":" if number in selected else "-"
                output.append(f"  {number}{marker} {lines[number - 1]}")
            if len(numbers) > per_file:
                notices.append(
                    f"{_label(file, root)}: showing {per_file} of {len(numbers)} matching lines; "
                    f"{len(numbers) - per_file} omitted. "
                    + (
                        f"Continue with path={str(file)!r}, skip={skip + per_file}, limit={per_file}."
                        if single
                        else f"Use path={str(file)!r} for up to 200 matches."
                    )
                )
        if not single and skip + page_size < len(found):
            notices.append(
                f"File limit reached: showing {min(page_size, len(found) - skip)} of "
                f"{len(found)} matching files; {len(found) - skip - page_size} later files omitted. "
                f"Continue with skip={skip + page_size}, limit={page_size}."
            )
        if not output:
            output.append(
                "No matches found." if not found else "No more matching files at this skip."
            )
        output.extend(output_notice(notice, kind="truncated") for notice in notices)
        return "\n".join(output)

    @tool
    def glob(
        pattern: str, path: str | None = None, limit: int = 200, include_hidden: bool = False
    ) -> str:
        """Find files by glob, newest first and grouped by directory; use instead of find.

        Examples: **/*.py, src/**/*.ts, *.md. Slash-free patterns match basenames
        recursively. path defaults to the workspace.
        Gitignore rules are respected; include_hidden enables dotfiles. Results are
        capped at limit (default 200). Narrow pattern/path to retrieve omitted files.
        A five-second search budget returns partial discoveries with a notice.
        """
        if limit < 1:
            raise ToolException("Error: limit must be positive.")
        try:
            root = active_policy.resolve(path or ".")
        except (OSError, ValueError) as exc:
            raise ToolException(f"Error: {exc}") from exc
        if not root.exists():
            raise ToolException(f"Error: path does not exist: {root}")
        deadline = time.monotonic() + GLOB_TIMEOUT
        matches: list[tuple[float, Path]] = []
        candidates = (
            iter([root])
            if root.is_file()
            else _files(root, include_hidden, deadline, active_policy)
        )
        for file in candidates:
            if time.monotonic() >= deadline:
                break
            if _matches(_label(file, root), pattern):
                try:
                    matches.append((active_policy.stat(file).st_mtime, file))
                except OSError:
                    continue
        timed_out = time.monotonic() >= deadline
        matches.sort(key=lambda item: (-item[0], str(item[1])))
        groups: dict[str, list[str]] = {}
        for _, file in matches[:limit]:
            relative = Path(_label(file, root))
            groups.setdefault(str(relative.parent), []).append(relative.name)
        output = [
            f"{directory}/\n" + "\n".join(f"  {name}" for name in names)
            for directory, names in groups.items()
        ]
        if not output:
            output.append("No matching files found.")
        if len(matches) > limit:
            output.append(
                output_notice(
                    f"Showing {limit} of {len(matches)} discovered files; "
                    f"{len(matches) - limit} omitted. Use limit={len(matches)} or narrow pattern/path.",
                    kind="truncated",
                    omitted_files=len(matches) - limit,
                )
            )
        if timed_out:
            output.append(
                output_notice(
                    f"Search timed out after {GLOB_TIMEOUT:g} s; "
                    f"{len(matches)} matches discovered, unvisited paths omitted (count unknown). "
                    "Narrow pattern/path to complete the search.",
                    kind="truncated",
                    timed_out=True,
                )
            )
        return "\n".join(output)

    grep.handle_tool_error = True
    glob.handle_tool_error = True
    return [grep, glob]
