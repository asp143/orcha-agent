"""Composer command, file-path, and plugin completion."""

from __future__ import annotations

import fnmatch
import inspect
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


def _sensitive(name: str) -> bool:
    return name == "Credentials" or name.startswith(".env")


@dataclass(frozen=True, slots=True)
class _IgnoreRule:
    base: str
    pattern: str
    negated: bool
    directory_only: bool
    regex: re.Pattern[str]

    def matches(self, path: str) -> bool:
        if self.base:
            prefix = self.base + "/"
            if path == self.base:
                local = ""
            elif path.startswith(prefix):
                local = path[len(prefix) :]
            else:
                return False
        else:
            local = path
        return bool(local and self.regex.search(local))

    def can_reinclude_below(self, directory: str) -> bool:
        if not self.negated:
            return False
        if self.base and not (
            directory == self.base
            or directory.startswith(self.base + "/")
            or self.base.startswith(directory + "/")
        ):
            return False
        local = (
            directory[len(self.base) + 1 :]
            if self.base and directory.startswith(self.base + "/")
            else ""
        )
        wildcard = min(
            (position for position, value in enumerate(self.pattern) if value in "*?["),
            default=len(self.pattern),
        )
        static = self.pattern[:wildcard].rstrip("/")
        return not static or not local or static.startswith(local) or local.startswith(static)


def _glob_regex(pattern: str, *, basename: bool, directory_only: bool) -> re.Pattern[str]:
    pieces: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    pieces.append("(?:.*/)?")
                    index += 1
                else:
                    pieces.append(".*")
                continue
            pieces.append("[^/]*")
        elif character == "?":
            pieces.append("[^/]")
        elif character == "[":
            close = pattern.find("]", index + 1)
            if close < 0:
                pieces.append(r"\[")
            else:
                content = pattern[index + 1 : close]
                if content.startswith("!"):
                    content = "^" + content[1:]
                pieces.append("[" + content + "]")
                index = close
        elif character == "\\" and index + 1 < len(pattern):
            index += 1
            pieces.append(re.escape(pattern[index]))
        else:
            pieces.append(re.escape(character))
        index += 1
    body = "".join(pieces)
    prefix = r"(?:^|/)" if basename else "^"
    suffix = r"(?:$|/)"
    return re.compile(prefix + body + suffix)


def _ignored(path: str, rules: Iterable[_IgnoreRule]) -> bool:
    ignored = False
    for rule in rules:
        if rule.matches(path):
            ignored = not rule.negated
    return ignored


class PathIndex:
    """Bounded, short-lived filename index that never enters sensitive paths."""

    def __init__(self, cwd: str | Path, *, cap: int = 20_000, ttl: float = 10.0) -> None:
        self.cwd = Path(cwd).resolve()
        self.cap = cap
        self.ttl = ttl
        self._cache: tuple[str, ...] = ()
        self._cached_at = 0.0

    def _read_rules(self, directory: Path) -> list[_IgnoreRule]:
        path = directory / ".gitignore"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        base_path = directory.relative_to(self.cwd).as_posix()
        base = "" if base_path == "." else base_path
        rules: list[_IgnoreRule] = []
        for line in lines:
            value = line.rstrip()
            if not value:
                continue
            if value.startswith(r"\#"):
                value = value[1:]
            elif value.startswith("#"):
                continue
            negated = value.startswith("!")
            if negated:
                value = value[1:]
            elif value.startswith(r"\!"):
                value = value[1:]
            anchored = value.startswith("/")
            if anchored:
                value = value[1:]
            directory_only = value.endswith("/")
            value = value.rstrip("/")
            if not value:
                continue
            basename = not anchored and "/" not in value
            rules.append(
                _IgnoreRule(
                    base=base,
                    pattern=value,
                    negated=negated,
                    directory_only=directory_only,
                    regex=_glob_regex(
                        value,
                        basename=basename,
                        directory_only=directory_only,
                    ),
                )
            )
        return rules

    def _walk(self) -> tuple[str, ...]:
        root_rules = self._read_rules(self.cwd)
        found: list[str] = []
        stack: list[tuple[Path, list[_IgnoreRule]]] = [(self.cwd, root_rules)]
        while stack and len(found) < self.cap:
            directory, rules = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError:
                continue
            for entry in entries:
                if len(found) >= self.cap:
                    break
                if _sensitive(entry.name) or entry.name == ".git":
                    continue
                try:
                    relative = Path(entry.path).relative_to(self.cwd).as_posix()
                    is_directory = entry.is_dir(follow_symlinks=False)
                except (OSError, ValueError):
                    continue
                ignored = _ignored(relative, rules)
                if not ignored:
                    found.append(relative + "/" if is_directory else relative)
                if is_directory and (
                    not ignored
                    or any(rule.can_reinclude_below(relative) for rule in rules)
                ):
                    child = Path(entry.path)
                    stack.append((child, [*rules, *self._read_rules(child)]))
        return tuple(sorted(found))

    def paths(self) -> tuple[str, ...]:
        now = time.monotonic()
        if not self._cache or now - self._cached_at >= self.ttl:
            self._cache = self._walk()
            self._cached_at = now
        return self._cache


def _fuzzy_score(candidate: str, query: str) -> tuple[int, int, str] | None:
    if not query:
        return (0, len(candidate), candidate)
    lower = candidate.lower()
    query = query.lower()
    position = 0
    gaps = 0
    for character in query:
        found = lower.find(character, position)
        if found < 0:
            return None
        gaps += found - position
        position = found + 1
    return (gaps, len(candidate), candidate)


def _quote_path(path: str) -> str:
    if any(character.isspace() for character in path):
        return '"' + path.replace('"', '\\"') + '"'
    return path


class ComposerCompleter(Completer):
    """Completion fan-in for core commands, safe paths, and plugins."""

    def __init__(self, registry: Any, cwd: str | Path) -> None:
        self.registry = registry
        self.path_index = PathIndex(cwd)

    def _commands(self, document: Document) -> Iterable[Completion]:
        before = document.text_before_cursor
        if not before.startswith("/") or any(character.isspace() for character in before):
            return ()
        query = before[1:]
        return (
            Completion(
                name,
                start_position=-len(query),
                display=name,
                display_meta=registration.help,
            )
            for name, registration in sorted(self.registry.commands.items())
            if _fuzzy_score(name, query) is not None
        )

    def _at_path(self, document: Document) -> Iterable[Completion]:
        before = document.text_before_cursor
        at = before.rfind("@")
        if at < 0 or (at > 0 and not before[at - 1].isspace()):
            return ()
        fragment = before[at:]
        if any(character.isspace() for character in fragment) and not fragment.startswith('@"'):
            return ()
        raw_query = fragment[2:] if fragment.startswith('@"') else fragment[1:]
        raw_query = raw_query.rstrip('"')
        ranked = sorted(
            (score, path)
            for path in self.path_index.paths()
            if (score := _fuzzy_score(path, raw_query)) is not None
        )
        return (
            Completion("@" + _quote_path(path), start_position=-len(fragment), display=path)
            for _score, path in ranked
        )

    def _bare_path(self, document: Document) -> Iterable[Completion]:
        fragment = document.get_word_before_cursor(WORD=True)
        if not fragment or fragment.startswith(("/", "@")):
            return ()
        ranked = sorted(
            (score, path)
            for path in self.path_index.paths()
            if (score := _fuzzy_score(path, fragment.strip('"'))) is not None
        )
        return (
            Completion(_quote_path(path), start_position=-len(fragment), display=path)
            for _score, path in ranked
        )

    def _plugins(self, document: Document) -> Iterable[Completion]:
        for registration in self.registry.completers:
            if not document.text_before_cursor.startswith(registration.trigger):
                continue
            result = registration.fn(document)
            if inspect.isawaitable(result):
                continue
            for item in result or ():
                yield item if isinstance(item, Completion) else Completion(str(item))

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        before = document.text_before_cursor
        if before.startswith("/"):
            yield from self._commands(document)
        elif "@" in before:
            yield from self._at_path(document)
        elif complete_event.completion_requested:
            yield from self._bare_path(document)
        yield from self._plugins(document)


__all__ = ["ComposerCompleter", "PathIndex"]
