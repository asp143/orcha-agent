"""Composer command, file-path, and plugin completion."""

from __future__ import annotations

import fnmatch
import inspect
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


def _sensitive(name: str) -> bool:
    return name == "Credentials" or name.startswith(".env")


def _ignored(path: str, patterns: list[tuple[str, bool]]) -> bool:
    ignored = False
    path = path.replace(os.sep, "/")
    for pattern, negated in patterns:
        candidate = pattern.rstrip("/")
        directory_pattern = pattern.endswith("/")
        matched = (
            fnmatch.fnmatch(path, candidate)
            or fnmatch.fnmatch(path, f"{candidate}/*")
            or ("/" not in candidate and fnmatch.fnmatch(Path(path).name, candidate))
        )
        if directory_pattern and (path == candidate or path.startswith(candidate + "/")):
            matched = True
        if matched:
            ignored = not negated
    return ignored


class PathIndex:
    """Bounded, short-lived filename index that never enters sensitive paths."""

    def __init__(self, cwd: str | Path, *, cap: int = 20_000, ttl: float = 10.0) -> None:
        self.cwd = Path(cwd).resolve()
        self.cap = cap
        self.ttl = ttl
        self._cache: tuple[str, ...] = ()
        self._cached_at = 0.0

    def _patterns(self) -> list[tuple[str, bool]]:
        path = self.cwd / ".gitignore"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        patterns: list[tuple[str, bool]] = []
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            negated = value.startswith("!")
            value = value[1:] if negated else value
            if value.startswith("/"):
                value = value[1:]
            if value:
                patterns.append((value, negated))
        return patterns

    def _walk(self) -> tuple[str, ...]:
        patterns = self._patterns()
        found: list[str] = []
        stack = [self.cwd]
        while stack and len(found) < self.cap:
            directory = stack.pop()
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
                except ValueError:
                    continue
                if _ignored(relative, patterns):
                    continue
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_directory:
                    found.append(relative + "/")
                    stack.append(Path(entry.path))
                else:
                    found.append(relative)
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
        del complete_event
        before = document.text_before_cursor
        if before.startswith("/"):
            yield from self._commands(document)
        elif "@" in before:
            yield from self._at_path(document)
        else:
            yield from self._bare_path(document)
        yield from self._plugins(document)


__all__ = ["ComposerCompleter", "PathIndex"]
