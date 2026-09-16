"""Bounded, layered repository instructions loaded through plugin lifecycle hooks."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

from orcha_agent.core.config import DEFAULT_MEMORY
from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="context_files")
_SKIP = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "Credentials",
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}
_IMPORT = re.compile(r"(?<!\S)@([./~\w-][^\s]*)")


def _safe(path: Path) -> bool:
    return not any(part == "Credentials" or part.startswith(".env") for part in path.parts)


def ancestor_dirs(cwd: Path) -> list[Path]:
    """Return closest first, stopping at a repository (including worktree .git files)."""
    result = []
    current = cwd.resolve()
    while True:
        result.append(current)
        if (current / ".git").exists() or current.parent == current:
            return result
        current = current.parent


def _ladder(options: Mapping[str, Any]) -> list[str]:
    names = [".orcha-agent/AGENTS.md", "AGENTS.md"]
    if options.get("import_claude", True):
        names.append("CLAUDE.md")
    if options.get("import_cursor", True):
        names.append(".cursorrules")
    if options.get("import_github", True):
        names.append(".github/copilot-instructions.md")
    return names


def _pick(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if _safe(candidate.resolve()) and candidate.is_file():
            return candidate
    return None


def expand_imports(path: Path, *, max_bytes: int, max_depth: int = 5) -> str:
    """Read with a shared I/O budget; imports inside Markdown code remain literal."""
    remaining = max(0, max_bytes)
    seen: set[Path] = set()

    def read(source: Path, depth: int) -> str | None:
        nonlocal remaining
        source = source.resolve()
        if source in seen or not _safe(source) or remaining <= 0:
            return None
        try:
            with source.open("rb") as handle:
                data = handle.read(remaining)
        except OSError:
            return None
        seen.add(source)
        remaining -= len(data)
        content = data.decode("utf-8", errors="replace")
        if depth >= max_depth:
            return content
        output: list[str] = []
        fence: str | None = None
        inline_ticks: int | None = None
        for line in content.splitlines(keepends=True):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker and inline_ticks is None:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                output.append(line)
                continue
            if fence:
                output.append(line)
                continue

            def replace(match: re.Match[str]) -> str:
                token = match.group(1)
                name = token.rstrip(".,;:!?)]}\"'")
                target = Path(name).expanduser()
                if not target.is_absolute():
                    target = source.parent / target
                expanded = read(target, depth + 1)
                return match.group(0) if expanded is None else expanded + token[len(name) :]

            # Backtick runs close only matching runs, including across lines.
            for part in re.split(r"(`+)", line):
                if part.startswith("`"):
                    if inline_ticks is None:
                        inline_ticks = len(part)
                    elif len(part) == inline_ticks:
                        inline_ticks = None
                    output.append(part)
                else:
                    output.append(part if inline_ticks is not None else _IMPORT.sub(replace, part))
        return "".join(output)

    return read(path, 0) or ""


def render_context(cwd: Path, options: Mapping[str, Any], *, home: Path | None = None) -> str:
    """Select one instruction at each depth and emit bounded descendant pointers."""
    home = home or Path.home()
    cap = max(0, int(options.get("max_bytes", 65_536)))
    if cap < 32:
        return ""
    names = _ladder(options)
    ancestors = ancestor_dirs(cwd)
    paths = [home / ".config/orcha-agent/AGENTS.md"]
    if options.get("import_claude", True):
        paths.append(home / ".claude/CLAUDE.md")
    paths.extend(path for directory in reversed(ancestors) if (path := _pick(directory, names)))
    opening, closing = "<repo-rules>\n", "</repo-rules>"
    fragments = [opening]
    used = len((opening + closing).encode())
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file() or not _safe(resolved):
            continue
        seen.add(resolved)
        prefix = f'<file path="{escape(str(path), quote=True)}">\n'
        suffix = "\n</file>\n"
        budget = cap - used - len((prefix + suffix).encode())
        if budget <= 0:
            break
        content = escape(
            expand_imports(
                path, max_bytes=budget, max_depth=max(0, int(options.get("max_import_depth", 5)))
            )
        )
        # Escape first and keep complete entities/Unicode characters within the cap.
        while len(content.encode()) > budget:
            content = content[: max(0, len(content) - max(1, (len(content.encode()) - budget)))]
        if content.rfind("&") > content.rfind(";"):
            content = content[: content.rfind("&")]
        fragment = prefix + content + suffix
        fragments.append(fragment)
        used += len(fragment.encode())
    # Pointer-only discovery: never open descendant instruction bodies.
    count = 0
    for root, dirs, _files in os.walk(cwd, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith(".env"))
        count += 1
        if count > 4096:
            break
        directory = Path(root)
        if directory.resolve() == cwd.resolve():
            continue
        path = _pick(directory, names)
        if path is None or path.resolve() in seen:
            continue
        pointer = f'<dir-context path="{escape(str(path), quote=True)}" />\n'
        if used + len(pointer.encode()) > cap:
            break
        fragments.append(pointer)
        used += len(pointer.encode())
    fragments.append(closing)
    return "".join(fragments)


def register(api: PluginAPI) -> None:
    task: asyncio.Task[str] | None = None
    context: Any = None
    fragment = ""

    async def start(event: AppStart) -> None:
        nonlocal task, context
        context = event.ctx
        if getattr(getattr(context, "cfg", None), "cwd", None) is None:
            return
        if (
            not api.config.get("enabled", True)
            or getattr(getattr(context.cfg, "memory_store", None), "backend", "files") == "turso"
        ):
            return
        task = asyncio.create_task(
            asyncio.to_thread(render_context, Path(context.cfg.cwd), api.config)
        )

    async def build(event: AgentBuildBefore) -> None:
        nonlocal fragment
        if task is None:
            return
        text = await asyncio.shield(task)
        if not fragment and text:
            fragment = text
            api.system_prompt_fragment(text, priority=20)
        prompt = event.kwargs.get("system_prompt", "")
        if text and text not in prompt:
            event.kwargs["system_prompt"] = "\n\n".join(filter(None, (prompt, text)))
        # The ladder replaces legacy default root-file loading, but explicit
        # custom memory sources and structured-memory backend semantics survive.
        if (
            context is not None
            and tuple(getattr(context.cfg, "memory", DEFAULT_MEMORY)) == DEFAULT_MEMORY
        ):
            event.kwargs["memory"] = []

    async def stop(_event: AppExit) -> None:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    api.on(AppStart, start)
    api.on(AgentBuildBefore, build)
    api.on(AppExit, stop)
