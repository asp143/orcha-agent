"""Bounded, layered repository instructions loaded through plugin lifecycle hooks."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

from orcha_agent.core.config import DEFAULT_MEMORY
from orcha_agent.core.events import AgentBuildBefore, AppExit, AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec

_LOG = logging.getLogger(__name__)

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
    return not any(
        part
        in {
            "Credentials",
            ".ssh",
            ".aws",
            ".gnupg",
            ".netrc",
            ".git-credentials",
            ".npmrc",
            ".pypirc",
            ".docker",
            ".kube",
            "credentials.json",
        }
        or part.startswith((".env", "secrets."))
        or part.endswith((".pem", ".key", ".p12", ".pfx"))
        for part in path.parts
    )


def _allowed(path: Path, boundary: Path | None = None) -> bool:
    resolved = path.resolve()
    return (
        _safe(path) and _safe(resolved) and (boundary is None or resolved.is_relative_to(boundary))
    )


def ancestor_dirs(cwd: Path, *, home: Path | None = None) -> list[Path]:
    """Stop at the repository root; without a repo, never ascend above home."""
    cwd = cwd.resolve()
    home = (home or Path.home()).resolve()
    candidates = [cwd, *cwd.parents]
    for index, directory in enumerate(candidates):
        if (directory / ".git").exists():
            return candidates[: index + 1]
    if cwd.is_relative_to(home):
        return candidates[: candidates.index(home) + 1]
    return [cwd]


def _ladder(options: Mapping[str, Any]) -> list[str]:
    names = [".orcha-agent/AGENTS.md", "AGENTS.md"]
    if options.get("import_claude", True):
        names.append("CLAUDE.md")
    if options.get("import_cursor", True):
        names.append(".cursorrules")
    if options.get("import_github", True):
        names.append(".github/copilot-instructions.md")
    return names


def _pick(directory: Path, names: list[str], boundary: Path | None = None) -> Path | None:
    for name in names:
        candidate = directory / name
        if _allowed(candidate, boundary) and candidate.is_file():
            return candidate
    return None


def expand_imports(
    path: Path,
    *,
    max_bytes: int,
    max_depth: int = 5,
    trust_cwd: bool = True,
    project_root: Path | None = None,
) -> str:
    """Read with a shared I/O budget; imports inside Markdown code remain literal."""
    boundary = project_root.resolve() if project_root is not None and not trust_cwd else None
    remaining = max(0, max_bytes)
    seen: set[Path] = set()

    def read(source: Path, depth: int) -> str | None:
        nonlocal remaining
        if not _allowed(source, boundary):
            return None
        source = source.resolve()
        if source in seen or remaining <= 0:
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
                if not trust_cwd and (name.startswith("~") or Path(name).is_absolute()):
                    return match.group(0)
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


def render_context(
    cwd: Path,
    options: Mapping[str, Any],
    *,
    home: Path | None = None,
    trust_cwd: bool = True,
) -> str:
    """Select one instruction at each depth and emit bounded descendant pointers."""
    home = home or Path.home()
    cap = max(0, int(options.get("max_bytes", 65_536)))
    if cap < 32:
        return ""
    names = _ladder(options)
    ancestors = ancestor_dirs(cwd, home=home)
    # Home bounds discovery, but it never grants project files access to all of
    # home. Only an actual repository root may widen the untrusted cwd boundary.
    project_root = ancestors[-1] if (ancestors[-1] / ".git").exists() else cwd
    boundary = project_root.resolve() if not trust_cwd else None
    paths = [home / ".config/orcha-agent/AGENTS.md"]
    if options.get("import_claude", True):
        paths.append(home / ".claude/CLAUDE.md")
    home_paths = set(paths)
    paths.extend(
        path for directory in reversed(ancestors) if (path := _pick(directory, names, boundary))
    )
    opening, closing = "<repo-rules>\n", "</repo-rules>"
    fragments = [opening]
    used = len((opening + closing).encode())
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        is_home = path in home_paths
        if (
            resolved in seen
            or not _allowed(path, None if is_home else boundary)
            or not path.is_file()
        ):
            continue
        seen.add(resolved)
        prefix = f'<file path="{escape(str(path), quote=True)}">\n'
        suffix = "\n</file>\n"
        budget = cap - used - len((prefix + suffix).encode())
        if budget <= 0:
            break
        content = escape(
            expand_imports(
                path,
                max_bytes=budget,
                max_depth=max(0, int(options.get("max_import_depth", 5))),
                trust_cwd=trust_cwd or is_home,
                project_root=boundary,
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
        path = _pick(directory, names, boundary)
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
        task = asyncio.create_task(discover())

    async def discover() -> str:
        try:
            text = await asyncio.to_thread(
                render_context,
                Path(context.cfg.cwd),
                api.config,
                trust_cwd=getattr(context.cfg, "trust_cwd", False),
            )
        except Exception:
            _LOG.exception("Context file discovery failed")
            text = ""
        api.request_rebuild()
        return text

    async def build(event: AgentBuildBefore) -> None:
        nonlocal fragment
        if task is None:
            return
        # Disable legacy reads before waiting: a timeout or discovery failure must
        # never fall back to reading an untrusted root instruction symlink.
        if (
            context is not None
            and tuple(getattr(context.cfg, "memory", DEFAULT_MEMORY)) == DEFAULT_MEMORY
        ):
            event.kwargs["memory"] = []
        try:
            text = await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
        except TimeoutError:
            return
        if not fragment and text:
            fragment = text
            api.system_prompt_fragment(text, priority=20)
        prompt = event.kwargs.get("system_prompt", "")
        if text and text not in prompt:
            event.kwargs["system_prompt"] = "\n\n".join(filter(None, (prompt, text)))

    async def stop(_event: AppExit) -> None:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    api.on(AppStart, start)
    api.on(AgentBuildBefore, build)
    api.on(AppExit, stop)
