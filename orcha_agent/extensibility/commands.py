"""Discover and render Markdown slash-command templates."""

from __future__ import annotations

import asyncio
import os
import re
import signal
from dataclasses import dataclass, replace
from pathlib import Path

from .frontmatter import parse_frontmatter

MAX_COMMAND_BYTES = 256 * 1024
_ARGUMENT = re.compile(r"\$@\[(\d+)(?::(\d*))?\]|\$ARGUMENTS|\$@|\$(\d+)")
_SHELL = re.compile(r"!`([^`]+)`")


@dataclass(frozen=True)
class FileCommand:
    name: str
    path: Path
    body: str
    description: str
    argument_hint: str = ""
    model: str | None = None
    trusted: bool = False
    ignored_model: bool = False


def discover_commands(
    cwd: Path,
    *,
    home: Path | None = None,
    trust_cwd: bool = False,
    import_claude: bool = True,
) -> dict[str, FileCommand]:
    """Trusted user commands take precedence over all untrusted project aliases."""
    home = home or Path.home()
    roots = [
        (cwd / ".orcha-agent/commands", False, trust_cwd, cwd),
        (home / ".config/orcha-agent/commands", False, True, home),
    ]
    if import_claude:
        roots.extend(
            [
                (cwd / ".claude/commands", True, trust_cwd, cwd),
                (home / ".claude/commands", True, True, home),
            ]
        )
    if not trust_cwd:
        roots.sort(key=lambda entry: not entry[2])
    result: dict[str, FileCommand] = {}
    short_aliases: list[tuple[str, str]] = []
    trusted_names: dict[str, set[str]] = {}
    for root, recursive, trusted, scope in roots:
        try:
            resolved_root = root.resolve()
            if not resolved_root.is_relative_to(scope.resolve()):
                continue
        except (OSError, RuntimeError):
            continue
        if any(
            part == "Credentials" or part.startswith(".env")
            for path in (root, resolved_root)
            for part in path.parts
        ):
            continue
        for path in sorted(root.glob("**/*.md" if recursive else "*.md")):
            try:
                relative = path.relative_to(root)
                # Do not traverse secret directories or follow links outside the scope.
                if any(part == "Credentials" or part.startswith(".env") for part in relative.parts):
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(resolved_root):
                    continue
                if any(part == "Credentials" or part.startswith(".env") for part in resolved.parts):
                    continue
                with path.open("rb") as source:
                    data = source.read(MAX_COMMAND_BYTES + 1)
                if len(data) > MAX_COMMAND_BYTES:
                    continue
                metadata, body = parse_frontmatter(data.decode("utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            description = metadata.get("description")
            if not isinstance(description, str):
                description = next(
                    (line.strip()[:60] for line in body.splitlines() if line.strip()), ""
                )
            hint = metadata.get("argument-hint", "")
            model = metadata.get("model")
            nested = recursive and len(relative.parts) > 1
            canonical = ":".join(relative.with_suffix("").parts) if nested else path.stem
            if trusted:
                trusted_names.setdefault(path.stem, set()).add(canonical)
                if nested:
                    short_aliases.append((path.stem, canonical))
            for name in [canonical]:
                if not re.fullmatch(r"[\w:-]+", name):
                    continue
                result.setdefault(
                    name,
                    FileCommand(
                        name,
                        path,
                        body,
                        description,
                        hint if isinstance(hint, str) else "",
                        model if trusted and isinstance(model, str) and model.strip() else None,
                        trusted,
                        bool(not trusted and isinstance(model, str) and model.strip()),
                    ),
                )
    for stem, canonical in short_aliases:
        command = result.get(canonical)
        if command is None or not command.trusted or len(trusted_names[stem]) != 1:
            continue
        existing = result.get(stem)
        if existing is None or not existing.trusted:
            result[stem] = replace(command, name=stem)
    return result


def parse_command_args(arguments: str) -> list[str]:
    """OMP-compatible quoted words; backslashes are literal, not shell escapes."""
    args: list[str] = []
    word = ""
    quote: str | None = None
    for char in arguments:
        if quote is not None:
            if char == quote:
                quote = None
            else:
                word += char
        elif char in {"'", '"'}:
            quote = char
        elif char in {" ", "\t"}:
            if word:
                args.append(word)
                word = ""
        else:
            word += char
    if word:
        args.append(word)
    return args


def substitute_arguments(body: str, arguments: str, *, append_args: bool = True) -> str:
    args = parse_command_args(arguments)

    def replace(match: re.Match[str]) -> str:
        start, length, position = match.groups()
        if position is not None:
            index = int(position) - 1
            return args[index] if 0 <= index < len(args) else ""
        if start is not None:
            index = int(start) - 1
            if index < 0:
                return ""
            end = index + int(length) if length else None
            return " ".join(args[index:end])
        return " ".join(args)

    uses_args = _ARGUMENT.search(body) is not None
    rendered = _ARGUMENT.sub(replace, body)
    if append_args and not uses_args and args:
        rendered = rendered.rstrip() + "\n\n" + " ".join(args)
    return rendered


async def _shell_output(command: str, cwd: Path, timeout: float) -> str:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout):
            assert process.stdout is not None
            chunks: list[bytes] = []
            size = 0
            while chunk := await process.stdout.read(8192):
                size += len(chunk)
                if size > MAX_COMMAND_BYTES:
                    raise ValueError("Command interpolation output exceeds 256 KiB")
                chunks.append(chunk)
            await process.wait()
        if process.returncode:
            raise ValueError(f"Command interpolation failed (exit {process.returncode})")
        return b"".join(chunks).decode("utf-8", errors="replace").rstrip("\n")
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Drain the killed process's bounded pipe buffer to let its transport close.
            await process.communicate()


async def render_command(
    command: FileCommand, arguments: str, *, cwd: Path, timeout: float = 10.0
) -> str:
    # Execute only interpolation authored in the file, never interpolation supplied
    # in user arguments. Argument replacement happens after shell expansion.
    body = command.body
    matches = list(_SHELL.finditer(body))
    if matches and not command.trusted:
        raise ValueError("Shell interpolation requires a trusted command scope (--trust-cwd)")
    segments: list[str] = []
    offset = 0
    for match in matches:
        segments.append(
            substitute_arguments(body[offset : match.start()], arguments, append_args=False)
        )
        segments.append(await _shell_output(match.group(1), cwd, timeout))
        offset = match.end()
    segments.append(substitute_arguments(body[offset:], arguments, append_args=False))
    rendered = "".join(segments)
    args = parse_command_args(arguments)
    if not _ARGUMENT.search(_SHELL.sub("", body)) and args:
        rendered = rendered.rstrip() + "\n\n" + " ".join(args)
    return rendered
