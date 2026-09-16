from __future__ import annotations

import json
import fnmatch
import os
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from pathlib import Path
import re
import stat
import tempfile
import time
import uuid
from typing import BinaryIO

DEFAULT_DENY = (
    ".env*",
    "Credentials/",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    ".ssh/id_*",
    "secrets.*",
    "credentials.json",
    "serviceAccountKey.json",
)


class PathPolicy:
    """Workspace policy shared by native tools and the artifact backend.

    Validate both the spelling and canonical target. Descriptor-relative I/O
    refuses links introduced between validation and use.
    """

    def __init__(
        self,
        cwd: Path,
        allowed_roots: Sequence[str | Path] = (),
        deny: Sequence[str] = DEFAULT_DENY,
    ) -> None:
        self.cwd = cwd.resolve()
        roots = [self.cwd]
        for value in allowed_roots:
            path = Path(value).expanduser()
            roots.append((path if path.is_absolute() else self.cwd / path).resolve())
        self.roots = tuple(dict.fromkeys(roots))
        self.deny = tuple(deny)

    def _denied(self, path: Path) -> bool:
        for root in self.roots:
            if not path.is_relative_to(root):
                continue
            parts = path.parts[1:]
            for pattern in self.deny:
                pattern = pattern.rstrip("/").casefold()
                if "/" not in pattern:
                    if any(fnmatch.fnmatchcase(part.casefold(), pattern) for part in parts):
                        return True
                elif any(
                    fnmatch.fnmatchcase("/".join(parts[start:end]).casefold(), pattern)
                    for start in range(len(parts))
                    for end in range(start + 1, len(parts) + 1)
                ):
                    return True
        return False

    def resolve(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        lexical = Path(os.path.abspath(value if value.is_absolute() else self.cwd / value))
        try:
            resolved = lexical.resolve()
        except (OSError, RuntimeError) as exc:
            raise PermissionError(f"Cannot resolve path safely: {lexical}") from exc
        if not any(lexical.is_relative_to(root) for root in self.roots) or not any(
            resolved.is_relative_to(root) for root in self.roots
        ):
            raise PermissionError(f"Path outside allowed workspace roots: {resolved}")
        if self._denied(lexical) or self._denied(resolved):
            raise PermissionError(f"Path denied by tools policy: {lexical}")
        return resolved

    def permits(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
            return True
        except (OSError, ValueError):
            return False

    @contextmanager
    def directory_fd(self, path: Path, *, create: bool = False) -> Iterator[int]:
        target = self.resolve(path)
        descriptor = os.open(target.anchor, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in target.parts[1:]:
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    def open_read(self, path: str | Path) -> BinaryIO:
        target = self.resolve(path)
        with self.directory_fd(target.parent) as parent:
            descriptor = os.open(
                target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"Not a regular file: {target}")
        return os.fdopen(descriptor, "rb")

    def read_bytes(self, path: str | Path, max_bytes: int) -> bytes:
        with self.open_read(path) as source:
            if os.fstat(source.fileno()).st_size > max_bytes:
                raise ValueError(f"File exceeds max_read_bytes={max_bytes}: {path}")
            data = source.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"File grew beyond max_read_bytes={max_bytes}: {path}")
            return data

    def stat(self, path: str | Path) -> os.stat_result:
        target = self.resolve(path)
        if target in self.roots:
            with self.directory_fd(target) as descriptor:
                return os.fstat(descriptor)
        with self.directory_fd(target.parent) as parent:
            return os.stat(target.name, dir_fd=parent, follow_symlinks=False)

    def listdir(self, path: str | Path) -> list[Path]:
        target = self.resolve(path)
        with self.directory_fd(target) as descriptor:
            return [target / name for name in os.listdir(descriptor) if self.permits(target / name)]

    def unlink(self, path: str | Path) -> None:
        target = self.resolve(path)
        if target in self.roots:
            raise PermissionError("Cannot remove a workspace root")
        with self.directory_fd(target.parent) as parent:
            os.unlink(target.name, dir_fd=parent)

    def chmod(self, path: str | Path, mode: int) -> None:
        with self.open_read(path) as stream:
            os.fchmod(stream.fileno(), stat.S_IMODE(mode))

    def mkdir(self, path: str | Path, *, mode: int = 0o755) -> Path:
        target = self.resolve(path)
        with self.directory_fd(target, create=True) as descriptor:
            os.fchmod(descriptor, mode)
        return target


def notice(message: str, **metadata: object) -> str:
    """Machine-readable metadata with a human-readable recovery instruction."""
    return "[notice] " + json.dumps({"message": message, **metadata}, ensure_ascii=False)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~])", "", text)


def resolve_path(cwd: Path, path: str) -> Path:
    return PathPolicy(cwd).resolve(path)


def atomic_write(path: Path, content: str, *, policy: PathPolicy | None = None) -> None:
    if policy is not None:
        target = policy.resolve(path)
        with policy.directory_fd(target.parent, create=True) as parent:
            try:
                existing = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(existing.st_mode):
                    raise ValueError(f"Not a regular file: {target}")
                mode = stat.S_IMODE(existing.st_mode)
            except FileNotFoundError:
                mode = 0o600
            name = f".orcha-write-{uuid.uuid4().hex}"
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.fchmod(stream.fileno(), mode)
                os.replace(name, target.name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(name, dir_fd=parent)
                except FileNotFoundError:
                    pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def ensure_artifacts(cwd: Path, policy: PathPolicy | None = None) -> Path:
    policy = policy or PathPolicy(cwd)
    private = policy.mkdir(cwd / ".orcha", mode=0o700)
    atomic_write(private / ".gitignore", "*\n", policy=policy)
    policy.mkdir(private / "artifacts", mode=0o700)
    return policy.mkdir(private / "artifacts" / "large_tool_results", mode=0o700)


def prune_artifacts(root: Path, *, count: int = 200, max_age: float = 7 * 86400) -> None:
    policy = PathPolicy(root)
    now = time.time()
    entries = sorted(
        (
            (policy.stat(path).st_mtime, path)
            for path in policy.listdir(root)
            if stat.S_ISREG(policy.stat(path).st_mode)
        ),
        reverse=True,
    )
    for index, (modified, path) in enumerate(entries):
        if index >= count or now - modified > max_age:
            try:
                policy.unlink(path)
            except FileNotFoundError:
                pass


def split_text_lines(content: str, *, keepends: bool = False) -> list[str]:
    """Address source lines by LF; strip CR only when it belongs to CRLF."""
    if not content:
        return []
    parts = content.split("\n")
    lines = [part + "\n" if keepends else part.removesuffix("\r") for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines
