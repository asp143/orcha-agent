"""Contain the deepagents operations retained beneath the native tool layer."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import DeleteResult, FileData, ReadResult, WriteResult
from deepagents.backends.utils import slice_read_response

from .common import PathPolicy, atomic_write, ensure_artifacts, prune_artifacts


class GuardedLocalShellBackend(LocalShellBackend):
    """Use native path spelling with explicit containment and no-follow I/O.

    The inherited shell is never exposed in native mode. File operations kept by
    deepagents (delete, memory and overflow) share the native policy.
    """

    def __init__(self, policy: PathPolicy, max_read_bytes: int = 64 * 1024 * 1024):
        super().__init__(root_dir=policy.cwd, virtual_mode=True, inherit_env=False)
        self.policy = policy
        self.max_read_bytes = max_read_bytes

    def _resolve_path(self, key: str) -> Path:
        return self.policy.resolve(key)

    def _display_path(self, path: Path) -> str:
        return str(self.policy.resolve(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            data = self.policy.read_bytes(file_path, self.max_read_bytes)
            return slice_read_response(
                FileData(content=data.decode("utf-8"), encoding="utf-8"), offset, limit
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return ReadResult(error=f"Error reading file: {exc}")

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            target = self.policy.resolve(file_path)
            private = self.policy.cwd / ".orcha" / "artifacts"
            if target.is_relative_to(private):
                ensure_artifacts(self.policy.cwd, self.policy)
            atomic_write(target, content, policy=self.policy)
            if target.is_relative_to(private):
                self.policy.chmod(target, 0o600)
                root = private / "large_tool_results"
                if target.is_relative_to(root):
                    prune_artifacts(root)
            return WriteResult(path=file_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return WriteResult(error=f"Error writing file: {exc}")

    def _validate_delete(self, target: Path) -> None:
        self.policy.resolve(target)
        if target in self.policy.roots:
            raise PermissionError("Cannot delete a workspace or allowed root")
        with self.policy.directory_fd(target.parent) as parent:
            metadata = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            with self.policy.directory_fd(target) as descriptor:
                # Do not filter: a denied descendant rejects the entire request.
                for name in os.listdir(descriptor):
                    self._validate_delete(target / name)
        elif not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"Refusing to delete a special file or symlink: {target}")

    def _remove(self, target: Path) -> None:
        self._validate_delete(target)
        if stat.S_ISDIR(self.policy.stat(target).st_mode):
            with self.policy.directory_fd(target) as descriptor:
                for name in os.listdir(descriptor):
                    self._remove(target / name)
            with self.policy.directory_fd(target.parent) as parent:
                os.rmdir(target.name, dir_fd=parent)
        else:
            with self.policy.directory_fd(target.parent) as parent:
                os.unlink(target.name, dir_fd=parent)

    def delete(self, file_path: str) -> DeleteResult:
        try:
            target = self.policy.resolve(file_path)
            self._validate_delete(target)
            self._remove(target)
            return DeleteResult(path=file_path)
        except (OSError, ValueError) as exc:
            return DeleteResult(error=f"Error deleting file: {exc}")
