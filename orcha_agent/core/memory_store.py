"""Structured, revisioned memory over a SQLite-compatible connection.

The store deliberately does not own the connection.  This keeps the memory
model independent from session persistence and lets callers share a connection
and lock with another SQLite-compatible component.
"""

from __future__ import annotations

import posixpath
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from os import PathLike
from typing import Any, Protocol, Self, runtime_checkable


class MemoryScope(StrEnum):
    """The context in which a memory document applies."""

    GLOBAL = "global"
    WORKSPACE = "workspace"
    PATH = "path"


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryDocument:
    """A single named memory document.

    ``id`` is a logical name, not a globally unique row identifier.  Defining
    the same id at a more specific scope overrides the less-specific document.
    Revisions start at zero before the first save and increase on every write,
    including deletion.
    """

    id: str
    content: str
    scope: MemoryScope | str = MemoryScope.GLOBAL
    workspace: str | None = None
    path: str | PathLike[str] | None = None
    revision: int = 0
    deleted: bool = False
    created_at: str = ""
    updated_at: str = ""

    @property
    def document_id(self) -> str:
        """Return the logical document id (an explicit readability alias)."""

        return self.id

    @property
    def tombstone(self) -> bool:
        """Whether this revision represents a deletion."""

        return self.deleted

    @classmethod
    def global_document(cls, id: str, content: str) -> Self:
        return cls(id=id, content=content)

    @classmethod
    def workspace_document(cls, id: str, content: str, workspace: str) -> Self:
        return cls(
            id=id,
            content=content,
            scope=MemoryScope.WORKSPACE,
            workspace=workspace,
        )

    @classmethod
    def path_document(
        cls,
        id: str,
        content: str,
        workspace: str,
        path: str | PathLike[str],
    ) -> Self:
        return cls(
            id=id,
            content=content,
            scope=MemoryScope.PATH,
            workspace=workspace,
            path=path,
        )


class MemoryConflictError(RuntimeError):
    """Raised when a write is based on a stale or nonexistent revision."""

    def __init__(
        self,
        document_id: str,
        expected_revision: int,
        actual_revision: int | None,
    ) -> None:
        self.document_id = document_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        actual = "missing" if actual_revision is None else str(actual_revision)
        super().__init__(
            f"Memory document {document_id!r} revision conflict: "
            f"expected {expected_revision}, found {actual}"
        )


# A descriptive alias is useful to callers that expose revision semantics in
# their own API.  It is the same exception type, so either name can be caught.
RevisionConflictError = MemoryConflictError


class CredentialContentError(ValueError):
    """Raised when content appears to contain a secret or credential."""

    def __init__(self) -> None:
        super().__init__("Memory content must not contain credentials or secrets")


@runtime_checkable
class CursorLike(Protocol):
    """The cursor surface required by :class:`MemoryStore`."""

    def fetchone(self) -> Any: ...

    def fetchall(self) -> Sequence[Any]: ...


@runtime_checkable
class ConnectionLike(Protocol):
    """A minimal DB-API connection implemented by sqlite3 and libSQL clients."""

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> CursorLike: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


LockLike = AbstractContextManager[Any]


_TABLE = "memory_documents"
_COLUMNS = (
    "document_id",
    "scope",
    "workspace",
    "path",
    "content",
    "revision",
    "deleted",
    "created_at",
    "updated_at",
)
_SELECT_COLUMNS = ", ".join(_COLUMNS)

_CREDENTIAL_PATTERNS = tuple(
    re.compile(pattern, flags)
    for pattern, flags in (
        (r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE),
        (
            r"\b(?:Proxy-)?Authorization['\"]?\s*:\s*['\"]?"
            r"(?:Basic|Bearer)\s+[A-Za-z0-9._~+/=-]{8,}",
            re.IGNORECASE,
        ),
        (r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{8,}\b", re.IGNORECASE),
        (r"\bsk-[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
        (r"\bgh[opusr]_[A-Za-z0-9]{16,}\b", 0),
        (r"\bAKIA[0-9A-Z]{16}\b", 0),
        (
            r"(?im)(?:^|[,{;\s])['\"]?"
            r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
            r"client[_-]?secret|secret[_-]?key|password|passwd)"
            r"['\"]?\s*(?:=|:)\s*['\"]?[^\s,'\";}]{4,}",
            0,
        ),
        (r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@[^\s]+", re.IGNORECASE),
    )
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _row_value(row: Any, column: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[column]
    try:
        return row[column]
    except (IndexError, TypeError):
        return row[index]


def _scope(value: MemoryScope | str) -> MemoryScope:
    try:
        return MemoryScope(value)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(scope.value for scope in MemoryScope)
        raise ValueError(f"Memory scope must be one of: {allowed}") from error


def _workspace(value: str | PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    if not normalized or normalized != normalized.strip():
        raise ValueError("Memory workspace must be a non-empty, trimmed string")
    if "\x00" in normalized:
        raise ValueError("Memory workspace must not contain NUL characters")
    return normalized


def _path(value: str | PathLike[str] | None) -> str | None:
    if value is None:
        return None
    raw = str(value).replace("\\", "/")
    if not raw or raw != raw.strip():
        raise ValueError("Memory path must be a non-empty, trimmed path")
    if "\x00" in raw:
        raise ValueError("Memory path must not contain NUL characters")
    normalized = posixpath.normpath(raw)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


def _validate_revision(revision: int) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Memory revision must be a non-negative integer")
    return revision


def _normalize_document(document: MemoryDocument) -> MemoryDocument:
    if not isinstance(document, MemoryDocument):
        raise TypeError("document must be a MemoryDocument")
    if not document.id or document.id != document.id.strip():
        raise ValueError("Memory document id must be a non-empty, trimmed string")
    if "\x00" in document.id:
        raise ValueError("Memory document id must not contain NUL characters")
    if not isinstance(document.content, str):
        raise TypeError("Memory document content must be a string")

    scope = _scope(document.scope)
    workspace = _workspace(document.workspace)
    path = _path(document.path)
    revision = _validate_revision(document.revision)

    if scope is MemoryScope.GLOBAL:
        if workspace is not None or path is not None:
            raise ValueError("Global memory must not specify a workspace or path")
    elif scope is MemoryScope.WORKSPACE:
        if workspace is None or path is not None:
            raise ValueError("Workspace memory requires a workspace and no path")
    elif workspace is None or path is None:
        raise ValueError("Path memory requires both a workspace and path")

    return replace(
        document,
        scope=scope,
        workspace=workspace,
        path=path,
        revision=revision,
    )


def _contains_credential(content: str) -> bool:
    return any(pattern.search(content) is not None for pattern in _CREDENTIAL_PATTERNS)


def _document_from_row(row: Any) -> MemoryDocument:
    values = {column: _row_value(row, column, index) for index, column in enumerate(_COLUMNS)}
    return MemoryDocument(
        id=str(values["document_id"]),
        content=str(values["content"]),
        scope=MemoryScope(str(values["scope"])),
        workspace=str(values["workspace"]) or None,
        path=str(values["path"]) or None,
        revision=int(values["revision"]),
        deleted=bool(values["deleted"]),
        created_at=str(values["created_at"]),
        updated_at=str(values["updated_at"]),
    )


def _address(document: MemoryDocument) -> tuple[str, str, str, str]:
    return (
        document.id,
        str(document.scope),
        document.workspace or "",
        str(document.path) if document.path is not None else "",
    )


def _path_parts(path: str) -> tuple[str, ...]:
    if path == ".":
        return ()
    return tuple(part for part in path.split("/") if part and part != ".")


def _path_applies(candidate: str, target: str) -> bool:
    candidate_absolute = candidate.startswith("/")
    target_absolute = target.startswith("/")
    if candidate_absolute != target_absolute:
        return False
    candidate_parts = _path_parts(candidate)
    target_parts = _path_parts(target)
    return target_parts[: len(candidate_parts)] == candidate_parts


def _precedence(document: MemoryDocument) -> tuple[int, int, str, str]:
    if document.scope is MemoryScope.GLOBAL:
        return (0, 0, "", document.id)
    if document.scope is MemoryScope.WORKSPACE:
        return (1, 0, "", document.id)
    path = str(document.path)
    return (2, len(_path_parts(path)), path, document.id)


class MemoryStore:
    """Persist and resolve scoped memory documents.

    Resolution is deterministic.  A path document overrides a workspace
    document with the same id, which overrides a global document.  Among path
    documents, the deepest matching ancestor wins.  Returned documents are
    ordered from least-specific to most-specific, then by id, so callers that
    concatenate them give the strongest instructions the final position.
    Tombstones participate in override selection but are omitted from results.
    """

    def __init__(
        self,
        connection: ConnectionLike,
        lock: LockLike | None = None,
        *,
        setup: bool = True,
    ) -> None:
        if not isinstance(connection, ConnectionLike):
            raise TypeError("connection must provide execute(), commit(), and rollback()")
        self.connection = connection
        self.lock: LockLike = lock if lock is not None else threading.RLock()
        if setup:
            self.setup()

    def setup(self) -> None:
        """Create the idempotent memory schema on the supplied connection."""

        with self.lock, self._transaction():
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    document_id TEXT NOT NULL,
                    scope TEXT NOT NULL
                        CHECK (scope IN ('global', 'workspace', 'path')),
                    workspace TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    deleted INTEGER NOT NULL DEFAULT 0
                        CHECK (deleted IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, scope, workspace, path)
                )
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS memory_documents_resolution
                ON {_TABLE}(workspace, scope, path, document_id)
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Isolate one operation without committing a caller-owned transaction."""

        savepoint = "orcha_memory_store"
        self.connection.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except BaseException:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")

    def _get_row(self, document: MemoryDocument) -> Any:
        return self.connection.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM {_TABLE}
            WHERE document_id = ? AND scope = ? AND workspace = ? AND path = ?
            """,
            _address(document),
        ).fetchone()

    def save(
        self,
        document: MemoryDocument,
        *,
        expected_revision: int | None = None,
    ) -> MemoryDocument:
        """Create or replace a document using optimistic revision checking.

        A new document has expected revision zero and is returned at revision
        one.  When ``expected_revision`` is omitted, ``document.revision`` is
        used.  Saving a tombstone directly is disallowed; use :meth:`delete`.
        """

        document = _normalize_document(document)
        expected = _validate_revision(
            document.revision if expected_revision is None else expected_revision
        )
        if document.deleted:
            raise ValueError("Use delete() to create a memory tombstone")
        if _contains_credential(document.content):
            raise CredentialContentError()

        with self.lock, self._transaction():
            row = self._get_row(document)
            current = None if row is None else _document_from_row(row)
            actual = None if current is None else current.revision
            if (current is None and expected != 0) or (current is not None and actual != expected):
                raise MemoryConflictError(document.id, expected, actual)

            now = _timestamp()
            revision = expected + 1
            if current is None:
                created_at = now
                self.connection.execute(
                    f"""
                    INSERT INTO {_TABLE}(
                        document_id, scope, workspace, path, content,
                        revision, deleted, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (*_address(document), document.content, revision, now, now),
                )
            else:
                created_at = current.created_at
                self.connection.execute(
                    f"""
                    UPDATE {_TABLE}
                    SET content = ?, revision = ?, deleted = 0, updated_at = ?
                    WHERE document_id = ? AND scope = ?
                        AND workspace = ? AND path = ? AND revision = ?
                    """,
                    (
                        document.content,
                        revision,
                        now,
                        *_address(document),
                        expected,
                    ),
                )

        return replace(
            document,
            revision=revision,
            deleted=False,
            created_at=created_at,
            updated_at=now,
        )

    def delete(
        self,
        document: MemoryDocument | str,
        *,
        scope: MemoryScope | str = MemoryScope.GLOBAL,
        workspace: str | None = None,
        path: str | PathLike[str] | None = None,
        expected_revision: int | None = None,
    ) -> MemoryDocument:
        """Write a tombstone after checking the expected current revision.

        Passing a document infers its address and revision.  Passing an id is
        also supported, but requires ``expected_revision`` explicitly.  An
        expected revision of zero can create a tombstone at a new, more
        specific scope, suppressing a less-specific document during resolve.
        """

        if isinstance(document, MemoryDocument):
            if scope != MemoryScope.GLOBAL or workspace is not None or path is not None:
                raise ValueError("Scope arguments cannot accompany a MemoryDocument")
            normalized = _normalize_document(document)
            expected = _validate_revision(
                normalized.revision if expected_revision is None else expected_revision
            )
        elif isinstance(document, str):
            if expected_revision is None:
                raise TypeError("expected_revision is required when deleting by id")
            normalized = _normalize_document(
                MemoryDocument(
                    id=document,
                    content="",
                    scope=scope,
                    workspace=workspace,
                    path=path,
                )
            )
            expected = _validate_revision(expected_revision)
        else:
            raise TypeError("document must be a MemoryDocument or document id")

        with self.lock, self._transaction():
            row = self._get_row(normalized)
            current = None if row is None else _document_from_row(row)
            actual = None if current is None else current.revision
            if (current is None and expected != 0) or (current is not None and actual != expected):
                raise MemoryConflictError(normalized.id, expected, actual)

            now = _timestamp()
            revision = expected + 1
            if current is None:
                created_at = now
                self.connection.execute(
                    f"""
                    INSERT INTO {_TABLE}(
                        document_id, scope, workspace, path, content,
                        revision, deleted, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, '', ?, 1, ?, ?)
                    """,
                    (*_address(normalized), revision, now, now),
                )
            else:
                created_at = current.created_at
                self.connection.execute(
                    f"""
                    UPDATE {_TABLE}
                    SET content = '', revision = ?, deleted = 1, updated_at = ?
                    WHERE document_id = ? AND scope = ?
                        AND workspace = ? AND path = ? AND revision = ?
                    """,
                    (revision, now, *_address(normalized), expected),
                )

        return replace(
            normalized,
            content="",
            revision=revision,
            deleted=True,
            created_at=created_at,
            updated_at=now,
        )

    def get(
        self,
        document_id: str,
        *,
        scope: MemoryScope | str = MemoryScope.GLOBAL,
        workspace: str | None = None,
        path: str | PathLike[str] | None = None,
        include_deleted: bool = False,
    ) -> MemoryDocument | None:
        """Get the current revision at one exact scope address."""

        address = _normalize_document(
            MemoryDocument(
                id=document_id,
                content="",
                scope=scope,
                workspace=workspace,
                path=path,
            )
        )
        with self.lock:
            row = self._get_row(address)
        if row is None:
            return None
        document = _document_from_row(row)
        if document.deleted and not include_deleted:
            return None
        return document

    def all(self, *, include_deleted: bool = False) -> list[MemoryDocument]:
        """Return every stored document in stable address order."""

        with self.lock:
            rows = self.connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM {_TABLE}
                ORDER BY scope, workspace, path, document_id
                """
            ).fetchall()
        documents = [_document_from_row(row) for row in rows]
        if not include_deleted:
            documents = [document for document in documents if not document.deleted]
        return documents

    def resolve(
        self,
        *,
        workspace: str | None = None,
        path: str | PathLike[str] | None = None,
    ) -> list[MemoryDocument]:
        """Resolve the effective memory for a workspace and optional path."""

        normalized_workspace = _workspace(workspace)
        normalized_path = _path(path)
        if normalized_path is not None and normalized_workspace is None:
            raise ValueError("Resolving a path requires a workspace")

        with self.lock:
            rows = self.connection.execute(f"SELECT {_SELECT_COLUMNS} FROM {_TABLE}").fetchall()

        applicable: list[MemoryDocument] = []
        for row in rows:
            document = _document_from_row(row)
            if document.scope is MemoryScope.GLOBAL:
                applicable.append(document)
            elif (
                normalized_workspace is not None
                and document.workspace == normalized_workspace
                and document.scope is MemoryScope.WORKSPACE
            ):
                applicable.append(document)
            elif (
                normalized_workspace is not None
                and normalized_path is not None
                and document.workspace == normalized_workspace
                and document.scope is MemoryScope.PATH
                and document.path is not None
                and _path_applies(str(document.path), normalized_path)
            ):
                applicable.append(document)

        # Iterating weak-to-strong means assignment naturally retains the most
        # specific definition of each logical document id.
        winners: dict[str, MemoryDocument] = {}
        for document in sorted(applicable, key=_precedence):
            winners[document.id] = document
        return sorted(
            (document for document in winners.values() if not document.deleted),
            key=_precedence,
        )


__all__ = [
    "ConnectionLike",
    "CredentialContentError",
    "CursorLike",
    "MemoryConflictError",
    "MemoryDocument",
    "MemoryScope",
    "MemoryStore",
    "RevisionConflictError",
]
