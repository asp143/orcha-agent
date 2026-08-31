from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator

import pytest

from orcha_agent.core.memory_store import (
    CredentialContentError,
    MemoryConflictError,
    MemoryDocument,
    MemoryScope,
    MemoryStore,
)


@pytest.fixture
def memory_store() -> Iterator[MemoryStore]:
    connection = sqlite3.connect(":memory:")
    try:
        yield MemoryStore(connection, threading.RLock())
    finally:
        connection.close()


def _save(
    store: MemoryStore,
    id: str,
    content: str,
    *,
    scope: MemoryScope = MemoryScope.GLOBAL,
    workspace: str | None = None,
    path: str | None = None,
) -> MemoryDocument:
    return store.save(
        MemoryDocument(
            id=id,
            content=content,
            scope=scope,
            workspace=workspace,
            path=path,
        )
    )


def test_setup_is_idempotent_and_works_with_default_sqlite_rows() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        store = MemoryStore(connection, setup=False)
        store.setup()
        store.setup()

        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "memory_documents" in tables
        assert store.all() == []
    finally:
        connection.close()


def test_scope_resolution_uses_the_most_specific_definition(
    memory_store: MemoryStore,
) -> None:
    _save(memory_store, "shared", "global")
    _save(memory_store, "global-only", "everywhere")
    _save(
        memory_store,
        "shared",
        "workspace",
        scope=MemoryScope.WORKSPACE,
        workspace="alpha",
    )
    _save(
        memory_store,
        "workspace-only",
        "alpha only",
        scope=MemoryScope.WORKSPACE,
        workspace="alpha",
    )
    _save(
        memory_store,
        "shared",
        "path",
        scope=MemoryScope.PATH,
        workspace="alpha",
        path="src",
    )

    assert [(doc.id, doc.content) for doc in memory_store.resolve()] == [
        ("global-only", "everywhere"),
        ("shared", "global"),
    ]
    assert [(doc.id, doc.content) for doc in memory_store.resolve(workspace="alpha")] == [
        ("global-only", "everywhere"),
        ("shared", "workspace"),
        ("workspace-only", "alpha only"),
    ]
    assert [
        (doc.id, doc.content, doc.scope)
        for doc in memory_store.resolve(workspace="alpha", path="src/api/client.py")
    ] == [
        ("global-only", "everywhere", MemoryScope.GLOBAL),
        ("workspace-only", "alpha only", MemoryScope.WORKSPACE),
        ("shared", "path", MemoryScope.PATH),
    ]


def test_path_resolution_prefers_the_deepest_segment_ancestor(
    memory_store: MemoryStore,
) -> None:
    for path, content in (
        (".", "root"),
        ("src", "source"),
        ("src/api", "api"),
        ("src/api/client.py", "client"),
        ("src/application", "not a segment ancestor"),
    ):
        _save(
            memory_store,
            "style",
            content,
            scope=MemoryScope.PATH,
            workspace="project",
            path=path,
        )

    assert [
        document.content
        for document in memory_store.resolve(
            workspace="project",
            path="src/api/client.py",
        )
    ] == ["client"]
    assert [
        document.content
        for document in memory_store.resolve(
            workspace="project",
            path="src/api/other.py",
        )
    ] == ["api"]
    assert [
        document.content
        for document in memory_store.resolve(
            workspace="project",
            path="src/app.py",
        )
    ] == ["source"]


def test_workspace_memories_never_leak_between_workspaces(
    memory_store: MemoryStore,
) -> None:
    _save(memory_store, "global", "global")
    _save(
        memory_store,
        "workspace",
        "alpha",
        scope=MemoryScope.WORKSPACE,
        workspace="alpha",
    )
    _save(
        memory_store,
        "workspace",
        "beta",
        scope=MemoryScope.WORKSPACE,
        workspace="beta",
    )
    _save(
        memory_store,
        "path",
        "alpha source",
        scope=MemoryScope.PATH,
        workspace="alpha",
        path="src",
    )
    _save(
        memory_store,
        "path",
        "beta source",
        scope=MemoryScope.PATH,
        workspace="beta",
        path="src",
    )

    assert [doc.content for doc in memory_store.resolve(workspace="alpha", path="src")] == [
        "global",
        "alpha",
        "alpha source",
    ]
    assert [doc.content for doc in memory_store.resolve(workspace="beta", path="src")] == [
        "global",
        "beta",
        "beta source",
    ]


def test_save_increments_revisions_and_rejects_stale_writes(
    memory_store: MemoryStore,
) -> None:
    first = _save(memory_store, "preferences", "first")
    assert first.revision == 1
    assert first.created_at
    assert first.updated_at

    second = memory_store.save(
        MemoryDocument(
            id=first.id,
            content="second",
            revision=first.revision,
        )
    )
    assert second.revision == 2
    assert second.created_at == first.created_at

    with pytest.raises(MemoryConflictError) as raised:
        memory_store.save(
            MemoryDocument(
                id=first.id,
                content="stale",
                revision=first.revision,
            )
        )

    assert raised.value.document_id == "preferences"
    assert raised.value.expected_revision == 1
    assert raised.value.actual_revision == 2
    assert memory_store.get("preferences") == second


def test_delete_persists_a_tombstone_and_can_suppress_a_fallback(
    memory_store: MemoryStore,
) -> None:
    global_document = _save(memory_store, "rules", "global rules")
    tombstone = memory_store.delete(
        "rules",
        scope=MemoryScope.WORKSPACE,
        workspace="alpha",
        expected_revision=0,
    )

    assert tombstone.deleted
    assert tombstone.content == ""
    assert tombstone.revision == 1
    assert (
        memory_store.get(
            "rules",
            scope=MemoryScope.WORKSPACE,
            workspace="alpha",
        )
        is None
    )
    assert (
        memory_store.get(
            "rules",
            scope=MemoryScope.WORKSPACE,
            workspace="alpha",
            include_deleted=True,
        )
        == tombstone
    )
    assert memory_store.resolve(workspace="alpha") == []
    assert memory_store.resolve(workspace="beta") == [global_document]

    revived = memory_store.save(
        MemoryDocument(
            id="rules",
            content="alpha rules",
            scope=MemoryScope.WORKSPACE,
            workspace="alpha",
            revision=tombstone.revision,
        )
    )
    assert not revived.deleted
    assert revived.revision == 2
    assert memory_store.resolve(workspace="alpha") == [revived]


def test_deleting_a_saved_document_checks_its_revision(
    memory_store: MemoryStore,
) -> None:
    saved = _save(memory_store, "facts", "some facts")
    updated = memory_store.save(
        MemoryDocument(id="facts", content="new facts", revision=saved.revision)
    )

    with pytest.raises(MemoryConflictError):
        memory_store.delete(saved)

    tombstone = memory_store.delete(updated)
    assert tombstone.revision == 3
    assert memory_store.all() == []
    assert memory_store.all(include_deleted=True) == [tombstone]


@pytest.mark.parametrize(
    "content",
    [
        "API_KEY=super-secret-value",
        '"access_token": "token-value-123"',
        "Authorization: Bearer abcdefghijklmnop",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret material",
        "postgresql://user:password@database.example/app",
        "Use sk-abcdefghijklmnopqrstuvwxyz for the API",
    ],
)
def test_save_rejects_content_that_looks_like_a_credential(
    memory_store: MemoryStore,
    content: str,
) -> None:
    with pytest.raises(CredentialContentError, match="credentials|secrets"):
        _save(memory_store, "unsafe", content)

    assert memory_store.all(include_deleted=True) == []


def test_credential_detector_allows_security_guidance_without_a_secret(
    memory_store: MemoryStore,
) -> None:
    saved = _save(
        memory_store,
        "security-guidance",
        "Never store passwords, API keys, access tokens, or private keys in memory.",
    )

    assert memory_store.resolve() == [saved]


def test_resolution_order_is_stable_across_insertion_order_and_reopen() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        store = MemoryStore(connection)
        _save(
            store,
            "z-path",
            "z path",
            scope=MemoryScope.PATH,
            workspace="alpha",
            path="src/api",
        )
        _save(store, "z-global", "z global")
        _save(
            store,
            "b-workspace",
            "b workspace",
            scope=MemoryScope.WORKSPACE,
            workspace="alpha",
        )
        _save(store, "a-global", "a global")
        _save(
            store,
            "a-path",
            "a path",
            scope=MemoryScope.PATH,
            workspace="alpha",
            path="src",
        )
        _save(
            store,
            "a-workspace",
            "a workspace",
            scope=MemoryScope.WORKSPACE,
            workspace="alpha",
        )

        expected = [
            "a-global",
            "z-global",
            "a-workspace",
            "b-workspace",
            "a-path",
            "z-path",
        ]
        assert [
            document.id for document in store.resolve(workspace="alpha", path="src/api/file.py")
        ] == expected

        reopened = MemoryStore(connection)
        assert [
            document.id for document in reopened.resolve(workspace="alpha", path="src/api/file.py")
        ] == expected
    finally:
        connection.close()
