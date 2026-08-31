import sqlite3
from pathlib import Path

import pytest

from orcha_agent.core.session import SessionStore


def test_session_discovery_hides_agents_but_lists_user_forks(tmp_path: Path) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        root = store.create(tmp_path, "fake:main", thread_id="root-session")
        user_fork = store.create(
            tmp_path,
            "fake:main",
            thread_id="fork-session",
            parent_session=root.thread_id,
        )
        advisor = store.create(
            tmp_path,
            "fake:advisor",
            thread_id="advisor-session",
            parent_session=user_fork.thread_id,
            kind="agent",
        )
        child = store.create(
            tmp_path,
            "fake:child",
            thread_id="child-session",
            parent_session=advisor.thread_id,
            kind="agent",
        )

        assert store.list() == [user_fork, root]
        assert store.list(include_children=True) == [child, advisor, user_fork, root]


def test_session_discovery_resolves_users_but_not_agent_sessions(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        root = store.create(tmp_path, "fake:main", thread_id="team-alpha")
        user_fork = store.create(
            tmp_path,
            "fake:main",
            thread_id="team-fork",
            parent_session=root.thread_id,
        )
        store.create(
            tmp_path,
            "fake:advisor",
            thread_id="team-agent",
            parent_session=root.thread_id,
            kind="agent",
        )
        store.create(
            tmp_path,
            "fake:child",
            thread_id="child-hidden",
            parent_session=root.thread_id,
            kind="agent",
        )

        assert store.resolve_session("team-alpha") == root
        assert store.resolve_session("team-a") == root
        assert store.resolve_session("team-fork") == user_fork
        assert store.resolve_session("team-f") == user_fork
        with pytest.raises(LookupError, match="No session matches"):
            store.resolve_session("team-agent")
        with pytest.raises(LookupError, match="No session matches"):
            store.resolve_session("child-h")


def test_internal_session_apis_still_expose_agents_for_hydration(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    with SessionStore(db_path) as store:
        root = store.create(tmp_path, "fake:main", thread_id="main-session")
        agent = store.create(
            tmp_path,
            "fake:child",
            thread_id="agent-session",
            parent_session=root.thread_id,
            kind="agent",
        )

    with SessionStore(db_path) as reopened:
        assert agent.kind == "agent"
        assert reopened.get(agent.thread_id) == agent
        assert reopened.children(root.thread_id) == [agent]
        assert agent in reopened.list(include_children=True)
        assert agent not in reopened.list()


def test_ambiguous_session_prefix_considers_user_sessions_only(
    tmp_path: Path,
) -> None:
    with SessionStore(tmp_path / "sessions.db") as store:
        root = store.create(tmp_path, "fake:main", thread_id="work-root")
        store.create(
            tmp_path,
            "fake:child",
            thread_id="work-runner",
            parent_session=root.thread_id,
            kind="agent",
        )

        assert store.resolve_session("work-r") == root

        user_fork = store.create(
            tmp_path,
            "fake:main",
            thread_id="work-research",
            parent_session=root.thread_id,
        )
        with pytest.raises(LookupError) as raised:
            store.resolve_session("work-r")

        message = str(raised.value)
        assert root.thread_id in message
        assert user_fork.thread_id in message
        assert "work-runner" not in message


def test_v1_sessions_migrate_as_user_sessions(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                thread_id TEXT PRIMARY KEY,
                cwd TEXT NOT NULL,
                model TEXT NOT NULL,
                created TEXT NOT NULL,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'ask',
                leaf_id TEXT,
                current_thread TEXT,
                parent_session TEXT
            );
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            INSERT INTO sessions(
                thread_id, cwd, model, created, title, mode,
                leaf_id, current_thread, parent_session
            ) VALUES
                ('legacy-root', '/tmp', 'fake:model', '2026-01-01T00:00:00',
                 NULL, 'ask', NULL, NULL, NULL),
                ('legacy-fork', '/tmp', 'fake:model', '2026-01-02T00:00:00',
                 NULL, 'ask', NULL, NULL, 'legacy-root');
            """
        )

    with SessionStore(db_path) as store:
        legacy_root = store.get("legacy-root")
        legacy_fork = store.get("legacy-fork")

        assert legacy_root is not None
        assert legacy_fork is not None
        assert legacy_root.kind == "user"
        assert legacy_fork.kind == "user"
        assert store.list() == [legacy_fork, legacy_root]
        assert store.resolve_session("legacy-f") == legacy_fork
