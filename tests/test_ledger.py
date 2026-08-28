import errno
import json
import re
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    message_to_dict,
)

from orcha_agent.core.export import (
    entry_from_envelope,
    entry_to_envelope,
    export_session,
)
from orcha_agent.core.ledger import (
    AmbiguousEntry,
    CompactionEntry,
    CustomEntry,
    EntryNotFound,
    Ledger,
    LedgerCycleError,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
    OpaqueEntry,
    ResetBoundaryEntry,
    ToolCallRef,
    build_context,
)
from orcha_agent.core.session import SessionStore


@pytest.fixture
def ledger_session(
    tmp_path: Path,
) -> Iterator[tuple[Ledger, SessionStore, str]]:
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "fake:model", thread_id="source-session")
        yield Ledger(store), store, session.thread_id


def _message(message: BaseMessage, **fields: Any) -> MessageEntry:
    return MessageEntry(message=message_to_dict(message), **fields)


def test_append_links_to_the_current_leaf_and_advances_it(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session

    first = ledger.append(session_id, _message(HumanMessage(content="first")))
    second = ledger.append(session_id, _message(AIMessage(content="second")))

    assert re.fullmatch(r"[0-9a-f]{8}", first.id)
    assert re.fullmatch(r"[0-9a-f]{8}", second.id)
    assert first.parent_id is None
    assert second.parent_id == first.id
    assert ledger.leaf(session_id) == second.id


def test_append_many_preserves_order_builds_one_parent_chain_and_sets_final_leaf(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))

    appended = ledger.append_many(
        session_id,
        [
            ModelChangeEntry(model=["fake:primary", "fake:fallback"]),
            ModeChangeEntry(mode="plan"),
            _message(AIMessage(content="done")),
        ],
    )

    assert [entry.parent_id for entry in appended] == [
        root.id,
        appended[0].id,
        appended[1].id,
    ]
    assert ledger.all(session_id) == [root, *appended]
    assert ledger.leaf(session_id) == appended[-1].id


def test_append_many_rolls_back_every_entry_and_the_leaf_on_storage_failure(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    store._connection.execute(
        """
        CREATE TRIGGER reject_mode_change
        BEFORE INSERT ON entries
        WHEN NEW.type = 'mode_change'
        BEGIN
            SELECT RAISE(ABORT, 'forced append failure');
        END
        """
    )
    store._connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced append failure"):
        ledger.append_many(
            session_id,
            [
                _message(AIMessage(content="must roll back")),
                ModeChangeEntry(mode="plan"),
            ],
        )

    assert ledger.all(session_id) == [root]
    assert ledger.count(session_id) == 1
    assert ledger.leaf(session_id) == root.id


def test_branch_creates_divergent_paths_without_deleting_the_old_branch(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    original = ledger.append(session_id, _message(AIMessage(content="original")))

    ledger.branch(session_id, root.id)
    alternate = ledger.append(session_id, _message(AIMessage(content="alternate")))

    assert original.parent_id == root.id
    assert alternate.parent_id == root.id
    assert ledger.path(session_id, original.id) == [root, original]
    assert ledger.path(session_id) == [root, alternate]
    assert ledger.all(session_id) == [root, original, alternate]
    assert ledger.get(session_id, original.id) == original
    assert ledger.count(session_id) == 3
    assert ledger.leaf(session_id) == alternate.id


def test_set_position_updates_leaf_and_active_thread(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    child = ledger.append(session_id, _message(AIMessage(content="child")))
    replacement_thread = store.create_thread(session_id)

    ledger.set_position(
        session_id,
        leaf_id=root.id,
        thread_id=replacement_thread.thread_id,
    )

    session = store.get(session_id)
    assert session is not None
    assert ledger.leaf(session_id) == root.id
    assert session.current_thread == replacement_thread.thread_id
    assert ledger.get(session_id, child.id) == child


def test_set_position_rollback_discards_newest_orphan_and_restores_position(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    original_thread = store.get(session_id)
    assert original_thread is not None
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    rollback_entry = ledger.append(
        session_id,
        ResetBoundaryEntry(),
    )
    pending_thread = store.create_thread(session_id)
    store.set_current_thread(session_id, pending_thread.thread_id)

    ledger.set_position(
        session_id,
        leaf_id=root.id,
        thread_id=original_thread.current_thread,
        discard_entry_id=rollback_entry.id,
    )

    restored = store.get(session_id)
    assert restored is not None
    assert ledger.leaf(session_id) == root.id
    assert restored.current_thread == original_thread.current_thread
    assert ledger.get(session_id, rollback_entry.id) is None
    assert ledger.all(session_id) == [root]


def test_set_position_rejects_discarding_an_older_orphan_atomically(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    older_orphan = ledger.append(
        session_id,
        _message(AIMessage(content="abandoned")),
    )
    ledger.branch(session_id, root.id)
    newest = ledger.append(session_id, _message(AIMessage(content="active")))
    pending_thread = store.create_thread(session_id)
    store.set_current_thread(session_id, pending_thread.thread_id)

    with pytest.raises(ValueError):
        ledger.set_position(
            session_id,
            leaf_id=root.id,
            thread_id=f"{session_id}.0",
            discard_entry_id=older_orphan.id,
        )

    unchanged = store.get(session_id)
    assert unchanged is not None
    assert ledger.leaf(session_id) == newest.id
    assert unchanged.current_thread == pending_thread.thread_id
    assert ledger.get(session_id, older_orphan.id) == older_orphan
    assert ledger.all(session_id) == [root, older_orphan, newest]


def test_set_position_rejects_discarding_the_restored_leaf_atomically(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    leaf = ledger.append(session_id, _message(HumanMessage(content="kept")))
    pending_thread = store.create_thread(session_id)
    store.set_current_thread(session_id, pending_thread.thread_id)

    with pytest.raises(ValueError):
        ledger.set_position(
            session_id,
            leaf_id=leaf.id,
            thread_id=f"{session_id}.0",
            discard_entry_id=leaf.id,
        )

    unchanged = store.get(session_id)
    assert unchanged is not None
    assert ledger.leaf(session_id) == leaf.id
    assert unchanged.current_thread == pending_thread.thread_id
    assert ledger.get(session_id, leaf.id) == leaf
    assert ledger.all(session_id) == [leaf]


def test_get_leaf_all_and_count_report_persisted_entries(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session

    assert ledger.get(session_id, "ffffffff") is None
    assert ledger.leaf(session_id) is None
    assert ledger.all(session_id) == []
    assert ledger.count(session_id) == 0

    entry = ledger.append(session_id, ModelChangeEntry(model="fake:new"))

    assert ledger.get(session_id, entry.id) == entry
    assert ledger.leaf(session_id) == entry.id
    assert ledger.all(session_id) == [entry]
    assert ledger.count(session_id) == 1


def test_path_returns_entries_in_root_to_requested_leaf_order(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    middle = ledger.append(session_id, _message(AIMessage(content="middle")))
    tip = ledger.append(session_id, _message(HumanMessage(content="tip")))

    assert ledger.path(session_id) == [root, middle, tip]
    assert ledger.path(session_id, middle.id) == [root, middle]


def test_fork_copies_only_the_active_path_with_ids_and_parents_verbatim(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
) -> None:
    ledger, store, source_id = ledger_session
    root = ledger.append(source_id, _message(HumanMessage(content="root")))
    abandoned = ledger.append(source_id, _message(AIMessage(content="abandoned")))
    ledger.branch(source_id, root.id)
    active = ledger.append(source_id, _message(AIMessage(content="active")))
    target = store.create(tmp_path, "fake:model", thread_id="forked-session")

    ledger.fork(source_id, target.thread_id)

    assert ledger.all(target.thread_id) == [root, active]
    assert [entry.id for entry in ledger.all(target.thread_id)] == [root.id, active.id]
    assert [entry.parent_id for entry in ledger.all(target.thread_id)] == [
        None,
        root.id,
    ]
    assert ledger.get(target.thread_id, abandoned.id) is None
    assert ledger.leaf(target.thread_id) == active.id


def test_resolve_returns_an_exact_entry_id_match(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    entry = ledger.append(session_id, _message(HumanMessage(content="entry")))

    assert ledger.resolve(session_id, entry.id) == entry


def test_resolve_returns_the_only_entry_matching_a_proper_prefix(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    entry = ledger.append(session_id, _message(HumanMessage(content="entry")))

    assert ledger.resolve(session_id, entry.id[:4]) == entry


def test_resolve_rejects_an_ambiguous_prefix(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    entries = [
        ledger.append(session_id, _message(HumanMessage(content=f"entry {index}")))
        for index in range(17)
    ]
    ambiguous_prefix = next(
        prefix
        for prefix in "0123456789abcdef"
        if sum(entry.id.startswith(prefix) for entry in entries) > 1
    )

    with pytest.raises(AmbiguousEntry):
        ledger.resolve(session_id, ambiguous_prefix)


def test_resolve_rejects_a_missing_prefix(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    ledger.append(session_id, _message(HumanMessage(content="entry")))

    with pytest.raises(EntryNotFound):
        ledger.resolve(session_id, "not-an-entry")


def test_path_raises_cycle_error_for_a_corrupt_parent_chain(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, store, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    child = ledger.append(session_id, _message(AIMessage(content="child")))
    store._connection.execute(
        "UPDATE entries SET parent_id = ? WHERE session_id = ? AND id = ?",
        (child.id, session_id, root.id),
    )
    store._connection.commit()

    with pytest.raises(LedgerCycleError):
        ledger.path(session_id)


def test_build_context_discards_everything_through_the_last_reset() -> None:
    path = [
        _message(HumanMessage(content="before first reset")),
        ResetBoundaryEntry(),
        _message(HumanMessage(content="between resets")),
        ModelChangeEntry(model="discarded:model"),
        ResetBoundaryEntry(),
        _message(HumanMessage(content="after last reset")),
        ModelChangeEntry(model="kept:model"),
    ]

    context = build_context(path)

    assert context.messages == [HumanMessage(content="after last reset")]
    assert context.model == "kept:model"
    assert context.compacted is False


def test_build_context_with_null_first_kept_starts_after_compaction() -> None:
    path = [
        _message(HumanMessage(content="old human")),
        _message(AIMessage(content="old assistant")),
        CompactionEntry(
            summary="Facts retained from the old conversation.",
            first_kept_id=None,
            tokens_before=1234,
        ),
        _message(HumanMessage(content="new human")),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(
            content="[Conversation summary]\nFacts retained from the old conversation."
        ),
        HumanMessage(content="new human"),
    ]
    assert context.compacted is True


def test_build_context_with_first_kept_id_starts_after_that_entry() -> None:
    path = [
        _message(HumanMessage(content="discarded"), id="00000001"),
        _message(HumanMessage(content="first-kept marker"), id="00000002"),
        _message(HumanMessage(content="retained after marker"), id="00000003"),
        CompactionEntry(
            id="00000004",
            summary="Earlier work was summarized.",
            first_kept_id="00000002",
            tokens_before=800,
        ),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(content="[Conversation summary]\nEarlier work was summarized."),
        HumanMessage(content="retained after marker"),
    ]
    assert context.compacted is True


def test_build_context_uses_latest_compaction_when_its_kept_marker_is_missing() -> None:
    path = [
        _message(HumanMessage(content="discarded by older"), id="00000001"),
        CompactionEntry(
            id="00000002",
            summary="Older summary must not win.",
            first_kept_id=None,
        ),
        _message(HumanMessage(content="between compactions"), id="00000003"),
        CompactionEntry(
            id="00000004",
            summary="Latest summary is authoritative.",
            first_kept_id="missing-entry",
        ),
        _message(HumanMessage(content="after latest"), id="00000005"),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(
            content="[Conversation summary]\nLatest summary is authoritative."
        ),
        HumanMessage(content="after latest"),
    ]
    assert context.compacted is True


def test_build_context_preserves_langchain_message_metadata() -> None:
    human = HumanMessage(
        content="look this up",
        id="human-1",
        name="operator",
        additional_kwargs={"source": "terminal"},
        response_metadata={"trace_id": "trace-human"},
    )
    assistant = AIMessage(
        content=[{"type": "text", "text": "checking"}],
        id="assistant-1",
        name="worker",
        additional_kwargs={"provider_field": {"value": 1}},
        response_metadata={"finish_reason": "tool_calls"},
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "ledger"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    )
    tool = ToolMessage(
        content="found",
        tool_call_id="call-1",
        id="tool-1",
        name="lookup",
        artifact={"raw": [1, 2, 3]},
        status="success",
        additional_kwargs={"elapsed_ms": 8},
        response_metadata={"cached": False},
    )

    context = build_context([_message(human), _message(assistant), _message(tool)])

    assert context.messages == [human, assistant, tool]


def test_build_context_strips_only_requested_foreign_ai_blocks_without_mutation() -> None:
    assistant = AIMessage(
        content=[
            {"type": "text", "text": "answer"},
            {"type": "reasoning", "reasoning": "private"},
            {"type": "thinking", "thinking": "kept"},
        ],
        id="assistant-foreign",
        additional_kwargs={"reasoning": {"private": True}, "kept": "yes"},
        response_metadata={"reasoning": "private", "kept": "yes"},
    )
    entry = _message(assistant)

    context = build_context([entry], strip={"reasoning"})

    assert context.messages == [
        assistant.model_copy(
            update={
                "content": [
                    {"type": "text", "text": "answer"},
                    {"type": "thinking", "thinking": "kept"},
                ],
                "additional_kwargs": {"kept": "yes"},
                "response_metadata": {"kept": "yes"},
            }
        )
    ]
    assert entry.message == message_to_dict(assistant)


def test_build_context_drops_assistant_with_any_dangling_call_and_its_tool_results() -> None:
    assistant = AIMessage(
        content="calling tools",
        tool_calls=[
            {
                "name": "lookup",
                "args": {"query": "ok"},
                "id": "call-matched",
                "type": "tool_call",
            },
            {
                "name": "write_file",
                "args": {"path": "missing.txt"},
                "id": "call-missing",
                "type": "tool_call",
            },
        ],
    )
    path = [
        _message(HumanMessage(content="start")),
        _message(assistant),
        _message(ToolMessage(content="result", tool_call_id="call-matched")),
        _message(AIMessage(content="later response")),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(content="start"),
        AIMessage(content="later response"),
    ]
    assert context.dangling == [ToolCallRef(id="call-missing", name="write_file")]


def test_build_context_removes_orphan_tool_messages() -> None:
    path = [
        _message(HumanMessage(content="start")),
        _message(ToolMessage(content="orphan", tool_call_id="unknown-call")),
        _message(AIMessage(content="finish")),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(content="start"),
        AIMessage(content="finish"),
    ]
    assert context.dangling == []


def test_build_context_uses_latest_model_mode_and_turn_state() -> None:
    path = [
        ModelChangeEntry(model="fake:old"),
        ModeChangeEntry(mode="ask"),
        CustomEntry(
            custom_type="turn_state",
            data={
                "todos": [{"content": "old", "status": "completed"}],
                "files": {"old.py": "old"},
            },
        ),
        CustomEntry(custom_type="plugin.snapshot", data={"ignored": True}),
        ModelChangeEntry(model=["fake:primary", "fake:fallback"]),
        ModeChangeEntry(mode="plan"),
        CustomEntry(
            custom_type="turn_state",
            data={
                "todos": [{"content": "new", "status": "pending"}],
                "files": {"new.py": "print('new')\n"},
            },
        ),
    ]

    context = build_context(path)

    assert context.model == ["fake:primary", "fake:fallback"]
    assert context.mode == "plan"
    assert context.todos == [{"content": "new", "status": "pending"}]
    assert context.files == {"new.py": "print('new')\n"}


def test_build_context_keeps_post_reset_state_discarded_by_compaction() -> None:
    path = [
        ResetBoundaryEntry(),
        ModelChangeEntry(model=["fake:primary", "fake:fallback"]),
        ModeChangeEntry(mode="plan"),
        CustomEntry(
            custom_type="turn_state",
            data={
                "todos": [{"content": "preserve state", "status": "pending"}],
                "files": {"state.py": "print('kept')\n"},
            },
        ),
        _message(HumanMessage(content="discarded by compaction")),
        CompactionEntry(
            summary="Messages were compacted.",
            first_kept_id=None,
            tokens_before=100,
        ),
    ]

    context = build_context(path)

    assert context.messages == [
        HumanMessage(content="[Conversation summary]\nMessages were compacted.")
    ]
    assert context.model == ["fake:primary", "fake:fallback"]
    assert context.mode == "plan"
    assert context.todos == [{"content": "preserve state", "status": "pending"}]
    assert context.files == {"state.py": "print('kept')\n"}


def test_opaque_entries_round_trip_through_ledger_without_affecting_context(
    ledger_session: tuple[Ledger, SessionStore, str],
) -> None:
    ledger, _, session_id = ledger_session
    human = ledger.append(session_id, _message(HumanMessage(content="before")))
    opaque = ledger.append(
        session_id,
        OpaqueEntry(
            entry_type="vendor_future",
            payload={
                "futureField": {"nested": [1, True, None]},
                "model": "opaque:ignored",
                "mode": "ignored",
                "todos": [{"content": "ignored", "status": "pending"}],
            },
        ),
    )
    assistant = ledger.append(session_id, _message(AIMessage(content="after")))

    assert ledger.get(session_id, opaque.id) == opaque
    assert isinstance(ledger.get(session_id, opaque.id), OpaqueEntry)
    assert opaque.entry_type == "vendor_future"
    assert opaque.payload == {
        "futureField": {"nested": [1, True, None]},
        "model": "opaque:ignored",
        "mode": "ignored",
        "todos": [{"content": "ignored", "status": "pending"}],
    }
    context = build_context(ledger.path(session_id))
    assert context.messages == [
        HumanMessage(content="before"),
        AIMessage(content="after"),
    ]
    assert context == build_context([human, assistant])
    assert ledger.all(session_id) == [human, opaque, assistant]


@pytest.mark.parametrize(
    ("stored_type", "stored_payload"),
    [
        pytest.param("custom", "{not-json", id="malformed-json"),
        pytest.param("mode_change", "[]", id="non-object-payload"),
        pytest.param("message", "{}", id="known-type-missing-field"),
        pytest.param(
            "message",
            '{"message":null}',
            id="known-type-invalid-field",
        ),
    ],
)
def test_corrupt_rows_decode_to_export_compatible_opaque_entries(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
    stored_type: str,
    stored_payload: str,
) -> None:
    ledger, store, session_id = ledger_session
    root = ledger.append(session_id, _message(HumanMessage(content="root")))
    corrupt = ledger.append(
        session_id,
        OpaqueEntry(entry_type="placeholder", payload={"valid": True}),
    )
    store._connection.execute(
        """
        UPDATE entries
        SET type = ?, payload = ?
        WHERE session_id = ? AND id = ?
        """,
        (stored_type, stored_payload, session_id, corrupt.id),
    )
    store._connection.commit()

    recovered = ledger.get(session_id, corrupt.id)

    assert isinstance(recovered, OpaqueEntry)
    assert (recovered.id, recovered.parent_id, recovered.ts) == (
        corrupt.id,
        corrupt.parent_id,
        corrupt.ts,
    )
    assert recovered.entry_type not in {
        "message",
        "model_change",
        "mode_change",
        "compaction",
        "reset_boundary",
        "custom",
    }
    all_entries = ledger.all(session_id)
    path_entries = ledger.path(session_id)
    assert [entry.id for entry in all_entries] == [root.id, corrupt.id]
    assert [entry.id for entry in path_entries] == [root.id, corrupt.id]
    assert isinstance(all_entries[-1], OpaqueEntry)
    assert isinstance(path_entries[-1], OpaqueEntry)

    export_path = tmp_path / f"{stored_type}.jsonl"
    export_session(store, session_id, export_path)
    export_text = export_path.read_text(encoding="utf-8")
    exported = [json.loads(line) for line in export_text.splitlines()]
    corrupt_envelope = exported[-1]
    assert corrupt_envelope["id"] == corrupt.id
    assert corrupt_envelope["parentId"] == corrupt.parent_id
    assert corrupt_envelope["timestamp"] == corrupt.ts
    assert corrupt_envelope["type"] == recovered.entry_type
    assert isinstance(entry_from_envelope(corrupt_envelope), OpaqueEntry)


def test_opaque_envelope_round_trip_preserves_reserved_payload_fields() -> None:
    entry = OpaqueEntry(
        id="deadbeef",
        parent_id="a0000001",
        ts="2026-08-27T12:00:00.000000+00:00",
        entry_type="vendor_reserved",
        payload={
            "type": "payload-type",
            "id": "payload-id",
            "parentId": "payload-parent",
            "timestamp": "payload-time",
            "opaqueWrapped": True,
            "opaquePayload": {"nested": "payload"},
            "futureField": [1, True, None],
        },
    )

    envelope = entry_to_envelope(entry)

    assert envelope["type"] == "vendor_reserved"
    assert envelope["id"] == "deadbeef"
    assert envelope["parentId"] == "a0000001"
    assert envelope["timestamp"] == "2026-08-27T12:00:00.000000+00:00"
    assert entry_from_envelope(envelope) == entry


def test_version_3_jsonl_entries_parse_losslessly_and_rebuild_equal_context(
    tmp_path: Path,
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sessions" / "session-tree.jsonl"
    records = [json.loads(line) for line in fixture_path.read_text().splitlines()]
    header, *envelopes = records

    assert header == {
        "type": "session",
        "version": 3,
        "id": "fixture-session",
        "timestamp": "2026-08-27T12:00:00.000000+00:00",
        "cwd": "/fixture/project",
        "title": "Session tree fixture",
        "parentSession": "parent-session",
    }
    parsed = [entry_from_envelope(envelope) for envelope in envelopes]
    assert [entry_to_envelope(entry) for entry in parsed] == envelopes

    source_context = build_context(parsed)
    assert source_context.compacted is True
    assert source_context.model == ["fake:primary", "fake:fallback"]
    assert source_context.mode == "plan"
    assert source_context.todos == [
        {"content": "verify fixture", "status": "pending"}
    ]
    assert source_context.files == {"fixture.py": "print('fixture')\n"}
    assert source_context.dangling == [
        ToolCallRef(id="call-pending", name="write_file")
    ]
    assert [message.content for message in source_context.messages] == [
        "[Conversation summary]\nThe fixture was compacted.",
        "Continue after summary",
        "The compacted work is ready.",
    ]
    assert source_context.messages[1].additional_kwargs == {"source": "fixture"}
    assert source_context.messages[2].usage_metadata == {
        "input_tokens": 9,
        "output_tokens": 5,
        "total_tokens": 14,
    }

    with SessionStore(tmp_path / "fresh.db") as store:
        session = store.create(tmp_path, "fake:model", thread_id="imported-session")
        ledger = Ledger(store)
        ledger.append_many(session.thread_id, parsed)

        rebuilt_context = build_context(ledger.path(session.thread_id))

    assert rebuilt_context == source_context


def test_export_session_round_trips_source_ledger_context_and_entry_order(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "source.jsonl"
    with SessionStore(tmp_path / "source.db") as store:
        session = store.create(
            "/source/project",
            "fake:initial",
            title="Source session",
            thread_id="source-export",
            parent_session="parent-export",
        )
        ledger = Ledger(store)
        source_entries = ledger.append_many(
            session.thread_id,
            [
                _message(HumanMessage(content="old message")),
                CompactionEntry(
                    summary="Source summary.",
                    first_kept_id=None,
                    tokens_before=10,
                ),
                ModelChangeEntry(model=["fake:primary", "fake:fallback"]),
                ModeChangeEntry(mode="plan"),
                CustomEntry(
                    custom_type="turn_state",
                    data={
                        "todos": [{"content": "round trip", "status": "pending"}],
                        "files": {"unicode.py": "print('π')\n"},
                    },
                ),
                _message(
                    HumanMessage(
                        content="continue",
                        additional_kwargs={"source": "export-test"},
                    )
                ),
                _message(AIMessage(content="done")),
            ],
        )
        source_context = build_context(ledger.path(session.thread_id))

        result = export_session(store, session.thread_id, export_path)

        assert result == export_path
        lines = export_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert records[0] == {
            "type": "session",
            "version": 3,
            "id": session.thread_id,
            "timestamp": session.created,
            "cwd": session.cwd,
            "title": session.title,
            "parentSession": session.parent_session,
        }
        assert [record["id"] for record in records[1:]] == [
            entry.id for entry in source_entries
        ]
        assert records[5]["data"]["files"] == {
            "unicode.py": "print('π')\n"
        }

    with SessionStore(tmp_path / "rebuilt.db") as store:
        rebuilt = store.create(
            tmp_path,
            "fake:initial",
            thread_id="rebuilt-export",
        )
        ledger = Ledger(store)
        ledger.append_many(
            rebuilt.thread_id,
            [entry_from_envelope(record) for record in records[1:]],
        )
        rebuilt_context = build_context(ledger.path(rebuilt.thread_id))

    assert rebuilt_context == source_context


def test_export_session_creates_a_private_file_exclusively(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
) -> None:
    _, store, session_id = ledger_session
    export_path = tmp_path / "private.jsonl"

    export_session(store, session_id, export_path)

    assert stat.S_IMODE(export_path.stat().st_mode) == 0o600


def test_export_session_refuses_to_overwrite_an_existing_file(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
) -> None:
    _, store, session_id = ledger_session
    export_path = tmp_path / "existing.jsonl"
    export_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_session(store, session_id, export_path)

    assert export_path.read_text(encoding="utf-8") == "keep me"


def test_export_session_force_truncates_an_existing_regular_file(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
) -> None:
    _, store, session_id = ledger_session
    export_path = tmp_path / "existing.jsonl"
    export_path.write_text("stale trailing data\n" * 100, encoding="utf-8")
    session = store.get(session_id)
    assert session is not None

    export_session(store, session_id, export_path, force=True)

    export_text = export_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in export_text.splitlines()]
    assert records == [
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": session.created,
            "cwd": session.cwd,
            "title": None,
            "parentSession": None,
        }
    ]


def test_export_session_force_refuses_to_follow_a_symlink(
    ledger_session: tuple[Ledger, SessionStore, str],
    tmp_path: Path,
) -> None:
    _, store, session_id = ledger_session
    target = tmp_path / "target.jsonl"
    target.write_text("keep target", encoding="utf-8")
    export_path = tmp_path / "export.jsonl"
    export_path.symlink_to(target)

    with pytest.raises(OSError) as error:
        export_session(store, session_id, export_path, force=True)

    assert error.value.errno == errno.ELOOP
    assert target.read_text(encoding="utf-8") == "keep target"
