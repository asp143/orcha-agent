"""Digest-aware incremental graph-state capture for main and child turns."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage, message_to_dict

from .capture_cursor import CaptureBatch, FingerprintCache, message_digest
from .ledger import (
    CompactionEntry,
    CustomEntry,
    Entry,
    Ledger,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
    ResetBoundaryEntry,
    build_context,
)
from .session import SessionStore

_SUMMARIZATION_PREFIX = "Here is a summary of the conversation to date:\n\n"

_CaptureReport = tuple[Callable[[str], None], str]
_DEFERRED_ERRORS: ContextVar[list[_CaptureReport] | None] = ContextVar(
    "capture_deferred_errors", default=None
)


@contextmanager
def defer_capture_errors(reports: list[_CaptureReport]) -> Iterator[None]:
    """Collect worker reports for delivery by its owning UI task."""
    token = _DEFERRED_ERRORS.set(reports)
    try:
        yield
    finally:
        _DEFERRED_ERRORS.reset(token)


def capture_graph_values(
    store: SessionStore,
    session_id: str,
    thread_id: str,
    values: Mapping[str, Any],
    *,
    only_if_new: bool,
    report_error: Callable[[str], None] | None = None,
) -> bool:
    """Capture changed messages/state and their normalized cursors atomically.

    Messages are mutable and have no revision contract. Digest-check the prefix
    to detect in-place edits, but write only changed cursor rows and live state.
    A non-append checkpoint gets a reset snapshot so duplicates and ordering are
    preserved exactly rather than silently deduplicated by message ID.
    """
    cache = store._capture_message_cache.get(thread_id)
    if cache is None:
        cache = FingerprintCache()
        if len(store._capture_message_cache) >= 4:
            del store._capture_message_cache[next(iter(store._capture_message_cache))]
        store._capture_message_cache[thread_id] = cache
    with store.saver.lock:
        thread = store._connection.execute(
            "SELECT thread.session_id, thread.captured, session.leaf_id "
            "FROM threads AS thread JOIN sessions AS session "
            "ON session.thread_id = thread.session_id WHERE thread.thread_id = ?",
            (thread_id,),
        ).fetchone()
        if thread is None or thread["session_id"] != session_id:
            raise LookupError(f"Unknown graph thread: {thread_id}")
        if cache.cursor is not None and cache.leaf_id == thread["leaf_id"]:
            persisted = cache.cursor
        else:
            rows = store._connection.execute(
                "SELECT message_id, digest FROM capture_messages "
                "WHERE thread_id = ? ORDER BY position",
                (thread_id,),
            ).fetchall()
            persisted = [(row["message_id"], row["digest"]) for row in rows]
        state_row = store._connection.execute(
            "SELECT digest FROM capture_state WHERE thread_id = ?", (thread_id,)
        ).fetchone()

    messages = values.get("messages", ())
    fingerprints = cache.messages_digest(messages)
    previous = persisted
    if not previous and thread["captured"]:
        # One-time migration of old cursor rows. Use ledger content, not current
        # graph content, so same-ID changes during recovery remain detectable.
        previous_messages = build_context(Ledger(store).path(session_id)).messages
        previous = [(message.id, message_digest(message)) for message in previous_messages]
    unchanged_prefix = len(fingerprints) >= len(previous) and all(
        current == old or (not old[1] and current[0] == old[0])
        for current, old in zip(fingerprints, previous, strict=False)
    )
    entries: list[Entry] = []
    reset = bool(previous) and not unchanged_prefix
    summary_index = (
        next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
                and message.additional_kwargs.get("lc_source") == "summarization"
            ),
            None,
        )
        if reset
        else None
    )
    if summary_index is not None:
        summary = messages[summary_index].content
        entries.append(CompactionEntry(summary=str(summary).removeprefix(_SUMMARIZATION_PREFIX)))
        candidates = messages[summary_index + 1 :]
    elif reset:
        context = build_context(Ledger(store).path(session_id))
        entries.append(ResetBoundaryEntry())
        if context.model is not None:
            entries.append(ModelChangeEntry(model=context.model))
        if context.mode is not None:
            entries.append(ModeChangeEntry(mode=context.mode))
        candidates = messages
    else:
        candidates = messages[len(previous) :]
    entries.extend(MessageEntry(message=message_to_dict(message)) for message in candidates)
    state = {"todos": values.get("todos", []), "files": values.get("files", {})}
    digest = cache.live_state_digest(state)
    if reset or state_row is None or state_row["digest"] != digest:
        entries.append(CustomEntry(custom_type="turn_state", data=state))
    # The only_if_new API still means no entries for an identical checkpoint.
    # Stable-state ordinary turn completion is also a no-op.
    if not entries:
        return False
    updates = [
        (index, message_id, fingerprint)
        for index, (message_id, fingerprint) in enumerate(fingerprints)
        if index >= len(persisted) or (message_id, fingerprint) != persisted[index]
    ]
    try:
        appended = Ledger(store).capture(
            session_id,
            thread_id,
            CaptureBatch(entries, updates, digest),
            captured=len(messages),
            captured_message_ids=tuple(
                message.id for message in messages if isinstance(message.id, str)
            ),
        )
    except Exception as exc:
        message = f"Failed to capture session {session_id} thread {thread_id}: {exc}"
        if report_error is not None:
            deferred = _DEFERRED_ERRORS.get()
            if deferred is None:
                report_error(message)
            else:
                deferred.append((report_error, message))
        raise RuntimeError(message) from exc
    cache.cursor = fingerprints
    cache.leaf_id = appended[-1].id
    return True


__all__ = ["capture_graph_values"]
