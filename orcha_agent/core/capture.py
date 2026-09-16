"""Digest-aware incremental graph-state capture for main and child turns."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, message_to_dict

from .capture_cursor import CaptureBatch, FingerprintCache, message_digest
from .ledger import (
    CompactionEntry,
    CustomEntry,
    Entry,
    Ledger,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
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
    Both values of only_if_new skip unchanged state; False does not force a
    turn_state write.
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
    same_order = False
    summary_changed = False
    if not unchanged_prefix:
        ids = [fingerprint[0] for fingerprint in fingerprints]
        same_order = (
            len(fingerprints) >= len(previous)
            and all(isinstance(message_id, str) for message_id in ids)
            and len(set(ids)) == len(ids)
            and ids[: len(previous)] == [fingerprint[0] for fingerprint in previous]
        )
        summary_changed = any(
            current != old
            and isinstance(messages[index], HumanMessage)
            and messages[index].additional_kwargs.get("lc_source") == "summarization"
            for index, (current, old) in enumerate(zip(fingerprints, previous, strict=False))
        )
    reset = bool(previous) and not unchanged_prefix and (not same_order or summary_changed)
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
    changed = []
    if summary_index is not None:
        path = Ledger(store).path(session_id)
        context = build_context(path)
        candidates = messages[summary_index + 1 :]
        # Reuse only an ordered suffix of the live persisted messages. Arbitrary
        # reorder/drop still needs a snapshot, never ID-based deduplication.
        retained_ids = [message.id for message in candidates]
        live_ids = [message.id for message in context.messages]
        retained_count = 0
        if (
            retained_ids
            and len(set(retained_ids)) == len(retained_ids)
            and all(isinstance(message_id, str) for message_id in retained_ids)
            and len(set(live_ids)) == len(live_ids)
        ):
            try:
                start = live_ids.index(retained_ids[0])
            except ValueError:
                pass
            else:
                suffix = live_ids[start:]
                if retained_ids[: len(suffix)] == suffix:
                    retained_count = len(suffix)
        marker = None
        if retained_count:
            first_id = retained_ids[0]
            first_at = next(
                (
                    index
                    for index in range(len(path) - 1, -1, -1)
                    if isinstance(entry := path[index], MessageEntry)
                    and entry.message["data"].get("id") == first_id
                ),
                None,
            )
            if first_at is not None:
                marker = path[first_at - 1].id if first_at else ""
            else:
                retained_count = 0
        summary = messages[summary_index].content
        entries.append(
            CompactionEntry(
                summary=str(summary).removeprefix(_SUMMARIZATION_PREFIX),
                first_kept_id=marker,
            )
        )
        if retained_count:
            changed = [
                message
                for message, old in zip(
                    candidates[:retained_count], context.messages[-retained_count:], strict=True
                )
                if message_digest(message) != message_digest(old)
            ]
        candidates = candidates[retained_count:]
    elif reset:
        context = build_context(Ledger(store).path(session_id))
        entries.append(CustomEntry(custom_type="checkpoint_reset", data={}))
        if any(
            isinstance(message, HumanMessage)
            and isinstance(message.content, str)
            and message.content.startswith(_SUMMARIZATION_PREFIX)
            for message in messages
        ):
            entries.append(
                CustomEntry(
                    custom_type="unrecognized_summary",
                    data={"reason": "summary prefix without lc_source=summarization"},
                )
            )
        if context.model is not None:
            entries.append(ModelChangeEntry(model=context.model))
        if context.mode is not None:
            entries.append(ModeChangeEntry(mode=context.mode))
        candidates = messages
    else:
        if not unchanged_prefix:
            changed = [
                messages[index]
                for index, (current, old) in enumerate(zip(fingerprints, previous, strict=False))
                if old[1] and current != old
            ]
        candidates = messages[len(previous) :]
    if changed:
        entries.append(
            CustomEntry(
                custom_type="message_replaced",
                data={"messages": [message_to_dict(message) for message in changed]},
            )
        )
    entries.extend(MessageEntry(message=message_to_dict(message)) for message in candidates)
    state = {"todos": values.get("todos", []), "files": values.get("files", {})}
    reminders = values.get("rule_reminders", {})
    if isinstance(reminders, Mapping) and reminders:
        state["rule_reminders"] = {
            name: message_to_dict(message)
            for name, message in reminders.items()
            if isinstance(message, SystemMessage)
        }
    digest = cache.live_state_digest(state)
    if state_row is None or state_row["digest"] != digest:
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
