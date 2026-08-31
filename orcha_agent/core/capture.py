"""Shared ID-aware graph-state capture for main and child turns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import HumanMessage, message_to_dict, messages_from_dict

from .ledger import CompactionEntry, CustomEntry, Ledger, MessageEntry
from .session import SessionStore

_SUMMARIZATION_PREFIX = "Here is a summary of the conversation to date:\n\n"


def capture_graph_values(
    store: SessionStore,
    session_id: str,
    thread_id: str,
    values: Mapping[str, Any],
    *,
    only_if_new: bool,
    report_error: Callable[[str], None] | None = None,
) -> bool:
    """Append unseen graph messages and turn state without losing compactions."""

    thread = store.get_thread(thread_id)
    if thread is None or thread.session_id != session_id:
        raise LookupError(f"Unknown graph thread: {thread_id}")
    messages = list(values.get("messages", ()))
    current_message_ids = tuple(
        message.id for message in messages if isinstance(message.id, str)
    )
    previous_message_ids = thread.captured_message_ids
    previous_id_set = set(previous_message_ids)
    current_id_set = set(current_message_ids)
    shrunk = (
        bool(previous_message_ids) and not previous_id_set.issubset(current_id_set)
    ) or (not previous_message_ids and len(messages) < thread.captured)
    entries: list[CompactionEntry | MessageEntry | CustomEntry] = []
    candidates = messages
    summary_index: int | None = None
    if shrunk:
        summary_index = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
                and message.additional_kwargs.get("lc_source") == "summarization"
            ),
            None,
        )
        if summary_index is not None:
            summary_message = messages[summary_index]
            summary = (
                summary_message.content
                if isinstance(summary_message.content, str)
                else str(summary_message.content)
            )
            candidates = messages[summary_index + 1 :]
            first_retained_id = next(
                (
                    message.id
                    for message in candidates
                    if isinstance(message.id, str)
                    and message.id in previous_id_set
                ),
                None,
            )
            first_kept_id = None
            if first_retained_id is not None:
                path = Ledger(store).path(session_id)
                retained_at = next(
                    (
                        index
                        for index, entry in enumerate(path)
                        if isinstance(entry, MessageEntry)
                        and messages_from_dict([entry.message])[0].id
                        == first_retained_id
                    ),
                    None,
                )
                if retained_at is not None and retained_at > 0:
                    first_kept_id = path[retained_at - 1].id
            entries.append(
                CompactionEntry(
                    summary=summary.removeprefix(_SUMMARIZATION_PREFIX),
                    first_kept_id=first_kept_id,
                )
            )

    if previous_message_ids:
        for index, message in enumerate(candidates):
            message_id = message.id
            if isinstance(message_id, str):
                unseen = message_id not in previous_id_set
            else:
                absolute_index = (
                    index if summary_index is None else summary_index + 1 + index
                )
                unseen = absolute_index >= thread.captured
            if unseen:
                entries.append(MessageEntry(message=message_to_dict(message)))
    elif summary_index is not None:
        entries.extend(
            MessageEntry(message=message_to_dict(message)) for message in candidates
        )
    else:
        entries.extend(
            MessageEntry(message=message_to_dict(message))
            for message in messages[thread.captured :]
        )

    if only_if_new and not entries:
        return False
    entries.append(
        CustomEntry(
            custom_type="turn_state",
            data={
                "todos": values.get("todos", []),
                "files": values.get("files", {}),
            },
        )
    )
    try:
        Ledger(store).capture(
            session_id,
            thread_id,
            entries,
            captured=len(messages),
            captured_message_ids=current_message_ids,
        )
    except Exception as exc:
        message = f"Failed to capture session {session_id} thread {thread_id}: {exc}"
        if report_error is not None:
            report_error(message)
        raise RuntimeError(message) from exc
    return True


__all__ = ["capture_graph_values"]
