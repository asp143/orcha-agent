"""Lossless version-3 JSONL export for session ledgers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ledger import (
    Entry,
    Ledger,
    OpaqueEntry,
    _decode_entry,
    _entry_type_and_payload,
)

if TYPE_CHECKING:
    from .session import SessionStore


_BASE_FIELDS = frozenset({"type", "id", "parentId", "timestamp"})
_OPAQUE_MARKER = "opaqueWrapped"
_OPAQUE_PAYLOAD = "opaquePayload"
_OPAQUE_RESERVED = _BASE_FIELDS | {_OPAQUE_MARKER, _OPAQUE_PAYLOAD}
_KNOWN_ENTRY_TYPES = frozenset(
    {
        "message",
        "model_change",
        "mode_change",
        "compaction",
        "reset_boundary",
        "custom",
    }
)


def entry_to_envelope(entry: Entry) -> dict[str, Any]:
    """Convert an entry to its flattened camel-case JSONL envelope."""
    entry_type, payload = _entry_type_and_payload(entry)
    envelope: dict[str, Any] = {
        "type": entry_type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": entry.ts,
    }
    if entry_type == "compaction":
        envelope.update(
            {
                "summary": payload["summary"],
                "firstKeptId": payload["first_kept_id"],
                "tokensBefore": payload["tokens_before"],
            }
        )
    elif entry_type == "custom":
        envelope.update(
            {
                "customType": payload["custom_type"],
                "data": payload["data"],
            }
        )
    elif isinstance(entry, OpaqueEntry) and _OPAQUE_RESERVED.intersection(payload):
        envelope.update({_OPAQUE_MARKER: True, _OPAQUE_PAYLOAD: payload})
    else:
        envelope.update(payload)
    return envelope


def entry_from_envelope(envelope: Mapping[str, Any]) -> Entry:
    """Parse one flattened version-3 envelope without discarding unknown data."""
    entry_type = envelope["type"]
    if not isinstance(entry_type, str):
        raise TypeError("Entry envelope type must be a string")
    entry_id = envelope.get("id", "")
    parent_id = envelope.get("parentId")
    timestamp = envelope.get("timestamp", "")
    if not isinstance(entry_id, str):
        raise TypeError("Entry envelope id must be a string")
    if parent_id is not None and not isinstance(parent_id, str):
        raise TypeError("Entry envelope parentId must be a string or null")
    if not isinstance(timestamp, str):
        raise TypeError("Entry envelope timestamp must be a string")

    if entry_type == "compaction":
        payload: dict[str, Any] = {
            "summary": envelope["summary"],
            "first_kept_id": envelope.get("firstKeptId"),
            "tokens_before": envelope.get("tokensBefore"),
        }
    elif entry_type == "custom":
        payload = {
            "custom_type": envelope["customType"],
            "data": envelope["data"],
        }
    else:
        payload = {
            key: value for key, value in envelope.items() if key not in _BASE_FIELDS
        }
        if (
            entry_type not in _KNOWN_ENTRY_TYPES
            and set(payload) == {_OPAQUE_MARKER, _OPAQUE_PAYLOAD}
            and payload[_OPAQUE_MARKER] is True
            and isinstance(payload[_OPAQUE_PAYLOAD], Mapping)
        ):
            payload = dict(payload[_OPAQUE_PAYLOAD])

    return _decode_entry(
        entry_type,
        payload,
        id=entry_id,
        parent_id=parent_id,
        ts=timestamp,
    )


def export_session(
    store: SessionStore, session_id: str, path: str | Path
) -> Path:
    """Export all branches of a session as compact version-3 JSONL."""
    session = store.get(session_id)
    if session is None:
        raise LookupError(f"Session not found: {session_id}")

    header = {
        "type": "session",
        "version": 3,
        "id": session.thread_id,
        "timestamp": session.created,
        "cwd": session.cwd,
        "title": session.title,
        "parentSession": session.parent_session,
    }
    records = [
        header,
        *(entry_to_envelope(entry) for entry in Ledger(store).all(session_id)),
    ]
    output = Path(path)
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )
    output.write_text(f"{text}\n", encoding="utf-8")
    return output
