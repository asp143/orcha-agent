"""Stable capture fingerprints shared by seeding and incremental capture."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage


def message_digest(message: BaseMessage) -> str:
    # Include tool calls, content blocks and metadata, not only the public ID.
    return hashlib.sha256(message.model_dump_json().encode()).hexdigest()


def state_digest(state: Mapping[str, Any]) -> str:
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


class CaptureBatch(list[Any]):
    """Entries plus cursor changes committed together by Ledger.capture.

    Keeping this an iterable preserves the capture API used by integrations.
    """

    def __init__(
        self,
        entries: list[Any],
        updates: list[tuple[int, str | None, str]],
        state: str,
    ) -> None:
        super().__init__(entries)
        self.updates = updates
        self.state = state


class FingerprintCache:
    """Bounded to active threads; compare snapshots before expensive encoding.

    Snapshot equality includes nested lists/dicts, so in-place tool-call and
    content-block mutations cannot bypass the digest safety check.
    """

    def __init__(self) -> None:
        self.messages: list[tuple[dict[str, Any], Any, str]] = []
        self.state: dict[str, Any] | None = None
        self.state_hash = ""
        self.leaf_id: str | None = None
        self.cursor: list[tuple[str | None, str]] | None = None

    def messages_digest(self, messages: Any) -> list[tuple[str | None, str]]:
        from copy import deepcopy

        result: list[tuple[str | None, str]] = []
        for index, message in enumerate(messages):
            extra = message.__pydantic_extra__
            if index < len(self.messages):
                snapshot, saved_extra, digest = self.messages[index]
                if message.__dict__ == snapshot and extra == saved_extra:
                    result.append((message.id, digest))
                    continue
            digest = message_digest(message)
            saved = (deepcopy(message.__dict__), deepcopy(extra), digest)
            if index < len(self.messages):
                self.messages[index] = saved
            else:
                self.messages.append(saved)
            result.append((message.id, digest))
        del self.messages[len(messages) :]
        return result

    def live_state_digest(self, state: dict[str, Any]) -> str:
        from copy import deepcopy

        if self.state is None or self.state != state:
            self.state_hash = state_digest(state)
            self.state = deepcopy(state)
        return self.state_hash
