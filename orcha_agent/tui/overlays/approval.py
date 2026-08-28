"""Bottom-anchored tool approval dialog."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from .select import SelectList


def _preview(name: str, args: Mapping[str, Any], description: str | None) -> str:
    if description:
        return description
    if name in {"execute", "bash", "shell"}:
        command = args.get("command")
        if isinstance(command, str):
            return f"$ {command}"
    if name in {"edit", "write_file", "apply_patch"}:
        before = args.get("old_string") or args.get("before")
        after = args.get("new_string") or args.get("content") or args.get("after")
        if isinstance(before, str) or isinstance(after, str):
            old_lines = "" if not isinstance(before, str) else "\n".join(
                f"- {line}" for line in before.splitlines()
            )
            new_lines = "" if not isinstance(after, str) else "\n".join(
                f"+ {line}" for line in after.splitlines()
            )
            return "\n".join(part for part in (old_lines, new_lines) if part)
    return json.dumps(dict(args), ensure_ascii=False, indent=2, default=str)


class ApprovalOverlay(SelectList[str]):
    """Approve, reject, or permanently allow one tool action."""

    def __init__(self, action: Mapping[str, Any] | None = None, **payload: Any) -> None:
        value = dict(action or payload)
        name = value.get("name")
        args = value.get("args")
        description = value.get("description")
        tool_name = name if isinstance(name, str) else "unknown tool"
        tool_args = args if isinstance(args, Mapping) else {}
        detail = _preview(
            tool_name,
            tool_args,
            description if isinstance(description, str) else None,
        )
        preview = Window(
            FormattedTextControl(FormattedText([("class:overlay.preview", detail)])),
            height=min(10, max(1, detail.count("\n") + 1)),
            wrap_lines=True,
        )
        decisions = ("Approve", "Reject", "Always")
        super().__init__(
            f"Approve {tool_name}?",
            decisions,
            label=lambda item: item,
            anchor="bottom",
            prefix=preview,
            on_accept=lambda item: str(item).casefold(),
        )

        for key, result in (("y", "approve"), ("n", "reject"), ("a", "always")):
            self.bindings.add(key)(lambda _event, result=result: self.resolve(result))


__all__ = ["ApprovalOverlay"]
