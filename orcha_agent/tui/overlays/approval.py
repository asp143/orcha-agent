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
    if name in {"execute", "bash", "shell"}:
        command = args.get("command")
        if isinstance(command, str):
            return f"$ {command}"
    if name in {"edit", "edit_file", "write_file", "apply_patch"}:
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
    if description:
        return description
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
        self.preview_text = detail
        detail_rows = max(1, detail.count("\n") + 1)

        def tail_scroll(window: Window) -> int:
            info = window.render_info
            visible = info.window_height if info is not None else min(10, detail_rows)
            return max(0, detail_rows - visible)

        preview = Window(
            FormattedTextControl(FormattedText([("class:overlay.preview", detail)])),
            height=min(10, detail_rows),
            wrap_lines=True,
            get_vertical_scroll=tail_scroll,
        )
        decisions = ("Approve", "Reject", "Always")
        super().__init__(
            f"Approve {tool_name}?",
            decisions,
            label=lambda item: item,
            anchor="bottom",
            prefix=preview,
            show_filter=False,
            on_accept=lambda item: str(item).casefold(),
        )

        for key, result in (("y", "approve"), ("n", "reject"), ("a", "always")):
            self.bindings.add(key)(lambda _event, result=result: self.resolve(result))


__all__ = ["ApprovalOverlay"]
