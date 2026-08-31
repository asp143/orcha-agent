"""Registry-backed two-pane agent hub overlay."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput

from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.tui.frame import Block, Frame

from .base import Overlay

_STATUS_GLYPHS = {
    "running": "⟳",
    "idle": "•",
    "parked": "⏸",
    "done": "✔",
    "failed": "✘",
    "aborted": "⏹",
}
_STATUS_STYLES = {
    "running": "class:accent",
    "idle": "class:warning",
    "parked": "class:muted",
    "done": "class:success",
    "failed": "class:error",
    "aborted": "class:muted",
}
_TERMINAL = frozenset({"done", "failed", "aborted"})
_REFRESH_INTERVAL = 0.25


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "thinking", "summary"):
            if key in value:
                return _plain(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "".join(_plain(item) for item in value)
    return str(value)


def _one_line(value: Any, limit: int | None = None) -> str:
    text = " ".join(_plain(value).split())
    if limit is not None and len(text) > limit:
        return f"{text[: max(0, limit - 1)]}…"
    return text


def _fuzzy(query: str, text: str) -> bool:
    needle = iter(query.casefold())
    current = next(needle, None)
    if current is None:
        return True
    for character in text.casefold():
        if character == current:
            current = next(needle, None)
            if current is None:
                return True
    return False


def _run_type(run: Any) -> str:
    value = getattr(run, "agent_type", None)
    return str(getattr(value, "name", value) or "agent")


def _run_id(run: Any) -> str:
    return str(getattr(run, "id", getattr(run, "run_id", "")))


def _run_label(run: Any) -> str:
    return f"{getattr(run, 'name', 'agent')}·{_run_type(run)}"


def _age(value: Any) -> str:
    if not isinstance(value, datetime):
        return "—"
    timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    seconds = max(0, int((datetime.now(UTC) - timestamp).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3_600:
        return f"{seconds // 60}m"
    if seconds < 86_400:
        return f"{seconds // 3_600}h"
    return f"{seconds // 86_400}d"


def _result_text(run: Any) -> str:
    value = getattr(run, "result", None)
    if value is None:
        value = getattr(run, "partial_findings", None)
    if value in (None, [], {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _ledger_messages(ctx: Any, run: Any) -> list[Any]:
    session_id = getattr(run, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        return []
    store = getattr(run, "session", None) or getattr(ctx, "session", None)
    if store is None:
        return []
    try:
        return list(build_context(Ledger(store).path(session_id)).messages)
    except Exception:
        return []


def transcript_rows(ctx: Any, run: Any, *, limit: int = 20) -> list[str]:
    """Return the last logical transcript rows for an agent ledger."""

    rows: list[str] = []
    labels = {"human": "You", "ai": "Agent", "tool": "Tool", "system": "System"}
    for message in _ledger_messages(ctx, run):
        role = str(getattr(message, "type", "message"))
        label = labels.get(role, role.title())
        content = _plain(getattr(message, "content", ""))
        lines = content.splitlines() or ([""] if content else [])
        for index, line in enumerate(lines):
            rows.append(f"{label}: {line}" if index == 0 else f"  {line}")
    return rows[-max(0, limit) :]


def ledger_transcript_frame(ctx: Any, run: Any) -> Frame:
    """Reconstruct renderer-native transcript blocks from an agent ledger."""

    frame = Frame()
    tools: dict[str, Block] = {}
    for offset, message in enumerate(_ledger_messages(ctx, run)):
        role = str(getattr(message, "type", ""))
        content = _plain(getattr(message, "content", ""))
        block_id = f"agent-{_run_id(run)}-{offset}"
        if role == "human":
            if content:
                frame.add("user", {"text": content}, source_id=_run_id(run), block_id=block_id)
            continue
        if role == "ai":
            if content:
                frame.add(
                    "assistant",
                    {"text": content, "role": "subagent", "subagent": True},
                    source_id=_run_id(run),
                    block_id=block_id,
                )
            for call_offset, call in enumerate(getattr(message, "tool_calls", ()) or ()):
                if not isinstance(call, Mapping):
                    continue
                call_id = str(call.get("id") or f"{block_id}-tool-{call_offset}")
                tool = frame.add(
                    "tool",
                    {
                        "id": call_id,
                        "name": str(call.get("name") or "tool"),
                        "args": call.get("args") if isinstance(call.get("args"), Mapping) else {},
                    },
                    source_id=_run_id(run),
                    block_id=f"{block_id}-tool-{call_offset}",
                )
                tools[call_id] = tool
            continue
        if role == "tool":
            call_id = str(getattr(message, "tool_call_id", ""))
            tool = tools.get(call_id)
            if tool is not None:
                tool.update(result=content)
            else:
                frame.add(
                    "tool",
                    {
                        "id": call_id or block_id,
                        "name": str(getattr(message, "name", None) or "tool"),
                        "args": {},
                        "result": content,
                    },
                    source_id=_run_id(run),
                    block_id=block_id,
                )
            continue
        if content:
            frame.add("marker", {"text": content}, source_id=_run_id(run), block_id=block_id)
    return frame


class HubOverlay(Overlay):
    """Inspect and control every run in the application agent registry."""

    def __init__(
        self,
        ctx: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        clipboard: Callable[[str], Any] | None = None,
    ) -> None:
        self.ctx = ctx
        self.agents = getattr(ctx, "agents", None)
        self._clock = clock
        self._clipboard = clipboard
        self._last_refresh = float("-inf")
        self.refresh_count = 0
        self.index = 0
        self._selected_id: str | None = None
        self.tree_mode = False
        self.mode = "roster"
        self.notice: str | None = None
        self.copied_text: str | None = None

        self.filter = Buffer(multiline=False)
        self.message = Buffer(multiline=False)
        self.filter.on_text_changed += self._filter_changed
        self.roster_control = FormattedTextControl(self._roster_fragments, focusable=True)
        self.inspector_control = FormattedTextControl(self._inspector_fragments)
        self.header_control = FormattedTextControl(self._header_fragments)
        self.footer_control = FormattedTextControl(self._footer_fragments)
        self.filter_control = BufferControl(
            buffer=self.filter,
            input_processors=[BeforeInput(FormattedText([("class:accent", "Filter: ")]))],
        )
        self.message_control = BufferControl(
            buffer=self.message,
            input_processors=[BeforeInput(FormattedText([("class:accent", "Message: ")]))],
        )

        panes = VSplit(
            [
                Window(
                    self.roster_control,
                    width=Dimension(weight=11, min=24),
                    wrap_lines=False,
                    always_hide_cursor=True,
                ),
                Window(char="│", width=1, style="class:bordermuted"),
                Window(
                    self.inspector_control,
                    width=Dimension(weight=9, min=24),
                    wrap_lines=True,
                    always_hide_cursor=True,
                ),
            ]
        )
        body = HSplit(
            [
                Window(self.header_control, height=1),
                Window(char="─", height=1, style="class:bordermuted"),
                panes,
                Window(char="─", height=1, style="class:bordermuted"),
                ConditionalContainer(
                    Window(self.filter_control, height=1, style="class:selectedbg"),
                    filter=Condition(lambda: self.mode == "filter"),
                ),
                ConditionalContainer(
                    Window(self.message_control, height=1, style="class:selectedbg"),
                    filter=Condition(lambda: self.mode == "message"),
                ),
                ConditionalContainer(
                    Window(self.footer_control, height=1),
                    filter=Condition(lambda: self.mode == "roster"),
                ),
            ]
        )

        bindings = KeyBindings()
        roster = Condition(lambda: self.mode == "roster")

        @bindings.add("j", filter=roster)
        @bindings.add("down", filter=roster)
        def _down(event: Any) -> None:
            self.move(1)
            event.app.invalidate()

        @bindings.add("k", filter=roster)
        @bindings.add("up", filter=roster)
        def _up(event: Any) -> None:
            self.move(-1)
            event.app.invalidate()

        @bindings.add("/", filter=roster)
        def _filter(event: Any) -> None:
            self.mode = "filter"
            self.notice = None
            event.app.layout.focus(self.filter_control)
            event.app.invalidate()

        @bindings.add("t", filter=roster)
        def _tree(event: Any) -> None:
            self.toggle_tree()
            event.app.invalidate()

        @bindings.add("m", filter=roster)
        def _message(event: Any) -> None:
            if self.selected_run is None:
                return
            self.mode = "message"
            self.notice = None
            self.message.reset()
            event.app.layout.focus(self.message_control)
            event.app.invalidate()

        @bindings.add("x", filter=roster)
        def _cancel(event: Any) -> None:
            self._start_action(self.cancel_selected(self.selected_run), event)

        @bindings.add("r", filter=roster)
        def _revive(event: Any) -> None:
            self._start_action(self.revive_selected(self.selected_run), event)

        @bindings.add("y", filter=roster)
        def _copy(event: Any) -> None:
            self.copy_selected()
            event.app.invalidate()

        @bindings.add("enter")
        def _enter(event: Any) -> None:
            if self.mode == "filter":
                self.mode = "roster"
                event.app.layout.focus(self.roster_control)
                event.app.invalidate()
                return
            if self.mode == "message":
                text = self.message.text.strip()
                if text:
                    self._start_action(self.send_message(text), event)
                return
            selected = self.selected_run
            if selected is not None:
                self.resolve(_run_id(selected))

        super().__init__(
            "Agent Hub",
            body,
            width=0.90,
            height=0.80,
            min_height=8,
            bindings=bindings,
        )
        self._sync_selection()

    @property
    def focus_target(self) -> Any:
        return self.roster_control

    def _width(self) -> int:
        columns, _rows = self._terminal_size()
        available = max(4, columns - self.margin * 2)
        target = max(4, int(columns * self.width_percent))
        return min(available, target)

    def _needed_height(self, available: int) -> int:
        roster_rows = max(1, len(self.filtered_runs))
        inspector_rows = max(1, len(self.render_inspector_text().splitlines()))
        return min(available, 6 + max(roster_rows, inspector_rows))

    @property
    def filtered_runs(self) -> tuple[Any, ...]:
        list_runs = getattr(self.agents, "tree" if self.tree_mode else "list", None)
        runs = list(list_runs()) if callable(list_runs) else []
        query = self.filter.text.strip()
        if not query:
            return tuple(runs)
        return tuple(run for run in runs if _fuzzy(query, self._search_text(run)))

    @property
    def selected_run(self) -> Any | None:
        runs = self.filtered_runs
        if not runs:
            return None
        self.index = min(max(0, self.index), len(runs) - 1)
        return runs[self.index]

    def _search_text(self, run: Any) -> str:
        return " ".join(
            str(value or "")
            for value in (
                _run_id(run),
                getattr(run, "name", ""),
                _run_type(run),
                getattr(run, "parent_id", ""),
                getattr(run, "model_label", ""),
                getattr(run, "description", ""),
                getattr(run, "status", ""),
                getattr(run, "current_tool", None) or getattr(run, "last_tool", ""),
            )
        )

    def _sync_selection(self) -> None:
        runs = self.filtered_runs
        if not runs:
            self.index = 0
            self._selected_id = None
            return
        if self._selected_id is not None:
            for offset, run in enumerate(runs):
                if _run_id(run) == self._selected_id:
                    self.index = offset
                    return
        self.index = min(self.index, len(runs) - 1)
        self._selected_id = _run_id(runs[self.index])

    def _filter_changed(self, _buffer: Buffer) -> None:
        self.index = 0
        self._selected_id = None
        self._sync_selection()
        self._invalidate()

    def move(self, delta: int) -> Any | None:
        runs = self.filtered_runs
        if not runs:
            self.index = 0
            self._selected_id = None
            return None
        self.index = min(len(runs) - 1, max(0, self.index + delta))
        self._selected_id = _run_id(runs[self.index])
        self.notice = None
        return runs[self.index]

    def toggle_tree(self) -> bool:
        selected = self.selected_run
        self._selected_id = _run_id(selected) if selected is not None else None
        self.tree_mode = not self.tree_mode
        self._sync_selection()
        return self.tree_mode

    def refresh_from_event(self, _event: Any = None) -> bool:
        """Refresh selection and invalidate at most four times per second."""

        now = self._clock()
        if now - self._last_refresh < _REFRESH_INTERVAL:
            return False
        self._last_refresh = now
        self.refresh_count += 1
        self._sync_selection()
        self._invalidate()
        return True

    async def send_message(self, text: str) -> bool:
        selected = self.selected_run
        send = getattr(self.agents, "send", None)
        if selected is None or not callable(send) or not text.strip():
            self.notice = "No agent selected"
            return False
        try:
            await send(_run_id(selected), text.strip())
        except Exception as exc:
            self.notice = f"{type(exc).__name__}: {exc}"
            return False
        self.message.reset()
        self.mode = "roster"
        self.notice = f"Message sent to {getattr(selected, 'name', 'agent')}"
        self.refresh_from_event()
        return True

    async def cancel_selected(self, selected: Any | None = None) -> bool:
        if selected is None:
            selected = self.selected_run
        cancel = getattr(self.agents, "cancel", None)
        if selected is None or not callable(cancel):
            self.notice = "No agent selected"
            return False
        if str(getattr(selected, "status", "")) in _TERMINAL:
            self.notice = f"{getattr(selected, 'name', 'Agent')} is already settled"
            return False
        try:
            await cancel(_run_id(selected))
        except Exception as exc:
            self.notice = f"{type(exc).__name__}: {exc}"
            return False
        self.notice = f"Cancelled {getattr(selected, 'name', 'agent')}"
        self.refresh_from_event()
        return True

    async def revive_selected(self, selected: Any | None = None) -> bool:
        if selected is None:
            selected = self.selected_run
        revive = getattr(self.agents, "revive", None)
        if selected is None or not callable(revive):
            self.notice = "No agent selected"
            return False
        if str(getattr(selected, "status", "")) != "parked":
            self.notice = f"{getattr(selected, 'name', 'Agent')} is not parked"
            return False
        try:
            await revive(str(getattr(selected, "session_id", "")))
        except Exception as exc:
            self.notice = f"{type(exc).__name__}: {exc}"
            return False
        self.notice = f"Revived {getattr(selected, 'name', 'agent')}"
        self.refresh_from_event()
        return True

    def copy_selected(self) -> str | None:
        selected = self.selected_run
        if selected is None:
            self.notice = "No agent selected"
            return None
        text = _result_text(selected)
        if not text:
            self.notice = "Selected agent has no result"
            return None
        self.copied_text = text
        try:
            if self._clipboard is not None:
                self._clipboard(text)
            else:
                payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
                output = get_app().output
                output.write_raw(f"\x1b]52;c;{payload}\a")
                output.flush()
        except Exception as exc:
            self.notice = f"{type(exc).__name__}: {exc}"
            return None
        self.notice = "Result copied"
        return text

    def _start_action(self, action: Any, event: Any) -> None:
        async def complete() -> None:
            await action
            if self.mode == "roster":
                try:
                    event.app.layout.focus(self.roster_control)
                except ValueError:
                    pass
            event.app.invalidate()

        event.app.create_background_task(complete())

    def cancel(self) -> None:
        if self.mode != "roster":
            self.mode = "roster"
            self.notice = None
            try:
                app = get_app()
                app.layout.focus(self.roster_control)
                app.invalidate()
            except Exception:
                pass
            return
        super().cancel()

    def _invalidate(self) -> None:
        try:
            get_app().invalidate()
        except Exception:
            pass

    def _aggregate_text(self) -> str:
        runs = self.filtered_runs
        statuses: dict[str, int] = {}
        for run in runs:
            status = str(getattr(run, "status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        parts = [f"{len(runs)} agents"]
        parts.extend(f"{count} {status}" for status, count in statuses.items() if count)
        requests = sum(int(getattr(run, "requests", 0) or 0) for run in runs)
        tokens = sum(
            int(getattr(run, "tokens_in", 0) or 0) + int(getattr(run, "tokens_out", 0) or 0)
            for run in runs
        )
        cost = sum(float(getattr(run, "cost", 0.0) or 0.0) for run in runs)
        parts.extend((f"{requests} req", f"{tokens} tok", f"${cost:.4f}"))
        suffix = " · tree" if self.tree_mode else ""
        return " · ".join(parts) + suffix

    def _header_fragments(self) -> StyleAndTextTuples:
        return [("class:accent", self._aggregate_text())]

    def _roster_fragments(self) -> StyleAndTextTuples:
        runs = self.filtered_runs
        if not runs:
            return [("class:muted", "  No matching agents\n")]
        fragments: StyleAndTextTuples = []
        width = max(24, int(self.inner_width * 0.55) - 1)
        for offset, run in enumerate(runs):
            current = offset == self.index
            if current:
                fragments.append(("[SetCursorPosition]", ""))
            status = str(getattr(run, "status", "parked"))
            glyph = _STATUS_GLYPHS.get(status, "?")
            parent = str(getattr(run, "parent_id", "main"))
            model = str(getattr(run, "model_label", "") or "—")
            tool = getattr(run, "current_tool", None) or getattr(run, "last_tool", None)
            args = getattr(run, "current_tool_args", None) or getattr(run, "last_tool_args", None)
            tool_text = f" {tool}·{_one_line(args, 40)}" if tool else ""
            tokens = int(getattr(run, "tokens_in", 0) or 0) + int(getattr(run, "tokens_out", 0) or 0)
            cost = float(getattr(run, "cost", 0.0) or 0.0)
            unread_count = getattr(self.agents, "unread_count", None)
            unread = int(unread_count(_run_id(run))) if callable(unread_count) else 0
            unread_text = f" ↑{unread}" if unread else ""
            depth = max(0, int(getattr(run, "depth", 0) or 0)) if self.tree_mode else 0
            tree_prefix = f"{'  ' * depth}{'└ ' if depth else ''}"
            marker = "›" if current else " "
            line = (
                f" {marker} {tree_prefix}{glyph} {_run_id(run)} {_run_label(run)} "
                f"←{parent} {model} {_age(getattr(run, 'created_at', None))}"
                f"{tool_text} {tokens}t ${cost:.4f}{unread_text}"
            )
            style = "class:selectedbg" if current else _STATUS_STYLES.get(status, "class:text")
            fragments.append((style, f"{line[:width]}\n"))
        return fragments

    def render_roster_text(self) -> str:
        return "".join(text for _style, text in self._roster_fragments())

    def _inspector_lines(self) -> list[str]:
        run = self.selected_run
        if run is None:
            return ["No agent selected"]
        status = str(getattr(run, "status", "unknown"))
        requests = int(getattr(run, "requests", 0) or 0)
        cfg = getattr(run, "cfg", None)
        budget = getattr(getattr(cfg, "agents", None), "soft_request_budget", None)
        budget_text = f"{requests}/{budget} requests" if isinstance(budget, int) else f"{requests} requests"
        tokens = int(getattr(run, "tokens_in", 0) or 0) + int(getattr(run, "tokens_out", 0) or 0)
        lines = [
            f"{_STATUS_GLYPHS.get(status, '?')} {_run_label(run)}  ⟦{status}⟧",
            _one_line(getattr(run, "description", "")) or "No description",
            f"id {_run_id(run)} · parent {getattr(run, 'parent_id', 'main')}",
            f"session {getattr(run, 'session_id', '—')}",
            f"budget {budget_text} · {tokens} tokens · ${float(getattr(run, 'cost', 0.0) or 0.0):.4f}",
        ]
        tool = getattr(run, "current_tool", None) or getattr(run, "last_tool", None)
        args = getattr(run, "current_tool_args", None) or getattr(run, "last_tool_args", None)
        if tool:
            lines.extend(("", f"Last tool · {tool}", _plain(args) or "{}"))
        result = _result_text(run)
        if result:
            heading = "Result" if getattr(run, "result", None) is not None else "Partial findings"
            lines.extend(("", heading, _one_line(result, 240)))
        transcript = transcript_rows(self.ctx, run, limit=20)
        lines.extend(("", "Transcript"))
        lines.extend(transcript or ["No transcript yet"])
        return lines

    def _inspector_fragments(self) -> StyleAndTextTuples:
        lines = self._inspector_lines()
        fragments: StyleAndTextTuples = []
        for offset, line in enumerate(lines):
            style = "class:accent" if offset == 0 or line in {"Result", "Partial findings", "Transcript"} else "class:text"
            fragments.append((style, f" {line}\n"))
        return fragments

    def render_inspector_text(self) -> str:
        return "".join(text for _style, text in self._inspector_fragments())

    def _footer_fragments(self) -> StyleAndTextTuples:
        if self.notice:
            style = "class:error" if ":" in self.notice and self.notice.split(":", 1)[0].endswith("Error") else "class:accent"
            return [(style, self.notice)]
        return [("class:muted", "j/k move · / filter · t tree · Enter open · m message · x cancel · r revive · y copy · Esc close")]

    def render_text(self) -> str:
        return f"{self._aggregate_text()}\n{self.render_roster_text()}\n{self.render_inspector_text()}"


__all__ = [
    "HubOverlay",
    "ledger_transcript_frame",
    "transcript_rows",
]
