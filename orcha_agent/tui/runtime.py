"""Prompt-toolkit input loop and graph stream dispatch."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.application.current import set_app
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import History
from prompt_toolkit.filters import Condition, vi_mode, vi_navigation_mode
from prompt_toolkit.key_binding import DynamicKeyBindings, KeyBindings, merge_key_bindings
from prompt_toolkit.layout import FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from orcha_agent.builtin.advisor import AdvisorService
from orcha_agent.builtin.commands_review import review

from orcha_agent.core.config import Config, is_trusted_cwd
from orcha_agent.core.events import (
    AgentDelivered,
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    AppExit,
    AppStart,
    Event,
    EventBus,
    InterruptRaised,
    ModelChunk,
    SessionSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.models import ModelResolver
from orcha_agent.core.persistence import TursoPersistenceError, open_session_store
from orcha_agent.core.registry import CommandRegistration, Registry
from orcha_agent.core.session import SessionStore

from .blocks.image import image_protocol
from .blocks.terminal import clear_terminal_cache
from .blocks import (
    BlockRendererDispatcher,
    DEFAULT_RENDERERS,
    DEFAULT_THEME,
    LEADING_SPACER_KINDS,
    render_delivery,
    render_task,
    theme_spinner,
)
from .console import ConsoleOutput
from .complete import ComposerCompleter
from .composer import Composer
from .context import (
    AppContext,
    _primary_provider_prefix,
    _session_resolution_error,
    _stored_model,
    _uncheckpointed_seed_target,
)
from .frame import Block, BlockState, Frame, FrameScheduler
from .history import SQLiteHistory, history_path
from .keys import create_key_bindings, format_key_bindings, load_keybindings
from .queue import PromptQueue, split_submission
from .notify import DesktopNotifier
from .transcript import Transcript
from .statusline import agent_counts, render_statusline
from .theme import Theme, ThemeWatcher, apply_colorblind, load_themes, select_theme, theme_from_background
from .title import TerminalTitle
from .turn import _run_cancellable_turn
from .overlays import HubOverlay, KeyBindingsOverlay, register_builtin_overlays
from .overlays.base import Overlay
from .overlays.paste import PasteOverlay
from .overlays.hub import ledger_transcript_frame


class StdoutStallWatchdog:
    """Track drain progress, rather than rejecting a large healthy frame."""

    def __init__(self, arm_bytes: int = 262144, clear_bytes: int = 65536,
                 stall_seconds: float = 1.0) -> None:
        self.arm_bytes = arm_bytes
        self.clear_bytes = clear_bytes
        self.stall_seconds = stall_seconds
        self.armed = False
        self.low_water = 0
        self.since = 0.0

    def sample(self, pending: int, now: float) -> bool:
        if pending <= self.clear_bytes:
            self.armed = False
        elif not self.armed and pending > self.arm_bytes:
            self.armed = True
            self.low_water, self.since = pending, now
        elif self.armed and pending < self.low_water:
            self.low_water, self.since = pending, now
        return self.armed and now - self.since >= self.stall_seconds


class _TerminalPump:
    """One ordered writer keeps a stalled PTY off the application event loop."""

    def __init__(self, stream: Any) -> None:
        import queue
        import threading

        self.stream = stream
        self.encoding = getattr(stream, "encoding", "utf-8") or "utf-8"
        self.pending = 0
        self.error: Exception | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._write, name="orcha-terminal", daemon=True)
        self._thread.start()

    def isatty(self) -> bool:
        return True

    def write(self, data: str) -> int:
        if self.error is not None:
            return len(data)
        with self._lock:
            self.pending += len(data.encode(self.encoding, "replace"))
        self._queue.put(data)
        return len(data)

    def flush(self) -> None:
        pass

    def _write(self) -> None:
        while (data := self._queue.get()) is not None:
            try:
                # Small chunks expose ongoing progress even for a large frame.
                for offset in range(0, len(data), 1024):
                    chunk = data[offset:offset + 1024]
                    self.stream.write(chunk)
                    self.stream.flush()
                    with self._lock:
                        self.pending -= len(chunk.encode(self.encoding, "replace"))
            except Exception as exc:
                self.error = exc
                with self._lock:
                    self.pending = 0

    def close(self) -> None:
        self._queue.put(None)


class _TerminalReplies:
    """Remove fragmented terminal reports before PT's keyboard/paste parser."""

    def __init__(self, feed: Callable[[str], None], report: Callable[[str], None]) -> None:
        self.feed = feed
        self.report = report
        self.pending = ""
        self.pasting = False

    def __call__(self, text: str) -> None:
        import re

        text = self.pending + text
        self.pending = ""
        while text:
            if not text.startswith("\x1b"):
                boundary = text.find("\x1b")
                if boundary < 0:
                    self.feed(text)
                    return
                self.feed(text[:boundary])
                text = text[boundary:]
            if self.pasting and "\x1b[201~".startswith(text):
                if text != "\x1b[201~":
                    self.pending = text
                    return
            if text.startswith("\x1b[200~"):
                self.pasting = True
                self.feed(text[:6])
                text = text[6:]
                continue
            if text.startswith("\x1b[201~"):
                self.pasting = False
                self.feed(text[:6])
                text = text[6:]
                continue
            if not self.pasting:
                if text.startswith(("\x1b[I", "\x1b[O")):
                    self.report(text[:3])
                    text = text[3:]
                    continue
                match = re.match(r"\x1b\[\?2026;[0-4]\$y|\x1b\]11;[^\x07\x1b]*(?:\x07|\x1b\\)", text)
                if match:
                    self.report(match[0])
                    text = text[len(match[0]):]
                    continue
                prefixes = ("\x1b[?2026;", "\x1b]11;", "\x1b[200~", "\x1b[201~", "\x1b[I", "\x1b[O")
                if any(prefix.startswith(text) for prefix in prefixes) or (
                    text.startswith(("\x1b[?2026;", "\x1b]11;")) and len(text) < 128
                ):
                    self.pending = text
                    return
            self.feed(text[0])
            text = text[1:]

    def flush(self) -> None:
        if self.pasting:
            return
        if self.pending:
            pending, self.pending = self.pending, ""
            self.feed(pending)


class _PaintOutput:
    """Synchronized flushes, bounded repaint backlog and a loop-lag probe."""

    def __init__(self, application: Any, *, enabled: bool = True) -> None:
        import logging
        from prompt_toolkit.output.vt100 import Vt100_Output

        self.application = application
        self.output = application.output
        self.logger = logging.getLogger("orcha.tui.output")
        self.watchdog = StdoutStallWatchdog()
        self.pump: _TerminalPump | None = None
        self._probe_task: asyncio.Task[Any] | None = None
        self._degraded = False
        self._redraw_skipped = False
        self._sync_depth = 0
        self._sync_open = False
        self._closed = False
        self.background: str | None = None
        self.focus_changed: Callable[[bool], None] = lambda _focused: None
        terminal = os.environ.get("TERM", "").lower()
        program = os.environ.get("TERM_PROGRAM", "").lower()
        override = os.environ.get("ORCHA_SYNC_OUTPUT", "").lower()
        self._detected_synchronized = (
            override in {"1", "true"} or (override not in {"0", "false"} and (
                any(name in terminal for name in ("kitty", "foot", "wezterm", "ghostty"))
                or program in {"wezterm", "ghostty", "iterm.app", "vscode"}
            ))
        )
        self.synchronized = enabled and self._detected_synchronized
        self._enabled = enabled
        self._original_flush = self.output.flush
        self._original_redraw = application._redraw
        self._parser: Any = None
        self._original_feed: Any = None
        self._original_parser_flush: Any = None
        if isinstance(self.output, Vt100_Output) and self.output.stdout.isatty():
            self.pump = _TerminalPump(self.output.stdout)
            self.output.flush = self.flush
            application._redraw = self.redraw
            parser = getattr(application.input, "vt100_parser", None)
            if parser is not None:
                self._parser = parser
                self._original_feed = parser.feed
                self._original_parser_flush = parser.flush
                replies = _TerminalReplies(parser.feed, self.report)
                parser.feed = replies
                def flush_parser() -> None:
                    replies.flush()
                    self._original_parser_flush()
                parser.flush = flush_parser

    def report(self, value: str) -> None:
        if value in {"\x1b[I", "\x1b[O"}:
            self.focus_changed(value == "\x1b[I")
        if value.startswith("\x1b[?2026;"):
            self._detected_synchronized = value[-3] in "12"
            self.synchronized = self._enabled and self._detected_synchronized
        elif value.startswith("\x1b]11;"):
            self.background = value[5:].rstrip("\x07\x1b\\")

    def begin_frame(self) -> None:
        """Keep erase, settled output and redraw in one terminal transaction."""
        if self._sync_depth == 0 and self.pump is not None:
            self.flush()
            self._sync_open = self.synchronized
            if self._sync_open:
                self.pump.write("\x1b[?2026h")
        self._sync_depth += 1

    def end_frame(self) -> None:
        if self._sync_depth <= 0:
            return
        if self._sync_depth == 1 and self.pump is not None:
            self.flush()
            if self._sync_open:
                self.pump.write("\x1b[?2026l")
            self._sync_open = False
        self._sync_depth -= 1

    def flush(self) -> None:
        data = "".join(self.output._buffer)
        self.output._buffer.clear()
        if data and self.pump is not None:
            if self.synchronized and self._sync_depth == 0:
                data = "\x1b[?2026h" + data + "\x1b[?2026l"
            self.pump.write(data)

    def redraw(self, render_as_done: bool = False) -> None:
        if not render_as_done and self.pump is not None and self.pump.pending > 262144:
            # Skip before PT advances its differential screen. Never drop bytes
            # from an already-rendered frame or enqueue another invalidation.
            self._redraw_skipped = True
            return
        self._redraw_skipped = False
        self._original_redraw(render_as_done)

    def start(self) -> None:
        if self.pump is None or self._probe_task is not None:
            return
        self.output.write_raw("\x1b[?2026$p\x1b]11;?\x07\x1b[?1004h")
        self.output.flush()
        self._probe_task = asyncio.create_task(self._probe())

    async def _probe(self) -> None:
        previous = time.monotonic()
        while True:
            await asyncio.sleep(0.25)
            now = time.monotonic()
            lag = max(0.0, now - previous - 0.25)
            previous = now
            if not self._sample_output(now=now, lag=lag):
                return

    def _sample_output(self, *, now: float, lag: float) -> bool:
        pending = self.pump.pending if self.pump is not None else 0
        stalled = self.watchdog.sample(pending, now)
        if self.pump is not None and self.pump.error is not None:
            self.logger.error("Terminal output failed: %s", type(self.pump.error).__name__)
            if self.application.is_running:
                self.application.exit(exception=OSError("terminal output disconnected"))
            return False
        degraded = stalled or lag > 0.25
        transitioned = degraded != self._degraded
        if degraded and transitioned:
            self.logger.warning("Terminal repaint degraded: pending=%d loop_lag=%.3fs", pending, lag)
        self.application.min_redraw_interval = 0.1 if degraded else 1 / 60
        self._degraded = degraded
        # FrameScheduler owns activity ticks. An idle watchdog must not paint.
        if transitioned or (self._redraw_skipped and pending < 65536):
            self.application.invalidate()
        return True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._probe_task is not None:
            self._probe_task.cancel()
            await asyncio.gather(self._probe_task, return_exceptions=True)
        if self.pump is not None:
            self.output.write_raw("\x1b[?1004l\x1b[?2026l")
            self.flush()
        if self.pump is not None:
            deadline = time.monotonic() + 1.0
            while self.pump.pending and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            self.pump.close()
        self.output.flush = self._original_flush
        self.application._redraw = self._original_redraw
        if self._parser is not None:
            self._parser.feed = self._original_feed
            self._parser.flush = self._original_parser_flush


def _completion_style(theme: Any) -> Any:
    base = getattr(theme, "pt", None)
    colors = getattr(theme, "colors", None)
    if base is None or not isinstance(colors, Mapping):
        return base

    def color(token: str) -> str:
        value = base.get_attrs_for_style_str(f"class:{token.lower()}").color
        return f"#{value}" if value and not value.startswith("#") else value

    menu = Style.from_dict(
        {
            "completion": f"fg:{color('text')}",
            "completion.arrow": f"fg:{color('accent')} bold",
            "completion.label": f"fg:{color('text')}",
            "completion.match": f"fg:{color('accent')} bold",
            "completion.meta": f"fg:{color('muted')}",
            "completion.counter": f"fg:{color('dim')}",
            "composer.placeholder": f"fg:{color('dim')}",
        }
    )
    return merge_styles([base, menu])


def _history_path() -> Path:
    return history_path()


def _bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _accept(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return bindings





def _bottom_toolbar(ctx: Any) -> Any:
    if not bool(getattr(ctx.cfg, "statusbar", True)):
        return []
    ui = getattr(ctx, "ui", None)
    theme = getattr(ui, "theme", DEFAULT_THEME)
    width_source = getattr(ui, "status_width", None)
    width = width_source() if callable(width_source) else None
    return render_statusline(
        ctx,
        theme,
        width=width,
        composer_shape=getattr(ctx.cfg, "composer", "box"),
    )

_MAIN_ACCOUNTING_EVENTS = (TurnStart, ModelChunk, TurnEnd)


def _scope_main_statusbar_accounting(bus: EventBus) -> None:
    """Keep main-session status accounting isolated from forwarded agent turns."""

    for index, registration in enumerate(bus.handlers):
        if (
            registration.plugin != "statusbar"
            or registration.event_type not in _MAIN_ACCOUNTING_EVENTS
        ):
            continue
        handler = registration.handler

        async def main_only(event: Any, *, _handler: Any = handler) -> Any:
            if getattr(event, "source_id", "main") != "main":
                return None
            return await _handler(event)

        bus.handlers[index] = replace(registration, handler=main_only)


async def dispatch_command(registry: Registry, ctx: Any, text: str) -> bool:
    """Dispatch slash commands without invoking the model."""

    if not text.startswith("/"):
        return False
    command_text = text[1:]
    name, separator, args = command_text.partition(" ")
    if name == "keys" and not separator:
        await ctx.ui.show(KeyBindingsOverlay(ctx))
        return True
    if name == "agents":
        await ctx.ui.show("hub")
        return True
    registration = registry.commands.get(name)
    if registration is None:
        ctx.console.error(f"Unknown command: /{name}")
        return True
    await registration.handler(ctx, args if separator else "")
    return True





def _compat(name: str, default: Any) -> Any:
    facade = sys.modules.get("orcha_agent.tui.app")
    return getattr(facade, name, default) if facade is not None else default


def _resolve_runtime_themes(
    cfg: Any,
    plugin_states: Mapping[str, Mapping[str, Any]],
    warn: Callable[[str], None],
) -> tuple[dict[str, Theme], Theme]:
    themes = load_themes(
        cwd=cfg.cwd,
        trusted=cfg.trust_cwd,
        warn=warn,
        symbols=cfg.symbols,
    )
    selected = plugin_states.get("commands_core", {}).get("theme", cfg.theme)
    if not isinstance(selected, str):
        selected = cfg.theme
    try:
        active = select_theme(themes, selected)
    except KeyError:
        warn(f"Unknown theme '{selected}'; using dark.")
        active = themes["dark"]
    return themes, active



class UIFacade:
    """Stable awaitable UI surface backed by the active application."""

    def __init__(
        self,
        *,
        show_overlay: Callable[..., Awaitable[Any]] | None = None,
        toggle_overlay: Callable[..., Awaitable[Any]] | None = None,
        notify: Callable[[str], None] | None = None,
        clear: Callable[[], Awaitable[None]] | None = None,
        set_theme: Callable[[str], Any] | None = None,
    ) -> None:
        self._show_overlay = show_overlay
        self._toggle_overlay = toggle_overlay
        self._notify = notify
        self._clear = clear
        self._set_theme = set_theme
        self.notifications: list[str] = []
        self.thinking_visible = True
        self.tools_expanded = False
        self.todos: list[Any] = []
        self.subagents: list[Any] = []

    def notify(self, text: str) -> None:
        self.notifications.append(text)
        if self._notify is not None:
            self._notify(text)

    async def show(self, overlay: object, *args: Any, **kwargs: Any) -> Any:
        if self._show_overlay is None:
            raise RuntimeError(f"overlay {overlay!r} is unavailable")
        return await self._show_overlay(overlay, *args, **kwargs)

    async def toggle(self, overlay: object, *args: Any, **kwargs: Any) -> Any:
        if self._toggle_overlay is None:
            raise RuntimeError(f"overlay {overlay!r} is unavailable")
        return await self._toggle_overlay(overlay, *args, **kwargs)

    async def ask(self, questions: object) -> Any:
        if self._show_overlay is None:
            raise RuntimeError("ask overlay is unavailable")
        try:
            inspect.signature(self._show_overlay).bind("ask", questions=questions)
        except (TypeError, ValueError):
            return await self.show(questions)
        return await self.show("ask", questions=questions)

    async def clear(self) -> None:
        if self._clear is not None:
            await self._clear()

    def set_theme(self, name: str) -> Any:
        if self._set_theme is None:
            raise RuntimeError("theme selection is unavailable")
        return self._set_theme(name)

    def set_todos(self, todos: Any) -> None:
        self.todos = list(todos) if isinstance(todos, (list, tuple)) else []
        invalidate = getattr(self, "invalidate", None)
        if callable(invalidate):
            invalidate()


    def toggle_thinking(self) -> bool:
        self.thinking_visible = not self.thinking_visible
        return self.thinking_visible

    def expand_tools(self, expanded: bool) -> None:
        self.tools_expanded = expanded


class ApplicationRuntime:
    """One inline prompt-toolkit application for a complete TUI session."""

    def __init__(
        self,
        submit: Callable[[str], Awaitable[None]],
        *,
        registry: Registry | None = None,
        history: History | None = None,
        status: Callable[[], Any] | None = None,
        input: Any = None,
        output: Any = None,
        console: Console | None = None,
        theme: Any = DEFAULT_THEME,
        themes: Mapping[str, Any] | None = None,
        ctx: Any = None,
        composer_shape: str = "box",
        keybindings_path: str | Path | None = None,
        shell_runner: Callable[[str, Path, float], Any] | None = None,
        editor_runner: Callable[[str], str] | None = None,
    ) -> None:
        self._submit = submit
        self.ctx = ctx
        self.registry = registry
        if registry is not None:
            _ensure_agent_command(registry)
            _ensure_review_command(registry)
        completion_registry = registry or Registry()
        self.frame = Frame()
        self._tui_config = getattr(getattr(ctx, "cfg", None), "tui", None)
        self._base_theme = theme
        if isinstance(theme, Theme):
            theme = replace(apply_colorblind(theme, getattr(self._tui_config, "colorblind", False)), hyperlinks=getattr(self._tui_config, "hyperlinks", True))
        self.theme: Any = theme
        self._viewport_scroll = 0
        self._turn_started = 0.0
        self._todo_completed_at: dict[str, float] = {}
        self._expanded_tool_id: str | None = None
        self._last_tool_card: Block | None = None
        self._theme_poll_task: asyncio.Task[Any] | None = None
        self.composer_shape = composer_shape
        self._themes = dict(themes or {})
        current_theme_id = str(
            getattr(theme, "id", theme.get("id", "default") if isinstance(theme, Mapping) else "default")
        )
        self._themes.setdefault(current_theme_id, self._base_theme)
        if registry is None:
            block_renderers: Any = {
                **DEFAULT_RENDERERS,
                "task": render_task,
                "delivery": render_delivery,
            }
        else:
            block_renderers = {
                entry.kind: entry.render for entry in registry.block_renderers
            }
            block_renderers.setdefault("task", render_task)
            block_renderers.setdefault("delivery", render_delivery)
        self._block_dispatcher = BlockRendererDispatcher(block_renderers)
        previous_ui = getattr(ctx, "ui", None)
        self._fallback_show = getattr(previous_ui, "_show_overlay", None)
        self.ui = UIFacade(
            show_overlay=self._show_overlay,
            toggle_overlay=self._toggle_overlay,
            notify=self._notify,
            clear=self._clear_scrollback,
            set_theme=self._set_theme,
        )
        self.ui.theme = theme
        self.ui.frame = self.frame
        self.ui.active_agent = None
        self.ui.themes = self._themes
        self.ui.history = history
        self._status = status or self._status_text
        self._pending: set[asyncio.Future[Any]] = set()
        self._terminal_pending: set[asyncio.Future[Any]] = set()
        self._submit_lock = asyncio.Lock()
        self._commit_lock = asyncio.Lock()
        self._early_notifications: list[str] = []
        self._scrollback = console or Console()
        self.queue = PromptQueue()
        self.ui.queue = self.queue
        if ctx is not None:
            ctx.ui = self.ui
            ctx.queue = self.queue
        self.streaming = False
        self._shutting_down = False
        self._active_turn: asyncio.Task[Any] | None = None
        advisor_cfg = getattr(getattr(ctx, "cfg", None), "advisor", None)
        self.advisor = (
            AdvisorService(ctx, submit_followup=self._submit_advisor_followup)
            if ctx is not None and bool(getattr(advisor_cfg, "enabled", False))
            else None
        )
        self._active_overlay: Overlay | None = None
        self._overlay_lock = asyncio.Lock()
        self._last_escape = 0.0
        self._last_interrupt = 0.0
        self._last_left = 0.0
        self._drilled_run_id: str | None = None
        self._drilled_frame: Frame | None = None
        self._last_drill_refresh = float("-inf")
        self._drill_refresh_task: asyncio.Future[Any] | None = None
        self._turn_active = False
        self._spinner_frame = 0
        self._hud_sections = {
            kind: Block(f"hud-{kind}", kind) for kind in ("todo", "queue")
        }
        self._approval_notification_sent = False
        self._shell_runner = shell_runner
        self._shell_process: asyncio.subprocess.Process | None = None
        self._custom_editor = editor_runner is not None
        self._editor_runner = editor_runner or self._run_editor_process
        self.thinking_level = self._restore_thinking_level()
        self.ui.thinking_level = self.thinking_level

        self._completion_registry = completion_registry
        completer = ComposerCompleter(completion_registry, self._cwd())
        self.composer = Composer(
            shape=composer_shape,
            theme=theme,
            model=self._model_label,
            thinking=lambda: self.thinking_level,
            history=history,
            completer=completer,
            accept_handler=self._accept,
        )
        self.composer.on_paste_peek = lambda text: self._track(self.ui.show(PasteOverlay(text)))
        self.buffer = self.composer.buffer
        self._restore_draft()
        effective = load_keybindings(
            user_path=keybindings_path,
            registry=completion_registry,
            warn=self._notify,
        )
        self.composer.configure_keys(effective)
        self._effective_keys = effective
        self.ui.effective_keys = effective
        self.ui.prepare_session_switch = self.prepare_session_switch
        handlers = self._action_handlers()
        bindings = create_key_bindings(effective, handlers)
        self._tree_handler = handlers["tree"]
        self._tree_double_escape = "escape escape" in effective.get("tree", ())
        core_bindings = KeyBindings()

        @core_bindings.add(
            "escape",
            filter=Condition(lambda: self._active_overlay is None) & (~vi_mode | vi_navigation_mode),
        )
        def _escape(event: Any) -> None:
            self._escape_ladder(event)

        if self._tree_double_escape:
            core_bindings.add("s-escape")(self._tree_handler)

        @core_bindings.add(
            "left",
            filter=Condition(
                lambda: (
                    self._active_overlay is None
                    and self._drilled_run_id is not None
                    and self.buffer.cursor_position == 0
                )
            ),
        )
        def _drill_back(event: Any) -> None:
            now = time.monotonic()
            if now - self._last_left <= 0.5:
                self._last_left = 0.0
                self._leave_agent()
            else:
                self._last_left = now
            event.app.invalidate()

        @core_bindings.add("?")
        def _help_or_insert(event: Any) -> None:
            if (
                self._active_overlay is None
                and event.current_buffer is self.buffer
                and not self.buffer.text
            ):
                self._track(self.ui.show("help"))
            else:
                event.current_buffer.insert_text("?")

        overlay_bindings = DynamicKeyBindings(
            lambda: self._active_overlay.bindings
            if self._active_overlay is not None
            else KeyBindings()
        )
        bindings = merge_key_bindings([bindings, core_bindings, overlay_bindings])

        root = FloatContainer(
            content=HSplit(
                [
                    Window(height=Dimension(weight=1)),
                    Window(
                        FormattedTextControl(self._viewport_fragments),
                        height=Dimension(min=0),
                        dont_extend_height=True,
                    ),
                    Window(
                        FormattedTextControl(self._hud_text),
                        height=self._hud_height,
                        dont_extend_height=True,
                    ),
                    self.composer.completion_container,
                    self.composer.container,
                    Window(
                        FormattedTextControl(self._status),
                        height=1,
                    ),
                ],
                height=self._root_height,
            ),
            floats=[],
        )
        self._root = root
        kwargs: dict[str, Any] = {}
        prompt_style = _completion_style(theme)
        if prompt_style is not None:
            kwargs["style"] = prompt_style
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        self.application: Application[None] = Application(
            layout=Layout(root, focused_element=self.buffer),
            key_bindings=bindings,
            full_screen=False,
            min_redraw_interval=1 / 60,
            max_render_postpone_time=0.05,
            mouse_support=Condition(lambda: self._active_overlay is not None or self._mouse_mode() == "full"),
            editing_mode=EditingMode.VI if getattr(self._tui_config, "vim", False) else EditingMode.EMACS,
            cursor=ModalCursorShapeConfig(),
            **kwargs,
        )
        self._paint_output = _PaintOutput(self.application, enabled=getattr(self._tui_config, "synchronized_output", True))
        self.ui.apply_settings = self._apply_settings
        self.ui.application = self.application
        self._original_scrollback_file = self._scrollback.file
        if self._paint_output.pump is not None:
            # Production passes an existing Rich console; it must use the same
            # ordered writer as PT so commits cannot race differential frames.
            self._scrollback.file = self._paint_output.pump
        original_resize = self.application._on_resize
        def resize() -> None:
            if getattr(self._tui_config, "resize", "preserve") == "rebuild":
                self._block_dispatcher.clear_cache()
                self._viewport_scroll = 0
            original_resize()
        self.application._on_resize = resize
        self.application.ttimeoutlen = 0.1
        self.application.timeoutlen = 0.1
        self.ui.invalidate = self.application.invalidate
        self.ui.status_width = lambda: self.application.output.get_size().columns
        symbols = str(getattr(getattr(ctx, "cfg", None), "symbols", "unicode"))
        encoding_value = getattr(self.application.output, "encoding", "utf-8")
        try:
            encoding = encoding_value() if callable(encoding_value) else encoding_value
        except Exception:
            encoding = ""
        unicode_title = symbols != "ascii" and "utf" in str(encoding).casefold()
        self.title = TerminalTitle(self.application.output, unicode=unicode_title)
        self.notifier = DesktopNotifier(
            enabled=bool(getattr(getattr(ctx, "cfg", None), "notify", False)),
            output=self.application.output,
        )
        self._paint_output.focus_changed = self.notifier.set_focused
        self._outstanding_agents = agent_counts(ctx)[2] if ctx is not None else 0
        self.application.key_processor.before_key_press += self._record_keypress
        self._refresh_title()
        self.scheduler = FrameScheduler(
            self.frame,
            commit=self._commit_blocks,
            invalidate=self.application.invalidate,
            spinning=self._has_spinner_activity,
            on_spinner_tick=self._spinner_tick,
        )
        self.transcript = Transcript(
            self.frame,
            registry=registry,
            scheduler=self.scheduler,
        )
        self.ui.subagents = []
        self.ui.todos = []
        self._refresh_title()

    @property
    def active_overlay(self) -> Overlay | None:
        return self._active_overlay

    @property
    def drilled_run_id(self) -> str | None:
        return self._drilled_run_id

    def _drill_in(self, run_id: str) -> bool:
        agents = getattr(self.ctx, "agents", None)
        get_run = getattr(agents, "get", None)
        run = get_run(run_id) if callable(get_run) else None
        if run is None:
            return False
        self._drilled_run_id = run_id
        self.ui.active_agent = run
        self._last_left = 0.0
        self._refresh_drilled_frame(force=True)
        self._refresh_title()
        self.application.invalidate()
        return True

    def _leave_agent(self) -> bool:
        if self._drilled_run_id is None:
            return False
        self._cancel_drilled_refresh()
        self._drilled_run_id = None
        self._drilled_frame = None
        self.ui.active_agent = None
        self._last_left = 0.0
        self._refresh_title()
        try:
            self.application.layout.focus(self.buffer)
        except ValueError:
            pass
        self.application.invalidate()
        return True

    def _refresh_drilled_frame(
        self,
        *,
        force: bool = False,
        include_live: bool = True,
    ) -> bool:
        run_id = self._drilled_run_id
        if run_id is None:
            return False
        now = time.monotonic()
        if not force and now - self._last_drill_refresh < 0.25:
            return False
        agents = getattr(self.ctx, "agents", None)
        get_run = getattr(agents, "get", None)
        run = get_run(run_id) if callable(get_run) else None
        if run is None:
            return self._leave_agent()
        self._last_drill_refresh = now
        frame = ledger_transcript_frame(self.ctx, run)
        if include_live:
            frame.blocks.extend(
                deepcopy(block)
                for block in self.frame.blocks
                if block.source_id == run_id and block.state is not BlockState.COMMITTED
            )
        self._drilled_frame = frame
        self.ui.active_agent = run
        return True

    def _cancel_drilled_refresh(self) -> None:
        pending = self._drill_refresh_task
        if pending is not None and not pending.done():
            self._pending.discard(pending)
            self._terminal_pending.discard(pending)
            pending.cancel()
        self._drill_refresh_task = None

    def _schedule_drilled_refresh(self) -> None:
        pending = self._drill_refresh_task
        if pending is not None and not pending.done():
            return
        delay = max(0.0, 0.25 - (time.monotonic() - self._last_drill_refresh))

        async def refresh() -> None:
            await asyncio.sleep(delay)
            if self._drilled_run_id is not None:
                self._refresh_drilled_frame(force=True, include_live=True)
                self.application.invalidate()

        self._drill_refresh_task = self._track(refresh())

    async def _send_to_drilled(self, text: str) -> bool:
        run_id = self._drilled_run_id
        send = getattr(getattr(self.ctx, "agents", None), "send", None)
        if run_id is None or not callable(send) or not text.strip():
            return False
        try:
            await send(run_id, text.strip())
        except Exception as exc:
            self.ui.notify(f"{type(exc).__name__}: {exc}")
            return False
        self._refresh_drilled_frame(force=True)
        self.application.invalidate()
        return True

    async def _send_drilled_batch(self, prompts: list[str]) -> None:
        for prompt in prompts:
            if not await self._send_to_drilled(prompt):
                break

    def _record_keypress(self, _event: Any = None) -> None:
        self.notifier.record_keypress()

    def _session_title(self) -> str:
        if self._drilled_run_id is not None:
            run = getattr(self.ui, "active_agent", None)
            if run is not None:
                return f"{getattr(run, 'name', 'agent')} · {self._drilled_run_id}"
        session = getattr(self.ctx, "session", None)
        session_id = getattr(self.ctx, "session_id", None)
        try:
            info = session.get(session_id) if session is not None and session_id is not None else None
        except Exception:
            info = None
        return str(getattr(info, "title", None) or "new session")

    def _refresh_title(self) -> None:
        self.title.set_session(self._session_title())
        outstanding = agent_counts(self.ctx)[2] if self.ctx is not None else 0
        self.title.set_agents(outstanding)

    def _has_spinner_activity(self) -> bool:
        outstanding = agent_counts(self.ctx)[2] if self.ctx is not None else 0
        return self._turn_active or outstanding > 0 or any(time.monotonic() - value < 0.3 for value in self._todo_completed_at.values())

    def _spinner_tick(self, frame: int) -> None:
        self._spinner_frame = frame
        self.ui._spinner_frame = frame
        spinner = theme_spinner(self.theme, "spinner.status", frame, ("✻",))
        self.title.set_spinner(spinner)

    def set_todos(self, todos: Any) -> None:
        previous = {str(item.get("content", item.get("text", ""))): item.get("status") for item in self.ui.todos if isinstance(item, Mapping)}
        self.ui.set_todos(todos)
        current = {str(item.get("content", item.get("text", ""))): item for item in self.ui.todos if isinstance(item, Mapping)}
        self._todo_completed_at = {key: value for key, value in self._todo_completed_at.items() if key in current}
        for key, item in current.items():
            if item.get("status") == "completed" and previous.get(key) != "completed":
                self._todo_completed_at[key] = time.monotonic()
        if self._todo_completed_at:
            self.scheduler.start_spinner()
        self.application.invalidate()

    def _hud_block(self, kind: str, data: Mapping[str, Any]) -> Block:
        block = self._hud_sections[kind]
        snapshot = deepcopy(dict(data))
        if block.data != snapshot:
            block.data.clear()
            block.update(snapshot)
        return block

    def _hud_blocks(self) -> list[Block]:
        blocks: list[Block] = []
        if self.ui.todos:
            items = []
            for item in self.ui.todos[:7]:
                if isinstance(item, Mapping):
                    item = dict(item)
                    key = str(item.get("content", item.get("text", "")))
                    if key in self._todo_completed_at:
                        item["completion_progress"] = min(1.0, (time.monotonic() - self._todo_completed_at[key]) / 0.25)
                items.append(item)
            blocks.append(self._hud_block("todo", {"items": items}))
        if self.queue:
            blocks.append(
                self._hud_block(
                    "queue",
                    {
                        "prompts": [
                            {"text": item.text, "mode": item.mode}
                            for item in self.queue.entries
                        ],
                        "dequeue_hint": format_key_bindings(
                            effective
                            for effective in self._effective_keys.get("dequeue", ())
                        ),
                    },
                )
            )
        return blocks

    @staticmethod
    def _clip_visual_rows(value: str, rows: int = 8) -> str:
        return "\n".join(value.rstrip("\n").splitlines()[:rows])

    def _hud_text(self) -> Any:
        width = max(1, self.application.output.get_size().columns)
        rendered = [
            self._clip_visual_rows(
                self._capture_block(
                    block,
                    width,
                    8,
                    force_terminal=True,
                )
            )
            for block in self._hud_blocks()
        ]
        return ANSI("\n".join(value for value in rendered if value))

    def _hud_height(self) -> int:
        value = self._hud_text().value
        return len(value.splitlines()) if value else 0

    async def handle_presentation(self, event: Event) -> None:
        if getattr(self.ctx, "agents", None) is None:
            if isinstance(event, ToolCallStart) and event.name == "task":
                self.ui.subagents.append(
                    {
                        "id": event.id,
                        "name": event.id,
                        "description": str(
                            event.args.get("description")
                            or event.args.get("task")
                            or "task"
                        ),
                        "status": "running",
                    }
                )
            elif isinstance(event, ToolCallEnd) and event.name == "task":
                self.ui.subagents[:] = [
                    agent
                    for agent in self.ui.subagents
                    if not isinstance(agent, Mapping) or agent.get("id") != event.id
                ]

        active_overlay = self._active_overlay
        if isinstance(active_overlay, HubOverlay):
            active_overlay.refresh_from_event(event)

        source_id = str(getattr(event, "source_id", "main"))
        drilled_event = (
            source_id == self._drilled_run_id
            or getattr(event, "run_id", None) == self._drilled_run_id
        )
        if self._drilled_run_id is not None and (
            drilled_event
            or isinstance(event, (AgentSpawned, AgentStatus, AgentFinished, AgentDelivered))
        ):
            lifecycle_refresh = isinstance(
                event, (TurnEnd, AgentStatus, AgentFinished, AgentDelivered)
            )
            if lifecycle_refresh:
                self._cancel_drilled_refresh()
            refreshed = self._refresh_drilled_frame(
                force=lifecycle_refresh,
                include_live=not lifecycle_refresh,
            )
            if not refreshed and drilled_event:
                self._schedule_drilled_refresh()

        if isinstance(event, TurnStart) and source_id == "main":
            self._turn_active = True
            self._turn_started = time.monotonic()
            self._viewport_scroll = 0
            self._expanded_tool_id = None
            self._refresh_title()
            spinner = theme_spinner(self.theme, "spinner.status", self._spinner_frame, ("✻",))
            self.title.set_turn(True, spinner=spinner)
            self.scheduler.start_spinner()
        elif isinstance(event, TurnEnd) and source_id == "main":
            self._turn_active = False
            self.title.set_turn(False)
            await self.notifier.notify("Orcha", "Turn complete")
        elif (
            isinstance(event, AgentFinished)
            and event.parent_id == "main"
            and not self._shutting_down
        ):
            self._track(self._submit_serially(None))
        elif isinstance(event, InterruptRaised):
            if self._approval_notification_sent:
                self._approval_notification_sent = False
            else:
                await self.notifier.notify("Orcha", "Approval required")
        elif isinstance(event, SessionSwitch):
            self._leave_agent()
            self._outstanding_agents = agent_counts(self.ctx)[2]
            self._refresh_title()

        if isinstance(event, (AgentSpawned, AgentStatus, AgentFinished, AgentDelivered)):
            previous = self._outstanding_agents
            self._outstanding_agents = agent_counts(self.ctx)[2]
            self.title.set_agents(self._outstanding_agents)
            if previous > 0 and self._outstanding_agents == 0 and not self._shutting_down:
                await self.notifier.notify("Orcha", "All agent jobs settled")
            if self._outstanding_agents:
                self.scheduler.start_spinner()
        self.application.invalidate()

    def _resolve_overlay(
        self,
        overlay: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Overlay | None:
        if isinstance(overlay, Overlay):
            if args or kwargs:
                raise TypeError("arguments cannot be passed with an overlay instance")
            return overlay
        if not isinstance(overlay, str):
            raise TypeError("overlay must be an Overlay instance or registered name")
        registration = (
            None if self.registry is None else self.registry.overlays.get(overlay)
        )
        if registration is None:
            return None
        created = registration.factory(self.ctx, *args, **kwargs)
        if not isinstance(created, Overlay):
            raise TypeError(f"overlay factory {overlay!r} did not return Overlay")
        return created

    async def _toggle_overlay(
        self,
        overlay: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if overlay == "hub" and isinstance(self._active_overlay, HubOverlay):
            self._active_overlay.cancel()
            return None
        return await self._show_overlay(overlay, *args, **kwargs)

    async def _show_overlay(
        self,
        overlay: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        approval = isinstance(overlay, str) and overlay == "approval"
        if approval:
            self.title.set_approval(True)
            await self.notifier.notify("Orcha", "Approval required")
            self._approval_notification_sent = True
        resolved = self._resolve_overlay(overlay, args, kwargs)
        if resolved is None:
            try:
                if self._fallback_show is not None:
                    return await self._fallback_show(overlay, *args, **kwargs)
                raise RuntimeError(f"overlay {overlay!r} is unavailable")
            finally:
                if approval:
                    self.title.set_approval(False)
        async with self._overlay_lock:
            if self._shutting_down:
                resolved.cancel()
                if approval:
                    self.title.set_approval(False)
                return None
            self._active_overlay = resolved
            self._root.floats.extend((resolved.backdrop, resolved))
            try:
                try:
                    self.application.layout.focus(resolved.focus_target)
                except ValueError:
                    pass
                result = await resolved.wait()
                if isinstance(resolved, HubOverlay) and isinstance(result, str):
                    self._drill_in(result)
                return result
            finally:
                for overlay_float in (resolved, resolved.backdrop):
                    if overlay_float in self._root.floats:
                        self._root.floats.remove(overlay_float)
                self._active_overlay = None
                if approval:
                    self.title.set_approval(False)
                self.application.layout.focus(self.buffer)
                self.application.invalidate()


    def _cwd(self) -> Path:
        value = getattr(getattr(self.ctx, "cfg", None), "cwd", Path.cwd())
        return Path(value)

    def _model_label(self) -> str:
        value = getattr(getattr(self.ctx, "cfg", None), "model", "model")
        return value[0] if isinstance(value, list) and value else str(value)

    def _status_text(self) -> Any:
        self.ui.vim_mode = (str(self.application.vi_state.input_mode.value) if self.application.editing_mode == EditingMode.VI else None)
        if self.ctx is None:
            if self._turn_active:
                elapsed = int(time.monotonic() - self._turn_started)
                return [("class:accent", f"{self.title.spinner} orcha · {elapsed}s")]
            return [("class:accent", "orcha")]
        return render_statusline(
            self.ctx,
            self.theme,
            width=self.application.output.get_size().columns,
            composer_shape=self.composer_shape,
        )

    def _composer_state(self) -> dict[str, Any]:
        states = getattr(self.ctx, "plugin_states", None)
        if not isinstance(states, dict):
            return {}
        return states.setdefault("composer", {})

    def _restore_thinking_level(self) -> str:
        value = self._composer_state().get("thinking_level", "off")
        return value if value in {"off", "low", "medium", "high", "max"} else "off"

    def _restore_draft(self) -> None:
        state = self._composer_state()
        draft = state.get("draft")
        saved_queue = state.get("queue")
        has_draft = isinstance(draft, str) and bool(draft)
        restored_queue = (
            isinstance(saved_queue, list)
            and bool(saved_queue)
            and self.queue.restore(saved_queue)
        )
        if not has_draft and not restored_queue:
            return
        if has_draft:
            self.buffer.text = draft
            self.buffer.cursor_position = len(draft)
            state.pop("draft", None)
        if restored_queue:
            state.pop("queue", None)
        self._persist_state()

    def prepare_session_switch(self) -> None:
        """Copy the outgoing editor state into the current session state."""

        state = self._composer_state()
        draft = self.composer.expanded_text(self.buffer.text)
        if draft:
            state["draft"] = draft
        else:
            state.pop("draft", None)
        if self.queue:
            state["queue"] = self.queue.dump()
        else:
            state.pop("queue", None)
        state["thinking_level"] = self.thinking_level

    async def rebind_session(self, _event: SessionSwitch) -> None:
        """Restore editor-local state after AppContext activates a session."""

        self.buffer.reset(append_to_history=False)
        self.composer.forget_pastes()
        clear_terminal_cache()
        self._last_tool_card = None
        self._expanded_tool_id = None
        self.queue.clear()
        self.thinking_level = self._restore_thinking_level()
        self.ui.thinking_level = self.thinking_level
        self.buffer.completer = ComposerCompleter(self._completion_registry, self._cwd())
        history = self.buffer.history
        if isinstance(history, SQLiteHistory):
            history.rebind(cwd=self._cwd(), session_id=str(self.ctx.session_id))
        self._restore_draft()
        prefix = self._thinking_provider_prefix()
        if prefix is not None:
            self._apply_thinking_level(prefix)
            self._persist_state()
            await self.ctx.rebuild()
        self.application.invalidate()

    def _persist_state(self) -> None:
        persist = getattr(self.ctx, "persist_plugin_states", None)
        if persist is not None:
            persist()

    def _accept(self, buffer: Any) -> bool:
        raw = self.composer.expanded_text(buffer.text)
        text = raw.strip()
        if not text:
            if self.streaming and self.queue:
                buffer.reset(append_to_history=False)
                self._abort_turn()
            return False
        self.transcript.dismiss_error()
        buffer.text = raw
        buffer.reset(append_to_history=True)
        self.composer.forget_pastes()
        if text == ".":
            text = "keep going"
        if self.streaming and text.startswith("/"):
            self._track(self._dispatch_submission(text))
            return False
        prompts = split_submission(text)
        if self._drilled_run_id is not None:
            self._track(self._send_drilled_batch(prompts))
            return False
        if self.advisor is not None:
            self.advisor.before_user_prompt()
        is_batch = len(prompts) > 1
        if self.streaming:
            mode = "steer" if self.queue.steering_open else "follow_up"
            self.queue.extend(prompts, mode=mode)
            self.application.invalidate()
            return False
        first = prompts.pop(0)
        if is_batch:
            self.queue.extend(prompts, mode="follow_up")
        self._track(self._submit_serially(first))
        return False

    def _action_handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "submit": self._submit_action,
            "newline": self._newline_or_followup,
            "queue": self._queue_draft,
            "dequeue": self._dequeue,
            "toggle_thinking": lambda _event: self._toggle_thinking(),
            "cycle_thinking_level": lambda _event: self._track(self._cycle_thinking_level()),
            "expand_tools": lambda _event: self._toggle_last_tool(),
            "model_picker": lambda _event: self._track(self.ui.show("model")),
            "cycle_model": lambda _event: self._track(self._cycle_model()),
            "history_search": lambda _event: self._track(self._history_search()),
            "external_editor": lambda _event: self._track(self._external_editor()),
            "clear_screen": lambda _event: self._track(self.ui.clear()),
            "tree": lambda _event: self._track(self.ui.show("tree")),
            "interrupt": self._interrupt,
            "exit": self._exit,
            **self._plugin_handlers(),
        }

    def _plugin_handlers(self) -> dict[str, Callable[[Any], None]]:
        handlers: dict[str, Callable[[Any], None]] = {}
        if self.registry is None:
            return handlers
        for action, registration in self.registry.keybindings.items():
            def invoke(event: Any, registration: Any = registration) -> None:
                result = registration.handler(self.ctx, event)
                if inspect.isawaitable(result):
                    self._track(result)
            handlers[action] = invoke
        return handlers

    def _toggle_last_tool(self) -> None:
        frame = self._drilled_frame or self.frame
        last = next((block for block in reversed(frame.blocks) if block.kind == "tool"), self._last_tool_card)
        if last is not None:
            self._expanded_tool_id = None if self._expanded_tool_id == last.id else last.id
        self.ui.expand_tools(not self.ui.tools_expanded)
        self.application.invalidate()

    def _submit_action(self, event: Any) -> None:
        buffer = event.current_buffer
        if buffer.text.endswith("\\"):
            buffer.text = buffer.text[:-1] + "\n"
            buffer.cursor_position = len(buffer.text)
            return
        buffer.validate_and_handle()

    def _queue_draft(self, event: Any) -> None:
        if not self.streaming:
            return
        text = self.composer.expanded_text(event.current_buffer.text).strip()
        if text:
            self.queue.extend(split_submission(text), mode="follow_up")
            event.current_buffer.reset(append_to_history=False)
            self.composer.forget_pastes()
            event.app.invalidate()

    def _newline_or_followup(self, event: Any) -> None:
        if self.streaming:
            self._queue_draft(event)
        else:
            event.current_buffer.insert_text("\n")

    def _dequeue(self, event: Any) -> None:
        text = self.queue.pop_last()
        if text is not None:
            self._merge_restored_draft(event.current_buffer, text)

    @staticmethod
    def _merge_restored_draft(buffer: Any, restored: str) -> None:
        active = buffer.text
        text = "\n\n".join(value for value in (restored, active) if value)
        buffer.text = text
        buffer.cursor_position = len(text)

    def _abort_turn(self) -> None:
        if self._active_turn is not None and not self._active_turn.done():
            self._active_turn.cancel()

    def _escape_ladder(self, event: Any) -> None:
        if self._drilled_run_id is not None:
            self._leave_agent()
            return
        buffer = event.current_buffer
        if buffer.complete_state is not None:
            buffer.cancel_completion()
            return
        if self.streaming:
            restored = self.queue.restore_text()
            if restored:
                self._merge_restored_draft(buffer, restored)
            self._abort_turn()
            return
        if buffer.text or not self._tree_double_escape:
            return
        now = time.monotonic()
        if now - self._last_escape <= 0.5:
            self._last_escape = 0.0
            self._tree_handler(event)
        else:
            self._last_escape = now

    def _interrupt(self, event: Any) -> None:
        buffer = event.current_buffer
        if buffer.text:
            self.composer.clear_draft()
            buffer.reset(append_to_history=False)
            self._last_interrupt = 0.0
            return
        if self.streaming:
            self._abort_turn()
            self._last_interrupt = 0.0
            return
        now = time.monotonic()
        if now - self._last_interrupt <= 1.0:
            event.app.exit()
        else:
            self._last_interrupt = now

    def _exit(self, event: Any) -> None:
        self._shutting_down = True
        state = self._composer_state()
        text = self.composer.expanded_text(event.current_buffer.text)
        if text:
            state["draft"] = text
        if self.queue:
            state["queue"] = self.queue.dump()
        if text or self.queue:
            self._persist_state()
        if self.streaming:
            self._abort_turn()
        event.app.exit()

    async def _history_search(self) -> None:
        selected = await self.ui.show("history")
        if isinstance(selected, str):
            self.buffer.text = selected
            self.buffer.cursor_position = len(selected)

    def _thinking_provider_prefix(self) -> str | None:
        if self.registry is None or self.ctx is None:
            return None
        cfg = self.ctx.cfg
        prefix = _primary_provider_prefix(cfg.model, getattr(cfg, "models", {}))
        registration = self.registry.providers.get(prefix) if prefix else None
        if registration is None or not registration.capabilities.thinking:
            return None
        return prefix

    def _thinking_supported(self) -> bool:
        return self._thinking_provider_prefix() is not None

    def _apply_thinking_level(self, prefix: str) -> None:
        providers = getattr(self.ctx.cfg, "providers", None)
        if isinstance(providers, dict):
            options = providers.setdefault(prefix, {})
            if prefix == "anthropic" and self.thinking_level == "off":
                options.pop("reasoning_effort", None)
            else:
                options["reasoning_effort"] = self.thinking_level
        if prefix == "anthropic":
            self.ctx.plugin_states.setdefault("provider_anthropic", {})[
                "thinking"
            ] = "off" if self.thinking_level == "off" else "summary"

    def _toggle_thinking(self) -> None:
        state = self.ctx.plugin_states.setdefault("render_default", {})
        configured = str(getattr(self.ctx.cfg, "thinking", "summary"))
        current = str(state.get("thinking", configured))
        state["thinking"] = "summary" if current == "off" else "off"
        self.ui.thinking_visible = state["thinking"] != "off"
        self._persist_state()
        self.application.invalidate()

    async def _cycle_thinking_level(self) -> None:
        prefix = self._thinking_provider_prefix()
        if prefix is None:
            self.ui.notify("Thinking levels are unavailable for the active provider.")
            return
        levels = ("off", "low", "medium", "high", "max")
        self.thinking_level = levels[(levels.index(self.thinking_level) + 1) % len(levels)]
        self.ui.thinking_level = self.thinking_level
        self._composer_state()["thinking_level"] = self.thinking_level
        self._apply_thinking_level(prefix)
        self._persist_state()
        await self.ctx.rebuild()
        self.application.invalidate()

    async def _cycle_model(self) -> None:
        if self.registry is None or self.ctx is None:
            return
        available = self._available_model_aliases()
        if not available:
            self.ui.notify("No configured model aliases are available.")
            return
        current = self._model_label()
        index = available.index(current) + 1 if current in available else 0
        await self.ctx.switch_model(available[index % len(available)])

    def _available_model_aliases(self) -> list[str]:
        if self.registry is None or self.ctx is None:
            return []
        aliases = getattr(self.ctx.cfg, "models", {})
        if not isinstance(aliases, Mapping):
            return []
        resolver = ModelResolver(self.registry, self.ctx.cfg)
        available: list[str] = []
        for alias in aliases:
            try:
                resolver.resolve(alias, "main")
            except (RuntimeError, ValueError):
                continue
            available.append(alias)
        return available

    async def _external_editor(self) -> None:
        if not self._custom_editor and not (os.environ.get("VISUAL") or os.environ.get("EDITOR")):
            self.ui.notify("Set $VISUAL or $EDITOR to edit the draft externally.")
            return
        original = self.composer.expanded_text(self.buffer.text)
        try:
            edited = await run_in_terminal(lambda: self._editor_runner(original))
        except (OSError, subprocess.SubprocessError) as exc:
            self.ui.notify(f"External editor failed: {exc}")
            return
        if isinstance(edited, str):
            self.buffer.text = edited
            self.buffer.cursor_position = len(edited)

    @staticmethod
    def _run_editor_process(text: str) -> str:
        command = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not command:
            return text
        path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md", delete=False) as stream:
                stream.write(text)
                path = stream.name
            subprocess.run([*shlex.split(command), path], check=True)
            return Path(path).read_text(encoding="utf-8")
        finally:
            if path:
                Path(path).unlink(missing_ok=True)


    async def _run_shell(self, command: str) -> None:
        identifier = f"execute-{time.monotonic_ns()}"
        bus = getattr(self.ctx, "_bus", None)
        if bus is not None:
            await bus.emit(ToolCallStart(name="execute", args={"command": command}, id=identifier))
        result: dict[str, Any]
        try:
            if self._shell_runner is None:
                result = await self._run_shell_process(command)
            else:
                # Injected runners are deterministic test adapters. Keeping them
                # on-loop avoids leaving asyncio's process-wide default executor
                # alive after a headless prompt-toolkit application exits.
                completed = self._shell_runner(command, self._cwd(), 60.0)
                result = {
                    "returncode": int(getattr(completed, "returncode", 0)),
                    "stdout": str(getattr(completed, "stdout", "")),
                    "stderr": str(getattr(completed, "stderr", "")),
                }
        except (TimeoutError, subprocess.TimeoutExpired):
            await self._stop_shell_process()
            result = {
                "returncode": 124,
                "stdout": "",
                "stderr": "timed out after 60 seconds",
            }
        except asyncio.CancelledError:
            result = {"returncode": 130, "stdout": "", "stderr": "cancelled"}
            process = self._shell_process
            self._shell_process = None
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                asyncio.create_task(process.wait())
            raise
        except Exception as exc:
            await self._stop_shell_process()
            result = {
                "returncode": 1,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if bus is not None:
                await bus.emit(ToolCallEnd(name="execute", id=identifier, result=result))

    async def _run_shell_process(self, command: str) -> Any:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self._cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._shell_process = process
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
        self._shell_process = None
        return {
            "returncode": int(process.returncode or 0),
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def _stop_shell_process(self) -> None:
        process = self._shell_process
        if process is None:
            return
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        self._shell_process = None

    async def _dispatch_submission(self, text: str) -> None:
        if text.strip() == "/settings":
            await self.ui.show("settings")
            return
        if text.startswith("!"):
            await self._run_shell(text[1:].strip())
        else:
            await self._submit(text)

    @staticmethod
    def _agent_delivery_payload(run: Any) -> Any:
        findings = getattr(run, "partial_findings", None)
        if not findings:
            return run.result
        return {
            "result": run.result,
            "partial_findings": findings,
        }

    @staticmethod
    def _agent_delivery_snapshot(job: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = dict(job)
        findings = snapshot.get("partial_findings")
        if findings:
            snapshot["result"] = {
                "result": snapshot.get("result"),
                "partial_findings": findings,
            }
        return snapshot

    @classmethod
    def _agent_delivery_notification(cls, runs: list[Any]) -> str:
        lines = ["<system-notification>"]
        for index, run in enumerate(runs):
            if index:
                lines.append("")
            lines.append(f"Job {run.id} ({run.name}) finished: {run.status}")
            payload = json.dumps(
                cls._agent_delivery_payload(run),
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(
                payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
        lines.append("</system-notification>")
        return "\n".join(lines)

    @classmethod
    async def _prepare_agent_delivery(cls, event: AgentDelivered) -> None:
        event.jobs = tuple(
            cls._agent_delivery_snapshot(job) if isinstance(job, Mapping) else job
            for job in event.jobs
        )

    async def _claim_agent_delivery(self) -> str | None:
        agents = getattr(self.ctx, "agents", None)
        if agents is None:
            return None
        runs = [
            run
            for run in agents.jobs("main")
            if run.terminal and not run.delivered
        ]
        if not runs:
            return None
        notification = self._agent_delivery_notification(runs)
        delivered = await agents.deliver("main", (run.id for run in runs))
        if not delivered:
            return None
        if [run.id for run in delivered] != [run.id for run in runs]:
            return self._agent_delivery_notification(delivered)
        return notification

    def _track(
        self,
        awaitable: Awaitable[Any],
        *,
        terminal: bool = False,
    ) -> asyncio.Future[Any]:
        future = asyncio.ensure_future(awaitable)
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)
        if terminal:
            self._terminal_pending.add(future)
            future.add_done_callback(self._terminal_pending.discard)
        return future

    async def _submit_serially(
        self,
        text: str | None,
        *,
        user_prompt: bool = True,
        expected_session: str | None = None,
    ) -> None:
        if text is not None and user_prompt and self.advisor is not None:
            self.advisor.before_user_prompt()
        async with self._submit_lock:
            if self._shutting_down:
                return
            if (
                expected_session is not None
                and str(getattr(self.ctx, "session_id", "")) != expected_session
            ):
                return
            pending_user = text
            current = await self._claim_agent_delivery()
            if current is None:
                current = pending_user
                pending_user = None
            while current is not None:
                self.streaming = True
                self._active_turn = asyncio.create_task(self._dispatch_submission(current))
                try:
                    await self._active_turn
                except (KeyboardInterrupt, asyncio.CancelledError):
                    self.transcript.append_banner("interrupted", level="warning")
                except Exception as exc:
                    self.transcript.pin_error(f"{type(exc).__name__}: {exc}")
                finally:
                    self.queue.close_steering()
                    self._active_turn = None
                    self.streaming = False
                    self.application.invalidate()
                if self._shutting_down:
                    break
                current = await self._claim_agent_delivery()
                next_is_user = False
                if current is None and pending_user is not None:
                    current = pending_user
                    pending_user = None
                    next_is_user = user_prompt
                elif current is None:
                    current = self.queue.pop(mode="follow_up")
                    next_is_user = current is not None
                if next_is_user and self.advisor is not None:
                    self.advisor.before_user_prompt()
                if current is not None:
                    self.application.invalidate()

    async def _submit_advisor_followup(self, session_id: str, text: str) -> None:
        await self._submit_serially(
            text,
            user_prompt=False,
            expected_session=session_id,
        )

    def _notify(self, text: str) -> None:
        transcript = getattr(self, "transcript", None)
        application = getattr(self, "application", None)
        if transcript is None:
            self._early_notifications.append(text)
            return
        transcript.append_banner(text, level="info")
        if application is not None:
            application.invalidate()

    def flush_early_notifications(self) -> None:
        for notification in self._early_notifications:
            self.transcript.append_banner(notification, level="info")
        self._early_notifications.clear()
        self.application.invalidate()

    def _apply_settings(self, cfg: Any) -> None:
        self.ctx.cfg = cfg
        self._tui_config = getattr(cfg, "tui", None)
        self.notifier.enabled = cfg.notify
        self.composer.set_shape(cfg.composer)
        self.composer_shape = cfg.composer
        self.application.editing_mode = EditingMode.VI if getattr(self._tui_config, "vim", False) else EditingMode.EMACS
        self._paint_output._enabled = getattr(self._tui_config, "synchronized_output", True)
        self._paint_output.synchronized = self._paint_output._enabled and self._paint_output._detected_synchronized
        self._apply_theme(self._base_theme)

    def _apply_theme(self, selected: Any) -> Any:
        selected = self._themes.get(getattr(selected, "id", None), selected)
        self._base_theme = selected
        if isinstance(selected, Theme):
            selected = replace(apply_colorblind(selected, getattr(self._tui_config, "colorblind", False)), hyperlinks=getattr(self._tui_config, "hyperlinks", True))
        self.composer.theme = selected
        self._block_dispatcher.clear_cache()
        self.theme = selected
        self.ui.theme = selected
        prompt_style = _completion_style(selected)
        if prompt_style is not None:
            self.application.style = prompt_style
        self._spinner_tick(self._spinner_frame)
        self._refresh_title()
        self.application.invalidate()
        return selected

    def _set_theme(self, name: str) -> Any:
        return self._apply_theme(select_theme(self._themes, name))

    def replace_themes(
        self,
        themes: Mapping[str, Any],
        selected: Any,
    ) -> Any:
        self._block_dispatcher.clear_cache()
        self._themes = dict(themes)
        identifier = str(getattr(selected, "id", "default"))
        self._themes.setdefault(identifier, selected)
        self.ui.themes = self._themes
        return self._apply_theme(selected)


    def _root_height(self) -> int:
        return max(1, self.application.output.get_size().rows - 1)

    def _render_block(self, block: Block, width: int, rows: int) -> Any:
        return self._block_dispatcher.render(
            block,
            self.theme,
            width,
            rows,
            block.id == self._expanded_tool_id if self._expanded_tool_id is not None else False,
        )

    def _composer_height(self, width: int) -> int:
        return self.composer.text_rows(width)

    def _print_block(
        self,
        console: Console,
        block: Block,
        width: int,
        rows: int,
        *,
        viewport: bool,
    ) -> None:
        if block.kind == "raw":
            options = dict(block.data.get("options", {}))
            objects = block.data.get("objects")
            if objects is not None:
                console.print(*objects, **options)
            else:
                console.print(block.data.get("renderable", ""), **options)
            return
        rendered = self._render_block(block, width, rows)
        if rendered is not None:
            console.print(rendered, end="" if viewport else "\n")

    def _capture_block(
        self,
        block: Block,
        width: int,
        rows: int,
        *,
        force_terminal: bool,
    ) -> str:
        stream = StringIO()
        console = Console(
            file=stream,
            force_terminal=force_terminal,
            # Pin the color system so viewport output does not depend on
            # COLORTERM/NO_COLOR in the launching shell.
            color_system="truecolor" if force_terminal else None,
            width=max(1, width),
            theme=getattr(self.theme, "rich", None),
        )
        self._print_block(console, block, width, rows, viewport=True)
        # PT's ANSI parser understands SGR but not OSC 8. Keep hyperlinks on
        # the native scrollback path and strip only their wrappers in the live UI.
        return re.sub(r"\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)", "", stream.getvalue())

    def _measure_block(self, block: Block, width: int) -> int:
        rendered = self._capture_block(
            block,
            width,
            10_000,
            force_terminal=False,
        )
        lines = rendered.splitlines()
        return max(1, len(lines))

    def _mouse_mode(self) -> str:
        value = getattr(self._tui_config, "mouse", "scroll")
        return ("full" if value else "off") if isinstance(value, bool) else value

    def _viewport_mouse(self, event: Any) -> Any:
        mode = self._mouse_mode()
        if mode == "off":
            return NotImplemented
        if event.event_type == MouseEventType.SCROLL_UP:
            self._viewport_scroll += 3
        elif event.event_type == MouseEventType.SCROLL_DOWN:
            self._viewport_scroll = max(0, self._viewport_scroll - 3)
        elif event.event_type == MouseEventType.MOUSE_UP and mode == "full":
            self.application.layout.focus(self.buffer)
        else:
            return NotImplemented
        self.application.invalidate()
        return None

    def _viewport_fragments(self) -> Any:
        return [(style, text, self._viewport_mouse) for style, text, *_ in self._viewport_text().__pt_formatted_text__()]

    def _viewport_text(self) -> Any:
        size = self.application.output.get_size()
        width = max(1, size.columns)
        budget = Frame.row_budget(
            terminal_rows=self._root_height(),
            composer_rows=(
                self.composer.height_for_width(width)
                + self._hud_height()
                + self.composer.completion_row_count()
            ),
            status_rows=1,
        )
        frame = self._drilled_frame if self._drilled_run_id is not None else self.frame
        if frame is None:
            return ANSI("")
        # Working activity is represented by the status brand; retain retry
        # information as a distinct card and leave transcript accumulation alone.
        visible_frame = Frame()
        visible_frame.blocks = [block for block in frame.blocks if block.kind != "working" or "retry_deadline" in block.data]
        if self._last_tool_card is not None and self._last_tool_card.id == self._expanded_tool_id and not any(block.id == self._expanded_tool_id for block in visible_frame.blocks):
            visible_frame.blocks.append(replace(self._last_tool_card, state=BlockState.SETTLED))
        if self._viewport_scroll or self._expanded_tool_id is not None:
            all_lines = "\n".join(self._capture_block(block, width, 10000, force_terminal=True) for block in visible_frame.blocks).splitlines(keepends=True)
            self._viewport_scroll = min(self._viewport_scroll, max(0, len(all_lines) - budget))
            end = len(all_lines) - self._viewport_scroll
            return ANSI("".join(all_lines[max(0, end - budget):end]))
        rendered: list[str] = []
        plan = visible_frame.viewport_plan(
            budget,
            width=width,
            measure=self._measure_block,
            minimum=lambda block: (
                2 if block.kind in LEADING_SPACER_KINDS else 1
            ),
        )
        for index, item in enumerate(plan):
            render_rows = item.rows
            if (
                item.block.kind in LEADING_SPACER_KINDS
                and render_rows > 1
            ):
                render_rows -= 1
            value = self._capture_block(
                item.block,
                width,
                render_rows,
                force_terminal=True,
            )
            lines = value.splitlines(keepends=True)
            leading: list[str] = []
            content_rows = item.rows
            if (
                item.block.kind in LEADING_SPACER_KINDS
                and lines
                and not lines[0].strip()
            ):
                if content_rows > 1:
                    leading, lines = lines[:1], lines[1:]
                    content_rows -= 1
                else:
                    lines = lines[1:]
            if content_rows <= 0:
                lines = []
            elif len(lines) > content_rows:
                lines = (
                    lines[-content_rows:]
                    if item.block.kind in {"assistant", "thinking"}
                    else lines[:content_rows]
                )
            rendered.append("".join([*leading, *lines]))
            if (
                index + 1 < len(plan)
                and plan[index + 1].block.kind not in LEADING_SPACER_KINDS
            ):
                rendered.append("\n")
        output_lines = "".join(rendered).splitlines(keepends=True)
        if len(output_lines) > budget:
            output_lines = output_lines[-budget:] if budget else []
        return ANSI("".join(output_lines))

    def _write_blocks(self, blocks: list[Block]) -> None:
        width = max(1, self.application.output.get_size().columns)
        for index, block in enumerate(blocks):
            if block.kind == "tool":
                self._last_tool_card = block
            if index and block.kind not in LEADING_SPACER_KINDS:
                self._scrollback.print()
            self._print_block(
                self._scrollback,
                block,
                width,
                10_000,
                viewport=False,
            )
            if self._paint_output.pump is not None:
                protocol = image_protocol(block)
                if protocol:
                    self._paint_output.pump.write(protocol + "\n")

    async def _run_in_app_terminal(self, func: Callable[[], Any]) -> Any:
        """Run a terminal callback bound to this prompt-toolkit application."""
        if not self.application.is_running:
            return func()
        with set_app(self.application):
            return await run_in_terminal(func)

    def _commit_blocks(self, blocks: list[Block]) -> None:
        def write_and_prune() -> None:
            self._write_blocks(blocks)
            # Prune inside the suspended-app window: the redraw that follows
            # run_in_terminal must paint the frame WITHOUT the just-printed
            # blocks, or it re-renders them at full height and scrolls a
            # duplicate (plus composer/status rows) into scrollback.
            self.transcript.release_committed(blocks)
            self.frame.prune_committed(blocks)
            self._block_dispatcher.evict(blocks)
            if any(block.id == self._expanded_tool_id for block in blocks):
                self._expanded_tool_id = None

        async def write_and_release() -> None:
            pump = self._paint_output.pump
            while pump is not None and pump.pending > 262144 and pump.error is None:
                if self._shutting_down:
                    return
                await asyncio.sleep(0.05)
            async with self._commit_lock:
                self._paint_output.begin_frame()
                try:
                    await self._run_in_app_terminal(write_and_prune)
                finally:
                    # run_in_terminal's context exit has already repainted PT.
                    self._paint_output.end_frame()

        self._track(write_and_release(), terminal=True)

    async def _clear_scrollback(self) -> None:
        self.scheduler.commit_now()
        await self._drain(self._terminal_pending)
        await self._track(
            self._run_in_app_terminal(self._scrollback.clear),
            terminal=True,
        )
        self.transcript.clear()
        clear_terminal_cache()
        self._last_tool_card = None
        self._expanded_tool_id = None
        self.application.invalidate()

    async def _drain(self, pending: set[asyncio.Future[Any]]) -> None:
        while pending:
            await asyncio.gather(*tuple(pending))

    async def _drain_pending(self) -> None:
        await self._drain(self._pending)

    async def _poll_themes(self) -> None:
        watcher = ThemeWatcher(Path.home() / ".config/orcha-agent/themes")
        last_background = None
        while True:
            await asyncio.sleep(0.25)
            cfg = getattr(self.ctx, "cfg", None)
            if cfg is None:
                continue
            background = self._paint_output.background
            if background != last_background:
                last_background = background
                selected = theme_from_background("\x1b]11;" + background) if background else None
                if cfg.theme == "auto" and selected in self._themes:
                    self._apply_theme(self._themes[selected])
            if await asyncio.to_thread(watcher.changed, time.monotonic()):
                warnings: list[str] = []
                themes = await asyncio.to_thread(load_themes, cwd=cfg.cwd, trusted=cfg.trust_cwd,
                                                 symbols=cfg.symbols, warn=warnings.append)
                for warning in warnings:
                    self._notify(warning)
                selected = themes.get(getattr(self.theme, "id", "dark"), themes["dark"])
                self.replace_themes(themes, selected)

    async def run(self) -> None:
        self._track(self._submit_serially(None))
        self.application.after_render += lambda _app: self._paint_output.start()
        self._theme_poll_task = asyncio.create_task(self._poll_themes())
        try:
            await self.application.run_async()
        except EOFError:
            pass
        finally:
            self._shutting_down = True
            if self._theme_poll_task is not None:
                self._theme_poll_task.cancel()
                await asyncio.gather(self._theme_poll_task, return_exceptions=True)
            if self._active_overlay is not None:
                self._active_overlay.cancel()
            if self.advisor is not None:
                await self.advisor.aclose()
            await self._drain_pending()
            self.scheduler.commit_now()
            await self._drain_pending()
            await self.scheduler.aclose()
            self.title.set_turn(False)
            await self._paint_output.close()
            self._scrollback.file = self._original_scrollback_file


def _register_theme_refresh(
    bus: EventBus,
    ctx: Any,
    runtime: ApplicationRuntime,
) -> None:
    async def refresh_session(event: SessionSwitch) -> None:
        await runtime.rebind_session(event)
        themes, active = _resolve_runtime_themes(
            ctx.cfg,
            ctx.plugin_states,
            ctx.console.warning,
        )
        runtime.replace_themes(themes, active)

    bus.on(
        SessionSwitch,
        refresh_session,
        plugin="<tui-session>",
        priority=9_000,
    )

async def _run_runtime(ctx: AppContext, runtime: Any, bus: EventBus) -> int:
    shutdown_completed = False
    try:
        await bus.emit(AppStart(ctx=ctx))
        if hasattr(runtime, "flush_early_notifications"):
            runtime.flush_early_notifications()
        if ctx._reseed_pending() and ctx.agent is not None:
            await ctx.ensure_agent()
        if ctx.rebuild_requested:
            await ctx.rebuild()
        if ctx.cfg.resume:
            ctx._warn_interrupted_resume()
        await runtime.run()
        ctx.persist_plugin_states()
        ctx.record_exit("normal")
        if ctx.agents is not None:
            await ctx.agents.shutdown()
            shutdown_completed = True
        await bus.emit(AppExit())
        return 0
    finally:
        if ctx.agents is not None and not shutdown_completed:
            await ctx.agents.shutdown()


async def _run_app(cfg: Config) -> int:
    """Compose plugins and run the interactive terminal application."""

    store_type = _compat("SessionStore", SessionStore)
    store_factory = _compat("open_session_store", open_session_store)
    console_type = _compat("ConsoleOutput", ConsoleOutput)
    try:
        store = store_factory(cfg, sqlite_store_factory=store_type)
    except Exception as exc:
        persistence = getattr(cfg, "persistence", None)
        path = (
            getattr(persistence, "replica_path", cfg.db_path)
            if getattr(persistence, "backend", "sqlite") == "turso"
            else cfg.db_path
        )
        console_type().error(f"Cannot open session database {path}: {exc}")
        return 1
    with store:
        history_model: str | list[str] | None = None
        pending_switch_old_thread: str | None = None
        resume_live_thread: str | None = None
        if cfg.list_sessions:
            console = console_type()
            for session in store.list():
                console.print(f"{session.thread_id}  {session.cwd}  {session.title or ''}")
            return 0
        if cfg.resume:
            try:
                saved_session = store.resolve_session(cfg.resume)
            except LookupError as exc:
                console_type().error(_session_resolution_error(cfg.resume, exc))
                return 1
            history_model = _stored_model(saved_session.model)
            cfg = replace(
                cfg,
                cwd=Path(saved_session.cwd),
                model=(
                    cfg.model
                    if cfg.model_overridden
                    else history_model
                ),
                mode=saved_session.mode,
                trust_cwd=is_trusted_cwd(
                    saved_session.cwd,
                    cfg.trusted_dirs,
                    trust_all=cfg.trust_all_cwd,
                ),
            )
            session_id = saved_session.thread_id
            resume_live_thread = saved_session.current_thread
            checkpoint_live = (
                resume_live_thread is not None
                and store.checkpoint_exists(resume_live_thread)
            )
            if not checkpoint_live:
                thread_id = _uncheckpointed_seed_target(
                    store,
                    session_id,
                    resume_live_thread,
                )
                pending_switch_old_thread = resume_live_thread or thread_id
                ledger = Ledger(store)
                ledger.set_position(
                    session_id,
                    leaf_id=ledger.leaf(session_id),
                    thread_id=None,
                )
            else:
                thread_id = resume_live_thread
        else:
            created = store.create(
                cfg.cwd,
                cfg.model,
                mode=cfg.mode,
            )
            if created.current_thread is None:
                raise RuntimeError(f"Session {created.thread_id} has no graph thread")
            session_id = created.thread_id
            thread_id = created.current_thread

        registry = Registry()
        register_builtin_overlays(registry)
        bus = EventBus()
        states = store.all_plugin_state(session_id)
        holder: dict[str, AppContext] = {}

        def request_rebuild() -> None:
            if "ctx" in holder:
                holder["ctx"].request_rebuild()

        loader = _compat("load_plugins", load_plugins)
        records = loader(registry, bus, cfg, states, request_rebuild)
        _scope_main_statusbar_accounting(bus)
        ctx = AppContext(
            cfg=cfg,
            registry=registry,
            bus=bus,
            session=store,
            plugins=records,
            plugin_states=states,
            console=console_type(),
            thread_id=thread_id,
            session_id=session_id,
            history_model=history_model,
        )
        ctx._pending_switch_old_thread = pending_switch_old_thread
        holder["ctx"] = ctx
        if cfg.resume and resume_live_thread is not None and store.checkpoint_exists(
            resume_live_thread
        ):
            ctx.recover_checkpoint(session_id, resume_live_thread)
            context = build_context(ctx.ledger.path(session_id))
            pending_interrupt = store.checkpoint_has_pending_interrupt(
                resume_live_thread
            )
            if context.dangling and not pending_interrupt:
                old_thread = ctx.thread_id
                ctx.ledger.set_position(
                    session_id,
                    leaf_id=ctx.ledger.leaf(session_id),
                    thread_id=None,
                )
                ctx.thread_id = store.next_thread_id(session_id)
                ctx._pending_switch_old_thread = old_thread
        async def submit(text: str) -> None:
            try:
                if not text.startswith("/"):
                    first_word = text.split(maxsplit=1)[0]
                    if first_word in registry.commands:
                        ctx.console.warning(f"Did you mean /{text}?")
                        return
                command_dispatch = _compat("dispatch_command", dispatch_command)
                if await command_dispatch(registry, ctx, text):
                    if ctx.rebuild_requested:
                        await ctx.rebuild()
                    if ctx.exit_requested:
                        runtime.application.exit()
                    return
                await _run_cancellable_turn(ctx, text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                ctx.console.warning("interrupted")
            except Exception as exc:
                ctx.console.error(f"{type(exc).__name__}: {exc}")

        available_themes, active_theme = _resolve_runtime_themes(
            ctx.cfg,
            states,
            ctx.console.warning,
        )

        prompt_history_path = _compat("_history_path", _history_path)()
        prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_type = _compat("ApplicationRuntime", ApplicationRuntime)
        runtime = runtime_type(
            submit,
            registry=registry,
            history=SQLiteHistory(
                prompt_history_path,
                cwd=ctx.cfg.cwd,
                session_id=session_id,
            ),
            status=lambda: _bottom_toolbar(ctx),
            console=(
                ctx.console.console
                if isinstance(ctx.console, ConsoleOutput)
                else None
            ),
            theme=active_theme,
            themes=available_themes,
            ctx=ctx,
            composer_shape=ctx.cfg.composer,
        )
        if hasattr(runtime, "replace_themes"):
            _register_theme_refresh(bus, ctx, runtime)
        advisor = getattr(runtime, "advisor", None)
        if advisor is not None:
            bus.on(
                TurnEnd,
                advisor.on_main_turn_end,
                plugin="<advisor>",
                priority=8_500,
            )
        if hasattr(runtime, "transcript"):
            ctx.transcript = runtime.transcript
            ctx.ui = runtime.ui
            if hasattr(runtime, "handle_presentation"):
                bus.on(
                    Event,
                    runtime.handle_presentation,
                    plugin="<tui-presentation>",
                    priority=10_000,
                )
            if hasattr(runtime, "_prepare_agent_delivery"):
                bus.on(
                    AgentDelivered,
                    runtime._prepare_agent_delivery,
                    plugin="<tui-delivery>",
                    priority=8_900,
                )
            bus.on(
                Event,
                runtime.transcript.handle,
                plugin="<tui>",
                priority=9_000,
            )
            if isinstance(ctx.console, ConsoleOutput):
                ctx.console = ConsoleOutput(
                    ctx.console.console,
                    transcript=runtime.transcript,
                )
        return await _run_runtime(ctx, runtime, bus)


async def run_app(cfg: Config) -> int:
    """Run the application and turn Turso shutdown failures into clean errors."""

    try:
        return await _run_app(cfg)
    except TursoPersistenceError as exc:
        console_type = _compat("ConsoleOutput", ConsoleOutput)
        console_type().error(str(exc))
        return 1


async def _show_agents(ctx: Any, _args: str) -> None:
    await ctx.ui.show("hub")


def _ensure_agent_command(registry: Registry) -> None:
    registry.commands.setdefault(
        "agents",
        CommandRegistration(
            plugin="commands_core",
            handler=_show_agents,
            help="Open the agent hub",
        ),
    )


def _ensure_review_command(registry: Registry) -> None:
    registry.commands.setdefault(
        "review",
        CommandRegistration(
            plugin="commands_review",
            handler=review,
            help="Review code changes with parallel agents",
        ),
    )