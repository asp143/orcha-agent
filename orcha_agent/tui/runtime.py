"""Prompt-toolkit input loop and graph stream dispatch."""

from __future__ import annotations

import asyncio
import inspect
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import History
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import DynamicKeyBindings, KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from orcha_agent.core.config import Config, is_trusted_cwd
from orcha_agent.core.events import (
    AppExit,
    AppStart,
    Event,
    EventBus,
    SessionSwitch,
    ToolCallEnd,
    ToolCallStart,
)
from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore

from .blocks import BlockRendererDispatcher, DEFAULT_THEME
from .console import ConsoleOutput
from .complete import ComposerCompleter
from .composer import Composer
from .context import (
    AppContext,
    _session_resolution_error,
    _stored_model,
    _uncheckpointed_seed_target,
)
from .frame import Block, Frame, FrameScheduler
from .history import SQLiteHistory, history_path
from .keys import create_key_bindings, load_keybindings
from .queue import PromptQueue, split_submission
from .transcript import Transcript
from .statusline import render_statusline
from .theme import Theme, load_themes, select_theme
from .turn import _run_cancellable_turn
from .overlays import register_builtin_overlays
from .overlays.base import Overlay


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
            "completion-menu.completion": (
                f"fg:{color('text')} bg:{color('statusLineBg')}"
            ),
            "completion-menu.completion.current": (
                f"fg:{color('text')} bg:{color('selectedBg')}"
            ),
            "completion-menu.meta.completion": (
                f"fg:{color('muted')} bg:{color('statusLineBg')}"
            ),
            "completion-menu.meta.completion.current": (
                f"fg:{color('text')} bg:{color('selectedBg')}"
            ),
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


async def dispatch_command(registry: Registry, ctx: Any, text: str) -> bool:
    """Dispatch slash commands without invoking the model."""

    if not text.startswith("/"):
        return False
    command_text = text[1:]
    name, separator, args = command_text.partition(" ")
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
        notify: Callable[[str], None] | None = None,
        clear: Callable[[], Awaitable[None]] | None = None,
        set_theme: Callable[[str], Any] | None = None,
    ) -> None:
        self._show_overlay = show_overlay
        self._notify = notify
        self._clear = clear
        self._set_theme = set_theme
        self.notifications: list[str] = []
        self.thinking_visible = True
        self.tools_expanded = False

    def notify(self, text: str) -> None:
        self.notifications.append(text)
        if self._notify is not None:
            self._notify(text)

    async def show(self, overlay: object, *args: Any, **kwargs: Any) -> Any:
        if self._show_overlay is None:
            raise RuntimeError(f"overlay {overlay!r} is unavailable")
        return await self._show_overlay(overlay, *args, **kwargs)

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
        completion_registry = registry or Registry()
        self.frame = Frame()
        self.theme: Any = theme
        self.composer_shape = composer_shape
        self._themes = dict(themes or {})
        current_theme_id = str(
            getattr(theme, "id", theme.get("id", "default") if isinstance(theme, Mapping) else "default")
        )
        self._themes.setdefault(current_theme_id, theme)
        self._block_dispatcher = BlockRendererDispatcher(
            registry.block_renderers if registry is not None else {}
        )
        previous_ui = getattr(ctx, "ui", None)
        self._fallback_show = getattr(previous_ui, "_show_overlay", None)
        self.ui = UIFacade(
            show_overlay=self._show_overlay,
            notify=self._notify,
            clear=self._clear_scrollback,
            set_theme=self._set_theme,
        )
        self.ui.theme = theme
        self.ui.themes = self._themes
        self.ui.history = history
        self._status = status or self._status_text
        self._pending: set[asyncio.Future[Any]] = set()
        self._terminal_pending: set[asyncio.Future[Any]] = set()
        self._submit_lock = asyncio.Lock()
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
        self._active_overlay: Overlay | None = None
        self._overlay_lock = asyncio.Lock()
        self._last_escape = 0.0
        self._last_interrupt = 0.0
        self._shell_runner = shell_runner or self._run_shell_process
        self._custom_editor = editor_runner is not None
        self._editor_runner = editor_runner or self._run_editor_process
        self.thinking_level = self._restore_thinking_level()
        self.ui.thinking_level = self.thinking_level

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
        self.buffer = self.composer.buffer
        self._restore_draft()
        effective = load_keybindings(
            user_path=keybindings_path,
            registry=completion_registry,
            warn=self._notify,
        )
        self.ui.effective_keys = effective
        handlers = self._action_handlers()
        bindings = create_key_bindings(effective, handlers)
        self._tree_handler = handlers["tree"]
        self._tree_double_escape = "escape escape" in effective.get("tree", ())
        core_bindings = KeyBindings()

        @core_bindings.add(
            "escape",
            filter=Condition(lambda: self._active_overlay is None),
        )
        def _escape(event: Any) -> None:
            self._escape_ladder(event)

        if self._tree_double_escape:
            core_bindings.add("s-escape")(self._tree_handler)

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
                    Window(
                        FormattedTextControl(self._viewport_text),
                        height=Dimension(min=0),
                    ),
                    self.composer.container,
                    Window(
                        FormattedTextControl(self._status),
                        height=1,
                    ),
                ]
            ),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
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
            mouse_support=Condition(lambda: self._active_overlay is not None),
            **kwargs,
        )
        self.application.ttimeoutlen = 0.1
        self.application.timeoutlen = 0.1
        self.ui.invalidate = self.application.invalidate
        self.ui.status_width = lambda: self.application.output.get_size().columns
        self.scheduler = FrameScheduler(
            self.frame,
            commit=self._commit_blocks,
            invalidate=self.application.invalidate,
        )
        self.transcript = Transcript(
            self.frame,
            registry=registry,
            scheduler=self.scheduler,
        )
        for notification in self._early_notifications:
            self.transcript.append_banner(notification, level="info")
        self._early_notifications.clear()
    @property
    def active_overlay(self) -> Overlay | None:
        return self._active_overlay

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

    async def _show_overlay(
        self,
        overlay: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolved = self._resolve_overlay(overlay, args, kwargs)
        if resolved is None:
            if self._fallback_show is not None:
                return await self._fallback_show(overlay, *args, **kwargs)
            raise RuntimeError(f"overlay {overlay!r} is unavailable")
        async with self._overlay_lock:
            if self._shutting_down:
                resolved.cancel()
                return None
            self._active_overlay = resolved
            self._root.floats.append(resolved)
            try:
                try:
                    self.application.layout.focus(resolved.focus_target)
                except ValueError:
                    pass
                self.application.invalidate()
                return await resolved.wait()
            finally:
                if resolved in self._root.floats:
                    self._root.floats.remove(resolved)
                self._active_overlay = None
                self.application.layout.focus(self.buffer)
                self.application.invalidate()


    def _cwd(self) -> Path:
        value = getattr(getattr(self.ctx, "cfg", None), "cwd", Path.cwd())
        return Path(value)

    def _model_label(self) -> str:
        value = getattr(getattr(self.ctx, "cfg", None), "model", "model")
        return value[0] if isinstance(value, list) and value else str(value)

    def _status_text(self) -> Any:
        if self.ctx is None:
            return []
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
        has_queue = (
            isinstance(saved_queue, list)
            and bool(saved_queue)
            and all(isinstance(prompt, str) for prompt in saved_queue)
        )
        if not has_draft and not has_queue:
            return
        if has_draft:
            self.buffer.text = draft
            self.buffer.cursor_position = len(draft)
            state.pop("draft", None)
        if has_queue:
            self.queue.extend(saved_queue)
            state.pop("queue", None)
        self._persist_state()

    def _persist_state(self) -> None:
        persist = getattr(self.ctx, "persist_plugin_states", None)
        if persist is not None:
            persist()

    def _accept(self, buffer: Any) -> bool:
        raw = buffer.text
        text = raw.strip()
        if not text:
            if self.streaming and self.queue:
                buffer.reset(append_to_history=False)
                self._abort_turn()
            return False
        buffer.reset(append_to_history=True)
        if text == ".":
            text = "keep going"
        prompts = split_submission(text)
        is_batch = len(prompts) > 1
        if self.streaming:
            self.queue.extend(prompts)
            self.application.invalidate()
            return False
        first = prompts.pop(0)
        if is_batch:
            self.queue.extend(prompts)
        self._track(self._submit_serially(first))
        return False

    def _action_handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "submit": self._submit_action,
            "newline": lambda event: event.current_buffer.insert_text("\n"),
            "queue": self._queue_draft,
            "dequeue": self._dequeue,
            "toggle_thinking": lambda _event: self.ui.toggle_thinking(),
            "cycle_thinking_level": lambda _event: self._cycle_thinking_level(),
            "expand_tools": lambda _event: self.ui.expand_tools(not self.ui.tools_expanded),
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
        text = event.current_buffer.text.strip()
        if text:
            self.queue.extend(split_submission(text))
            event.current_buffer.reset(append_to_history=False)
            event.app.invalidate()

    def _dequeue(self, event: Any) -> None:
        text = self.queue.pop_last()
        if text is not None:
            event.current_buffer.text = text
            event.current_buffer.cursor_position = len(text)

    def _abort_turn(self) -> None:
        if self._active_turn is not None and not self._active_turn.done():
            self._active_turn.cancel()

    def _escape_ladder(self, event: Any) -> None:
        buffer = event.current_buffer
        if buffer.complete_state is not None:
            buffer.cancel_completion()
            return
        if self.streaming:
            restored = self.queue.restore_text()
            if restored:
                buffer.text = restored
                buffer.cursor_position = len(restored)
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
        text = event.current_buffer.text
        if text:
            state["draft"] = text
        if self.queue:
            state["queue"] = list(self.queue.items)
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

    def _thinking_supported(self) -> bool:
        if self.registry is None:
            return False
        model = self._model_label()
        prefix, separator, _name = model.partition(":")
        registration = self.registry.providers.get(prefix) if separator else None
        return bool(registration and registration.capabilities.thinking)

    def _cycle_thinking_level(self) -> None:
        if not self._thinking_supported():
            self.ui.notify("Thinking levels are unavailable for the active provider.")
            return
        levels = ("off", "low", "medium", "high", "max")
        self.thinking_level = levels[(levels.index(self.thinking_level) + 1) % len(levels)]
        self.ui.thinking_level = self.thinking_level
        self._composer_state()["thinking_level"] = self.thinking_level
        self._persist_state()
        self.application.invalidate()

    async def _cycle_model(self) -> None:
        if self.registry is None or self.ctx is None:
            return
        available = [
            f"{prefix}:{model}"
            for prefix, provider in sorted(self.registry.providers.items())
            for model in provider.models
            if provider.available() is None
        ]
        if not available:
            self.ui.notify("No models are available.")
            return
        current = self._model_label()
        index = available.index(current) + 1 if current in available else 0
        await self.ctx.switch_model(available[index % len(available)])

    async def _external_editor(self) -> None:
        if not self._custom_editor and not (os.environ.get("VISUAL") or os.environ.get("EDITOR")):
            self.ui.notify("Set $VISUAL or $EDITOR to edit the draft externally.")
            return
        original = self.buffer.text
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

    @staticmethod
    def _run_shell_process(command: str, cwd: Path, timeout: float) -> Any:
        return subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

    async def _run_shell(self, command: str) -> None:
        identifier = f"execute-{time.monotonic_ns()}"
        bus = getattr(self.ctx, "_bus", None)
        if bus is not None:
            await bus.emit(ToolCallStart(name="execute", args={"command": command}, id=identifier))
        try:
            completed = await asyncio.to_thread(self._shell_runner, command, self._cwd(), 60.0)
            result = {
                "returncode": int(getattr(completed, "returncode", 0)),
                "stdout": str(getattr(completed, "stdout", "")),
                "stderr": str(getattr(completed, "stderr", "")),
            }
        except subprocess.TimeoutExpired:
            result = {"returncode": 124, "stdout": "", "stderr": "timed out after 60 seconds"}
        if bus is not None:
            await bus.emit(ToolCallEnd(name="execute", id=identifier, result=result))

    async def _dispatch_submission(self, text: str) -> None:
        if text.startswith("!"):
            await self._run_shell(text[1:].strip())
        else:
            await self._submit(text)

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

    async def _submit_serially(self, text: str) -> None:
        async with self._submit_lock:
            current: str | None = text
            while current is not None:
                self.streaming = True
                self._active_turn = asyncio.create_task(self._dispatch_submission(current))
                try:
                    await self._active_turn
                except (KeyboardInterrupt, asyncio.CancelledError):
                    self.transcript.append_banner("interrupted", level="warning")
                except Exception as exc:
                    self.transcript.append_banner(f"{type(exc).__name__}: {exc}")
                finally:
                    self._active_turn = None
                    self.streaming = False
                    self.application.invalidate()
                if self._shutting_down:
                    break
                current = self.queue.pop()

    def _notify(self, text: str) -> None:
        transcript = getattr(self, "transcript", None)
        application = getattr(self, "application", None)
        if transcript is None:
            self._early_notifications.append(text)
            return
        transcript.append_banner(text, level="info")
        if application is not None:
            application.invalidate()

    def _apply_theme(self, selected: Any) -> Any:
        self.theme = selected
        self.ui.theme = selected
        prompt_style = _completion_style(selected)
        if prompt_style is not None:
            self.application.style = prompt_style
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


    def _render_block(self, block: Block, width: int, rows: int) -> Any:
        return self._block_dispatcher.render(
            block,
            self.theme,
            width,
            rows,
            self.ui.tools_expanded,
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
            width=max(1, width),
            theme=getattr(self.theme, "rich", None),
        )
        self._print_block(console, block, width, rows, viewport=True)
        return stream.getvalue()

    def _measure_block(self, block: Block, width: int) -> int:
        rendered = self._capture_block(
            block,
            width,
            10_000,
            force_terminal=False,
        )
        return max(1, len(rendered.splitlines()))

    def _viewport_text(self) -> Any:
        size = self.application.output.get_size()
        width = max(1, size.columns)
        budget = Frame.row_budget(
            terminal_rows=size.rows,
            composer_rows=self.composer.height_for_width(width),
            status_rows=1,
        )
        rendered: list[str] = []
        for item in self.frame.viewport_plan(
            budget,
            width=width,
            measure=self._measure_block,
        ):
            value = self._capture_block(
                item.block,
                width,
                item.rows,
                force_terminal=True,
            )
            lines = value.splitlines(keepends=True)
            if len(lines) > item.rows:
                lines = (
                    lines[-item.rows :]
                    if item.block.kind in {"assistant", "thinking"}
                    else lines[: item.rows]
                )
            rendered.append("".join(lines))
        return ANSI("\n".join(rendered))

    def _write_blocks(self, blocks: list[Block]) -> None:
        width = max(1, self.application.output.get_size().columns)
        for block in blocks:
            self._print_block(
                self._scrollback,
                block,
                width,
                10_000,
                viewport=False,
            )

    def _commit_blocks(self, blocks: list[Block]) -> None:
        self._track(
            run_in_terminal(lambda: self._write_blocks(blocks)),
            terminal=True,
        )

    async def _clear_scrollback(self) -> None:
        self.scheduler.commit_now()
        await self._drain(self._terminal_pending)
        await self._track(
            run_in_terminal(self._scrollback.clear),
            terminal=True,
        )
        self.transcript.clear()
        self.application.invalidate()

    async def _drain(self, pending: set[asyncio.Future[Any]]) -> None:
        while pending:
            await asyncio.gather(*tuple(pending))

    async def _drain_pending(self) -> None:
        await self._drain(self._pending)

    async def run(self) -> None:
        try:
            await self.application.run_async()
        except EOFError:
            pass
        finally:
            self._shutting_down = True
            if self._active_overlay is not None:
                self._active_overlay.cancel()
            await self._drain_pending()
            self.scheduler.commit_now()
            await self._drain_pending()
            await self.scheduler.aclose()


def _register_theme_refresh(
    bus: EventBus,
    ctx: Any,
    runtime: ApplicationRuntime,
) -> None:
    async def refresh_theme(_event: SessionSwitch) -> None:
        themes, active = _resolve_runtime_themes(
            ctx.cfg,
            ctx.plugin_states,
            ctx.console.warning,
        )
        runtime.replace_themes(themes, active)

    bus.on(
        SessionSwitch,
        refresh_theme,
        plugin="<tui-theme>",
        priority=9_000,
    )

async def run_app(cfg: Config) -> int:
    """Compose plugins and run the interactive terminal application."""

    store_type = _compat("SessionStore", SessionStore)
    console_type = _compat("ConsoleOutput", ConsoleOutput)
    try:
        store = store_type(cfg.db_path)
    except Exception as exc:
        console_type().error(f"Cannot open session database {cfg.db_path}: {exc}")
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
        if hasattr(runtime, "transcript"):
            ctx.transcript = runtime.transcript
            ctx.ui = runtime.ui
            bus.on(
                Event,
                runtime.transcript.handle,
                plugin="<tui>",
                priority=10_000,
            )
            if isinstance(ctx.console, ConsoleOutput):
                ctx.console = ConsoleOutput(
                    ctx.console.console,
                    transcript=runtime.transcript,
                )
        await bus.emit(AppStart(ctx=ctx))
        if ctx._reseed_pending() and ctx.agent is not None:
            await ctx.ensure_agent()
        if ctx.rebuild_requested:
            await ctx.rebuild()
        if cfg.resume:
            ctx._warn_interrupted_resume()
        await runtime.run()
        ctx.persist_plugin_states()
        ctx.record_exit("normal")
        await bus.emit(AppExit())
        return 0
