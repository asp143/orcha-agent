"""Prompt-toolkit input loop and graph stream dispatch."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from orcha_agent.core.config import Config, is_trusted_cwd
from orcha_agent.core.events import (
    AppExit,
    AppStart,
    Event,
    EventBus,
    SessionSwitch,
)
from orcha_agent.core.ledger import Ledger, build_context
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore

from .console import ConsoleOutput
from .context import (
    AppContext,
    _session_resolution_error,
    _stored_model,
    _uncheckpointed_seed_target,
)
from .frame import Block, Frame, FrameScheduler
from .blocks import BlockRendererDispatcher, DEFAULT_THEME
from .transcript import Transcript
from .theme import Theme, load_themes, select_theme
from .turn import _run_cancellable_turn



def _history_path() -> Path:
    return Path.home() / ".local/share/orcha-agent/history"


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
        return ""
    values: list[str] = []
    for segment in ctx.registry.status_segments:
        try:
            value = segment.render(ctx)
        except Exception:
            value = f"!{segment.name}"
        if value:
            values.append(value)
    return HTML(" · ".join(values)) if values else ""


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
    """Minimal stable UI surface for plugins while overlays remain optional."""

    def __init__(
        self,
        *,
        show_overlay: Callable[[object], Awaitable[Any]] | None = None,
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

    async def show(self, overlay: object) -> Any:
        if self._show_overlay is None:
            return None
        return await self._show_overlay(overlay)

    async def ask(self, questions: object) -> Any:
        return await self.show(questions)

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
        history: FileHistory | None = None,
        status: Callable[[], Any] | None = None,
        input: Any = None,
        output: Any = None,
        console: Console | None = None,
        theme: Any = DEFAULT_THEME,
        themes: Mapping[str, Any] | None = None,
    ) -> None:
        self._submit = submit
        self.registry = registry
        self.frame = Frame()
        self.theme: Any = theme
        self._themes = dict(themes or {})
        current_theme_id = str(
            getattr(theme, "id", theme.get("id", "default") if isinstance(theme, Mapping) else "default")
        )
        self._themes.setdefault(current_theme_id, theme)
        self._block_dispatcher = BlockRendererDispatcher(
            registry.block_renderers if registry is not None else {}
        )
        self.ui = UIFacade(
            notify=self._notify,
            clear=self._clear_scrollback,
            set_theme=self._set_theme,
        )
        self._status = status or (lambda: "")
        self._pending: set[asyncio.Future[Any]] = set()
        self._terminal_pending: set[asyncio.Future[Any]] = set()
        self._submit_lock = asyncio.Lock()
        self._scrollback = console or Console()

        self.buffer = Buffer(
            history=history,
            multiline=True,
            accept_handler=self._accept,
        )
        bindings = _bindings()

        @bindings.add("c-d")
        def _exit(event: Any) -> None:
            if event.current_buffer.text:
                event.current_buffer.delete()
            else:
                event.app.exit()

        root = FloatContainer(
            content=HSplit(
                [
                    Window(
                        FormattedTextControl(self._viewport_text),
                        height=Dimension(min=0),
                    ),
                    Window(
                        BufferControl(buffer=self.buffer),
                        height=Dimension(min=1, max=8),
                    ),
                    Window(
                        FormattedTextControl(self._status),
                        height=1,
                    ),
                ]
            ),
            floats=[],
        )
        kwargs: dict[str, Any] = {}
        prompt_style = getattr(theme, "pt", None)
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
            **kwargs,
        )
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

    def _accept(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        buffer.reset(append_to_history=bool(text))
        if not text:
            return False
        self._track(self._submit_serially(text))
        return False

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
            try:
                await self._submit(text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.transcript.append_banner("interrupted", level="warning")
            except Exception as exc:
                self.transcript.append_banner(f"{type(exc).__name__}: {exc}")

    def _notify(self, text: str) -> None:
        self.transcript.append_banner(text, level="info")
        self.application.invalidate()

    def _apply_theme(self, selected: Any) -> Any:
        self.theme = selected
        prompt_style = getattr(selected, "pt", None)
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
        width = max(1, width)
        rows = 0
        for line in self.buffer.text.split("\n"):
            columns = sum(get_cwidth(character) for character in line)
            rows += max(1, (columns + width - 1) // width)
        return min(8, max(1, rows))

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
            composer_rows=self._composer_height(width),
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

        history_path = _compat("_history_path", _history_path)()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_type = _compat("ApplicationRuntime", ApplicationRuntime)
        runtime = runtime_type(
            submit,
            registry=registry,
            history=FileHistory(str(history_path)),
            status=lambda: _bottom_toolbar(ctx),
            console=(
                ctx.console.console
                if isinstance(ctx.console, ConsoleOutput)
                else None
            ),
            theme=active_theme,
            themes=available_themes,
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
