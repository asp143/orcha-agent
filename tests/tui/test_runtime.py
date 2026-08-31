from __future__ import annotations

import asyncio
import json
import re
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout.dimension import to_dimension
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from orcha_agent.core.events import (
    EventBus,
    ModelChunk,
    SessionSwitch,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.registry import Registry
from orcha_agent.tui.frame import Block, Frame
from orcha_agent.tui.runtime import (
    ApplicationRuntime,
    UIFacade,
    _register_theme_refresh,
)
from orcha_agent.tui.theme import COLOR_TOKENS, Theme, load_themes
from orcha_agent.tui.turn import _run_turn


class _SizedDummyOutput(DummyOutput):
    def __init__(self, *, columns: int, rows: int = 24) -> None:
        self._size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self._size


@pytest.mark.asyncio
async def test_root_has_dynamic_terminal_height_and_bottom_filler() -> None:
    output = _SizedDummyOutput(columns=100, rows=30)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=output,
        )
        root = runtime.application.layout.container.content

        dimension = to_dimension(root.height)
        assert (dimension.min, dimension.preferred, dimension.max) == (29, 29, 29)
        assert to_dimension(root.children[0].height).weight == 1
        assert root.children[1].dont_extend_height()

        output._size = Size(rows=18, columns=70)
        resized = to_dimension(root.height)
        assert (resized.min, resized.preferred, resized.max) == (17, 17, 17)
        await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_viewport_budget_counts_every_fixed_layout_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _SizedDummyOutput(columns=100, rows=30)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=output,
        )
        runtime.buffer.text = "one\ntwo\nthree"
        monkeypatch.setattr(runtime.composer, "completion_row_count", lambda: 5)
        monkeypatch.setattr(runtime, "_hud_height", lambda: 3)
        runtime.frame.add(
            "raw",
            {"renderable": "\n".join(f"marker-{index}" for index in range(40))},
        )

        rendered = runtime._viewport_text()
        await runtime.scheduler.aclose()

    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered.value)
    assert len(plain.splitlines()) == 16


@pytest.mark.asyncio
async def test_application_is_headlessly_driveable_through_submit_and_exit() -> None:
    submitted: list[str] = []
    submitted_event = asyncio.Event()
    stream = StringIO()
    console = Console(file=stream, force_terminal=False)

    async def submit(text: str) -> None:
        await runtime.transcript.handle(TurnStart(thread_id="thread", text=text))
        submitted.append(text)
        submitted_event.set()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            input=pipe,
            output=DummyOutput(),
            console=console,
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("hello\n")
        await asyncio.wait_for(submitted_event.wait(), timeout=1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, timeout=1)

    assert submitted == ["hello"]
    assert runtime.application.full_screen is False
    assert "hello" in stream.getvalue()


@pytest.mark.asyncio
async def test_headless_fake_model_turn_keeps_full_width_composer_frame(
    wait_for_render,
) -> None:
    width = 72
    output = _SizedDummyOutput(columns=width)
    scrollback = StringIO()
    model = FakeListChatModel(responses=["A complete fake-model response."])
    completed = asyncio.Event()
    submitted: list[str] = []

    async def submit(text: str) -> None:
        submitted.append(text)
        start = TurnStart(thread_id="headless", text=text)
        await runtime.handle_presentation(start)
        await runtime.transcript.handle(start)

        response = await model.ainvoke([HumanMessage(content=text)])
        await runtime.transcript.handle(
            ModelChunk(
                chunk=AIMessageChunk(content=response.content),
                role="main",
                source_id="main",
            )
        )

        end = TurnEnd(thread_id="headless")
        await runtime.transcript.handle(end)
        await runtime.handle_presentation(end)
        completed.set()

    def full_width_composer_rendered() -> bool:
        if not completed.is_set():
            return False
        screen = runtime.application.renderer._last_screen
        rows = (
            "".join(screen.data_buffer[y][x].char for x in range(width))
            for y in range(screen.height)
        )
        return any(
            row.startswith("╭──") and row.endswith("╮") and len(row) == width
            for row in rows
        )

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            input=pipe,
            output=output,
            console=Console(file=scrollback, force_terminal=False, width=width),
            composer_shape="box",
        )
        task = asyncio.create_task(runtime.run())
        rendered = asyncio.create_task(
            wait_for_render(runtime, full_width_composer_rendered)
        )
        pipe.send_text("render the full turn\n")
        await asyncio.wait_for(completed.wait(), timeout=1)
        await rendered

        screen = runtime.application.renderer._last_screen
        rows = [
            "".join(screen.data_buffer[y][x].char for x in range(width))
            for y in range(screen.height)
        ]
        top_border = next(row for row in rows if row.startswith("╭──"))
        assert top_border.endswith("╮")
        assert len(top_border) == width

        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, timeout=1)

    assert submitted == ["render the full turn"]
    assert "A complete fake-model response." in scrollback.getvalue()


@pytest.mark.asyncio
async def test_no_tools_graph_turn_streams_without_tools_interrupt() -> None:
    errors: list[str] = []
    chunks: list[ModelChunk] = []
    real_graph = create_agent(
        model=FakeListChatModel(responses=["done"]),
        tools=[],
    )

    class CapabilityCheckingGraph:
        nodes = real_graph.nodes

        async def astream(self, *args: object, **kwargs: object):
            interrupt_after = set(kwargs.get("interrupt_after", ()))
            unknown = interrupt_after.difference(self.nodes)
            if unknown:
                raise ValueError(f"unknown interrupt nodes: {sorted(unknown)}")
            async for item in real_graph.astream(*args, **kwargs):
                yield item

    graph = CapabilityCheckingGraph()
    bus = EventBus()

    async def capture(event: ModelChunk) -> None:
        chunks.append(event)

    async def ensure_agent() -> bool:
        return True

    bus.on(ModelChunk, capture, plugin="test")
    ctx = SimpleNamespace(
        agent=graph,
        ensure_agent=ensure_agent,
        session=SimpleNamespace(get=lambda _session_id: None),
        session_id="session",
        thread_id="thread",
        thread_config={},
        _bus=bus,
        bus=bus,
        registry=Registry(),
        console=SimpleNamespace(
            warning=errors.append,
            error=errors.append,
            print=lambda: None,
        ),
        capture_turn=lambda: None,
        record_exit=lambda _reason: None,
        rebuild_requested=False,
    )

    await _run_turn(ctx, "hello")

    assert "".join(str(chunk.chunk.content) for chunk in chunks) == "done"
    assert errors == []


@pytest.mark.asyncio
async def test_ui_facade_controls_runtime_state_and_overlay_results() -> None:
    shown: list[object] = []

    async def show(value: object) -> str:
        shown.append(value)
        return "selected"

    facade = UIFacade(show_overlay=show)
    facade.notify("saved")
    facade.toggle_thinking()
    facade.expand_tools(True)

    assert await facade.show("picker") == "selected"
    assert await facade.ask([{"question": "Choose"}]) == "selected"
    assert shown == ["picker", [{"question": "Choose"}]]
    assert facade.notifications == ["saved"]
    assert facade.thinking_visible is False
    assert facade.tools_expanded is True


@pytest.mark.asyncio
async def test_exit_drains_the_final_coalesced_transcript_commit() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    stream = StringIO()

    async def submit(_text: str) -> None:
        await runtime.transcript.handle(
            ModelChunk(chunk="final answer", role="main", source_id="main")
        )
        started.set()
        await release.wait()
        await runtime.transcript.handle(TurnEnd(thread_id="thread"))

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            submit,
            input=pipe,
            output=DummyOutput(),
            console=Console(file=stream, force_terminal=False),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("go\n")
        await asyncio.wait_for(started.wait(), timeout=1)
        pipe.send_bytes(b"\x04")
        release.set()
        await asyncio.wait_for(task, timeout=1)

    assert "final answer" in stream.getvalue()


def test_composer_height_counts_newlines_and_wrapped_rows() -> None:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime.buffer.text = "one\ntwo\n123456789"

        assert runtime._composer_height(5) == 4


@pytest.mark.asyncio
async def test_runtime_clear_uses_application_terminal_operation_and_redraws() -> None:
    cleared = asyncio.Event()

    async def submit(_text: str) -> None:
        runtime.frame.add("assistant", {"text": "stale"})
        await runtime.ui.clear()
        cleared.set()

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        pipe.send_text("clear\n")
        await asyncio.wait_for(cleared.wait(), timeout=1)
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, timeout=1)

    assert runtime.frame.blocks == []


@pytest.mark.asyncio
async def test_transcript_print_replays_rich_sep_and_end_semantics() -> None:
    expected_stream = StringIO()
    actual_stream = StringIO()
    expected = Console(file=expected_stream, force_terminal=False)
    actual = Console(file=actual_stream, force_terminal=False)
    expected.print("alpha", "beta", sep="|", end="!")

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
            console=actual,
        )
        runtime.transcript.print("alpha", "beta", sep="|", end="!")
        runtime._write_blocks(runtime.frame.blocks)
        await runtime.scheduler.aclose()

    assert actual_stream.getvalue() == expected_stream.getvalue()


@pytest.mark.asyncio
async def test_scrollback_places_exactly_one_blank_row_between_blocks() -> None:
    stream = StringIO()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
            console=Console(file=stream, force_terminal=False, width=80, color_system=None),
        )
        runtime._write_blocks(
            [
                Block("first", "assistant", data={"text": "first"}),
                Block("second", "assistant", data={"text": "second"}),
            ]
        )
        await runtime.scheduler.aclose()

    lines = [line.rstrip() for line in stream.getvalue().splitlines()]
    assert lines == ["", "first", "", "second"]


@pytest.mark.asyncio
async def test_viewport_places_exactly_one_blank_row_between_blocks() -> None:
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime.frame.add("assistant", {"text": "first"})
        runtime.frame.add("assistant", {"text": "second"})
        rendered = runtime._viewport_text()
        await runtime.scheduler.aclose()

    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered.value)
    lines = [line.rstrip() for line in plain.splitlines()]
    assert lines == ["", "first", "", "second"]


@pytest.mark.asyncio
async def test_viewport_reserves_leading_spacer_rows_and_keeps_newest_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Frame,
        "row_budget",
        staticmethod(lambda **_kwargs: 2),
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime.frame.add("assistant", {"text": "first"})
        runtime.frame.add("assistant", {"text": "second"})
        rendered = runtime._viewport_text()
        await runtime.scheduler.aclose()

    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered.value)
    lines = [line.rstrip() for line in plain.splitlines()]
    assert lines == ["", "second"]


@pytest.mark.asyncio
async def test_tool_viewport_reserves_spacer_before_selecting_card_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Frame,
        "row_budget",
        staticmethod(lambda **_kwargs: 3),
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime.frame.add(
            "tool",
            {
                "name": "read_file",
                "args": {"path": "a.py"},
                "result": "one line",
            },
        )
        rendered = runtime._viewport_text()
        await runtime.scheduler.aclose()

    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered.value)
    lines = [line.rstrip() for line in plain.splitlines()]
    assert len(lines) == 3
    assert lines[0] == ""
    assert lines[1].startswith("╭")
    assert lines[2].startswith("╰")



@pytest.mark.asyncio
async def test_commit_uses_runtime_app_context_and_prunes_after_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_apps: list[object] = []

    async def immediate(callback: object) -> None:
        from prompt_toolkit.application.current import get_app

        active_apps.append(get_app())
        callback()

    monkeypatch.setattr("orcha_agent.tui.runtime.run_in_terminal", immediate)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=DummyOutput(),
        )
        runtime.application._is_running = True
        block = runtime.frame.add("assistant", {"text": "done"})
        runtime._render_block(block, 80, 3)
        runtime.frame.settle(block)
        ready = runtime.frame.commit_ready()

        runtime._commit_blocks(ready)
        await runtime._drain_pending()

        assert active_apps == [runtime.application]
        assert runtime.frame.blocks == []
        assert not runtime._block_dispatcher._cache
        runtime.application._is_running = False
        await runtime.scheduler.aclose()


def test_runtime_uses_memoized_registry_block_dispatcher() -> None:
    registry = Registry()
    calls: list[int] = []

    def render(
        block: Block,
        _theme: object,
        width: int,
        rows: int,
        expanded: bool,
    ) -> str:
        calls.append(block.revision)
        return f"{block.data['text']}:{width}:{rows}:{expanded}"

    registry._add_block_renderer("test", "assistant", render)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            input=pipe,
            output=DummyOutput(),
        )
        value = Block(id="answer", kind="assistant", data={"text": "hello"})

        assert runtime._render_block(value, 80, 3) == "hello:80:3:False"
        assert runtime._render_block(value, 80, 3) == "hello:80:3:False"

    assert calls == [0]


def test_runtime_theme_change_invalidates_only_active_viewport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = SimpleNamespace(id="one", pt=Style.from_dict({"one": "#ffffff"}))
    second = SimpleNamespace(id="two", pt=Style.from_dict({"two": "#000000"}))
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            theme=first,
            themes={"one": first, "two": second},
            input=pipe,
            output=DummyOutput(),
        )
        invalidations: list[None] = []
        writes: list[list[Block]] = []
        monkeypatch.setattr(
            runtime.application,
            "invalidate",
            lambda: invalidations.append(None),
        )
        monkeypatch.setattr(runtime, "_write_blocks", writes.append)

        selected = runtime.ui.set_theme("two")

    assert selected is second
    assert runtime.theme is second
    assert invalidations == [None]
    assert writes == []

    assert runtime.application.style is second.pt

def test_renderer_cache_is_separated_by_runtime_theme_id() -> None:
    registry = Registry()
    calls: list[str] = []

    def render(
        _block: Block,
        theme: object,
        _width: int,
        _rows: int,
        _expanded: bool,
    ) -> str:
        calls.append(str(getattr(theme, "id")))
        return calls[-1]

    registry._add_block_renderer("test", "assistant", render)
    one = SimpleNamespace(id="one")
    two = SimpleNamespace(id="two")
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            theme=one,
            themes={"one": one, "two": two},
            input=pipe,
            output=DummyOutput(),
        )
        block = Block(id="answer", kind="assistant", data={"text": "hello"})

        assert runtime._render_block(block, 80, 3) == "one"
        runtime.ui.set_theme("two")
        assert runtime._render_block(block, 80, 3) == "two"
        assert runtime._render_block(block, 80, 3) == "two"

    assert calls == ["one", "two"]


def test_replacing_theme_registry_clears_same_id_renderer_cache() -> None:
    registry = Registry()
    calls: list[str] = []

    def render(
        _block: Block,
        theme: object,
        _width: int,
        _rows: int,
        _expanded: bool,
    ) -> str:
        marker = str(getattr(theme, "marker"))
        calls.append(marker)
        return marker

    registry._add_block_renderer("test", "assistant", render)
    first = SimpleNamespace(id="project", marker="first")
    second = SimpleNamespace(id="project", marker="second")
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            theme=first,
            themes={"project": first},
            input=pipe,
            output=DummyOutput(),
        )
        block = Block(id="answer", kind="assistant", data={"text": "hello"})

        assert runtime._render_block(block, 80, 3) == "first"
        runtime.replace_themes({"project": second}, second)
        assert runtime._render_block(block, 80, 3) == "second"

    assert calls == ["first", "second"]


def test_active_block_capture_uses_selected_rich_theme() -> None:
    class ThemeProbe:
        def __init__(self) -> None:
            self.style: object | None = None

        def __rich_console__(
            self,
            console: Console,
            _options: object,
        ) -> list[Text]:
            self.style = console.get_style("accent")
            return [Text("named")]

    probe = ThemeProbe()
    registry = Registry()

    def render(
        _block: Block,
        _theme: object,
        _width: int,
        _rows: int,
        _expanded: bool,
    ) -> ThemeProbe:
        return probe

    registry._add_block_renderer("test", "named", render)
    theme = Theme(
        id="named",
        name="Named",
        colors={token: "#010203" for token in COLOR_TOKENS},
        symbols={},
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            theme=theme,
            themes={theme.id: theme},
            input=pipe,
            output=DummyOutput(),
        )
        runtime._capture_block(
            Block(id="named", kind="named", data={}),
            80,
            1,
            force_terminal=True,
        )

    assert probe.style is not None
    assert getattr(getattr(probe.style, "color"), "triplet").hex == "#010203"


@pytest.mark.asyncio
async def test_session_switch_reloads_project_themes_and_saved_selection(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    theme_dir = project / ".orcha-agent/themes"
    theme_dir.mkdir(parents=True)
    (theme_dir / "target.json").write_text(
        json.dumps(
            {
                "name": "Target",
                "colors": {"accent": "#123456"},
                "symbols": {"preset": "nerd", "overrides": {}},
            }
        ),
        encoding="utf-8",
    )
    initial_themes = load_themes(home=tmp_path, cwd=tmp_path)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=project,
            trust_cwd=True,
            symbols="ascii",
            theme="dark",
        ),
        plugin_states={"commands_core": {"theme": "target"}},
        console=SimpleNamespace(warning=lambda _message: None),
    )
    bus = EventBus()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            theme=initial_themes["dark"],
            themes=initial_themes,
            input=pipe,
            output=DummyOutput(),
        )
        _register_theme_refresh(bus, ctx, runtime)

        await bus.emit(SessionSwitch(old="old", new="target"))

    assert runtime.theme.id == "target"
    assert runtime.theme.name == "Target"
    assert runtime.theme.symbols["status.success"] == "+"
    assert "target" in runtime._themes


def test_runtime_owns_themed_statusline_and_keeps_inline_application(tmp_path: Path) -> None:
    registry = Registry()
    registry._add_status_segment(
        "test",
        "legacy",
        lambda _ctx: "legacy status",
    )
    themes = load_themes(home=tmp_path, cwd=tmp_path, symbols="ascii")
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=tmp_path,
            model="codex:gpt-5.6-sol",
            mode="ask",
            providers={},
            pricing={},
            statusbar=True,
            composer="box",
            statusline=SimpleNamespace(
                preset="minimal",
                separator="ascii",
                left=("legacy",),
                right=(),
                transparent=False,
            ),
        ),
        registry=registry,
        plugin_states={"statusbar": {}},
        console=SimpleNamespace(width=40, encoding="ascii"),
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            registry=registry,
            ctx=ctx,
            theme=themes["dark"],
            input=pipe,
            output=DummyOutput(),
        )
        rendered = "".join(text for _style, text in runtime._status())

    assert "legacy status" in rendered
    assert len(rendered) == runtime.application.output.get_size().columns
    assert ctx.ui.invalidate == runtime.application.invalidate
    assert runtime.application.full_screen is False
