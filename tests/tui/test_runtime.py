from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style
from rich.console import Console

from orcha_agent.core.events import ModelChunk, TurnEnd, TurnStart
from orcha_agent.core.registry import Registry
from orcha_agent.tui.frame import Block
from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade


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
