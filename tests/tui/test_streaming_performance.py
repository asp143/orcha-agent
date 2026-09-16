from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown

from orcha_agent.core.events import ModelChunk
from orcha_agent.tui.frame import StreamingData
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.transcript import Transcript


@pytest.mark.asyncio
async def test_streamed_text_materializes_only_when_read() -> None:
    transcript = Transcript()
    for _ in range(4096):
        await transcript.handle(ModelChunk(chunk=AIMessageChunk(content="x"), role="main"))
    data = transcript.frame.blocks[0].data
    assert isinstance(data, StreamingData)
    assert data.text_length == 4096
    # No prefix has been recopied while processing chunks.
    assert data.data["text"] == ""
    assert len(data._parts) == 4096
    assert data["text"] == "x" * 4096
    assert data._parts == []
    await transcript.handle(ModelChunk(chunk=AIMessageChunk(content="y"), role="main"))
    assert dict(data)["text"] == "x" * 4096 + "y"
    data.update(text="replacement")
    assert data.text_length == len("replacement")
    assert data.get("text") == "replacement"


@pytest.mark.asyncio
async def test_viewport_reuses_one_rich_layout_per_revision_and_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = Markdown.__rich_console__

    def counted(self: Markdown, *args: Any) -> Any:
        nonlocal calls
        calls += 1
        yield from original(self, *args)

    monkeypatch.setattr(Markdown, "__rich_console__", counted)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        block = runtime.frame.add("assistant", {"text": "**hello**\n" * 10})
        try:
            runtime._viewport_text()
            runtime._viewport_text()
            assert calls == 1
            block.update(text="new text")
            runtime._viewport_text()
            assert calls == 2
            for width in range(20, 40):
                runtime._measure_block(block, width)
                runtime._capture_block(block, width, 3, force_terminal=True)
            assert calls == 22
            assert len(block._rendered_rows) <= 4
        finally:
            await runtime.scheduler.aclose()


def test_renderer_budget_cache_is_bounded_and_drops_old_revisions() -> None:
    from orcha_agent.tui.blocks import BlockRendererDispatcher, DEFAULT_RENDERERS, DEFAULT_THEME
    from orcha_agent.tui.frame import Block

    dispatcher = BlockRendererDispatcher(DEFAULT_RENDERERS)
    block = Block("stream", "assistant", data={"text": "hello"})
    for rows in range(1, 100):
        dispatcher.render(block, DEFAULT_THEME, 80, rows, False)
    assert len(dispatcher._cache) == 4
    block.update(text="next")
    dispatcher.render(block, DEFAULT_THEME, 80, 5, False)
    assert len(dispatcher._cache) == 1


@pytest.mark.asyncio
async def test_visible_thinking_metrics_do_not_relayout_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = Markdown.__rich_console__

    def counted(self: Markdown, *args: Any) -> Any:
        nonlocal calls
        calls += 1
        yield from original(self, *args)

    monkeypatch.setattr(Markdown, "__rich_console__", counted)
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(lambda _: asyncio.sleep(0), input=pipe, output=DummyOutput())
        block = runtime.frame.add("thinking", {"text": "**reasoning**", "visible": True})
        try:
            before = runtime._viewport_text().value
            for tick in range(10):
                runtime.scheduler.tick_spinners(now=block.created + tick + 1)
                assert runtime._viewport_text().value == before
            assert calls == 1
            block.update(visible=False)
            collapsed = runtime._viewport_text().value
            runtime.scheduler.tick_spinners(now=block.created + 20)
            assert runtime._viewport_text().value != collapsed
            block.update(visible=True, text="new reasoning")
            runtime._viewport_text()
            assert calls == 2
        finally:
            await runtime.scheduler.aclose()
