from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from orcha_agent.core.events import ModelChunk
from orcha_agent.extensibility.stream_rules import StreamAborted
from orcha_agent.tui.frame import BlockState
from orcha_agent.tui.transcript import Transcript


@pytest.mark.asyncio
@pytest.mark.parametrize("aborted_source", ["main", "worker"])
async def test_abort_only_settles_matching_source(aborted_source):
    transcript = Transcript()
    blocks = {}
    for source in ("main", "worker"):
        await transcript.handle(
            ModelChunk(
                chunk=AIMessageChunk(content=f"{source} text"), role="main", source_id=source
            )
        )
        blocks[source] = transcript._source_blocks[(source, "assistant")]
    await transcript.handle(StreamAborted(source_id=aborted_source))
    for source, block in blocks.items():
        assert bool(block.data.get("aborted")) == (source == aborted_source)
        assert (block.state is BlockState.ACTIVE) == (source != aborted_source)
    await transcript.handle(
        ModelChunk(chunk=AIMessageChunk(content="retry"), role="main", source_id=aborted_source)
    )
    assert transcript._source_blocks[(aborted_source, "assistant")] is not blocks[aborted_source]


@pytest.mark.asyncio
async def test_interceptor_emits_host_source_on_abort():
    from orcha_agent.core.events import EventBus
    from orcha_agent.extensibility.stream_rules import (
        StreamInspect,
        StreamInterrupt,
        StreamRetry,
        intercepted_stream,
    )

    class Agent:
        calls = 0

        async def astream(self, value, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise StreamInterrupt(StreamRetry(messages=[]))
            yield "updates", {}

    bus = EventBus()
    seen = []

    async def inspect(event):
        return None

    async def record(event):
        seen.append(event)

    bus.on(StreamInspect, inspect)
    bus.on(StreamAborted, record)
    host = SimpleNamespace(agent=Agent(), bus=bus, source_id="worker")
    assert [item async for item in intercepted_stream(host, {})] == [("updates", {})]
    assert [event.source_id for event in seen] == ["worker"]
