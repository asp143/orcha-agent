from dataclasses import replace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from orcha_agent.core.compaction import CompactionMiddleware, Compactor
from orcha_agent.core.compaction_config import CompactionConfig
from orcha_agent.core.events import Compaction
from orcha_agent.extensibility.hooks import HooksMiddleware


class Summary:
    async def ainvoke(self, messages):
        return AIMessage(content="Keep the verified decision")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["summary", "shake"])
async def test_committed_compaction_emits_hook_once_across_checkpoint_resume(method):
    events = []

    async def emit(event):
        events.append(event)

    controller = Compactor(
        Summary(),
        CompactionConfig(
            threshold_tokens=150,
            keep_recent_tokens=40,
            speculative=False,
            method_order=(method,),
        ),
    )
    graph = create_agent(
        FakeListChatModel(responses=["done"]),
        middleware=[HooksMiddleware(emit), CompactionMiddleware(controller)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "compaction-hooks"}}
    if method == "summary":
        messages = [HumanMessage(content="old " * 400), AIMessage(content="answer")]
    else:
        messages = [
            HumanMessage(content="read the file"),
            AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "read1"}]),
            ToolMessage(
                content="old " * 400, tool_call_id="read1", additional_kwargs={"superseded": True}
            ),
            HumanMessage(content="continue"),
        ]
    result = await graph.ainvoke({"messages": messages}, config)
    marker_key = "compaction" if method == "summary" else "compaction_shake"
    assert any(marker_key in message.additional_kwargs for message in result["messages"])
    compacted = [event for event in events if isinstance(event, Compaction)]
    assert len(compacted) == 1
    assert compacted[0].session_id == "compaction-hooks"
    assert compacted[0].summary == (
        "Keep the verified decision" if method == "summary" else "Superseded tool results removed."
    )
    controller.policy = replace(controller.policy, threshold_tokens=100_000)
    await graph.ainvoke({"messages": [HumanMessage(content="continue")]}, config)
    assert [event for event in events if isinstance(event, Compaction)] == compacted


@pytest.mark.asyncio
async def test_uncommitted_speculative_summary_does_not_emit_hook():
    events = []

    async def emit(event):
        events.append(event)

    hooks = HooksMiddleware(emit)
    state = {"messages": [HumanMessage(content="old " * 400)]}
    state.update(await hooks.abefore_model(state, None))
    controller = Compactor(Summary(), CompactionConfig(keep_recent_tokens=40))
    prepared = await controller.compact(state["messages"])
    assert prepared.summary
    assert await hooks.aafter_model(state, None) is None
    assert events == []


@pytest.mark.asyncio
async def test_manual_compaction_emits_lifecycle_event(tmp_path):
    from langchain_core.messages import message_to_dict
    from langgraph.graph.message import add_messages
    from test_tui import _context, _HistoryGraph

    from orcha_agent.core.ledger import MessageEntry

    class Graph(_HistoryGraph):
        async def aupdate_state(self, config, values, *, as_node):
            if as_node == "__start__":
                self.messages = add_messages(self.messages, values["messages"])
            else:
                await super().aupdate_state(config, values, as_node=as_node)

    ctx = _context(tmp_path, agent=Graph([]))
    ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(keep_recent_tokens=30))
    ctx.summarizer = Summary()
    try:
        ctx.ledger.append_many(
            ctx.session_id,
            [
                MessageEntry(message=message_to_dict(message))
                for message in [
                    HumanMessage(content="old " * 400, id="old"),
                    AIMessage(content="old answer", id="answer"),
                    HumanMessage(content="latest request", id="latest"),
                    AIMessage(content="latest answer", id="latest-answer"),
                ]
            ],
        )
        await ctx.compact()
        assert [
            (event.session_id, event.summary)
            for event in ctx.bus.events
            if isinstance(event, Compaction)
        ] == [(ctx.session_id, "Keep the verified decision")]
    finally:
        ctx.session.close()


@pytest.mark.asyncio
async def test_automatic_shake_without_changes_does_not_emit_lifecycle():
    events = []

    async def emit(event):
        events.append(event)

    graph = create_agent(
        FakeListChatModel(responses=["done"]),
        middleware=[
            HooksMiddleware(emit),
            CompactionMiddleware(
                Compactor(
                    None,
                    CompactionConfig(
                        threshold_tokens=150,
                        keep_recent_tokens=40,
                        speculative=False,
                        method_order=("shake",),
                    ),
                )
            ),
        ],
    )
    await graph.ainvoke({"messages": [HumanMessage(content="old " * 400)]})
    assert not any(isinstance(event, Compaction) for event in events)
