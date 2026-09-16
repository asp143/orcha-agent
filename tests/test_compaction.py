from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orcha_agent.core.compaction import (
    CompactionMiddleware,
    Compactor,
    estimate_tokens,
    prune_results,
)
from orcha_agent.core.compaction_config import CompactionConfig, compaction_config
from orcha_agent.core.capture import capture_graph_values
from orcha_agent.core.ledger import CompactionEntry, Ledger, build_context
from orcha_agent.core.session import SessionStore


class Summary:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="Summary of decisions\nNext steps")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger", ["manual", "overflow", "length", "post_turn", "mid_turn", "idle"]
)
async def test_every_trigger_with_fake_summary(trigger):
    model = Summary()
    policy = CompactionConfig(threshold_tokens=100, keep_recent_tokens=30)
    controller = Compactor(model, policy)
    messages = [
        HumanMessage(content="old " * 400),
        AIMessage(content="old answer"),
        HumanMessage(content="continue"),
    ]
    assert controller.needed(messages, trigger)
    result = await controller.compact(
        messages, "Preserve test failures" if trigger == "manual" else ""
    )
    assert result.method == "summary"
    assert result.messages[-1] is messages[-1]
    assert result.tokens_before == estimate_tokens(messages)
    assert result.short_summary == "Summary of decisions"
    if trigger == "manual":
        assert "Preserve test failures" in model.calls[0][-1].content


@pytest.mark.asyncio
async def test_disabled_auto_and_length_hook():
    model = Summary()
    controller = Compactor(model, CompactionConfig(threshold_tokens=100_000))
    middleware = CompactionMiddleware(controller)
    messages = [
        HumanMessage(content="request"),
        AIMessage(content="partial", response_metadata={"finish_reason": "length"}),
    ]

    async def handler(request):
        return ModelResponse(result=[messages[-1]])

    assert await middleware.awrap_model_call(SimpleNamespace(messages=messages[:1]), handler)
    controller.policy = replace(controller.policy, enabled=False)
    assert not controller.needed(messages, "length")
    assert controller.needed(messages, "manual")


@pytest.mark.asyncio
async def test_overflow_retries_once_and_preserves_response():
    controller = Compactor(Summary(), CompactionConfig())
    middleware = CompactionMiddleware(controller)
    calls = []

    class Request:
        messages = [HumanMessage(content="request")]

        def override(self, **kwargs):
            return SimpleNamespace(**kwargs)

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise ValueError("context_length_exceeded")
        return ModelResponse(result=[AIMessage(content="success")])

    response = await middleware.awrap_model_call(Request(), handler)
    assert len(calls) == 2
    assert response.model_response.result[0].content == "success"
    assert calls[1].messages[0].additional_kwargs["lc_source"] == "summarization"
    calls.clear()

    async def always_fail(request):
        calls.append(request)
        raise ValueError("context_length_exceeded")

    with pytest.raises(ValueError, match="context_length_exceeded"):
        await middleware.awrap_model_call(Request(), always_fail)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_non_overflow_not_retried_and_pending_calls_retained():
    model = Summary()
    middleware = CompactionMiddleware(Compactor(model, CompactionConfig(threshold_tokens=1)))
    pending = AIMessage(
        content="", tool_calls=[{"id": "call", "name": "read", "args": {"path": "a"}}]
    )
    middleware.compactor.policy = CompactionConfig(threshold_tokens=100_000)

    async def pending_handler(request):
        return ModelResponse(result=[pending])

    await middleware.awrap_model_call(SimpleNamespace(messages=[]), pending_handler)

    async def handler(request):
        raise ValueError("authentication failed")

    with pytest.raises(ValueError, match="authentication"):
        await middleware.awrap_model_call(SimpleNamespace(messages=[]), handler)
    assert model.calls == []


def test_shake_preserves_pairs_and_partial_read_ranges():
    messages = [
        AIMessage(content="", tool_calls=[{"id": "1", "name": "read", "args": {"path": "a"}}]),
        ToolMessage(content="old", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"id": "2", "name": "read", "args": {"path": "a"}}]),
        ToolMessage(content="new", tool_call_id="2"),
    ]
    result = prune_results(messages, CompactionConfig())
    assert result[1].content == "[Superseded tool result]"
    assert result[3].content == "new"
    assert len(result) == len(messages)
    assert messages[1].content == "old"
    messages[2].tool_calls[0]["args"]["offset"] = 10
    assert prune_results(messages, CompactionConfig())[1].content == "old"


@pytest.mark.asyncio
async def test_handoff_fallback_and_speculative_prefix_reuse():
    class Fallback(Summary):
        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                raise ValueError("summary failed")
            return AIMessage(content="Goal\nNext steps")

    model = Fallback()
    controller = Compactor(model, CompactionConfig(threshold_tokens=100, keep_recent_tokens=20))
    old = HumanMessage(content="old " * 400)
    messages = [old, AIMessage(content="answer"), HumanMessage(content="next")]
    controller.speculate(messages)
    result = await controller.compact([*messages, AIMessage(content="response")])
    assert result.method == "handoff"
    assert len(model.calls) == 2
    assert result.messages[-1].content == "response"
    assert "structured handoff" in model.calls[-1][-1].content


def test_policy_validation_and_dynamic_reserve():
    assert CompactionConfig().threshold(200_000) == 160_000
    assert CompactionConfig(threshold_ratio=1).threshold(200_000) == 170_000
    for raw in (
        {"enabled": "true"},
        {"threshold_ratio": 2},
        {"method_order": []},
        {"reserve_tokens": -1},
    ):
        with pytest.raises(ValueError):
            compaction_config(raw)


@pytest.mark.asyncio
async def test_graph_compaction_checkpoint_ledger_roundtrip(tmp_path):
    policy = CompactionConfig(threshold_tokens=150, keep_recent_tokens=40, speculative=False)
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(model="fake:test", mode="yolo", cwd=str(tmp_path))
        # SessionStore exposes the session id through thread_id for compatibility.
        session_id = session.thread_id
        thread_id = f"{session_id}.0"
        store.activate_thread(session_id, thread_id)
        graph = create_agent(
            FakeListChatModel(responses=["done"]),
            middleware=[CompactionMiddleware(Compactor(Summary(), policy))],
            checkpointer=store.saver,
        )
        config = {"configurable": {"thread_id": thread_id}}
        messages = [
            HumanMessage(content="old " * 400, id="old"),
            AIMessage(content="answer", id="answer"),
        ]
        await graph.aupdate_state(config, {"messages": messages})
        capture_graph_values(
            store, session_id, thread_id, graph.get_state(config).values, only_if_new=False
        )
        await graph.ainvoke({"messages": [HumanMessage(content="continue", id="new")]}, config)
        capture_graph_values(
            store, session_id, thread_id, graph.get_state(config).values, only_if_new=False
        )
        entries = Ledger(store).path(session_id)
        compacted = [entry for entry in entries if isinstance(entry, CompactionEntry)]
        assert compacted and compacted[-1].tokens_before
        assert compacted[-1].method == "summary"
        assert compacted[-1].short_summary == "Summary of decisions"
        assert build_context(entries).messages[-1].content == "done"


@pytest.mark.asyncio
async def test_idle_compacts_and_new_turn_waits_for_cancellation(tmp_path):
    import asyncio
    from test_tui import _context, _HistoryGraph
    from orcha_agent.core.events import TurnEnd, TurnStart
    from orcha_agent.core.ledger import MessageEntry
    from langchain_core.messages import message_to_dict

    ctx = _context(tmp_path, agent=_HistoryGraph([]))
    ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(threshold_tokens=10, idle_seconds=0.001))
    ctx.summarizer = Summary()
    Ledger(ctx.session).append(
        ctx.session_id, MessageEntry(message=message_to_dict(HumanMessage(content="old " * 400)))
    )
    await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
    assert ctx._idle_compaction is not None
    await asyncio.wait_for(ctx._idle_compaction, 2)
    assert any(
        isinstance(entry, CompactionEntry) for entry in Ledger(ctx.session).path(ctx.session_id)
    )
    assert ctx.compaction_status == ""
    await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
    pending = ctx._idle_compaction
    await ctx._compaction_activity(TurnStart(thread_id=ctx.thread_id, text="next"))
    assert pending is not None and pending.done()
    assert ctx._idle_compaction is None
    ctx.session.close()


@pytest.mark.asyncio
async def test_shake_capture_keeps_history_and_roundtrips_metadata(tmp_path):
    from langchain_core.messages import message_to_dict

    messages = [
        HumanMessage(content="read file", id="user"),
        AIMessage(
            content="", id="call1", tool_calls=[{"id": "1", "name": "read", "args": {"path": "a"}}]
        ),
        ToolMessage(content="old", id="result1", tool_call_id="1"),
        AIMessage(
            content="", id="call2", tool_calls=[{"id": "2", "name": "read", "args": {"path": "a"}}]
        ),
        ToolMessage(content="new", id="result2", tool_call_id="2"),
    ]
    with SessionStore(tmp_path / "shake.db") as store:
        session = store.create(tmp_path, "fake:test")
        thread = f"{session.thread_id}.0"
        capture_graph_values(
            store, session.thread_id, thread, {"messages": messages}, only_if_new=False
        )
        result = await Compactor(None, CompactionConfig(method_order=("shake",))).compact(messages)
        capture_graph_values(
            store, session.thread_id, thread, {"messages": result.messages}, only_if_new=False
        )
        entries = Ledger(store).path(session.thread_id)
        compactions = [entry for entry in entries if isinstance(entry, CompactionEntry)]
        assert len(compactions) == 1
        assert compactions[0].method == "shake"
        assert compactions[0].tokens_before == result.tokens_before
        replayed = build_context(entries).messages
        assert [message_to_dict(m) for m in replayed] == [
            message_to_dict(m) for m in result.messages
        ]


@pytest.mark.asyncio
async def test_compaction_card_renders_metadata(tmp_path):
    from io import StringIO
    from rich.console import Console
    from test_tui import _context, _HistoryGraph
    from orcha_agent.tui.console import ConsoleOutput

    output = StringIO()
    ctx = _context(tmp_path, agent=_HistoryGraph([]))
    ctx.console = ConsoleOutput(Console(file=output, width=90, color_system=None))
    ctx._compaction_card(
        CompactionEntry(
            summary="Full summary",
            short_summary="Keep decisions",
            tokens_before=12345,
            method="handoff",
        )
    )
    rendered = output.getvalue()
    assert "Compacted" in rendered and "handoff" in rendered and "12,345 tokens before" in rendered
    assert "Keep decisions" in rendered
    ctx.session.close()


def test_export_roundtrip_preserves_compaction_method():
    from orcha_agent.core.export import entry_from_envelope, entry_to_envelope

    for method in ("summary", "handoff", "shake"):
        entry = CompactionEntry(
            summary="summary",
            short_summary="short",
            first_kept_id="",
            tokens_before=40_000,
            method=method,
        )
        assert entry_from_envelope(entry_to_envelope(entry)) == entry


def test_provider_usage_controls_threshold():
    message = AIMessage(
        content="short",
        usage_metadata={"input_tokens": 80_000, "output_tokens": 100, "total_tokens": 80_100},
    )
    compactor = Compactor(None, CompactionConfig(threshold_tokens=80_000))
    assert compactor.needed([message], "post_turn")
    assert estimate_tokens([message]) == 80_100
    assert estimate_tokens([message, ToolMessage(content="result", tool_call_id="a")]) > 80_100


@pytest.mark.asyncio
async def test_manual_retained_tail_pruning_and_usage_survive_reseed(tmp_path):
    from test_tui import _real_context

    with SessionStore(tmp_path / "manual.db") as store:
        session = store.create(tmp_path, "fake:test")
        graph = create_agent(FakeListChatModel(responses=["done"]), checkpointer=store.saver)
        ctx = _real_context(tmp_path, store, graph, session_id=session.thread_id)
        ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(keep_recent_tokens=200))
        ctx.summarizer = Summary()
        messages = [
            HumanMessage(content="old " * 400, id="old"),
            AIMessage(content="old answer", id="old-answer"),
            HumanMessage(content="new", id="new"),
            AIMessage(
                content="answer",
                id="new-answer",
                usage_metadata={
                    "input_tokens": 80_000,
                    "output_tokens": 20,
                    "total_tokens": 80_020,
                },
            ),
        ]
        await graph.aupdate_state(ctx.thread_config, {"messages": messages})
        ctx.capture_turn()
        await ctx.compact("keep the final turn")
        replayed = build_context(Ledger(store).path(session.thread_id)).messages
        assert replayed[-1].content == "answer"
        assert replayed[-1].additional_kwargs["compaction_usage_stale"] is True
        assert estimate_tokens(replayed) < 1000
        assert (
            graph.get_state(ctx.thread_config)
            .values["messages"][-1]
            .additional_kwargs["compaction_usage_stale"]
            is True
        )


@pytest.mark.parametrize("failed", [False, True])
def test_failed_or_pending_reread_does_not_supersede_success(failed):
    messages = [
        AIMessage(content="", tool_calls=[{"id": "old", "name": "read", "args": {"path": "a"}}]),
        ToolMessage(content="useful", tool_call_id="old"),
        AIMessage(content="", tool_calls=[{"id": "new", "name": "read", "args": {"path": "a"}}]),
    ]
    if failed:
        messages.append(
            ToolMessage(content="permission denied", tool_call_id="new", status="error")
        )
    assert prune_results(messages, CompactionConfig())[1].content == "useful"


@pytest.mark.asyncio
async def test_summary_stream_is_hidden_but_usage_is_recorded(tmp_path):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from orcha_agent.core.usage_store import UsageCallback

    with SessionStore(tmp_path / "stream.db") as store:
        session = store.create(tmp_path, "fake:main")
        summarizer = GenericFakeChatModel(
            # GenericFakeChatModel streaming drops usage metadata; the real
            # callback path is exercised with its complete response here.
            disable_streaming=True,
            messages=iter(
                [
                    AIMessage(
                        content="PRIVATE_COMPACTION_SUMMARY",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "total_tokens": 110,
                        },
                    )
                ]
            ),
            metadata={"orcha_model": "fake:summary", "orcha_role": "summarizer"},
        )
        middleware = CompactionMiddleware(
            Compactor(
                summarizer,
                CompactionConfig(threshold_tokens=150, keep_recent_tokens=40, speculative=False),
            )
        )
        graph = create_agent(
            FakeListChatModel(responses=["VISIBLE_MAIN_ANSWER"]),
            middleware=[middleware],
            checkpointer=store.saver,
        )
        callback = UsageCallback(store, SimpleNamespace(model="fake:main", pricing={}))
        config = {"configurable": {"thread_id": session.current_thread}, "callbacks": [callback]}
        emitted = []
        async for message, metadata in graph.astream(
            {
                "messages": [
                    HumanMessage(content="old " * 400),
                    AIMessage(content="old answer"),
                    HumanMessage(content="continue"),
                ]
            },
            config=config,
            stream_mode="messages",
        ):
            # State updates can emit the HumanMessage summary marker; the TUI
            # consumes only AI model output. No summary AI tokens may escape.
            if isinstance(message, AIMessage):
                emitted.append(str(message.content))
        assert "VISIBLE_MAIN_ANSWER" in "".join(emitted)
        assert "PRIVATE_COMPACTION_SUMMARY" not in "".join(emitted)
        rows = store._connection.execute(
            "SELECT * FROM usage_requests WHERE role='summarizer'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["output_tokens"] == 10
