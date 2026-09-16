from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict
from langgraph.graph.message import add_messages
from rich.console import Console
from rich.text import Text

from orcha_agent.core.capture import capture_graph_values, compaction_metadata, first_kept_marker
from orcha_agent.core.compaction import Compactor
from orcha_agent.core.compaction_config import CompactionConfig
from orcha_agent.core.events import ThreadSwitch, TurnEnd
from orcha_agent.core.ledger import CompactionEntry, Ledger, MessageEntry, build_context
from orcha_agent.core.session import SessionStore
from orcha_agent.core.summary import create_summary_message
from orcha_agent.tui.blocks.compaction import render
from orcha_agent.tui.frame import BlockState, Frame
from orcha_agent.tui.gallery import render_gallery_state
from orcha_agent.tui.blocks import DEFAULT_THEME
from orcha_agent.tui.transcript import Transcript


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "bad",
        {"method": [], "tokens_before": True},
        {"method": "bad", "tokens_before": -1, "short_summary": []},
    ],
)
def test_invalid_annotations_are_ignored(metadata):
    assert compaction_metadata(metadata) == {"method": "summary"}


def test_checkpoint_metadata_cannot_inject_ledger_identity(tmp_path):
    with SessionStore(tmp_path / "metadata.db") as store:
        session = store.create(tmp_path, "fake:test")
        thread = f"{session.thread_id}.0"
        marker = create_summary_message(
            "summary",
            message_id="marker",
            metadata={
                "compaction": {
                    "id": "injected",
                    "parent_id": "missing",
                    "summary": "forged",
                    "first_kept_id": "missing",
                    "ts": "forged",
                    "tokens_before": 40,
                    "short_summary": "short",
                    "method": "handoff",
                }
            },
        )
        capture_graph_values(
            store, session.thread_id, thread, {"messages": [marker]}, only_if_new=False
        )
        entry = next(
            e for e in Ledger(store).path(session.thread_id) if isinstance(e, CompactionEntry)
        )
        assert entry.id != "injected" and entry.ts != "forged"
        assert entry.summary == "summary" and entry.first_kept_id is None
        assert entry.tokens_before == 40 and entry.method == "handoff"


@pytest.mark.asyncio
async def test_manual_retention_uses_identity_after_dangling_tool_filter(tmp_path):
    from test_tui import _context, _HistoryGraph

    class Graph(_HistoryGraph):
        async def aupdate_state(self, config, values, *, as_node):
            if as_node == "__start__":
                self.messages = add_messages(self.messages, values["messages"])
            else:
                await super().aupdate_state(config, values, as_node=as_node)

    class Summary:
        async def ainvoke(self, _messages):
            return AIMessage(content="Retained decisions")

    graph = Graph([])
    ctx = _context(tmp_path, agent=graph)
    ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(keep_recent_tokens=30))
    ctx.summarizer = Summary()
    messages = [
        HumanMessage(content="old " * 400, id="old"),
        AIMessage(content="old answer", id="answer"),
        HumanMessage(content="latest request", id="latest"),
        AIMessage(
            content="", id="dangling", tool_calls=[{"id": "missing", "name": "read", "args": {}}]
        ),
        AIMessage(content="latest answer", id="latest-answer"),
    ]
    entries = ctx.ledger.append_many(
        ctx.session_id, [MessageEntry(message=message_to_dict(m)) for m in messages]
    )
    assert first_kept_marker(entries, messages[2]) == entries[1].id
    await ctx.compact()
    replay = build_context(ctx.ledger.path(ctx.session_id)).messages
    assert [m.content for m in replay[1:]] == ["latest request", "latest answer"]
    ctx.session.close()


@pytest.mark.asyncio
async def test_shake_replayed_in_reset_is_not_another_event(tmp_path):
    messages = [
        HumanMessage(content="read", id="u"),
        AIMessage(
            content="", id="a1", tool_calls=[{"id": "1", "name": "read", "args": {"path": "a"}}]
        ),
        ToolMessage(content="old", id="t1", tool_call_id="1"),
        AIMessage(
            content="", id="a2", tool_calls=[{"id": "2", "name": "read", "args": {"path": "a"}}]
        ),
        ToolMessage(content="new", id="t2", tool_call_id="2"),
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
        capture_graph_values(
            store,
            session.thread_id,
            thread,
            {"messages": [HumanMessage(content="reset", id="reset"), *result.messages]},
            only_if_new=False,
        )
        assert (
            len(
                [e for e in Ledger(store).path(session.thread_id) if isinstance(e, CompactionEntry)]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_historical_cards_seeded_and_turn_end_checks_only_new_leaf(tmp_path, monkeypatch):
    from test_tui import _context, _HistoryGraph

    ctx = _context(tmp_path, agent=_HistoryGraph([]))
    ctx.cfg = replace(ctx.cfg, compaction=CompactionConfig(enabled=False))
    old = ctx.ledger.append(ctx.session_id, CompactionEntry(summary="old"))
    await ctx._compaction_activity(
        ThreadSwitch(session_id=ctx.session_id, old="one", new="two", reason="branch")
    )
    assert old.id in ctx._shown_compactions
    cards = []
    monkeypatch.setattr(type(ctx), "_compaction_card", lambda self, entry: cards.append(entry.id))
    monkeypatch.setattr(
        Ledger,
        "path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("turn end must not scan path")
        ),
    )
    await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
    assert cards == []
    new = ctx.ledger.append(ctx.session_id, CompactionEntry(summary="new"))
    await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
    await ctx._compaction_activity(TurnEnd(thread_id=ctx.thread_id))
    assert cards == [new.id]
    ctx.session.close()


def test_compaction_block_uses_theme_literal_title_and_commits():
    transcript = Transcript(Frame())
    block = transcript.append_compaction(
        {
            "summary": "[bold]literal[/bold]",
            "method": "[link=x]handoff[/link]",
            "tokens_before": 1234,
        }
    )
    assert block.kind == "compaction" and block.state is BlockState.COMMITTED
    panel = render(block, {"colors": {"warning": "magenta", "text": "white"}}, 100, 20, False)
    assert isinstance(panel.title, Text) and "[link=x]" in panel.title.plain
    assert panel.border_style == "magenta"
    output = StringIO()
    Console(file=output, width=100, color_system=None).print(panel)
    assert "[bold]literal[/bold]" in output.getvalue()


def test_compaction_gallery_golden(update_goldens):
    actual = render_gallery_state(
        "compaction", "success", theme=DEFAULT_THEME, width=76, expanded=False, plain=True
    )
    path = Path(__file__).parent / "tui/golden/compaction.76.txt"
    if update_goldens:
        path.write_text(actual)
    assert path.read_text() == actual
