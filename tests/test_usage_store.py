from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from orcha_agent.core.session import SessionStore
from orcha_agent.core.usage_store import UsageCallback, UsageRequest, UsageStore, usage_table


def test_usage_round_trip_periods_children_and_dedup(tmp_path):
    now = datetime(2026, 9, 17, 12, tzinfo=UTC)
    path = tmp_path / "sessions.db"
    with SessionStore(path) as session:
        parent = session.create(tmp_path, "fake:a")
        child = session.create(tmp_path, "fake:b", parent_session=parent.thread_id)
        other = session.create(tmp_path, "fake:a")
        store = UsageStore(session)
        for index, (sid, days, model) in enumerate(
            [
                (parent.thread_id, 0, "fake:a"),
                (child.thread_id, 1, "fake:b"),
                (other.thread_id, 8, "fake:a"),
            ]
        ):
            record = UsageRequest(
                str(index),
                (now - timedelta(days=days)).timestamp(),
                sid,
                model,
                "fake",
                "main",
                100,
                20,
                50,
                10,
                0.1,
                0.01,
                0.5,
                "stop",
            )
            store.record(record)
            store.record(record)
        assert sum(row["requests"] for row in store.report("session", parent.thread_id)) == 2
        assert sum(row["requests"] for row in store.report("today", now=now)) == 1
        assert sum(row["requests"] for row in store.report("week", now=now)) == 2
        assert sum(row["requests"] for row in store.report("all")) == 3
        assert "Total" in str(usage_table(store.report("all"), "all").columns[0]._cells)
    with SessionStore(path) as session:
        assert len(UsageStore(session).report("all")) == 2
        row = session._connection.execute(
            "SELECT * FROM usage_requests WHERE request_id='0'"
        ).fetchone()
        assert row["cache_write"] == 10
        assert row["ttft"] == 0.01
        assert row["stop_reason"] == "stop"


@pytest.mark.asyncio
async def test_callback_records_real_usage_and_failed_requests_once(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as session:
        info = session.create(tmp_path, "fake:model")
        cfg = SimpleNamespace(model="fake:model", pricing={"fake:model": {"input": 1, "output": 2}})
        callback = UsageCallback(session, cfg)
        request = uuid4()
        await callback.on_chat_model_start(
            {},
            [],
            run_id=request,
            metadata={
                "thread_id": info.current_thread,
                "orcha_model": "fake:model",
                "orcha_role": "task",
            },
        )
        await callback.on_llm_new_token("hello", run_id=request)
        response = LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            "hello",
                            usage_metadata={
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "total_tokens": 120,
                                "input_token_details": {"cache_read": 50, "cache_creation": 10},
                            },
                            response_metadata={"finish_reason": "length"},
                        )
                    )
                ]
            ]
        )
        await callback.on_llm_end(response, run_id=request)
        await callback.on_llm_end(response, run_id=request)
        error = uuid4()
        await callback.on_chat_model_start(
            {}, [], run_id=error, metadata={"thread_id": info.current_thread}
        )
        await callback.on_llm_error(RuntimeError("private error"), run_id=error)
        rows = session._connection.execute("SELECT * FROM usage_requests ORDER BY ts").fetchall()
        assert len(rows) == 2
        assert rows[0]["session"] == info.thread_id
        assert rows[0]["role"] == "task"
        assert rows[0]["stop_reason"] == "length"
        assert rows[0]["cost"] == pytest.approx(0.00014)
        assert rows[0]["ttft"] >= 0
        assert rows[0]["duration"] >= rows[0]["ttft"]
        assert rows[1]["stop_reason"] == "error"
        assert callback.pending == {}


@pytest.mark.asyncio
async def test_inherited_callbacks_capture_actual_model_and_deduplicate(tmp_path):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.runnables import RunnableLambda

    with SessionStore(tmp_path / "sessions.db") as session:
        info = session.create(tmp_path, "fake:primary")
        cfg = SimpleNamespace(model="fake:primary", pricing={})
        inherited = UsageCallback(session, cfg)
        explicit = UsageCallback(session, cfg, session_id=info.thread_id, role="summarizer")
        model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        "summary",
                        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    )
                ]
            ),
            metadata={"orcha_model": "fake:actual-fallback", "orcha_role": "summarizer"},
        )

        async def invoke_model(value):
            return await model.ainvoke(value, config={"callbacks": [explicit]})

        graph = RunnableLambda(invoke_model).with_config(callbacks=[inherited])
        await graph.ainvoke("hello", config={"configurable": {"thread_id": info.current_thread}})
        rows = session._connection.execute("SELECT * FROM usage_requests").fetchall()
        assert len(rows) == 1
        assert rows[0]["model"] == "fake:actual-fallback"
        assert rows[0]["role"] == "summarizer"
        assert rows[0]["session"] == info.thread_id
        assert rows[0]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_usage_plugin_refreshes_after_requests_and_renders(tmp_path):
    from orcha_agent.builtin.usage import register
    from orcha_agent.core.events import AppStart, EventBus, TurnStart
    from orcha_agent.core.plugin import PluginAPI
    from orcha_agent.core.registry import Registry
    from orcha_agent.core.usage_store import UsageRecorded

    with SessionStore(tmp_path / "sessions.db") as session:
        info = session.create(tmp_path, "fake:model")
        state = {}
        bus = EventBus()
        registry = Registry()
        printed = []
        ctx = SimpleNamespace(
            session=session,
            session_id=info.thread_id,
            console=SimpleNamespace(print=printed.append, error=printed.append),
        )
        register(
            PluginAPI(
                name="usage",
                config={},
                state=state,
                registry=registry,
                bus=bus,
                request_rebuild=lambda: None,
            )
        )
        await bus.emit(AppStart(ctx))
        assert state["session_cost"] == 0
        UsageStore(session).record(
            UsageRequest(
                "request",
                datetime.now().timestamp(),
                info.thread_id,
                "fake:model",
                "fake",
                "main",
                cost=0.42,
            )
        )
        await bus.emit(UsageRecorded())
        assert state["session_cost"] == 0.42
        assert state["today_cost"] == 0.42
        await bus.emit(TurnStart(info.current_thread, "hello"))
        await registry.commands["usage"].handler(ctx, "all")
        assert printed[-1].title.startswith("Usage · all")
        await registry.commands["usage"].handler(ctx, "invalid")
        assert printed[-1].startswith("Usage: /usage")


def test_stats_cli_isolated_database_without_provider_startup(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    db = tmp_path / "sessions.db"
    with SessionStore(db) as session:
        info = session.create(tmp_path, "unavailable:missing", thread_id="stats-fixture")
        UsageStore(session).record(
            UsageRequest(
                "current",
                datetime.now().timestamp(),
                info.thread_id,
                "unavailable:missing",
                "unavailable",
                "main",
                100,
                20,
                cost=0.25,
            )
        )
        UsageStore(session).record(
            UsageRequest(
                "old",
                (datetime.now() - timedelta(days=10)).timestamp(),
                info.thread_id,
                "unavailable:older",
                "unavailable",
                "main",
                200,
                40,
                cost=0.5,
            )
        )
    environment = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "ORCHA_DB_PATH": str(db),
        "ORCHA_MODEL": "unavailable:missing",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "COLUMNS": "200",
        "NO_COLOR": "1",
    }

    def run(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "orcha_agent", "stats", *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )

    all_usage = run("all")
    assert all_usage.returncode == 0, all_usage.stderr
    assert "unavailable:missing" in all_usage.stdout
    assert "unavailable:older" in all_usage.stdout
    assert "$0.7500" in all_usage.stdout
    today = run("today")
    assert today.returncode == 0, today.stderr
    assert "unavailable:older" not in today.stdout
    assert "$0.2500" in today.stdout
    selected = run("session", "--session", "stats-fixt")
    assert selected.returncode == 0, selected.stderr
    assert "$0.7500" in selected.stdout
    missing = run("session")
    assert missing.returncode == 2
    assert "--session SESSION" in missing.stdout
    invalid = run("session", "--session", "unknown")
    assert invalid.returncode != 0
    assert "Traceback" not in invalid.stderr


@pytest.mark.asyncio
async def test_rule_interrupt_preserves_partial_usage_once_across_retry(tmp_path):
    from orcha_agent.extensibility.stream_rules import StreamInterrupt, StreamRetry

    with SessionStore(tmp_path / "usage.db") as session:
        info = session.create(tmp_path, "fake:model")
        cfg = SimpleNamespace(model="fake:model", pricing={"fake:model": {"input": 1, "output": 2}})
        callback = UsageCallback(session, cfg)
        interrupted, retry = uuid4(), uuid4()
        partial = AIMessage(
            content="forbidden",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 3,
                "total_tokens": 103,
                "input_token_details": {"cache_read": 40, "cache_creation": 10},
            },
        )
        error = StreamInterrupt(StreamRetry([]), [partial])
        response = LLMResult(generations=[[ChatGeneration(message=partial)]])
        await callback.on_chat_model_start(
            {}, [], run_id=interrupted, metadata={"thread_id": info.current_thread}
        )
        await callback.on_llm_error(error, run_id=interrupted)
        await callback.on_llm_error(error, run_id=interrupted)
        await callback.on_llm_end(response, run_id=interrupted)
        await callback.on_chat_model_start(
            {}, [], run_id=retry, metadata={"thread_id": info.current_thread}
        )
        await callback.on_llm_end(response, run_id=retry)
        rows = session._connection.execute("SELECT * FROM usage_requests ORDER BY ts").fetchall()
        assert len(rows) == 2
        assert [row["stop_reason"] for row in rows] == ["error", ""]
        assert [row["input_tokens"] for row in rows] == [100, 100]
        assert [row["output_tokens"] for row in rows] == [3, 3]
        assert rows[0]["cache_read"] == 40 and rows[0]["cache_write"] == 10
        assert rows[0]["cost"] == pytest.approx(0.000106)
        assert callback.pending == {}
