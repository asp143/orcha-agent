"""Regression contracts for documented compaction hooks and usage semantics."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from orcha_agent.core.compaction import prune_results
from orcha_agent.core.compaction_config import CompactionConfig
from orcha_agent.core.usage_store import UsageCallback


def test_m6_drop_useless_is_an_opt_in_plugin_metadata_hook():
    call = AIMessage(content="", tool_calls=[{"id": "call", "name": "custom", "args": {}}])
    result = ToolMessage(content="valuable history", tool_call_id="call")
    policy = CompactionConfig(supersede_reads=False)
    assert prune_results([call, result], policy)[1].content == "valuable history"
    marked = result.model_copy(update={"additional_kwargs": {"superseded": True}})
    cleaned = prune_results([call, marked], policy)
    assert cleaned[1].content == "[Superseded tool result]"
    assert cleaned[0].tool_calls[0]["id"] == cleaned[1].tool_call_id
    assert prune_results([call, marked], replace(policy, drop_useless=False))[1] == marked
    assert marked.content == "valuable history"


@pytest.mark.asyncio
async def test_m10_ttft_tracks_first_visible_token_and_stays_unknown_without_one(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("orcha_agent.core.usage_store.time.monotonic", lambda: clock[0])
    callback = UsageCallback(None, SimpleNamespace(model="fake:test"))
    request = uuid4()
    await callback.on_chat_model_start({}, [], run_id=request)
    clock[0] = 11.0
    await callback.on_llm_new_token("", run_id=request)
    assert callback.pending[request]["ttft"] is None
    clock[0] = 12.5
    await callback.on_llm_new_token("answer", run_id=request)
    assert callback.pending[request]["ttft"] == 2.5
    clock[0] = 13.0
    await callback.on_llm_new_token("later", run_id=request)
    assert callback.pending[request]["ttft"] == 2.5
    nonstreaming = uuid4()
    await callback.on_chat_model_start({}, [], run_id=nonstreaming)
    assert callback.pending[nonstreaming]["ttft"] is None


def test_m14_readme_documents_nondefault_boundaries():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "`drop_useless` is a plugin hook" in readme
    assert "`idle_seconds` is the quiet interval" in readme
    assert "`speculative` enables background summary preparation" in readme
    assert "`!command` in the **user-scope** file only" in readme
    assert "cached for the process lifetime" in readme
    assert "role smol is not configured, using main" in readme
    assert "first visible token" in readme
