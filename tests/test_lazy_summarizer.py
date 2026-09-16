from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore
from orcha_agent.tui.context import AppContext


@pytest.mark.asyncio
async def test_manual_summarizer_constructs_only_on_compact_and_rebuild_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(
        model="fake:main",
        subagent_model=None,
        summarizer_model="fake:summary",
        mode="ask",
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )
    registry = Registry()
    bus = EventBus()
    constructed: list[str] = []

    def factory(name: str, _config: dict[str, Any]) -> FakeListChatModel:
        constructed.append(name)
        return FakeListChatModel(responses=["summary"])

    PluginAPI(
        name="fake",
        config={},
        state={},
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    ).add_provider(
        "fake",
        factory,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=False,
            structured_output=False,
            max_context=None,
        ),
    )

    async def build(*_args: Any, **_kwargs: Any) -> object:
        return object()

    async def seed(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("orcha_agent.tui.app.build_agent", build)
    monkeypatch.setattr(AppContext, "_seed_ready_thread", seed)
    with SessionStore(cfg.db_path) as store:
        session = store.create(tmp_path, cfg.model)
        ctx = AppContext(
            cfg=cfg,
            registry=registry,
            bus=bus,
            session=store,
            plugins=[],
            plugin_states={},
            console=SimpleNamespace(print=lambda *_args: None),
            session_id=session.thread_id,
            thread_id=session.current_thread or "",
        )
        assert await ctx.ensure_agent()
        assert constructed == []
        await ctx.compact()
        assert constructed == ["summary"]
        await ctx.compact()
        assert constructed == ["summary"]
        await ctx.rebuild()
        assert ctx.summarizer is None
        assert constructed == ["summary"]
        await ctx.compact()
        assert constructed == ["summary", "summary"]


def test_provider_free_context_preserves_injected_summarizer() -> None:
    injected = object()
    ctx = SimpleNamespace(registry=Registry(), summarizer=injected)
    assert AppContext._resolve_summarizer(ctx, None) is injected
