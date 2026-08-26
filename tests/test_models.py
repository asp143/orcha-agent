from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.models import ModelResolver, strip_foreign_blocks
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry


class RaisingFakeChatModel(FakeListChatModel):
    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("primary unavailable at invocation")


def _caps() -> ProviderCaps:
    return ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=False,
        structured_output=False,
        max_context=None,
    )


def _config(
    tmp_path: Path,
    *,
    model: str | list[str] = "fake:main",
    subagent_model: str = "fake:subagent",
    summarizer_model: str = "fake:summarizer",
    models: dict[str, str] | None = None,
    providers: dict[str, dict[str, object]] | None = None,
) -> Config:
    return Config(
        model=model,
        subagent_model=subagent_model,
        summarizer_model=summarizer_model,
        mode="ask",
        backend="test",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models=models or {},
        providers=providers or {},
        plugins={},
    )


def _api(registry: Registry, *, name: str = "test") -> PluginAPI:
    return PluginAPI(
        name=name,
        config={},
        state={},
        registry=registry,
        bus=EventBus(),
        request_rebuild=lambda: None,
    )


def test_resolve_expands_alias_and_passes_provider_config(tmp_path: Path) -> None:
    registry = Registry()
    seen: list[tuple[str, dict[str, object]]] = []

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        seen.append((model_name, provider_config))
        return FakeListChatModel(responses=[model_name])

    _api(registry).add_provider("fake", factory, capabilities=_caps())
    cfg = _config(
        tmp_path,
        models={"fast": "fake:small"},
        providers={"fake": {"temperature": 0}},
    )

    model = ModelResolver(registry, cfg).resolve("fast", "summarizer")

    assert model.invoke("hello").content == "small"
    assert seen == [("small", {"temperature": 0})]


def test_unknown_prefix_lists_registered_prefixes(tmp_path: Path) -> None:
    registry = Registry()
    api = _api(registry)
    for prefix in ("beta", "alpha"):
        api.add_provider(
            prefix,
            lambda name, config: FakeListChatModel(responses=[name]),
            capabilities=_caps(),
        )

    with pytest.raises(ValueError) as exc_info:
        ModelResolver(registry, _config(tmp_path)).resolve("missing:model", "main")

    message = str(exc_info.value)
    assert "missing" in message
    assert "alpha" in message
    assert "beta" in message


def test_unavailable_provider_error_includes_install_hint(tmp_path: Path) -> None:
    registry = Registry()
    factory_calls: list[str] = []

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        factory_calls.append(model_name)
        return FakeListChatModel(responses=[model_name])

    _api(registry).add_provider(
        "optional",
        factory,
        capabilities=_caps(),
        available=lambda: "pip install orcha-agent[optional]",
    )

    with pytest.raises(RuntimeError, match=r"pip install orcha-agent\[optional\]"):
        ModelResolver(registry, _config(tmp_path)).resolve("optional:model", "main")

    assert factory_calls == []


def test_resolve_roles_constructs_three_distinct_models(tmp_path: Path) -> None:
    registry = Registry()
    created: list[FakeListChatModel] = []

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        model = FakeListChatModel(responses=[model_name])
        created.append(model)
        return model

    _api(registry).add_provider("fake", factory, capabilities=_caps())
    cfg = _config(
        tmp_path,
        model="fake:main",
        subagent_model="fake:worker",
        summarizer_model="fake:summary",
    )

    roles = ModelResolver(registry, cfg).resolve_roles()

    assert set(roles) == {"main", "subagent", "summarizer"}
    assert roles["main"].invoke("hello").content == "main"
    assert roles["subagent"].invoke("hello").content == "worker"
    assert roles["summarizer"].invoke("hello").content == "summary"
    assert len(created) == 3
    assert len({id(model) for model in roles.values()}) == 3


def test_fallback_model_rolls_to_next_fake_on_invoke_error(tmp_path: Path) -> None:
    registry = Registry()

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        if model_name == "primary":
            return RaisingFakeChatModel(responses=["unused"])
        return FakeListChatModel(responses=["fallback response"])

    _api(registry).add_provider("fake", factory, capabilities=_caps())
    cfg = _config(tmp_path, model=["fake:primary", "fake:secondary"])

    model = ModelResolver(registry, cfg).resolve(cfg.model, "main")

    assert model.invoke("hello").content == "fallback response"


def test_strip_foreign_blocks_replaces_history_with_cleaned_messages() -> None:
    thread_config = {"configurable": {"thread_id": "thread-1"}}
    human = HumanMessage(content="question", id="human-1")
    assistant = AIMessage(
        id="assistant-1",
        content=[
            {"type": "thinking", "thinking": "provider-private"},
            {"type": "text", "text": "visible answer"},
            {"type": "reasoning", "summary": "provider-private"},
        ],
    )

    class RecordingGraph:
        def __init__(self) -> None:
            self.updates: list[tuple[dict[str, object], dict[str, object]]] = []

        def get_state(self, config: dict[str, object]) -> SimpleNamespace:
            assert config is thread_config
            return SimpleNamespace(values={"messages": [human, assistant]})

        def update_state(
            self,
            config: dict[str, object],
            values: dict[str, object],
        ) -> None:
            self.updates.append((config, values))

    graph = RecordingGraph()

    strip_foreign_blocks(graph, thread_config, {"thinking", "reasoning"})

    assert len(graph.updates) == 1
    updated_config, update = graph.updates[0]
    assert updated_config is thread_config
    messages = update["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == REMOVE_ALL_MESSAGES
    assert messages[1] == human
    assert isinstance(messages[2], AIMessage)
    assert messages[2].id == "assistant-1"
    assert messages[2].content == [{"type": "text", "text": "visible answer"}]
