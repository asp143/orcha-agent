from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from orcha_agent.builtin import provider_ollama
from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.models import ModelResolver, strip_foreign_blocks
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry


class RaisingFakeChatModel(FakeListChatModel):
    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("primary unavailable at invocation")


def _caps(*, thinking: bool = False) -> ProviderCaps:
    return ProviderCaps(
        tool_calling=True,
        streaming=True,
        thinking=thinking,
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


def test_missing_provider_key_error_names_accepted_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    keys = ("FIRST_FAKE_API_KEY", "SECOND_FAKE_API_KEY")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    factory_calls: list[str] = []

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        factory_calls.append(model_name)
        return FakeListChatModel(responses=[model_name])

    _api(registry).add_provider(
        "guarded",
        factory,
        capabilities=_caps(),
        env_keys=keys,
    )

    with pytest.raises(RuntimeError) as exc_info:
        ModelResolver(registry, _config(tmp_path)).resolve("guarded:model", "main")

    message = str(exc_info.value)
    assert "FIRST_FAKE_API_KEY" in message
    assert "SECOND_FAKE_API_KEY" in message
    assert factory_calls == []


def test_fallback_chain_skips_provider_with_no_accepted_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    monkeypatch.delenv("PRIMARY_FAKE_API_KEY", raising=False)
    factory_calls: list[tuple[str, str]] = []

    def guarded_factory(
        model_name: str,
        provider_config: dict[str, object],
    ) -> FakeListChatModel:
        factory_calls.append(("guarded", model_name))
        return FakeListChatModel(responses=["unavailable"])

    def fallback_factory(
        model_name: str,
        provider_config: dict[str, object],
    ) -> FakeListChatModel:
        factory_calls.append(("fallback", model_name))
        return FakeListChatModel(responses=["fallback response"])

    api = _api(registry)
    api.add_provider(
        "guarded",
        guarded_factory,
        capabilities=_caps(),
        env_keys=("PRIMARY_FAKE_API_KEY",),
    )
    api.add_provider("fallback", fallback_factory, capabilities=_caps())

    models = ModelResolver(registry, _config(tmp_path)).resolve_chain(
        ["guarded:primary", "fallback:secondary"],
        "main",
    )

    assert len(models) == 1
    assert models[0].invoke("hello").content == "fallback response"
    assert factory_calls == [("fallback", "secondary")]


def test_ollama_provider_does_not_require_an_environment_variable() -> None:
    registry = Registry()

    provider_ollama.register(_api(registry))
    assert registry.providers["ollama"].env_keys == ()



def test_availability_hints_skip_fallbacks_and_report_all_failures(
    tmp_path: Path,
) -> None:
    registry = Registry()
    api = _api(registry)
    api.add_provider(
        "missing_one",
        lambda name, config: FakeListChatModel(responses=[name]),
        capabilities=_caps(),
        available=lambda: "pip install missing-one",
    )
    api.add_provider(
        "working",
        lambda name, config: FakeListChatModel(responses=["working"]),
        capabilities=_caps(),
    )

    resolved = ModelResolver(registry, _config(tmp_path)).resolve_chain(
        ["missing_one:primary", "working:fallback"],
        "main",
    )

    assert len(resolved) == 1
    assert resolved[0].invoke("hello").content == "working"

    api.add_provider(
        "missing_two",
        lambda name, config: FakeListChatModel(responses=[name]),
        capabilities=_caps(),
        available=lambda: "pip install missing-two",
    )
    with pytest.raises(RuntimeError) as exc_info:
        ModelResolver(registry, _config(tmp_path)).resolve_chain(
            ["missing_one:first", "missing_two:second"],
            "main",
        )

    message = str(exc_info.value)
    assert "pip install missing-one" in message
    assert "pip install missing-two" in message


def test_non_thinking_provider_does_not_receive_thinking_config(tmp_path: Path) -> None:
    registry = Registry()
    seen: list[dict[str, object]] = []

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        seen.append(provider_config)
        return FakeListChatModel(responses=[model_name])

    _api(registry).add_provider("fake", factory, capabilities=_caps(thinking=False))
    cfg = _config(
        tmp_path,
        providers={
            "fake": {
                "temperature": 0,
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            }
        },
    )

    ModelResolver(registry, cfg).resolve("fake:model", "main")

    assert seen == [{"temperature": 0}]


def test_thinking_provider_receives_thinking_config_unchanged(tmp_path: Path) -> None:
    registry = Registry()
    seen: list[dict[str, object]] = []
    thinking = {"type": "enabled", "budget_tokens": 2048}

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        seen.append(provider_config)
        return FakeListChatModel(responses=[model_name])

    _api(registry).add_provider("fake", factory, capabilities=_caps(thinking=True))
    cfg = _config(
        tmp_path,
        providers={"fake": {"temperature": 0, "thinking": thinking}},
    )

    ModelResolver(registry, cfg).resolve("fake:model", "main")

    assert seen == [{"temperature": 0, "thinking": thinking}]


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


def test_resolve_chain_returns_primary_and_fallback_chat_models(tmp_path: Path) -> None:
    registry = Registry()

    def factory(model_name: str, provider_config: dict[str, object]) -> FakeListChatModel:
        if model_name == "primary":
            return RaisingFakeChatModel(responses=["unused"])
        return FakeListChatModel(responses=["fallback response"])

    _api(registry).add_provider("fake", factory, capabilities=_caps())
    cfg = _config(tmp_path, model=["fake:primary", "fake:secondary"])

    models = ModelResolver(registry, cfg).resolve_chain(cfg.model, "main")

    assert len(models) == 2
    with pytest.raises(RuntimeError, match="primary unavailable"):
        models[0].invoke("hello")
    assert models[1].invoke("hello").content == "fallback response"


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
        additional_kwargs={
            "reasoning": {"encrypted": "private"},
            "safe": "keep",
        },
        response_metadata={
            "reasoning": {"summary": "private"},
            "usage": {"input_tokens": 1},
        },
    )

    class RecordingGraph:
        def __init__(self) -> None:
            self.updates: list[tuple[dict[str, object], dict[str, object]]] = []
            self.as_nodes: list[str | None] = []

        def get_state(self, config: dict[str, object]) -> SimpleNamespace:
            assert config is thread_config
            return SimpleNamespace(values={"messages": [human, assistant]})

        def update_state(
            self,
            config: dict[str, object],
            values: dict[str, object],
            *,
            as_node: str | None = None,
        ) -> None:
            self.updates.append((config, values))
            self.as_nodes.append(as_node)

    graph = RecordingGraph()

    strip_foreign_blocks(graph, thread_config, {"thinking", "reasoning"})

    assert len(graph.updates) == 1
    updated_config, update = graph.updates[0]
    assert graph.as_nodes == ["model"]
    assert updated_config is thread_config
    messages = update["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == REMOVE_ALL_MESSAGES
    assert messages[1] == human
    assert isinstance(messages[2], AIMessage)
    assert messages[2].id == "assistant-1"
    assert messages[2].content == [{"type": "text", "text": "visible answer"}]
    assert messages[2].additional_kwargs == {"safe": "keep"}
    assert messages[2].response_metadata == {"usage": {"input_tokens": 1}}
