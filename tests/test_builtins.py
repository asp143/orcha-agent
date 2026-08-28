from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import pytest

import orcha_agent.builtin
from orcha_agent.core.config import Config
from orcha_agent.core.events import EventBus
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


EXPECTED_BUILTIN_PLUGINS = {
    "approval_prompt",
    "banner",
    "commands_core",
    "commands_model",
    "commands_session",
    "filesystem",
    "memory",
    "modes",
    "provider_anthropic",
    "provider_codex",
    "provider_google",
    "provider_langchain",
    "provider_ollama",
    "provider_openai",
    "render_default",
    "statusbar",
}
EXPECTED_COMMANDS = {
    "branch",
    "clear",
    "compact",
    "export",
    "exit",
    "fork",
    "help",
    "login",
    "keys",
    "logout",
    "mode",
    "model",
    "new",
    "plugins",
    "providers",
    "resume",
    "sessions",
    "status",
    "theme",
    "thinking",
    "tree",
}
EXPECTED_PROVIDERS = {"anthropic", "codex", "google", "langchain", "ollama", "openai"}
EXPECTED_MODES = {"ask", "edit", "plan", "yolo"}


def config_for(tmp_path: Path) -> Config:
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    return Config(
        model="anthropic:test",
        subagent_model="anthropic:test",
        summarizer_model="anthropic:test",
        mode="ask",
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=cwd,
        resume=None,
        list_sessions=False,
        strict_plugins=False,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={"disabled": ()},
    )


def assert_registry_empty(registry: Registry) -> None:
    assert not registry.tools
    assert not registry.commands
    assert not registry.providers
    assert not registry.backends
    assert not registry.modes
    assert not registry.middleware
    assert not registry.renderers
    assert not registry.block_renderers
    assert not registry.subagents
    assert not registry.prompt_fragments


def builtin_modules() -> list[ModuleType]:
    return [
        importlib.import_module(module_info.name)
        for module_info in pkgutil.iter_modules(
            orcha_agent.builtin.__path__,
            prefix=f"{orcha_agent.builtin.__name__}.",
        )
    ]


def test_registry_stays_empty_until_builtins_are_loaded() -> None:
    registry = Registry()

    assert_registry_empty(registry)
    modules = builtin_modules()
    assert_registry_empty(registry)

    assert EXPECTED_BUILTIN_PLUGINS <= {
        module.__name__.rsplit(".", maxsplit=1)[-1] for module in modules
    }


def test_every_builtin_has_only_plugin_api_as_its_registration_boundary() -> None:
    modules = builtin_modules()

    for module in modules:
        signature = inspect.signature(module.register)
        parameters = list(signature.parameters.values())
        assert len(parameters) == 1, module.__name__
        parameter = parameters[0]
        assert parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }, module.__name__
        assert get_type_hints(module.register)[parameter.name] is PluginAPI

        result = module.register(
            PluginAPI(
                name=module.__name__.rsplit(".", maxsplit=1)[-1],
                config={},
                state={},
                registry=Registry(),
                bus=EventBus(),
                request_rebuild=lambda: None,
            )
        )
        assert result is None, module.__name__


def test_loading_builtins_registers_expected_plugins_and_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: [])
    registry = Registry()
    bus = EventBus()
    assert_registry_empty(registry)

    records = load_plugins(registry, bus, config_for(tmp_path))

    assert EXPECTED_BUILTIN_PLUGINS <= {record.name for record in records}
    assert set(registry.backends) == {"local_shell"}
    assert set(registry.modes) == EXPECTED_MODES
    assert set(registry.providers) == EXPECTED_PROVIDERS
    assert set(registry.commands) == EXPECTED_COMMANDS
    assert {entry.kind for entry in registry.block_renderers} == {
        "assistant",
        "banner",
        "diff",
        "marker",
        "queue",
        "subagents",
        "thinking",
        "todo",
        "tool",
        "user",
        "welcome",
    }


def test_anthropic_adaptive_thinking_config_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_anthropic

    from orcha_agent.builtin import provider_anthropic

    monkeypatch.setattr(
        langchain_anthropic,
        "ChatAnthropic",
        lambda **options: options,
    )
    registry = Registry()
    provider_anthropic.register(
        PluginAPI(
            name="provider_anthropic",
            config={"_ui_thinking": "summary"},
            state={},
            registry=registry,
            bus=EventBus(),
            request_rebuild=lambda: None,
        )
    )

    model = registry.providers["anthropic"].factory(
        "claude-opus-5",
        {"max_tokens": 16000},
    )

    assert model["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert model["max_tokens"] == 16000


def test_anthropic_omits_thinking_when_session_toggle_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_anthropic

    from orcha_agent.builtin import provider_anthropic

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **options: options)
    registry = Registry()
    provider_anthropic.register(
        PluginAPI(
            name="provider_anthropic",
            config={"_ui_thinking": "summary"},
            state={"thinking": "off"},
            registry=registry,
            bus=EventBus(),
            request_rebuild=lambda: None,
        )
    )

    model = registry.providers["anthropic"].factory(
        "claude-opus-5",
        {"thinking": "adaptive", "max_tokens": 16000},
    )

    assert "thinking" not in model


def test_openai_reasoning_effort_requests_auto_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    from orcha_agent.builtin import provider_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **options: options)
    registry = Registry()
    provider_openai.register(
        PluginAPI(
            name="provider_openai",
            config={},
            state={},
            registry=registry,
            bus=EventBus(),
            request_rebuild=lambda: None,
        )
    )

    model = registry.providers["openai"].factory(
        "gpt-5.6-sol",
        {"reasoning_effort": "high", "temperature": 0},
    )

    assert model["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in model
    assert model["temperature"] == 0
