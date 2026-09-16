from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orcha_agent.core.plugin import ModeSpec
from orcha_agent.extensibility.magic_keywords import MagicKeywordsMiddleware, keywords


@pytest.mark.parametrize(
    "text",
    [
        "ultrathink",
        "Please ultrathink!",
        "'ultrathink'.",
        "ultrathink, orchestrate",
        "計画 ultrathink",
        "`unclosed ultrathink",
    ],
)
def test_prose_keyword(text):
    assert "ultrathink" in keywords(text)


@pytest.mark.parametrize(
    "text",
    [
        "Ultrathink",
        "ultrathinking",
        "ultrathink.ts",
        "dir/ultrathink",
        "foo::ultrathink",
        "ultrathink()",
        "ultrathink-test",
        "éultrathink",
        "`ultrathink`",
        "``a ` ultrathink``",
        "```python\nultrathink\n```",
        "~~~\nultrathink\n~~~",
        "<note>ultrathink</note>",
        "<note><note>x</note>ultrathink</note>",
        '<note value="ultrathink"/>',
        "<!-- ultrathink -->",
    ],
)
def test_embedded_keyword_is_literal(text):
    assert "ultrathink" not in keywords(text)


def middleware():
    return MagicKeywordsMiddleware(
        SimpleNamespace(
            cfg=SimpleNamespace(model="anthropic:test", models={}),
            registry=SimpleNamespace(
                providers={
                    "anthropic": SimpleNamespace(capabilities=SimpleNamespace(thinking=True))
                },
                modes={"plan": ModeSpec("Read only", {}, {"read_file", "glob"})},
            ),
        )
    )


def request(text):
    return ModelRequest(
        model=FakeListChatModel(
            responses=["ok"],
            profile={"reasoning_output": True, "reasoning_effort_levels": ["high", "xhigh", "max"]},
        ),
        messages=[HumanMessage(text, additional_kwargs={"orcha_user_origin": True})],
        tools=[{"name": "read"}, {"name": "write"}, {"name": "task"}],
        system_message=SystemMessage("Base"),
        model_settings={"temperature": 0},
    )


@pytest.mark.asyncio
async def test_turn_notices_and_settings_do_not_mutate_original_or_next_turn():
    value = request("ultrathink and orchestrate")
    handler = AsyncMock()
    await middleware().awrap_model_call(value, handler)
    revised = handler.call_args.args[0]
    assert revised.model_settings == {
        "temperature": 0,
        "reasoning_effort": "max",
        "thinking": {"type": "adaptive"},
    }
    assert "task tool" in revised.system_message.text
    assert revised.messages == value.messages
    assert value.system_message.text == "Base"
    assert value.model_settings == {"temperature": 0}
    plain = value.override(messages=[*value.messages, AIMessage("Done"), HumanMessage("Hello")])
    await middleware().awrap_model_call(plain, handler)
    assert handler.call_args.args[0] is plain


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["plan", "implement the plan", "plan: the change"])
async def test_plan_is_ordinary_text_even_with_registered_plan_mode(text):
    value = request(text)
    handler = AsyncMock()
    await middleware().awrap_model_call(value, handler)
    assert handler.call_args.args[0] is value
    assert "plan" not in keywords(text)


@pytest.mark.asyncio
async def test_plan_without_registered_mode_is_ordinary_text():
    plugin = MagicKeywordsMiddleware(SimpleNamespace(registry=SimpleNamespace(modes={})))
    value = request("plan")
    handler = AsyncMock()
    await plugin.awrap_model_call(value, handler)
    assert handler.call_args.args[0] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,thinking", [("unknown", True), ("anthropic", False)])
async def test_unsupported_provider_receives_no_reasoning_kwargs(prefix, thinking):
    plugin = middleware()
    plugin.ctx.cfg.model = prefix + ":test"
    plugin.ctx.registry.providers = {
        prefix: SimpleNamespace(capabilities=SimpleNamespace(thinking=thinking))
    }
    handler = AsyncMock()
    await plugin.awrap_model_call(request("ultrathink"), handler)
    assert handler.call_args.args[0].model_settings == {"temperature": 0}


@pytest.mark.asyncio
async def test_ultrathink_enables_anthropic_thinking_when_display_off():
    value = request("ultrathink")
    value.model_settings["thinking"] = {"type": "disabled"}
    handler = AsyncMock()
    await middleware().awrap_model_call(value, handler)
    assert handler.call_args.args[0].model_settings["thinking"] == {"type": "adaptive"}
    assert value.model_settings["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,expected",
    [
        ("gemini-3.1-pro", {"thinking_level": "high", "thinking_budget": None}),
        (
            "gemini-2.5-pro",
            {"thinking_level": None, "reasoning_effort": None, "thinking_budget": 32768},
        ),
        (
            "gemini-2.5-flash",
            {"thinking_level": None, "reasoning_effort": None, "thinking_budget": 24576},
        ),
        ("gemini-2.0-flash", {}),
    ],
)
async def test_google_highest_supported_thinking(name, expected):
    plugin = middleware()
    plugin.ctx.cfg.model = "google:" + name
    plugin.ctx.registry.providers["google"] = SimpleNamespace(
        capabilities=SimpleNamespace(thinking=True)
    )
    handler = AsyncMock()
    await plugin.awrap_model_call(request("ultrathink"), handler)
    assert handler.call_args.args[0].model_settings == {"temperature": 0, **expected}


@pytest.mark.asyncio
async def test_temporary_model_adapter_overrides_main_provider():
    class TemporaryGoogleModel(FakeListChatModel):
        model: str = "gemini-3.1-pro"

    TemporaryGoogleModel.__module__ = "langchain_google_genai.chat_models"
    plugin = middleware()
    plugin.ctx.registry.providers["google"] = SimpleNamespace(
        capabilities=SimpleNamespace(thinking=True)
    )
    value = request("ultrathink").override(model=TemporaryGoogleModel(responses=["ok"]))
    handler = AsyncMock()
    await plugin.awrap_model_call(value, handler)
    settings = handler.call_args.args[0].model_settings
    assert settings["thinking_level"] == "high"
    assert "thinking" not in settings
    assert "reasoning_effort" not in settings


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,expected", [("gpt-4o", None), ("gpt-5.2", "xhigh")])
async def test_real_openai_payload_respects_model_profile(model_name, expected):
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=model_name, api_key="test")
    plugin = middleware()
    plugin.ctx.registry.providers["openai"] = SimpleNamespace(
        capabilities=SimpleNamespace(thinking=True)
    )
    value = request("ultrathink").override(model=model, model_settings={})
    handler = AsyncMock()
    await plugin.awrap_model_call(value, handler)
    modified = handler.call_args.args[0]
    payload = model._get_request_payload(modified.messages, **modified.model_settings)
    if expected is None:
        assert "reasoning" not in payload
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning"]["effort"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,expected", [("claude-opus-4-5", "high"), ("claude-sonnet-4-5", None)]
)
async def test_real_older_anthropic_payload_uses_enabled_thinking(model_name, expected):
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model=model_name, api_key="test", max_tokens=4096)
    plugin = middleware()
    value = request("ultrathink").override(model=model, model_settings={})
    handler = AsyncMock()
    await plugin.awrap_model_call(value, handler)
    modified = handler.call_args.args[0]
    payload = model._get_request_payload(modified.messages, **modified.model_settings)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert payload["max_tokens"] == 24192
    assert payload["max_tokens"] - payload["thinking"]["budget_tokens"] == 8192
    assert payload["temperature"] == 1
    assert payload.get("output_config", {}).get("effort") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["ultrathink orchestrate", "Summary: ultrathink", "Task: orchestrate"]
)
async def test_model_authored_human_messages_do_not_activate_keywords(text):
    value = request("ultrathink").override(
        messages=[
            HumanMessage("ultrathink", additional_kwargs={"orcha_user_origin": True}),
            AIMessage("Delegating now"),
            HumanMessage(text),
        ]
    )
    handler = AsyncMock()
    await middleware().awrap_model_call(value, handler)
    assert handler.call_args.args[0] is value


@pytest.mark.asyncio
@pytest.mark.parametrize("cap,expected_budget", [(8192, 4096), (16384, 8192), (20000, 11808)])
async def test_older_anthropic_respects_output_cap_with_visible_headroom(cap, expected_budget):
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key="test",
        max_tokens=4096,
        profile={"reasoning_output": True, "max_output_tokens": cap},
    )
    value = request("ultrathink").override(model=model, model_settings={})
    handler = AsyncMock()
    await middleware().awrap_model_call(value, handler)
    modified = handler.call_args.args[0]
    payload = model._get_request_payload(modified.messages, **modified.model_settings)
    assert payload["max_tokens"] == cap
    assert payload["thinking"]["budget_tokens"] == expected_budget
    assert payload["max_tokens"] - expected_budget >= min(8192, cap // 2)


def test_user_input_marks_only_user_originated_messages():
    from langchain_core.messages import convert_to_messages

    from orcha_agent.extensibility.magic_keywords import turn_keywords
    from orcha_agent.tui.turn import _user_input

    user = convert_to_messages(_user_input("ultrathink")["messages"])
    synthetic = convert_to_messages(_user_input("ultrathink", user_origin=False)["messages"])
    assert turn_keywords(user) == {"ultrathink"}
    assert turn_keywords(synthetic) == frozenset()


@pytest.mark.asyncio
async def test_plugin_expanded_prompt_is_not_user_authored(monkeypatch):
    from orcha_agent.tui.context import AppContext

    turn = AsyncMock()
    monkeypatch.setattr("orcha_agent.tui.turn._run_cancellable_turn", turn)
    ctx = SimpleNamespace(cfg=SimpleNamespace(model="anthropic:test"))
    await AppContext.submit_prompt(ctx, "Instructions from a file: ultrathink orchestrate")
    turn.assert_awaited_once_with(
        ctx, "Instructions from a file: ultrathink orchestrate", user_origin=False
    )
