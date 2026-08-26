from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from io import StringIO
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
from rich.console import Console

from orcha_agent.builtin import commands_core, provider_codex
from orcha_agent.core.auth import AuthFlow
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.app import dispatch_command
from orcha_agent.tui.console import ConsoleOutput


EXPECTED_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
)


def _fake_jwt(payload: Mapping[str, Any]) -> str:
    def encode(value: Mapping[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f'{encode({"alg": "none", "typ": "JWT"})}.{encode(payload)}.fake-signature'


class FakeTokenSource:
    def __init__(self) -> None:
        self.calls = 0

    def get_token(self) -> tuple[str, str]:
        self.calls += 1
        return f"fake-access-{self.calls}", f"fake-account-{self.calls}"


class _RecordingByteStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.iterations: list[str] = []

    def __iter__(self) -> Iterator[bytes]:
        self.iterations.append("sync")
        yield self.body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterations.append("async")
        yield self.body


def _successful_sse() -> httpx.Response:
    event = {
        "type": "response.output_text.delta",
        "sequence_number": 0,
        "item_id": "fake-message",
        "output_index": 0,
        "content_index": 0,
        "delta": "ok",
        "logprobs": [],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body.encode(),
    )


def _api(registry: Registry, *, config: Mapping[str, Any] | None = None) -> PluginAPI:
    return PluginAPI(
        name="provider_codex",
        config=config or {},
        state={},
        registry=registry,
        bus=EventBus(),
        request_rebuild=lambda: None,
    )


def test_codex_models_are_the_exact_supported_static_list() -> None:
    assert provider_codex.CODEX_MODELS == EXPECTED_MODELS


def test_extracts_account_and_email_from_fake_jwts() -> None:
    access_token = _fake_jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_fake_nested",
            }
        }
    )
    id_token = _fake_jwt({"email": "codex-user@example.test"})

    assert provider_codex.extract_account_id(access_token) == "acct_fake_nested"
    assert provider_codex.extract_email(id_token) == "codex-user@example.test"


def test_create_model_uses_exact_codex_responses_configuration() -> None:
    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(lambda _request: _successful_sse()),
    )

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5.6-sol"
    assert model.use_responses_api is True
    assert model.openai_api_base == "https://chatgpt.com/backend-api/codex"
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "<placeholder>"
    assert model.store is False
    assert model.streaming is True
    assert model.include == ["reasoning.encrypted_content"]
    assert model.default_headers == {
        "originator": "pi",
        "OpenAI-Beta": "responses=experimental",
    }
    assert model.reasoning is None
    assert model.max_tokens is None
    assert isinstance(model.http_client, httpx.Client)
    assert isinstance(model.http_async_client, httpx.AsyncClient)


def test_create_model_maps_optional_reasoning_and_originator_configuration() -> None:
    model = provider_codex.create_model(
        "gpt-5.6-luna",
        {"originator": "orcha-test", "reasoning_effort": "high"},
        FakeTokenSource(),
        transport=httpx.MockTransport(lambda _request: _successful_sse()),
    )

    assert model.reasoning == {"effort": "high", "summary": "auto"}
    assert model.default_headers == {
        "originator": "orcha-test",
        "OpenAI-Beta": "responses=experimental",
    }


def test_each_request_uses_fresh_auth_and_never_sends_max_output_tokens() -> None:
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _successful_sse()

    tokens = FakeTokenSource()
    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        tokens,
        transport=httpx.MockTransport(capture),
    )

    list(model.stream("first fake prompt"))
    list(model.stream("second fake prompt"))

    assert [request.url for request in requests] == [
        httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
        httpx.URL("https://chatgpt.com/backend-api/codex/responses"),
    ]
    assert [request.headers["authorization"] for request in requests] == [
        "Bearer fake-access-1",
        "Bearer fake-access-2",
    ]
    assert [request.headers["chatgpt-account-id"] for request in requests] == [
        "fake-account-1",
        "fake-account-2",
    ]
    assert all(request.headers["originator"] == "pi" for request in requests)
    assert all(
        request.headers["openai-beta"] == "responses=experimental"
        for request in requests
    )
    payloads = [json.loads(request.content) for request in requests]
    assert all("max_output_tokens" not in payload for payload in payloads)
    assert all(payload["stream"] is True for payload in payloads)
    assert all(payload["store"] is False for payload in payloads)
    assert all(
        payload["include"] == ["reasoning.encrypted_content"]
        for payload in payloads
    )
    assert tokens.calls == 2


@pytest.mark.asyncio
async def test_async_request_uses_the_same_fresh_auth_hook() -> None:
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _successful_sse()

    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(capture),
    )

    chunks = [chunk async for chunk in model.astream("fake async prompt")]

    assert chunks
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer fake-access-1"
    assert requests[0].headers["chatgpt-account-id"] == "fake-account-1"


def test_successful_sse_sync_hook_does_not_read_before_consumption() -> None:
    body = b"data: [DONE]\n\n"
    stream = _RecordingByteStream(body)
    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
    )

    with model.http_client.stream(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
    ) as response:
        assert response.status_code == 200
        assert stream.iterations == []
        assert response.read() == body

    assert stream.iterations == ["sync"]


@pytest.mark.asyncio
async def test_successful_sse_async_hook_does_not_aread_before_consumption() -> None:
    body = b"data: [DONE]\n\n"
    stream = _RecordingByteStream(body)
    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
    )

    async with model.http_async_client.stream(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
    ) as response:
        assert response.status_code == 200
        assert stream.iterations == []
        assert await response.aread() == body

    assert stream.iterations == ["async"]


def test_unauthorized_response_directs_user_to_codex_login() -> None:
    def unauthorized(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "fake access token expired"}},
        )

    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(unauthorized),
    )

    with pytest.raises(Exception) as raised:
        list(model.stream("fake prompt"))

    assert "run /login codex" in str(raised.value).lower()


def test_usage_limit_response_includes_friendly_plan_and_reset_details() -> None:
    def usage_limited(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Usage limit reached",
                    "plan_type": "plus",
                    "resets_at": "2026-08-28T12:00:00Z",
                }
            },
        )

    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(usage_limited),
    )

    with pytest.raises(Exception) as raised:
        list(model.stream("fake prompt"))

    message = str(raised.value).lower()
    assert "usage limit" in message
    assert "plus" in message
    assert "2026-08-28t12:00:00z" in message


@pytest.mark.asyncio
async def test_register_exposes_models_and_safe_auth_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = _fake_jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_fake_registered",
            }
        }
    )
    id_token = _fake_jwt({"email": "registered@example.test"})
    credential = {
        "type": "oauth",
        "access": access_token,
        "refresh": "fake-refresh-token",
        "expires": 4_102_444_800_000,
        "account_id": "acct_fake_registered",
        "email": "registered@example.test",
        "id_token": id_token,
    }

    class FakeCredentialStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.credential = dict(credential)

        def get(self, prefix: str) -> dict[str, Any] | None:
            assert prefix == "codex"
            return dict(self.credential)

        def set(self, prefix: str, value: Mapping[str, Any]) -> None:
            assert prefix == "codex"
            self.credential = dict(value)

        def delete(self, prefix: str) -> None:
            assert prefix == "codex"
            self.credential = {}

    oauth_options: dict[str, Any] = {}

    class FakeOAuthPKCEFlow:
        def __init__(self, **options: Any) -> None:
            oauth_options.update(options)

        def authorize(self, **_options: Any) -> dict[str, Any]:
            raise AssertionError("registering/status must not start OAuth")

        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            raise AssertionError("fresh fake credential must not refresh")

    monkeypatch.setattr(provider_codex, "CredentialStore", FakeCredentialStore)
    monkeypatch.setattr(provider_codex, "OAuthPKCEFlow", FakeOAuthPKCEFlow)
    registry = Registry()

    provider_codex.register(_api(registry))

    assert oauth_options == {
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "authorize_url": "https://auth.openai.com/oauth/authorize",
        "token_url": "https://auth.openai.com/oauth/token",
        "scopes": "openid profile email offline_access",
        "redirect_port": 1455,
        "redirect_path": "/auth/callback",
        "extra_authorize_params": {
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "pi",
        },
    }
    provider = registry.providers["codex"]
    assert provider.env_keys == ()
    assert provider.models == EXPECTED_MODELS
    assert provider.foreign_block_types == frozenset({"reasoning"})
    assert provider.capabilities.tool_calling is True
    assert provider.capabilities.streaming is True
    assert provider.capabilities.thinking is True

    auth = registry.auth["codex"].flow
    assert isinstance(auth, AuthFlow)
    status = auth.status()
    assert "registered@example.test" in status
    assert "acct_fake_registered" in status
    assert access_token not in status
    assert id_token not in status
    assert "fake-refresh-token" not in status

    commands_core.register(_api(registry))
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=240)
    ctx = SimpleNamespace(
        console=ConsoleOutput(console),
        registry=registry,
        agent=None,
    )

    assert await dispatch_command(registry, ctx, "/providers") is True
    rendered = output.getvalue()
    assert all(model_name in rendered for model_name in EXPECTED_MODELS)
    assert "registered@example.test" in rendered
    assert "acct_fake_registered" in rendered
    assert access_token not in rendered
    assert id_token not in rendered
    assert "fake-refresh-token" not in rendered
