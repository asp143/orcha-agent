from __future__ import annotations

import base64
import json
import webbrowser
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


@pytest.fixture
def codex_login_harness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    responses = {
        label: {
            "access_token": _fake_jwt(
                {
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": f"acct_fake_{label}",
                    }
                }
            ),
            "refresh_token": f"fake-{label}-refresh-token",
            "id_token": _fake_jwt({"email": f"{label}@example.test"}),
            "expires_in": 90,
        }
        for label in ("browser", "device", "paste")
    }
    login_paths: list[str] = []
    browser_options: list[dict[str, Any]] = []
    device_outputs: list[object] = []
    saved_credentials: list[tuple[str, dict[str, Any]]] = []
    behavior: dict[str, BaseException | None] = {"device_error": None}

    class FakeCredentialStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get(self, prefix: str) -> dict[str, Any] | None:
            assert prefix == "codex"
            if not saved_credentials:
                return None
            return dict(saved_credentials[-1][1])

        def set(self, prefix: str, value: Mapping[str, Any]) -> None:
            saved_credentials.append((prefix, dict(value)))

        def delete(self, prefix: str) -> None:
            assert prefix == "codex"

    class FakeOAuthPKCEFlow:
        def __init__(self, **_options: Any) -> None:
            pass

        def authorize(self, *, no_browser: bool = False) -> dict[str, Any]:
            label = "paste" if no_browser else "browser"
            login_paths.append(label)
            browser_options.append({"no_browser": no_browser})
            if not no_browser and not webbrowser.open(
                "https://auth.openai.test/oauth/authorize"
            ):
                raise RuntimeError("browser did not open")
            return dict(responses[label])

        def refresh(self, _refresh_token: str) -> dict[str, Any]:
            raise AssertionError("login must not refresh a new credential")

    class FakeOAuthDeviceFlow:
        def __init__(self, **_options: Any) -> None:
            pass

        def authorize(self, *, output: object) -> dict[str, Any]:
            login_paths.append("device")
            device_outputs.append(output)
            error = behavior["device_error"]
            if error is not None:
                raise error
            return dict(responses["device"])

    def unexpected_browser_open(_url: str) -> bool:
        raise AssertionError("login mode must not open a real browser")

    monkeypatch.setattr(provider_codex, "CredentialStore", FakeCredentialStore)
    monkeypatch.setattr(provider_codex, "OAuthPKCEFlow", FakeOAuthPKCEFlow)
    monkeypatch.setattr(
        provider_codex,
        "OAuthDeviceFlow",
        FakeOAuthDeviceFlow,
        raising=False,
    )
    monkeypatch.setattr(webbrowser, "open", unexpected_browser_open)
    registry = Registry()
    provider_codex.register(_api(registry))
    printed: list[str] = []
    ctx = SimpleNamespace(console=SimpleNamespace(print=printed.append))
    return SimpleNamespace(
        behavior=behavior,
        browser_options=browser_options,
        ctx=ctx,
        device_outputs=device_outputs,
        login_paths=login_paths,
        printed=printed,
        registry=registry,
        responses=responses,
        saved_credentials=saved_credentials,
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
    assert model.http_client.follow_redirects is False
    assert model.http_async_client.follow_redirects is False


def test_create_model_maps_optional_reasoning_and_originator_configuration() -> None:
    model = provider_codex.create_model(
        "gpt-5.6-luna",
        {"originator": "orcha_agent-42", "reasoning_effort": "high"},
        FakeTokenSource(),
        transport=httpx.MockTransport(lambda _request: _successful_sse()),
    )

    assert model.reasoning == {"effort": "high", "summary": "auto"}
    assert model.default_headers == {
        "originator": "orcha_agent-42",
        "OpenAI-Beta": "responses=experimental",
    }


def test_invalid_originator_is_rejected_for_models_and_oauth_registration() -> None:
    invalid_originator = "orcha agent\r\nx-injected: true"

    with pytest.raises(ValueError, match="originator"):
        provider_codex.create_model(
            "gpt-5.6-sol",
            {"originator": invalid_originator},
            FakeTokenSource(),
            transport=httpx.MockTransport(lambda _request: _successful_sse()),
        )

    with pytest.raises(ValueError, match="originator"):
        provider_codex.register(
            _api(Registry(), config={"originator": invalid_originator})
        )


@pytest.mark.asyncio
async def test_sync_and_async_clients_never_send_codex_auth_off_origin() -> None:
    requests: list[httpx.Request] = []
    codex_url = httpx.URL(f"{provider_codex.CODEX_BASE_URL}/responses")
    redirected_url = httpx.URL("https://evil.example/redirected")
    direct_https_url = httpx.URL("https://evil.example/direct")
    direct_http_url = httpx.URL(
        "http://chatgpt.com/backend-api/codex/responses"
    )

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == codex_url:
            return httpx.Response(302, headers={"location": str(redirected_url)})
        return httpx.Response(200)

    model = provider_codex.create_model(
        "gpt-5.6-sol",
        {},
        FakeTokenSource(),
        transport=httpx.MockTransport(capture),
    )
    assert isinstance(model.http_client, httpx.Client)
    assert isinstance(model.http_async_client, httpx.AsyncClient)
    model.http_client.follow_redirects = True
    model.http_async_client.follow_redirects = True

    model.http_client.get(codex_url)
    model.http_client.get(direct_https_url)
    model.http_client.get(direct_http_url)
    await model.http_async_client.get(codex_url)
    await model.http_async_client.get(direct_https_url)
    await model.http_async_client.get(direct_http_url)

    codex_requests = [request for request in requests if request.url == codex_url]
    assert len(codex_requests) == 2
    assert all(
        request.headers["authorization"].startswith("Bearer fake-access-")
        for request in codex_requests
    )
    assert all(
        request.headers["chatgpt-account-id"].startswith("fake-account-")
        for request in codex_requests
    )

    for untrusted_url in (redirected_url, direct_https_url, direct_http_url):
        untrusted_requests = [
            request for request in requests if request.url == untrusted_url
        ]
        assert len(untrusted_requests) == 2
        assert all(
            "authorization" not in request.headers
            and "chatgpt-account-id" not in request.headers
            for request in untrusted_requests
        )


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

    provider_codex.register(
        _api(registry, config={"originator": "orcha_agent-42"})
    )

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
            "originator": "orcha_agent-42",
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("display", "ssh", "browser_opens", "expected"),
    [
        ("DISPLAY", False, True, "browser"),
        ("WAYLAND_DISPLAY", False, True, "browser"),
        (None, False, True, "device"),
        ("DISPLAY", True, True, "device"),
        ("DISPLAY", False, False, "device"),
    ],
    ids=[
        "x11-browser",
        "wayland-browser",
        "headless-device",
        "ssh-device",
        "browser-open-failure-device",
    ],
)
async def test_auto_login_selects_browser_only_for_usable_local_display(
    monkeypatch: pytest.MonkeyPatch,
    codex_login_harness: SimpleNamespace,
    display: str | None,
    ssh: bool,
    browser_opens: bool,
    expected: str,
) -> None:
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    if display is not None:
        monkeypatch.setenv(display, "fake-display")
    if ssh:
        monkeypatch.setenv("SSH_TTY", "/dev/pts/fake")
    opened_urls: list[str] = []
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda url: opened_urls.append(url) or browser_opens,
    )

    await codex_login_harness.registry.auth["codex"].flow.login(
        codex_login_harness.ctx,
        "auto",
    )

    attempted_browser = display is not None and not ssh
    expected_paths = (
        ["browser", "device"]
        if attempted_browser and not browser_opens
        else [expected]
    )
    assert codex_login_harness.login_paths == expected_paths
    assert bool(opened_urls) is attempted_browser
    assert codex_login_harness.saved_credentials[-1][1]["account_id"] == (
        f"acct_fake_{expected}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["browser", "device", "paste"])
async def test_explicit_login_modes_route_exactly(
    monkeypatch: pytest.MonkeyPatch,
    codex_login_harness: SimpleNamespace,
    mode: str,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("SSH_TTY", "/dev/pts/fake")
    opened_urls: list[str] = []
    monkeypatch.setattr(
        webbrowser,
        "open",
        lambda url: opened_urls.append(url) or True,
    )

    await codex_login_harness.registry.auth["codex"].flow.login(
        codex_login_harness.ctx,
        mode,
    )

    assert codex_login_harness.login_paths == [mode]
    assert codex_login_harness.saved_credentials[-1][1]["account_id"] == (
        f"acct_fake_{mode}"
    )
    if mode == "browser":
        assert codex_login_harness.browser_options == [{"no_browser": False}]
        assert opened_urls
    elif mode == "paste":
        assert codex_login_harness.browser_options == [{"no_browser": True}]
        assert opened_urls == []
    else:
        assert codex_login_harness.browser_options == []
        assert opened_urls == []
        assert codex_login_harness.device_outputs == [
            codex_login_harness.ctx.console.print
        ]

@pytest.mark.asyncio
async def test_explicit_browser_does_not_fall_back_to_device(
    monkeypatch: pytest.MonkeyPatch,
    codex_login_harness: SimpleNamespace,
) -> None:
    monkeypatch.setattr(webbrowser, "open", lambda _url: False)

    with pytest.raises(RuntimeError, match="browser did not open"):
        await codex_login_harness.registry.auth["codex"].flow.login(
            codex_login_harness.ctx,
            "browser",
        )

    assert codex_login_harness.login_paths == ["browser"]
    assert codex_login_harness.device_outputs == []
    assert codex_login_harness.saved_credentials == []


@pytest.mark.asyncio
async def test_device_login_keyboard_interrupt_does_not_persist_credential(
    codex_login_harness: SimpleNamespace,
) -> None:
    codex_login_harness.behavior["device_error"] = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await codex_login_harness.registry.auth["codex"].flow.login(
            codex_login_harness.ctx,
            "device",
        )

    assert codex_login_harness.login_paths == ["device"]
    assert codex_login_harness.saved_credentials == []
    assert codex_login_harness.printed == []


@pytest.mark.asyncio
async def test_login_persists_only_refreshable_credential_fields(
    monkeypatch: pytest.MonkeyPatch,
    codex_login_harness: SimpleNamespace,
) -> None:
    monkeypatch.setattr(provider_codex.time, "time", lambda: 1_700_000_000.0)

    await codex_login_harness.registry.auth["codex"].flow.login(
        codex_login_harness.ctx,
        "paste",
    )

    assert len(codex_login_harness.saved_credentials) == 1
    prefix, credential = codex_login_harness.saved_credentials[0]
    response = codex_login_harness.responses["paste"]
    assert prefix == "codex"
    assert set(credential) == {
        "type",
        "access",
        "refresh",
        "expires",
        "account_id",
        "email",
    }
    assert credential == {
        "type": "oauth",
        "access": response["access_token"],
        "refresh": "fake-paste-refresh-token",
        "expires": 1_700_000_090_000,
        "account_id": "acct_fake_paste",
        "email": "paste@example.test",
    }
    assert "id_token" not in credential
