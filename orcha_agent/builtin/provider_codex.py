"""ChatGPT-subscription Codex provider and OAuth authentication."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from orcha_agent.core.auth import AuthFlow, CredentialStore, OAuthPKCEFlow, TokenSource
from orcha_agent.core.plugin import PluginAPI, PluginSpec, ProviderCaps

PLUGIN = PluginSpec(name="provider_codex", version="1.0.0")

CODEX_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
SCOPES = "openid profile email offline_access"
REDIRECT_PORT = 1455
REDIRECT_PATH = "/auth/callback"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_ORIGINATOR = "pi"


def _jwt_payload(token: str) -> dict[str, Any]:
    try:
        encoded = token.split(".", 2)[1]
        padding = "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def extract_account_id(access_token: str) -> str:
    auth = _jwt_payload(access_token).get("https://api.openai.com/auth", {})
    if not isinstance(auth, Mapping):
        return ""
    value = auth.get("chatgpt_account_id", "")
    return value if isinstance(value, str) else ""


def extract_email(id_token: str) -> str:
    value = _jwt_payload(id_token).get("email", "")
    return value if isinstance(value, str) else ""


def _friendly_error(response: httpx.Response) -> RuntimeError | None:
    if response.status_code == 401:
        return RuntimeError("Codex authentication failed; run /login codex")
    if response.status_code != 429:
        return None
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(error, Mapping):
        error = {}
    message = str(error.get("message") or "Codex usage limit reached")
    plan = error.get("plan_type")
    reset = error.get("resets_at") or error.get("reset_at")
    details = [message]
    if plan:
        details.append(f"plan: {plan}")
    if reset:
        details.append(f"resets at: {reset}")
    return RuntimeError("; ".join(details))


def _friendly_cause(exc: BaseException) -> RuntimeError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if isinstance(current, RuntimeError) and (
            "/login codex" in message or "usage limit" in message.lower()
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


class CodexChatOpenAI(ChatOpenAI):
    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        try:
            yield from super()._stream(*args, **kwargs)
        except Exception as exc:
            if friendly := _friendly_cause(exc):
                raise RuntimeError(str(friendly)) from exc
            raise

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        try:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk
        except Exception as exc:
            if friendly := _friendly_cause(exc):
                raise RuntimeError(str(friendly)) from exc
            raise


def create_model(
    name: str,
    config: Mapping[str, Any],
    token_source: TokenSource,
    *,
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
) -> ChatOpenAI:
    """Create a Codex Responses API model with per-request fresh OAuth headers."""

    originator = str(config.get("originator") or DEFAULT_ORIGINATOR)

    def authorize(request: httpx.Request) -> None:
        access, account_id = token_source.get_token()
        request.headers["Authorization"] = f"Bearer {access}"
        request.headers["chatgpt-account-id"] = account_id

    async def async_authorize(request: httpx.Request) -> None:
        access, account_id = await asyncio.to_thread(token_source.get_token)
        request.headers["Authorization"] = f"Bearer {access}"
        request.headers["chatgpt-account-id"] = account_id

    def check_response(response: httpx.Response) -> None:
        if response.status_code not in {401, 429}:
            return
        if response.status_code == 429:
            response.read()
        if error := _friendly_error(response):
            raise error

    async def async_check_response(response: httpx.Response) -> None:
        if response.status_code not in {401, 429}:
            return
        if response.status_code == 429:
            await response.aread()
        if error := _friendly_error(response):
            raise error

    sync_transport = transport if isinstance(transport, httpx.BaseTransport) else None
    async_transport = transport if isinstance(transport, httpx.AsyncBaseTransport) else None
    if isinstance(transport, httpx.MockTransport):
        sync_transport = transport
        async_transport = transport
    http_client = httpx.Client(
        transport=sync_transport,
        event_hooks={"request": [authorize], "response": [check_response]},
    )
    http_async_client = httpx.AsyncClient(
        transport=async_transport,
        event_hooks={
            "request": [async_authorize],
            "response": [async_check_response],
        },
    )
    options: dict[str, Any] = {
        "model": name,
        "use_responses_api": True,
        "base_url": CODEX_BASE_URL,
        "openai_api_key": "<placeholder>",
        "store": False,
        "streaming": True,
        "include": ["reasoning.encrypted_content"],
        "default_headers": {
            "originator": originator,
            "OpenAI-Beta": "responses=experimental",
        },
        "http_client": http_client,
        "http_async_client": http_async_client,
    }
    reasoning_effort = config.get("reasoning_effort")
    if reasoning_effort:
        options["reasoning"] = {
            "effort": str(reasoning_effort),
            "summary": "auto",
        }
    return CodexChatOpenAI(**options)


def register(api: PluginAPI) -> None:
    originator = str(api.config.get("originator") or DEFAULT_ORIGINATOR)
    store_path = api.config.get("auth_path")
    store = CredentialStore(store_path) if store_path else CredentialStore()
    oauth = OAuthPKCEFlow(
        client_id=CLIENT_ID,
        authorize_url=AUTHORIZE_URL,
        token_url=TOKEN_URL,
        scopes=SCOPES,
        redirect_port=REDIRECT_PORT,
        redirect_path=REDIRECT_PATH,
        extra_authorize_params={
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": originator,
        },
    )
    token_source = TokenSource(store, "codex", oauth)
    async def login(ctx: Any) -> None:
        response = await asyncio.to_thread(
            oauth.authorize,
            no_browser=bool(getattr(ctx, "no_browser", False)),
        )
        access = response.get("access_token")
        if not isinstance(access, str) or not access:
            raise RuntimeError("Codex OAuth response omitted access_token")
        refresh = response.get("refresh_token", "")
        id_token = response.get("id_token", "")
        credential = {
            "type": "oauth",
            "access": access,
            "refresh": refresh if isinstance(refresh, str) else "",
            "expires": int(time.time() * 1000) + int(response.get("expires_in", 3600)) * 1000,
            "account_id": extract_account_id(access),
            "email": extract_email(id_token) if isinstance(id_token, str) else "",
            "id_token": id_token if isinstance(id_token, str) else "",
        }
        store.set("codex", credential)
        ctx.console.print(f"Logged in to codex as {_status(store)}")

    async def logout(ctx: Any) -> None:
        store.delete("codex")
        ctx.console.print("Logged out of codex.")

    def status() -> str:
        return _status(store)

    api.add_auth("codex", AuthFlow(login=login, logout=logout, status=status))
    api.add_provider(
        "codex",
        lambda model_name, provider_config: create_model(
            model_name,
            provider_config,
            token_source,
        ),
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=True,
            thinking=True,
            structured_output=False,
            max_context=None,
        ),
        models=CODEX_MODELS,
        env_keys=(),
        foreign_block_types=("reasoning",),
    )


def _status(store: CredentialStore) -> str:
    credential = store.get("codex")
    if not credential:
        return "not logged in"
    email = credential.get("email")
    account = credential.get("account_id")
    identity = " / ".join(str(value) for value in (email, account) if value)
    return f"logged in as {identity}" if identity else "logged in"
