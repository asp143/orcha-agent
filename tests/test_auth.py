import json
import os
import secrets
import socket
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import httpx
import pytest

from orcha_agent.core.auth import CredentialStore, OAuthPKCEFlow, TokenSource


FAKE_NOW_MS = 1_800_000_000_000


def _flow(
    *,
    token_url: str = "https://tokens.invalid/token",
    redirect_port: int = 1455,
    http_client: httpx.Client | None = None,
    input_fn: Callable[[str], str] = input,
) -> OAuthPKCEFlow:
    return OAuthPKCEFlow(
        client_id="fake-client-id",
        authorize_url="https://accounts.invalid/authorize",
        token_url=token_url,
        scopes="openid profile offline_access",
        redirect_port=redirect_port,
        redirect_path="/auth/callback",
        extra_authorize_params={"audience": "fake-audience"},
        http_client=http_client,
        input_fn=input_fn,
    )


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _TokenServer(ThreadingHTTPServer):
    requests: list[dict[str, list[str]]]


class _TokenHandler(BaseHTTPRequestHandler):
    server: _TokenServer

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        form = parse_qs(body.decode("ascii"), keep_blank_values=True)
        self.server.requests.append(form)
        if form.get("grant_type") == ["authorization_code"]:
            payload = {
                "access_token": "fake-exchanged-access",
                "refresh_token": "fake-exchanged-refresh",
                "expires_in": 3600,
            }
        else:
            payload = {
                "access_token": "fake-refreshed-access",
                "expires_in": 1800,
            }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _token_server() -> Iterator[tuple[str, list[dict[str, list[str]]]]]:
    server = _TokenServer(("127.0.0.1", 0), _TokenHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/token", server.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_credential_store_writes_atomically_with_private_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "private-config" / "auth.json"
    store = CredentialStore(auth_path)
    credential = {
        "type": "oauth",
        "access": "fake-access",
        "refresh": "fake-refresh",
        "expires": FAKE_NOW_MS + 3_600_000,
        "account_id": "fake-account",
    }
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
    ) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", record_replace)

    store.set("codex", credential)

    assert json.loads(auth_path.read_text()) == {"codex": credential}
    assert stat.S_IMODE(auth_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    assert replacements
    temporary_path, destination = replacements[-1]
    assert temporary_path.parent == auth_path.parent
    assert destination == auth_path
    assert not temporary_path.exists()
    assert CredentialStore(auth_path).get("codex") == credential


def test_credential_store_write_failure_does_not_reclose_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "config" / "auth.json"
    store = CredentialStore(auth_path)
    real_fdopen = os.fdopen
    real_close = os.close
    handed_over_descriptors: list[int] = []
    explicitly_closed_descriptors: list[int] = []

    def record_fdopen(
        descriptor: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        handed_over_descriptors.append(descriptor)
        return real_fdopen(descriptor, *args, **kwargs)

    def record_close(descriptor: int) -> None:
        explicitly_closed_descriptors.append(descriptor)
        real_close(descriptor)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("fake credential write failure")

    monkeypatch.setattr(os, "fdopen", record_fdopen)
    monkeypatch.setattr(os, "close", record_close)
    monkeypatch.setattr("orcha_agent.core.auth.json.dump", fail_write)

    with pytest.raises(OSError, match="fake credential write failure"):
        store.set("codex", {"type": "oauth", "access": "fake-access"})

    assert len(handed_over_descriptors) == 1
    assert handed_over_descriptors[0] not in explicitly_closed_descriptors
    assert not auth_path.exists()
    assert list(auth_path.parent.glob(f".{auth_path.name}.*")) == []


def test_credential_store_delete_removes_only_requested_prefix(tmp_path: Path) -> None:
    auth_path = tmp_path / "config" / "auth.json"
    store = CredentialStore(auth_path)
    codex = {"type": "oauth", "access": "fake-access"}
    other = {"type": "api_key", "key": "fake-api-key"}
    store.set("codex", codex)
    store.set("other", other)

    store.delete("codex")

    assert store.get("codex") is None
    assert store.get("other") == other
    assert json.loads(auth_path.read_text()) == {"other": other}


def test_code_challenge_matches_rfc_7636_s256_example() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    assert _flow().code_challenge(verifier) == (
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


@pytest.mark.parametrize(
    ("pasted", "expected_state", "expected_code"),
    [
        (
            "http://127.0.0.1:1455/auth/callback?code=fake-url-code&state=fake-state",
            "fake-state",
            "fake-url-code",
        ),
        ("fake-fragment-code#fake-state", "fake-state", "fake-fragment-code"),
        ("fake-raw-code", "fake-state", "fake-raw-code"),
    ],
)
def test_parse_paste_accepts_redirect_fragment_and_raw_code(
    pasted: str,
    expected_state: str | None,
    expected_code: str,
) -> None:
    assert _flow().parse_paste(pasted, expected_state=expected_state) == expected_code


def test_parse_paste_uses_constant_time_state_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparisons: list[tuple[str, str]] = []

    def accept_state(callback_state: str, expected_state: str) -> bool:
        comparisons.append((callback_state, expected_state))
        return True

    monkeypatch.setattr(
        "orcha_agent.core.auth.secrets.compare_digest",
        accept_state,
    )

    code = _flow().parse_paste(
        "http://127.0.0.1:1455/auth/callback"
        "?code=fake-code&state=fake-callback-state",
        expected_state="fake-expected-state",
    )

    assert code == "fake-code"
    assert len(comparisons) == 1
    assert set(comparisons[0]) == {
        "fake-callback-state",
        "fake-expected-state",
    }


@pytest.mark.parametrize(
    "pasted",
    [
        "http://127.0.0.1:1455/auth/callback?code=fake-code&state=wrong-state",
        "fake-code#wrong-state",
    ],
)
def test_parse_paste_rejects_state_mismatch(pasted: str) -> None:
    with pytest.raises(ValueError, match="(?i)state"):
        _flow().parse_paste(pasted, expected_state="expected-state")


def test_no_browser_authorize_prompts_with_url_without_opening_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    prompts: list[str] = []
    opened_urls: list[str] = []

    def exchange(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "fake-access",
                "refresh_token": "fake-refresh",
                "expires_in": 3600,
            },
        )

    def paste_code(prompt: str) -> str:
        prompts.append(prompt)
        return "fake-pasted-code"

    monkeypatch.setattr(
        "orcha_agent.core.auth.webbrowser.open",
        lambda url: opened_urls.append(url) or True,
    )
    with httpx.Client(transport=httpx.MockTransport(exchange)) as client:
        response = _flow(http_client=client, input_fn=paste_code).authorize(
            no_browser=True
        )

    assert opened_urls == []
    assert len(prompts) == 1
    assert "https://accounts.invalid/authorize?" in prompts[0]
    assert response["access_token"] == "fake-access"
    assert len(requests) == 1
    assert parse_qs(requests[0].content.decode())["code"] == ["fake-pasted-code"]


def test_loopback_authorize_survives_invalid_requests_then_exchanges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_requests: list[httpx.Request] = []
    state_comparisons: list[tuple[str, str]] = []
    opened_urls: Queue[str] = Queue()
    outcome: Queue[dict[str, Any] | BaseException] = Queue()
    real_compare_digest = secrets.compare_digest

    def exchange(request: httpx.Request) -> httpx.Response:
        token_requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "fake-loopback-access",
                "refresh_token": "fake-loopback-refresh",
                "expires_in": 3600,
            },
        )

    def record_compare_digest(left: str, right: str) -> bool:
        state_comparisons.append((left, right))
        return real_compare_digest(left, right)

    def request(url: str) -> tuple[int, bytes]:
        try:
            with urlopen(url, timeout=2) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    monkeypatch.setattr(
        "orcha_agent.core.auth.webbrowser.open",
        lambda url: opened_urls.put(url) or True,
    )
    monkeypatch.setattr(
        "orcha_agent.core.auth.secrets.compare_digest",
        record_compare_digest,
    )

    with httpx.Client(transport=httpx.MockTransport(exchange)) as client:
        flow = _flow(
            redirect_port=_unused_loopback_port(),
            http_client=client,
        )

        def authorize() -> None:
            try:
                outcome.put(flow.authorize())
            except BaseException as exc:
                outcome.put(exc)

        authorize_thread = threading.Thread(target=authorize, daemon=True)
        authorize_thread.start()
        authorization_url = opened_urls.get(timeout=2)
        authorization_query = parse_qs(urlparse(authorization_url).query)
        redirect_uri = authorization_query["redirect_uri"][0]
        expected_state = authorization_query["state"][0]
        redirect = urlparse(redirect_uri)
        origin = f"{redirect.scheme}://{redirect.netloc}"

        favicon_status, _ = request(f"{origin}/favicon.ico")
        bad_state_status, _ = request(
            f"{redirect_uri}?"
            f"{urlencode({'code': 'fake-wrong-code', 'state': 'wrong-state'})}"
        )
        callback_status, callback_body = request(
            f"{redirect_uri}?"
            f"{urlencode({'code': 'fake-correct-code', 'state': expected_state})}"
        )
        result = outcome.get(timeout=2)
        authorize_thread.join(timeout=2)

    assert favicon_status == 404
    assert bad_state_status == 400
    assert callback_status == 200
    assert b"Authentication complete" in callback_body
    assert isinstance(result, dict)
    assert result["access_token"] == "fake-loopback-access"
    assert not authorize_thread.is_alive()
    assert len(token_requests) == 1
    assert parse_qs(token_requests[0].content.decode())["code"] == [
        "fake-correct-code"
    ]
    assert any(
        (left, right) in {
            ("wrong-state", expected_state),
            (expected_state, "wrong-state"),
        }
        for left, right in state_comparisons
    )
    assert (expected_state, expected_state) in state_comparisons


def test_exchange_and_refresh_post_oauth_forms_to_token_endpoint() -> None:
    with _token_server() as (token_url, requests):
        flow = _flow(token_url=token_url)

        exchanged = flow.exchange(
            "fake-code",
            "fake-verifier",
            "http://127.0.0.1:1455/auth/callback",
        )
        refreshed = flow.refresh("fake-old-refresh")

    assert exchanged == {
        "access_token": "fake-exchanged-access",
        "refresh_token": "fake-exchanged-refresh",
        "expires_in": 3600,
    }
    assert refreshed == {
        "access_token": "fake-refreshed-access",
        "expires_in": 1800,
    }
    assert requests == [
        {
            "grant_type": ["authorization_code"],
            "client_id": ["fake-client-id"],
            "code": ["fake-code"],
            "code_verifier": ["fake-verifier"],
            "redirect_uri": ["http://127.0.0.1:1455/auth/callback"],
        },
        {
            "grant_type": ["refresh_token"],
            "client_id": ["fake-client-id"],
            "refresh_token": ["fake-old-refresh"],
        },
    ]


class _RefreshingFlow:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        with self._lock:
            self.calls.append(refresh_token)
        self.entered.set()
        assert self.release.wait(timeout=2)
        return dict(self.response)


def _expired_credential() -> dict[str, Any]:
    return {
        "type": "oauth",
        "access": "fake-old-access",
        "refresh": "fake-old-refresh",
        "expires": FAKE_NOW_MS + 299_999,
        "account_id": "fake-account",
    }


def _start_token_read(
    source: TokenSource,
) -> tuple[threading.Thread, Queue[tuple[str, str] | BaseException]]:
    outcome: Queue[tuple[str, str] | BaseException] = Queue()

    def read_token() -> None:
        try:
            outcome.put(source.get_token())
        except BaseException as exc:
            outcome.put(exc)

    thread = threading.Thread(target=read_token, daemon=True)
    thread.start()
    return thread, outcome


def test_token_source_refreshes_inside_five_minutes_and_persists_result(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    flow = _RefreshingFlow(
        {"access_token": "fake-new-access", "expires_in": 3600}
    )
    flow.release.set()
    source = TokenSource(store, "codex", flow, now_ms=lambda: FAKE_NOW_MS)

    assert source.get_token() == ("fake-new-access", "fake-account")
    assert flow.calls == ["fake-old-refresh"]
    assert store.get("codex") == {
        "type": "oauth",
        "access": "fake-new-access",
        "refresh": "fake-old-refresh",
        "expires": FAKE_NOW_MS + 3_600_000,
        "account_id": "fake-account",
    }


def test_concurrent_token_callers_share_one_refresh(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    flow = _RefreshingFlow(
        {
            "access_token": "fake-single-flight-access",
            "refresh_token": "fake-new-refresh",
            "expires_in": 3600,
        }
    )
    source = TokenSource(store, "codex", flow, now_ms=lambda: FAKE_NOW_MS)
    caller_count = 8
    ready = threading.Barrier(caller_count + 1)
    results: Queue[tuple[str, str] | BaseException] = Queue()

    def get_token() -> None:
        ready.wait(timeout=2)
        try:
            results.put(source.get_token())
        except BaseException as exc:
            results.put(exc)

    callers = [
        threading.Thread(target=get_token, daemon=True) for _ in range(caller_count)
    ]
    for caller in callers:
        caller.start()
    ready.wait(timeout=2)
    assert flow.entered.wait(timeout=2)
    flow.release.set()

    observed = [results.get(timeout=2) for _ in callers]
    for caller in callers:
        caller.join(timeout=2)
        assert not caller.is_alive()

    assert observed == [
        ("fake-single-flight-access", "fake-account")
    ] * caller_count
    assert flow.calls == ["fake-old-refresh"]
    assert store.get("codex") == {
        "type": "oauth",
        "access": "fake-single-flight-access",
        "refresh": "fake-new-refresh",
        "expires": FAKE_NOW_MS + 3_600_000,
        "account_id": "fake-account",
    }


def test_logout_during_refresh_does_not_resurrect_deleted_credential(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    flow = _RefreshingFlow(
        {"access_token": "fake-refreshed-access", "expires_in": 3600}
    )
    source = TokenSource(store, "codex", flow, now_ms=lambda: FAKE_NOW_MS)

    reader, outcome = _start_token_read(source)
    assert flow.entered.wait(timeout=2)
    store.delete("codex")
    flow.release.set()
    result = outcome.get(timeout=2)
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert isinstance(result, RuntimeError)
    assert "run /login codex" in str(result)
    assert store.get("codex") is None


def test_login_replacement_during_refresh_is_not_overwritten(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    flow = _RefreshingFlow(
        {"access_token": "fake-refreshed-access", "expires_in": 3600}
    )
    source = TokenSource(store, "codex", flow, now_ms=lambda: FAKE_NOW_MS)
    replacement = {
        "type": "oauth",
        "access": "fake-replacement-access",
        "refresh": "fake-replacement-refresh",
        "expires": FAKE_NOW_MS + 3_600_000,
        "account_id": "fake-replacement-account",
    }

    reader, outcome = _start_token_read(source)
    assert flow.entered.wait(timeout=2)
    store.set("codex", replacement)
    flow.release.set()
    result = outcome.get(timeout=2)
    reader.join(timeout=2)

    assert not reader.is_alive()
    assert result == ("fake-replacement-access", "fake-replacement-account")
    assert store.get("codex") == replacement


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (400, {"error": "invalid_grant"}),
        (401, {"error": "unauthorized"}),
    ],
)
def test_definitive_refresh_failure_clears_unchanged_stale_credential(
    tmp_path: Path,
    status_code: int,
    payload: dict[str, str],
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    requests: list[httpx.Request] = []

    def reject_refresh(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json=payload)

    with httpx.Client(transport=httpx.MockTransport(reject_refresh)) as client:
        source = TokenSource(
            store,
            "codex",
            _flow(http_client=client),
            now_ms=lambda: FAKE_NOW_MS,
        )
        with pytest.raises(RuntimeError, match="run /login codex"):
            source.get_token()

    assert len(requests) == 1
    assert parse_qs(requests[0].content.decode())["refresh_token"] == [
        "fake-old-refresh"
    ]
    assert store.get("codex") is None


def test_definitive_refresh_failure_does_not_clear_replacement_credential(
    tmp_path: Path,
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.set("codex", _expired_credential())
    entered = threading.Event()
    release = threading.Event()
    replacement = {
        "type": "oauth",
        "access": "fake-replacement-access",
        "refresh": "fake-replacement-refresh",
        "expires": FAKE_NOW_MS + 3_600_000,
        "account_id": "fake-replacement-account",
    }

    def reject_refresh(_request: httpx.Request) -> httpx.Response:
        entered.set()
        assert release.wait(timeout=2)
        return httpx.Response(400, json={"error": "invalid_grant"})

    with httpx.Client(transport=httpx.MockTransport(reject_refresh)) as client:
        source = TokenSource(
            store,
            "codex",
            _flow(http_client=client),
            now_ms=lambda: FAKE_NOW_MS,
        )
        reader, outcome = _start_token_read(source)
        assert entered.wait(timeout=2)
        store.set("codex", replacement)
        release.set()
        outcome.get(timeout=2)
        reader.join(timeout=2)

    assert not reader.is_alive()
    assert store.get("codex") == replacement


def test_transient_refresh_failure_preserves_stale_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    stale = _expired_credential()
    store.set("codex", stale)

    def unavailable(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "fake-server-error"})

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
        source = TokenSource(
            store,
            "codex",
            _flow(http_client=client),
            now_ms=lambda: FAKE_NOW_MS,
        )
        with pytest.raises(httpx.HTTPStatusError):
            source.get_token()

    assert store.get("codex") == stale
