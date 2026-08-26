import json
import os
import socket
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import pytest

from orcha_agent.core.auth import CredentialStore, OAuthPKCEFlow, TokenSource


FAKE_NOW_MS = 1_800_000_000_000


def _flow(
    *,
    token_url: str = "https://tokens.invalid/token",
    redirect_port: int = 1455,
) -> OAuthPKCEFlow:
    return OAuthPKCEFlow(
        client_id="fake-client-id",
        authorize_url="https://accounts.invalid/authorize",
        token_url=token_url,
        scopes="openid profile offline_access",
        redirect_port=redirect_port,
        redirect_path="/auth/callback",
        extra_authorize_params={"audience": "fake-audience"},
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
        ("fake-raw-code", None, "fake-raw-code"),
    ],
)
def test_parse_paste_accepts_redirect_fragment_and_raw_code(
    pasted: str,
    expected_state: str | None,
    expected_code: str,
) -> None:
    assert _flow().parse_paste(pasted, expected_state=expected_state) == expected_code


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


def test_loopback_callback_rejects_state_mismatch_and_authorize_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = _flow(redirect_port=_unused_loopback_port())
    opened_urls: Queue[str] = Queue()
    outcome: Queue[dict[str, Any] | BaseException] = Queue()
    monkeypatch.setattr(
        "orcha_agent.core.auth.webbrowser.open",
        lambda url: opened_urls.put(url) or True,
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
    callback_url = f"{redirect_uri}?{urlencode({'code': 'fake-code', 'state': 'wrong-state'})}"

    with pytest.raises(HTTPError) as response:
        urlopen(callback_url, timeout=2)
    assert response.value.code == 400
    response.value.read()

    result = outcome.get(timeout=2)
    assert isinstance(result, Exception)
    assert "state" in str(result).lower()
    authorize_thread.join(timeout=2)
    assert not authorize_thread.is_alive()


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
