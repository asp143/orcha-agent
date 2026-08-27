"""Credential storage, OAuth PKCE, and refreshing token sources."""

from __future__ import annotations

import fcntl
import base64
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import webbrowser
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, TypeAlias
from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

Credential = dict[str, Any]
LoginMode: TypeAlias = Literal["auto", "browser", "device", "paste"]

_PATH_LOCKS: dict[Path, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_TOKEN_LOCKS: dict[tuple[Path, str], threading.Lock] = {}
_TOKEN_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(resolved, threading.RLock())


def _token_lock(path: Path, prefix: str) -> threading.Lock:
    key = (path.resolve(), prefix)
    with _TOKEN_LOCKS_GUARD:
        return _TOKEN_LOCKS.setdefault(key, threading.Lock())


@dataclass(frozen=True, slots=True)
class AuthFlow:
    login: Callable[[Any, LoginMode], Awaitable[None]]
    logout: Callable[[Any], Awaitable[None]]
    status: Callable[[], str]


class CredentialStore:
    """Private, atomic JSON credential storage keyed by provider prefix."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Path.home() / ".config/orcha-agent/auth.json")
        self._lock = _path_lock(self.path)

    @contextmanager
    def _locked(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(lock_path, 0o600)
        with self._lock:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read(self) -> dict[str, Credential]:
        if not self.path.is_file():
            return {}
        with self.path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("credential store must contain a JSON object")
        return {
            str(prefix): dict(credential)
            for prefix, credential in value.items()
            if isinstance(credential, Mapping)
        }

    def _write(self, credentials: Mapping[str, Credential]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = -1
            with stream:
                json.dump(credentials, stream, separators=(",", ":"), sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            temporary.unlink(missing_ok=True)
            raise

    def get(self, prefix: str) -> Credential | None:
        with self._locked():
            credential = self._read().get(prefix)
            return None if credential is None else dict(credential)

    def set(self, prefix: str, credential: Mapping[str, Any]) -> None:
        with self._locked():
            credentials = self._read()
            credentials[prefix] = dict(credential)
            self._write(credentials)

    def delete(self, prefix: str) -> None:
        with self._locked():
            credentials = self._read()
            credentials.pop(prefix, None)
            self._write(credentials)

    def compare_and_set(
        self,
        prefix: str,
        expected: Mapping[str, Any],
        replacement: Mapping[str, Any] | None,
    ) -> bool:
        with self._locked():
            credentials = self._read()
            if credentials.get(prefix) != dict(expected):
                return False
            if replacement is None:
                credentials.pop(prefix, None)
            else:
                credentials[prefix] = dict(replacement)
            self._write(credentials)
            return True


class OAuthPKCEFlow:
    """Generic synchronous OAuth authorization-code flow with S256 PKCE."""

    def __init__(
        self,
        client_id: str,
        authorize_url: str,
        token_url: str,
        scopes: str | Sequence[str],
        redirect_port: int,
        redirect_path: str,
        extra_authorize_params: Mapping[str, str] | None = None,
        *,
        http_client: httpx.Client | None = None,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.client_id = client_id
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.scopes = scopes if isinstance(scopes, str) else " ".join(scopes)
        self.redirect_port = redirect_port
        self.redirect_path = redirect_path
        self.extra_authorize_params = dict(extra_authorize_params or {})
        self.http_client = http_client
        self.input_fn = input_fn

    @staticmethod
    def code_challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    @staticmethod
    def parse_paste(value: str, *, expected_state: str | None = None) -> str:
        pasted = value.strip()
        state: str | None = None
        if "://" in pasted:
            query = parse_qs(urlparse(pasted).query)
            code = query.get("code", [""])[0]
            state = query.get("state", [None])[0]
        elif "#" in pasted:
            code, state = pasted.split("#", 1)
        else:
            # Raw codes carry no callback state; accepting them is the explicit
            # paste fallback for headless or unavailable loopback flows.
            code = pasted
        if not code:
            raise ValueError("OAuth callback did not contain a code")
        if (
            expected_state is not None
            and state is not None
            and not secrets.compare_digest(state, expected_state)
        ):
            raise ValueError("OAuth state mismatch")
        return code

    def _authorization_url(self, verifier: str, state: str, redirect_uri: str) -> str:
        parameters = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scopes,
            "state": state,
            "code_challenge": self.code_challenge(verifier),
            "code_challenge_method": "S256",
            **self.extra_authorize_params,
        }
        return f"{self.authorize_url}?{urlencode(parameters)}"

    def _paste_code(
        self,
        authorization_url: str,
        state: str,
        *,
        open_browser: bool = True,
    ) -> str:
        if open_browser and not webbrowser.open(authorization_url):
            raise RuntimeError("browser did not open")
        pasted = self.input_fn(
            f"Open this URL to authenticate:\n{authorization_url}\n"
            "Paste the redirect URL or authorization code: "
        )
        return self.parse_paste(pasted, expected_state=state)

    def authorize(self, *, no_browser: bool = False) -> dict[str, Any]:
        verifier = secrets.token_urlsafe(64)
        state = secrets.token_urlsafe(32)
        redirect_uri = f"http://127.0.0.1:{self.redirect_port}{self.redirect_path}"
        authorization_url = self._authorization_url(verifier, state, redirect_uri)
        if no_browser:
            code = self._paste_code(authorization_url, state, open_browser=False)
            return self.exchange(code, verifier, redirect_uri)

        result: dict[str, str | Exception] = {}
        callback_done = threading.Event()
        expected_path = self.redirect_path
        expected_state = state

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                code = query.get("code", [""])[0]
                callback_state = query.get("state", [None])[0]
                if parsed.path != expected_path:
                    body = b"Not found"
                    self.send_response(404)
                elif (
                    not isinstance(callback_state, str)
                    or not secrets.compare_digest(callback_state, expected_state)
                ):
                    body = b"OAuth state mismatch"
                    self.send_response(400)
                elif not code:
                    body = b"Missing authorization code"
                    self.send_response(400)
                else:
                    result["code"] = code
                    body = b"Authentication complete. You may close this window."
                    self.send_response(200)
                    callback_done.set()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        try:
            server = ThreadingHTTPServer(("127.0.0.1", self.redirect_port), CallbackHandler)
        except OSError:
            code = self._paste_code(authorization_url, state)
            return self.exchange(code, verifier, redirect_uri)

        try:
            deadline = time.monotonic() + 120
            if not webbrowser.open(authorization_url):
                raise RuntimeError("browser did not open")
            while not callback_done.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                server.timeout = min(1, remaining)
                server.handle_request()
        finally:
            server.server_close()
        error = result.get("error")
        if isinstance(error, Exception):
            raise error
        code = result.get("code")
        if not isinstance(code, str):
            raise TimeoutError("OAuth callback was not received")
        return self.exchange(code, verifier, redirect_uri)

    def _post_token(self, data: Mapping[str, str]) -> dict[str, Any]:
        if self.http_client is not None:
            response = self.http_client.post(self.token_url, data=data)
        else:
            response = httpx.post(self.token_url, data=data, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OAuth token endpoint returned invalid JSON")
        return dict(payload)

    def exchange(self, code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
        return self._post_token(
            {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            }
        )

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        return self._post_token(
            {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": refresh_token,
            }
        )


class DeviceAuthorizationCancelled(RuntimeError):
    pass


class OAuthDeviceFlow:
    """OAuth device authorization flow with polling and code exchange."""

    def __init__(
        self,
        client_id: str,
        user_code_url: str,
        device_token_url: str,
        token_url: str,
        verification_url: str,
        *,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        timeout: float = 900,
    ) -> None:
        self.client_id = client_id
        self.user_code_url = user_code_url
        self.device_token_url = device_token_url
        self.token_url = token_url
        self.verification_url = verification_url
        self.http_client = http_client
        self.sleep = sleep
        self.monotonic = monotonic
        self.timeout = timeout

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, **kwargs)
        return httpx.post(url, timeout=30, **kwargs)

    def authorize(
        self,
        *,
        output: Callable[[str], None] = print,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        response = self._post(
            self.user_code_url,
            json={"client_id": self.client_id},
        )
        response.raise_for_status()
        challenge = response.json()
        device_auth_id = challenge["device_auth_id"]
        user_code = challenge["user_code"]
        interval = max(2.0, float(challenge.get("interval", 2)))
        output(f"Open {self.verification_url} and enter code: {user_code}")
        complete_url = challenge.get("verification_uri_complete")
        if isinstance(complete_url, str) and complete_url:
            output(complete_url)

        deadline = self.monotonic() + self.timeout
        authorization_code: str | None = None
        code_verifier: str | None = None
        while self.monotonic() < deadline:
            remaining = deadline - self.monotonic()
            delay = min(interval, remaining)
            if cancel_event is not None:
                if cancel_event.wait(timeout=delay):
                    raise DeviceAuthorizationCancelled(
                        "OAuth device authorization cancelled"
                    )
            else:
                self.sleep(delay)
            if self.monotonic() >= deadline:
                break
            poll = self._post(
                self.device_token_url,
                json={
                    "device_auth_id": device_auth_id,
                    "user_code": user_code,
                },
            )
            if poll.status_code == 200:
                payload = poll.json()
                authorization_code = payload["authorization_code"]
                code_verifier = payload["code_verifier"]
                break
            error: Any = None
            try:
                payload = poll.json()
                error = payload.get("error") if isinstance(payload, Mapping) else None
            except (ValueError, json.JSONDecodeError):
                pass
            if error == "slow_down":
                interval += 5
                continue
            if poll.status_code in {403, 404}:
                continue
            poll.raise_for_status()

        if authorization_code is None or code_verifier is None:
            raise TimeoutError("OAuth device authorization timed out")
        exchange = self._post(
            self.token_url,
            data={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "code": authorization_code,
                "code_verifier": code_verifier,
                "redirect_uri": "https://auth.openai.com/deviceauth/callback",
            },
        )
        exchange.raise_for_status()
        payload = exchange.json()
        if not isinstance(payload, Mapping):
            raise ValueError("OAuth token endpoint returned invalid JSON")
        return dict(payload)


def _definitive_refresh_error(exc: httpx.HTTPStatusError) -> bool:
    response = exc.response
    if response.status_code == 401:
        return True
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return False
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping):
        error = error.get("type") or error.get("code")
    return error == "invalid_grant"


class TokenSource:
    """Read and single-flight refresh OAuth access tokens."""

    def __init__(
        self,
        store: CredentialStore,
        prefix: str,
        flow: Any,
        *,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.flow = flow
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        store_path = getattr(store, "path", Path(f".orcha-auth-{id(store)}"))
        self._refresh_lock = _token_lock(Path(store_path), prefix)

    def _needs_refresh(self, credential: Mapping[str, Any]) -> bool:
        expires = credential.get("expires")
        return not isinstance(expires, int) or expires - self.now_ms() < 300_000

    def get_token(self) -> tuple[str, str]:
        credential = self.store.get(self.prefix)
        if credential is None or credential.get("type") != "oauth":
            raise RuntimeError(f"Not logged in to {self.prefix}; run /login {self.prefix}")
        if self._needs_refresh(credential):
            with self._refresh_lock:
                credential = self.store.get(self.prefix)
                if credential is None:
                    raise RuntimeError(f"Not logged in to {self.prefix}; run /login {self.prefix}")
                if self._needs_refresh(credential):
                    stale = dict(credential)
                    refresh_token = stale.get("refresh")
                    if not isinstance(refresh_token, str) or not refresh_token:
                        raise RuntimeError(f"Login for {self.prefix} cannot be refreshed")
                    try:
                        refreshed = self.flow.refresh(refresh_token)
                    except httpx.HTTPStatusError as exc:
                        if not _definitive_refresh_error(exc):
                            raise
                        if self.store.compare_and_set(self.prefix, stale, None):
                            raise RuntimeError(
                                f"Refresh rejected; run /login {self.prefix}"
                            ) from exc
                        credential = self.store.get(self.prefix)
                        if credential is None:
                            raise RuntimeError(
                                f"Not logged in to {self.prefix}; run /login {self.prefix}"
                            ) from exc
                    else:
                        access = refreshed.get("access_token")
                        if not isinstance(access, str) or not access:
                            raise RuntimeError("OAuth refresh response omitted access_token")
                        expires_in = refreshed.get("expires_in", 3600)
                        updated = {
                            **stale,
                            "access": access,
                            "refresh": refreshed.get("refresh_token") or refresh_token,
                            "expires": self.now_ms() + int(expires_in) * 1000,
                        }
                        if self.store.compare_and_set(self.prefix, stale, updated):
                            credential = updated
                        else:
                            credential = self.store.get(self.prefix)
                            if credential is None:
                                raise RuntimeError(
                                    f"Not logged in to {self.prefix}; "
                                    f"run /login {self.prefix}"
                                )
        access = credential.get("access")
        account_id = credential.get("account_id", "")
        if not isinstance(access, str) or not access:
            raise RuntimeError(f"Login for {self.prefix} has no access token")
        return access, str(account_id)
