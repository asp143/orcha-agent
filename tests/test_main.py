import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent import __main__ as entrypoint
from orcha_agent.core.auth import AuthFlow
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.tui import app as tui_app


@pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
def test_main_loads_dotenv_only_from_trusted_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trusted: bool,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ORCHA_TEST_SENTINEL=from-temporary-dotenv\n")
    monkeypatch.delenv("ORCHA_TEST_SENTINEL", raising=False)
    dotenv_calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(dotenv_path: str | Path, *, override: bool) -> bool:
        dotenv_calls.append((Path(dotenv_path), override))
        monkeypatch.setenv("ORCHA_TEST_SENTINEL", "from-temporary-dotenv")
        return True

    async def fake_run_app(cfg: Any) -> int:
        assert cfg.cwd == tmp_path.resolve()
        return 0

    monkeypatch.setattr(entrypoint, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(entrypoint, "run_app", fake_run_app)
    argv = ["orcha", "--cwd", str(tmp_path)]
    if trusted:
        argv.append("--trust-cwd")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    if trusted:
        assert dotenv_calls == [(env_file.resolve(), False)]
        assert os.environ["ORCHA_TEST_SENTINEL"] == "from-temporary-dotenv"
    else:
        assert dotenv_calls == []
        assert "ORCHA_TEST_SENTINEL" not in os.environ


@pytest.mark.parametrize("login_mode", ["auto", "browser", "device", "paste"])
def test_main_login_passes_mode_to_auth_plugin_without_starting_repl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    login_mode: str,
) -> None:
    cfg = SimpleNamespace(
        command="login",
        login_prefix="codex",
        login_mode=login_mode,
        cwd=tmp_path,
        trust_cwd=False,
    )
    loaded_configs: list[object] = []
    login_calls: list[tuple[object, str]] = []

    async def login(ctx: object, mode: str) -> None:
        login_calls.append((ctx, mode))

    async def logout(_ctx: object) -> None:
        raise AssertionError("login command must not invoke logout")

    flow = AuthFlow(
        login=login,
        logout=logout,
        status=lambda: "not logged in",
    )

    def fake_load_plugins(
        registry: Any,
        bus: Any,
        cfg: object,
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        loaded_configs.append(cfg)
        PluginAPI(
            name="fake_auth_plugin",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        ).add_auth("codex", flow)
        return []

    async def unexpected_run_app(_cfg: object) -> int:
        raise AssertionError("login command must not start run_app")

    def unexpected_prompt_session(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("login command must not create PromptSession")

    monkeypatch.setattr(entrypoint, "load_config", lambda: cfg)
    monkeypatch.setattr(entrypoint, "load_plugins", fake_load_plugins)
    monkeypatch.setattr(entrypoint, "run_app", unexpected_run_app)
    monkeypatch.setattr(tui_app, "PromptSession", unexpected_prompt_session)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert loaded_configs == [cfg]
    assert len(login_calls) == 1
    assert login_calls[0][1] == login_mode


def test_main_login_reports_auth_failure_without_starting_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        command="login",
        login_prefix="codex",
        login_mode="auto",
        cwd=Path("/unused"),
        trust_cwd=False,
    )
    error_messages: list[str] = []
    login_calls: list[tuple[object, str]] = []

    async def login(ctx: object, mode: str) -> None:
        login_calls.append((ctx, mode))
        raise RuntimeError("authentication failed")

    async def logout(_ctx: object) -> None:
        raise AssertionError("login command must not invoke logout")

    flow = AuthFlow(
        login=login,
        logout=logout,
        status=lambda: "not logged in",
    )

    def fake_load_plugins(
        registry: Any,
        bus: Any,
        _cfg: object,
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        PluginAPI(
            name="fake_auth_plugin",
            config={},
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        ).add_auth("codex", flow)
        return []

    async def unexpected_run_app(_cfg: object) -> int:
        raise AssertionError("failed login must not start run_app")

    def unexpected_prompt_session(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("failed login must not create PromptSession")

    monkeypatch.setattr(entrypoint, "load_config", lambda: cfg)
    monkeypatch.setattr(entrypoint, "load_plugins", fake_load_plugins)
    monkeypatch.setattr(
        entrypoint,
        "ConsoleOutput",
        lambda: SimpleNamespace(error=error_messages.append),
    )
    monkeypatch.setattr(entrypoint, "run_app", unexpected_run_app)
    monkeypatch.setattr(tui_app, "PromptSession", unexpected_prompt_session)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 1
    assert error_messages == ["authentication failed"]
    assert len(login_calls) == 1
    assert login_calls[0][1] == "auto"
