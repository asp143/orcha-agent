from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from rich.console import Console

from orcha_agent.builtin.commands_model import _model, _models
from orcha_agent.core.catalog import get_catalog, provider_api_key
from orcha_agent.core.models import ModelResolver, role_fallback_notices
from orcha_agent.core.registry import Registry
from orcha_agent.core.usage_store import usage_table
from orcha_agent.tui.console import ConsoleOutput
from tests.test_models import _api, _caps, _config


def settings(tmp_path):
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    (project / ".orcha-agent").mkdir(parents=True)
    cfg = replace(_config(project), user_config_path=user / "config.toml", trust_cwd=True)
    return cfg, user / "models.yml", project / ".orcha-agent/models.yml"


@pytest.mark.parametrize("value", ['!command "printf private"', '"!printf private"'])
def test_project_commands_never_execute_even_trusted(tmp_path, monkeypatch, value):
    cfg, user, project = settings(tmp_path)
    project.write_text(f"fake:\n  api_key: {value}\n")
    calls = []
    monkeypatch.setattr("orcha_agent.core.catalog.subprocess.run", lambda *a, **kw: calls.append(a))
    get_catalog(cfg)
    with pytest.raises(ValueError, match="environment variable"):
        provider_api_key("fake", cfg)
    # A user filename pointing to project bytes does not elevate the helper.
    user.symlink_to(project)
    with pytest.raises(ValueError, match="environment variable"):
        provider_api_key("fake", replace(cfg, trust_cwd=False))
    assert calls == []


def test_user_helpers_cached_concurrently_by_user_stamp_only(tmp_path, monkeypatch):
    import subprocess

    cfg, user, project = settings(tmp_path)
    user.write_text('fake:\n  api_key: !command "printf secret"\n')
    calls = []

    def execute(command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        calls.append(command)
        return SimpleNamespace(stdout="secret\n")

    monkeypatch.setattr("orcha_agent.core.catalog.subprocess.run", execute)
    get_catalog(cfg)
    assert not calls
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert (
            list(executor.map(lambda _: provider_api_key("fake", cfg), range(20)))
            == ["secret"] * 20
        )
    assert len(calls) == 1
    project.write_text("fake:\n  models:\n    custom: {context_window: 50000}\n")
    assert provider_api_key("fake", cfg) == "secret"
    assert len(calls) == 1
    user.write_text('fake:\n  api_key: !command "printf refreshed-secret"\n')
    assert provider_api_key("fake", cfg) == "secret"
    assert len(calls) == 2


def test_environment_precedes_helper_and_project_env_override(tmp_path, monkeypatch):
    cfg, user, project = settings(tmp_path)
    user.write_text('fake:\n  api_key: !command "printf private"\n')
    registry = Registry()
    captured = []

    def factory(name, options):
        captured.append(options)
        return FakeListChatModel(responses=["ok"])

    _api(registry).add_provider("fake", factory, capabilities=_caps(), env_keys=("DIRECT_KEY",))
    monkeypatch.setenv("DIRECT_KEY", "direct")
    monkeypatch.setattr(
        "orcha_agent.core.catalog.provider_api_key",
        lambda *a: pytest.fail("helper must not be read"),
    )
    ModelResolver(registry, cfg).resolve("fake:model", "main")
    assert "api_key" not in captured[0]
    monkeypatch.undo()
    project.write_text("fake:\n  api_key: PROJECT_KEY\n")
    monkeypatch.setenv("PROJECT_KEY", "project-secret")
    assert provider_api_key("fake", cfg) == "project-secret"


def test_helper_failure_redacts_command_and_stderr(tmp_path, monkeypatch):
    import subprocess

    cfg, user, _ = settings(tmp_path)
    user.write_text('fake:\n  api_key: !command "private-command"\n')

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "private-command", stderr="private-secret")

    monkeypatch.setattr("orcha_agent.core.catalog.subprocess.run", fail)
    with pytest.raises(RuntimeError) as error:
        provider_api_key("fake", cfg)
    assert str(error.value) == "Model API key command failed"
    assert error.value.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["smol", "slow", "plan"])
async def test_model_fallback_notice_once_and_explicit_role_silent(tmp_path, role):
    cfg, _, _ = settings(tmp_path)
    output = StringIO()
    ctx = SimpleNamespace(
        cfg=cfg, console=ConsoleOutput(Console(file=output)), switch_model=AsyncMock()
    )
    await _model(ctx, f"@{role},@{role}:high")
    notice = f"role {role} is not configured, using main"
    assert output.getvalue().count(notice) == 1
    assert role_fallback_notices(f"@{role}", replace(cfg, model_roles={role: "fake:model"})) == []


@pytest.mark.asyncio
async def test_catalog_usage_and_model_display_preserve_literal_markup(tmp_path):
    cfg, user, _ = settings(tmp_path)
    name = "[red]literal[/red]"
    user.write_text(f'fake:\n  models:\n    "{name}": {{context_window: 1234}}\n')
    cfg = replace(cfg, model=f"fake:{name}", model_roles={"smol": f"fake:{name}"})
    stream = StringIO()
    ctx = SimpleNamespace(
        cfg=cfg, console=ConsoleOutput(Console(file=stream, width=180, color_system=None))
    )
    await _models(ctx, "literal")
    await _model(ctx, "")
    ctx.console.print(
        usage_table(
            [
                dict(
                    model=f"fake:{name}",
                    requests=1,
                    input_tokens=0,
                    output_tokens=0,
                    cache_read=0,
                    cache_write=0,
                    cost=0,
                    ttft=None,
                    duration=0,
                )
            ],
            "all",
        )
    )
    assert stream.getvalue().count(f"fake:{name}") >= 4


def test_google_invalid_effort_is_clear_value_error(monkeypatch):
    import sys
    from orcha_agent.builtin.provider_google import _factory

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=lambda **kw: kw),
    )
    with pytest.raises(
        ValueError, match="expected one of: off, none, minimal, low, medium, high, xhigh, max"
    ):
        _factory("gemini", {"reasoning_effort": "bogus"})


def test_project_cannot_spoof_user_credential_provenance(tmp_path, monkeypatch):
    cfg, _, project = settings(tmp_path)
    project.write_text(
        'fake:\n  api_key: "!private-helper"\n  _api_key_source: [user, 1, 1, true]\n'
    )
    monkeypatch.setattr(
        "orcha_agent.core.catalog.subprocess.run", lambda *a, **kw: pytest.fail("project executed")
    )
    with pytest.raises(ValueError, match="commands are user-only"):
        provider_api_key("fake", cfg)


@pytest.mark.parametrize("role", ["smol", "slow", "plan"])
def test_cli_role_fallback_notice_printed_once(tmp_path, monkeypatch, role):
    import sys
    from orcha_agent import __main__ as cli
    from orcha_agent.core.config import load_config

    cfg = load_config(["--" + role], env={"HOME": str(tmp_path)}, cwd=tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    runner = AsyncMock(return_value=0)
    monkeypatch.setitem(sys.modules, "orcha_agent.tui.app", SimpleNamespace(run_app=runner))
    output = StringIO()
    monkeypatch.setattr(
        "orcha_agent.tui.console.ConsoleOutput", lambda: ConsoleOutput(Console(file=output))
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    assert output.getvalue().count(f"role {role} is not configured, using main") == 1
    runner.assert_awaited_once_with(cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["vision", "commit", "advisor", "task", "subagent", "summarizer"])
async def test_other_unconfigured_roles_report_main_fallback(tmp_path, role):
    cfg, _, _ = settings(tmp_path)
    cfg = replace(cfg, subagent_model=None, summarizer_model=None)
    output = StringIO()
    ctx = SimpleNamespace(
        cfg=cfg, console=ConsoleOutput(Console(file=output)), switch_model=AsyncMock()
    )
    await _model(ctx, f"@{role},@{role}:high")
    assert output.getvalue().count(f"role {role} is not configured, using main") == 1


@pytest.mark.parametrize("role", ["task", "subagent", "summarizer"])
def test_explicit_worker_or_summarizer_inheritance_has_no_fallback_notice(tmp_path, role):
    cfg, _, _ = settings(tmp_path)
    assert role_fallback_notices(f"@{role}", cfg) == []
    cfg = replace(cfg, subagent_model="@vision", summarizer_model="@vision")
    assert role_fallback_notices(f"@{role}", cfg) == ["role vision is not configured, using main"]
