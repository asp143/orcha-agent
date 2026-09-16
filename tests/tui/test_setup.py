from __future__ import annotations

import tomllib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orcha_agent.builtin.setup import run_setup, setup_picker, should_setup
from orcha_agent.core.config import load_config


def context(tmp_path, answers=()):
    provider = SimpleNamespace(
        available=lambda: None,
        env_keys=("WIZARD_TEST_KEY",),
        models=("example",),
        default_model="example",
    )
    ctx = SimpleNamespace(
        cfg=load_config([], env={}, cwd=tmp_path, user_config_path=tmp_path / "user/config.toml"),
        registry=SimpleNamespace(providers={"example": provider}, auth={}),
        ui=SimpleNamespace(
            show=AsyncMock(side_effect=answers),
            themes={"dark": object(), "light": object()},
            set_theme=Mock(),
            apply_settings=Mock(),
        ),
        console=SimpleNamespace(print=Mock(), warning=Mock()),
        request_rebuild=Mock(),
    )

    async def switch_model(model):
        ctx.cfg = replace(ctx.cfg, model=model)

    ctx.switch_model = AsyncMock(side_effect=switch_model)
    return ctx


def test_first_run_requires_missing_config_and_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("WIZARD_TEST_KEY", raising=False)
    ctx = context(tmp_path)
    assert should_setup(ctx, interactive=True)
    assert not should_setup(ctx, interactive=False)
    monkeypatch.setenv("WIZARD_TEST_KEY", "available")
    assert not should_setup(ctx, interactive=True)
    monkeypatch.delenv("WIZARD_TEST_KEY")
    ctx.cfg.user_config_path.parent.mkdir()
    ctx.cfg.user_config_path.write_text("# configured\n")
    assert not should_setup(ctx, interactive=True)
    ctx.cfg = replace(ctx.cfg, command="setup")
    assert should_setup(ctx, interactive=True)


def test_generic_adapter_does_not_suppress_first_run(tmp_path):
    ctx = context(tmp_path)
    ctx.registry.providers = {"langchain": SimpleNamespace()}
    assert should_setup(ctx, interactive=True)


@pytest.mark.asyncio
async def test_setup_saves_preferences_without_secrets(tmp_path):
    ctx = context(
        tmp_path,
        ("light", "rail", "example", "Continue (API keys are never saved)", "example:example"),
    )
    await run_setup(ctx)
    values = tomllib.loads(ctx.cfg.user_config_path.read_text())
    assert values == {
        "tui": {"theme": "light", "composer": "rail"},
        "core": {"model": "example:example"},
    }
    assert ctx.cfg.model == "example:example"
    ctx.ui.set_theme.assert_called_once_with("light")
    ctx.ui.apply_settings.assert_called_once_with(ctx.cfg)
    ctx.switch_model.assert_awaited_once_with("example:example")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        (None,),
        ("dark", None),
        ("dark", "box", None),
        ("dark", "box", "example", None),
        ("dark", "box", "example", "Continue (API keys are never saved)", None),
    ],
)
async def test_cancel_at_any_step_does_not_write(tmp_path, answers):
    ctx = context(tmp_path, answers)
    await run_setup(ctx)
    assert not ctx.cfg.user_config_path.exists()
    ctx.ui.set_theme.assert_not_called()


@pytest.mark.asyncio
async def test_sign_in_uses_registered_auth_flow(tmp_path):
    ctx = context(tmp_path, ("dark", "box", "example", "Sign in now", "example:example"))
    login = AsyncMock()
    ctx.registry.auth["example"] = SimpleNamespace(flow=SimpleNamespace(login=login))
    await run_setup(ctx)
    login.assert_awaited_once_with(ctx, "auto")


def test_setup_picker_headless_frame():
    overlay = setup_picker(None, title="1/4 Theme", items=("dark", "light"))
    assert "Esc to skip" in overlay.title
    assert "› dark" in overlay.render_text()
    overlay.filter.text = "light"
    assert overlay.filtered_items == ("light",)
    assert "dark" not in overlay.render_text()


def test_setup_cli(tmp_path):
    cfg = load_config(["setup"], env={}, cwd=tmp_path, user_config_path=tmp_path / "absent.toml")
    assert cfg.command == "setup"


@pytest.mark.asyncio
async def test_provider_without_catalog_accepts_custom_model(tmp_path):
    ctx = context(
        tmp_path,
        ("dark", "box", "example", "Continue (API keys are never saved)", "Enter model name"),
    )
    ctx.registry.providers["example"].models = ()
    ctx.registry.providers["example"].default_model = None
    ctx.ui.ask = AsyncMock(
        return_value={"kind": "submit", "results": [{"id": "model", "customInput": "my-model"}]}
    )
    await run_setup(ctx)
    assert ctx.cfg.model == "example:my-model"
    assert (
        tomllib.loads(ctx.cfg.user_config_path.read_text())["core"]["model"] == "example:my-model"
    )


@pytest.mark.asyncio
async def test_unavailable_model_saves_default_without_mutating_active_session(tmp_path):
    ctx = context(
        tmp_path,
        ("light", "rail", "example", "Continue (API keys are never saved)", "example:example"),
    )
    original = ctx.cfg.model
    ctx.switch_model = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    await run_setup(ctx)
    assert ctx.cfg.model == original
    assert tomllib.loads(ctx.cfg.user_config_path.read_text())["core"]["model"] == "example:example"
    ctx.console.warning.assert_called_once()


@pytest.mark.asyncio
async def test_provider_picker_excludes_generic_langchain_adapter(tmp_path):
    ctx = context(tmp_path, ("dark", "box", None))
    ctx.registry.providers["langchain"] = SimpleNamespace()
    await run_setup(ctx)
    provider_pick = ctx.ui.show.call_args_list[2]
    assert provider_pick.kwargs["items"] == ("example",)
