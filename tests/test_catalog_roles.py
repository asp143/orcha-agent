from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from orcha_agent.core.catalog import get_catalog, get_model, provider_api_key
from orcha_agent.core.config import load_config
from orcha_agent.core.models import MODEL_ROLES, ModelResolver, expand_model_spec
from tests.test_models import _api, _caps, _config


def test_bundle_and_passthrough():
    catalog = get_catalog()
    assert catalog is get_catalog()
    assert len(catalog) > 100
    assert get_model("anthropic:claude-sonnet-4-5").context_window > 0
    assert get_model("langchain:custom").context_window == 0
    assert get_model("unknown:custom") is None


def test_override_merge_trust_and_cache(tmp_path):
    user = tmp_path / "user"
    project = tmp_path / ".orcha-agent"
    user.mkdir()
    project.mkdir()
    (user / "models.yml").write_text(
        "providers:\n  openai:\n    models:\n      local:\n        context_window: 32000\n        cost: {input: 1, output: 2}\n"
    )
    (project / "models.yml").write_text(
        "providers:\n  openai:\n    model_overrides:\n      local:\n        context_window: 64000\n        cost: {output: 3}\n"
    )
    cfg = replace(_config(tmp_path), user_config_path=user / "config.toml")
    assert get_model("openai:local", cfg).context_window == 32000
    cfg = replace(cfg, trust_cwd=True)
    info = get_model("openai:local", cfg)
    assert info.context_window == 64000
    assert info.cost == {"input": 1, "output": 3}
    assert get_catalog(cfg) is get_catalog(cfg)
    (project / "models.yml").write_text(
        "openai:\n  model_overrides:\n    local: {context_window: 99000}\n"
    )
    assert get_model("openai:local", cfg).context_window == 99000


def test_keys_are_lazy_and_project_trusted(tmp_path, monkeypatch):
    configdir = tmp_path / ".orcha-agent"
    configdir.mkdir()
    marker = tmp_path / "marker"
    (configdir / "models.yml").write_text(
        f'fake:\n  api_key: !command "touch {marker}; printf secret"\n'
    )
    cfg = _config(tmp_path)
    assert provider_api_key("fake", cfg) is None
    assert not marker.exists()
    cfg = replace(cfg, trust_cwd=True)
    get_catalog(cfg)
    assert not marker.exists()
    with pytest.raises(ValueError, match="commands are user-only"):
        provider_api_key("fake", cfg)
    assert not marker.exists()
    (configdir / "models.yml").write_text("fake:\n  api_key: TEST_CATALOG_KEY\n")
    monkeypatch.setenv("TEST_CATALOG_KEY", "env-secret")
    assert provider_api_key("fake", cfg) == "env-secret"


@pytest.mark.parametrize("role", MODEL_ROLES)
def test_roles_and_effort(tmp_path, role):
    cfg = replace(_config(tmp_path), model_roles={role: "tiny"}, models={"tiny": "fake:small"})
    assert expand_model_spec("@" + role + ":high", cfg) == ["fake:small:high"]


def test_role_cycles_unknown_and_effort(tmp_path):
    cfg = replace(_config(tmp_path), model_roles={"smol": "@slow", "slow": "@smol"})
    with pytest.raises(ValueError, match="cycle"):
        expand_model_spec("@smol", cfg)
    with pytest.raises(ValueError, match="Unknown model role"):
        expand_model_spec("@bogus", cfg)
    with pytest.raises(ValueError, match="effort"):
        expand_model_spec("@smol:bogus", cfg)


def test_factory_receives_effort_and_metadata(tmp_path):
    from orcha_agent.core.registry import Registry

    registry = Registry()
    seen = []

    def factory(name, options):
        seen.append((name, options))
        return FakeListChatModel(responses=["ok"])

    _api(registry).add_provider("fake", factory, capabilities=_caps(thinking=True))
    cfg = replace(_config(tmp_path), model_roles={"smol": "fake:small"})
    model = ModelResolver(registry, cfg).resolve("@smol:high", "task")
    assert seen == [("small", {"reasoning_effort": "high"})]
    assert model.metadata["orcha_model"] == "fake:small"
    assert model.metadata["orcha_role"] == "task"


@pytest.mark.parametrize("flag", ["smol", "slow", "plan"])
def test_cli_role_flags(tmp_path, flag):
    cfg = load_config(["--" + flag], env={"HOME": str(tmp_path)}, cwd=tmp_path)
    assert cfg.model == "@" + flag
    assert not expand_model_spec(cfg.model, cfg)[0].startswith("@")


def test_model_browser_has_metadata_and_search(tmp_path):
    from orcha_agent.core.registry import Registry
    from orcha_agent.tui.overlays.model import ModelOverlay

    registry = Registry()
    _api(registry).add_provider(
        "anthropic",
        lambda *_: FakeListChatModel(responses=["ok"]),
        capabilities=_caps(),
        models=["claude-sonnet-4-5"],
    )
    ctx = SimpleNamespace(cfg=_config(tmp_path), registry=registry)
    picker = ModelOverlay(ctx, browse=True)
    label = picker.label("anthropic:claude-sonnet-4-5")
    assert "ctx" in label and "/M" in label and "thinking" in label
    picker.filter.text = "sonnet"
    assert "anthropic:claude-sonnet-4-5" in picker.filtered_items
    assert len(picker.filtered_items) < len(picker.items)
    content = picker.list_control.create_content(160, 10)
    assert "anthropic" in "".join(text for _, text in content.get_line(0))
    assert "sonnet" in "".join(text for _, text in content.get_line(content.cursor_position.y))


def test_invalid_context_override(tmp_path):
    configdir = tmp_path / ".orcha-agent"
    configdir.mkdir()
    (configdir / "models.yml").write_text("openai:\n  models:\n    bad: {context_window: nope}\n")
    with pytest.raises(ValueError, match="context_window"):
        get_catalog(replace(_config(tmp_path), trust_cwd=True))


def test_nested_role_effort_overrides_and_ollama_tags(tmp_path):
    cfg = replace(_config(tmp_path), model_roles={"smol": "@slow:low", "slow": "fake:small"})
    assert expand_model_spec("@smol:high", cfg) == ["fake:small:high"]
    assert expand_model_spec("@slow:off", cfg) == ["fake:small:off"]
    cfg = replace(cfg, model_roles={"smol": "ollama:custom:high"})
    assert expand_model_spec("@smol:low", cfg) == ["ollama:custom:high"]


def test_same_section_overrides_merge_fields_and_cost(tmp_path):
    user = tmp_path / "user"
    project = tmp_path / ".orcha-agent"
    user.mkdir()
    project.mkdir()
    (user / "models.yml").write_text(
        "openai:\n  model_overrides:\n    custom:\n      context_window: 64000\n      cost: {input: 1, output: 2}\n"
    )
    (project / "models.yml").write_text(
        "openai:\n  model_overrides:\n    custom:\n      cost: {output: 3}\n"
    )
    cfg = replace(_config(tmp_path), user_config_path=user / "config.toml", trust_cwd=True)
    model = get_model("openai:custom", cfg)
    assert model.context_window == 64000
    assert model.cost == {"input": 1, "output": 3}


def test_secret_is_redacted_from_factory_failure(tmp_path, monkeypatch):
    from orcha_agent.core.registry import Registry

    registry = Registry()

    def factory(_name, options):
        raise ValueError(f"invalid key {options['api_key']}")

    _api(registry).add_provider("fake", factory, capabilities=_caps())
    directory = tmp_path / ".orcha-agent"
    directory.mkdir()
    (directory / "models.yml").write_text("fake:\n  api_key: CATALOG_TEST_SECRET\n")
    monkeypatch.setenv("CATALOG_TEST_SECRET", "secret-value")
    with pytest.raises(RuntimeError) as raised:
        ModelResolver(registry, replace(_config(tmp_path), trust_cwd=True)).resolve(
            "fake:small", "main"
        )
    assert "secret-value" not in str(raised.value)
    assert "[redacted]" in str(raised.value)


def test_unset_roles_follow_current_main_after_switch(tmp_path):
    cfg = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path)
    switched = replace(cfg, model="openai:new-main")
    assert expand_model_spec("@smol", switched) == ["openai:new-main"]
    assert expand_model_spec("@advisor", switched) == ["openai:new-main"]
    assert "main" not in cfg.model_roles
    selected = replace(switched, model="@smol", model_role_default=switched.model)
    assert expand_model_spec(selected.model, selected) == ["openai:new-main"]
