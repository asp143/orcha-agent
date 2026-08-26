from pathlib import Path
from typing import Mapping, Sequence

from orcha_agent.core.config import Config, load_config


def _load(
    tmp_path: Path,
    *,
    argv: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    user_config_path: Path | None = None,
    project_config_path: Path | None = None,
) -> Config:
    return load_config(
        list(argv),
        env={} if env is None else env,
        cwd=tmp_path,
        user_config_path=user_config_path or tmp_path / "missing-user.toml",
        project_config_path=project_config_path or tmp_path / "missing-project.toml",
    )


def test_model_precedence_walks_all_five_layers_without_skipping_project_config(
    tmp_path: Path,
) -> None:
    user_config = tmp_path / "user.toml"
    project_config = tmp_path / "project.toml"
    user_config.write_text('[core]\nmodel = "anthropic:user"\n')
    project_config.write_text('[core]\nmodel = "anthropic:project"\n')

    assert _load(
        tmp_path,
        argv=("--model", "anthropic:cli"),
        env={"ORCHA_MODEL": "anthropic:env"},
        user_config_path=user_config,
        project_config_path=project_config,
    ).model == "anthropic:cli"
    assert _load(
        tmp_path,
        env={"ORCHA_MODEL": "anthropic:env"},
        user_config_path=user_config,
        project_config_path=project_config,
    ).model == "anthropic:env"
    assert _load(
        tmp_path,
        user_config_path=user_config,
        project_config_path=project_config,
    ).model == "anthropic:project"
    assert _load(
        tmp_path,
        user_config_path=user_config,
    ).model == "anthropic:user"
    assert _load(tmp_path).model == "anthropic:claude-opus-5"


def test_model_aliases_and_per_plugin_sections_survive_toml_loading(tmp_path: Path) -> None:
    project_config = tmp_path / "project.toml"
    project_config.write_text(
        """
[core]
model = "fast"

[models]
fast = "anthropic:claude-haiku-4-5"
reviewer = "openai:gpt-5"

[plugins]
disabled = ["disabled_plugin"]

[plugins.external_plugin]
greeting = "hello"
retries = 3
""".strip()
        + "\n"
    )

    cfg = _load(tmp_path, project_config_path=project_config)

    assert cfg.model == "fast"
    assert cfg.models == {
        "fast": "anthropic:claude-haiku-4-5",
        "reviewer": "openai:gpt-5",
    }
    assert cfg.plugins["disabled"] == ["disabled_plugin"]
    assert cfg.plugin_config("external_plugin") == {"greeting": "hello", "retries": 3}
