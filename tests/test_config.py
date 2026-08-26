from pathlib import Path
from typing import Mapping, Sequence

import pytest

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
        argv=("--trust-cwd", "--model", "anthropic:cli"),
        env={"ORCHA_MODEL": "anthropic:env"},
        user_config_path=user_config,
        project_config_path=project_config,
    ).model == "anthropic:cli"
    assert _load(
        tmp_path,
        argv=("--trust-cwd",),
        env={"ORCHA_MODEL": "anthropic:env"},
        user_config_path=user_config,
        project_config_path=project_config,
    ).model == "anthropic:env"
    assert _load(
        tmp_path,
        argv=("--trust-cwd",),
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

    cfg = _load(
        tmp_path,
        argv=("--trust-cwd",),
        project_config_path=project_config,
    )

    assert cfg.model == "fast"
    assert cfg.models == {
        "fast": "anthropic:claude-haiku-4-5",
        "reviewer": "openai:gpt-5",
    }
    assert cfg.plugins["disabled"] == ["disabled_plugin"]
    assert cfg.plugin_config("external_plugin") == {"greeting": "hello", "retries": 3}


def test_untrusted_cwd_ignores_project_config_with_one_line_trust_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = tmp_path / ".orcha-agent"
    project_dir.mkdir()
    project_config = project_dir / "config.toml"
    project_config.write_text('[core]\nmode = "project-mode"\n')

    cfg = _load(tmp_path, project_config_path=project_config)
    captured = capsys.readouterr()
    notices = [*captured.out.splitlines(), *captured.err.splitlines()]

    assert cfg.mode == "ask"
    assert len(notices) == 1
    assert "trust" in notices[0].lower()
    assert "--trust-cwd" in notices[0]


@pytest.mark.parametrize("trust_source", ["user-config", "cli"])
def test_trusting_cwd_applies_project_config(
    tmp_path: Path,
    trust_source: str,
) -> None:
    project_dir = tmp_path / ".orcha-agent"
    project_dir.mkdir()
    project_config = project_dir / "config.toml"
    project_config.write_text('[core]\nmode = "project-mode"\n')
    user_config = tmp_path / "user.toml"
    argv: tuple[str, ...] = ()
    if trust_source == "user-config":
        user_config.write_text(f'[trust]\ndirs = ["{tmp_path}"]\n')
    else:
        argv = ("--trust-cwd",)
    cfg = _load(
        tmp_path,
        argv=argv,
        user_config_path=user_config,
        project_config_path=project_config,
    )

    assert cfg.mode == "project-mode"


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (("--model", "anthropic:first, openai:second"), {}),
        ((), {"ORCHA_MODEL": "anthropic:first, openai:second"}),
    ],
)
def test_comma_separated_cli_and_env_models_normalize_to_fallback_lists(
    tmp_path: Path,
    argv: tuple[str, ...],
    env: Mapping[str, str],
) -> None:
    assert _load(tmp_path, argv=argv, env=env).model == [
        "anthropic:first",
        "openai:second",
    ]
