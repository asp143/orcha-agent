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


def test_cli_parser_defaults_login_mode_to_auto(tmp_path: Path) -> None:
    repl = _load(
        tmp_path,
        argv=("repl",),
        env={"HOME": str(tmp_path)},
    )
    login = _load(
        tmp_path,
        argv=("login", "codex"),
        env={"HOME": str(tmp_path)},
    )

    assert repl.command == "repl"
    assert repl.login_prefix is None
    assert repl.login_mode == "auto"
    assert login.command == "login"
    assert login.login_prefix == "codex"
    assert login.login_mode == "auto"


@pytest.mark.parametrize(
    ("flag", "mode"),
    [
        ("--browser", "browser"),
        ("--device", "device"),
        ("--paste", "paste"),
    ],
)
def test_cli_parser_accepts_one_explicit_login_mode(
    tmp_path: Path,
    flag: str,
    mode: str,
) -> None:
    cfg = _load(
        tmp_path,
        argv=("login", "codex", flag),
        env={"HOME": str(tmp_path)},
    )

    assert cfg.command == "login"
    assert cfg.login_prefix == "codex"
    assert cfg.login_mode == mode


@pytest.mark.parametrize(
    "flags",
    [
        ("--browser", "--device"),
        ("--browser", "--paste"),
        ("--device", "--paste"),
    ],
)
def test_cli_parser_rejects_multiple_login_modes(
    tmp_path: Path,
    flags: tuple[str, str],
) -> None:
    with pytest.raises(SystemExit):
        _load(
            tmp_path,
            argv=("login", "codex", *flags),
            env={"HOME": str(tmp_path)},
        )


def test_cli_parser_rejects_obsolete_no_browser_option(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _load(
            tmp_path,
            argv=("login", "codex", "--no-browser"),
            env={"HOME": str(tmp_path)},
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


def test_unset_role_models_remain_unset_when_main_model_is_configured(
    tmp_path: Path,
) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text('[core]\nmodel = "codex:gpt-5.1-codex"\n')

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.model == "codex:gpt-5.1-codex"
    assert cfg.subagent_model is None
    assert cfg.summarizer_model is None


def test_explicit_role_models_are_preserved(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        """
[core]
model = "codex:gpt-5.1-codex"
subagent_model = "anthropic:claude-haiku-4-5"
summarizer_model = "openai:gpt-5-mini"
""".strip()
        + "\n"
    )

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.model == "codex:gpt-5.1-codex"
    assert cfg.subagent_model == "anthropic:claude-haiku-4-5"
    assert cfg.summarizer_model == "openai:gpt-5-mini"


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


def test_trust_is_recomputed_after_environment_changes_cwd(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trusted = tmp_path / "trusted"
    untrusted = tmp_path / "untrusted"
    trusted.mkdir()
    (untrusted / ".orcha-agent/plugins").mkdir(parents=True)
    user_config = tmp_path / "user.toml"
    user_config.write_text(f'[trust]\ndirs = ["{trusted}"]\n')

    cfg = load_config(
        [],
        env={
            "HOME": str(tmp_path),
            "ORCHA_CWD": str(untrusted),
        },
        cwd=trusted,
        user_config_path=user_config,
    )

    assert cfg.cwd == untrusted.resolve()
    assert cfg.trust_cwd is False
    assert "trust" in capsys.readouterr().err.lower()


def test_banner_defaults_to_enabled(tmp_path: Path) -> None:
    assert _load(tmp_path).banner is True


def test_core_config_can_disable_banner(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text("[core]\nbanner = false\n")

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.banner is False


def test_ui_flags_and_per_model_pricing_survive_toml_loading(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        """
[ui]
statusbar = false
icons = false

[pricing."codex:gpt-5.6-sol"]
input = 5
output = 30
cache_read = 0.5

[pricing."anthropic:claude-opus-4-1"]
input = 15
output = 75
""".strip()
        + "\n"
    )

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.statusbar is False
    assert cfg.icons is False
    assert cfg.pricing == {
        "codex:gpt-5.6-sol": {"input": 5, "output": 30, "cache_read": 0.5},
        "anthropic:claude-opus-4-1": {"input": 15, "output": 75},
    }
