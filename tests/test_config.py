from dataclasses import fields
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


def test_thinking_display_defaults_to_summary(tmp_path: Path) -> None:
    assert _load(tmp_path).thinking == "summary"


def test_thinking_display_rejects_unknown_mode(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text('[ui]\nthinking = "verbose"\n')

    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=user_config)


def test_core_config_can_disable_banner(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text("[core]\nbanner = false\n")

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.banner is False


def test_ui_banner_overrides_legacy_core_and_notify_defaults_off(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text("[core]\nbanner = false\n\n[ui]\nbanner = true\n")

    cfg = _load(tmp_path, user_config_path=user_config)

    assert cfg.banner is True
    assert cfg.notify is False


def test_ui_can_enable_notifications(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text("[ui]\nnotify = true\n")

    assert _load(tmp_path, user_config_path=user_config).notify is True


@pytest.mark.parametrize("name", ["banner", "notify"])
def test_stage7_ui_flags_require_toml_booleans(tmp_path: Path, name: str) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text(f'[ui]\n{name} = "yes"\n')

    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=user_config)


def test_ui_flags_and_per_model_pricing_survive_toml_loading(tmp_path: Path) -> None:
    user_config = tmp_path / "user.toml"
    user_config.write_text(
        """
[ui]
statusbar = false
icons = false
thinking = "all"

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
    assert cfg.thinking == "all"
    assert cfg.pricing == {
        "codex:gpt-5.6-sol": {"input": 5, "output": 30, "cache_read": 0.5},
        "anthropic:claude-opus-4-1": {"input": 15, "output": 75},
    }


def test_yolo_flag_is_shorthand_for_mode_yolo(tmp_path: Path) -> None:
    assert _load(tmp_path, argv=("--yolo",)).mode == "yolo"


def test_yolo_flag_agrees_with_explicit_mode(tmp_path: Path) -> None:
    assert _load(tmp_path, argv=("--yolo", "--mode", "yolo")).mode == "yolo"


def test_yolo_flag_conflicting_with_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _load(tmp_path, argv=("--yolo", "--mode", "ask"))


def test_config_records_user_config_path(tmp_path: Path) -> None:
    user_path = tmp_path / "user.toml"
    assert _load(tmp_path, user_config_path=user_path).user_config_path == user_path


def test_save_core_value_creates_file_and_core_table(tmp_path: Path) -> None:
    from orcha_agent.core.config import save_core_value

    path = tmp_path / "nested" / "config.toml"
    save_core_value(path, "model", "codex:gpt-5.6-sol")
    assert path.read_text() == '[core]\nmodel = "codex:gpt-5.6-sol"\n'
    assert _load(tmp_path, user_config_path=path).model == "codex:gpt-5.6-sol"


def test_save_core_value_replaces_existing_key_and_keeps_everything_else(
    tmp_path: Path,
) -> None:
    from orcha_agent.core.config import save_core_value

    path = tmp_path / "config.toml"
    path.write_text(
        "# top comment\n"
        "[core]\n"
        "mode = \"edit\"\n"
        "model = \"old:model\"  # trailing\n"
        "\n"
        "[ui]\n"
        "icons = false\n"
    )
    save_core_value(path, "model", ["a:x", "b:y"])
    assert path.read_text() == (
        "# top comment\n"
        "[core]\n"
        "mode = \"edit\"\n"
        'model = ["a:x", "b:y"]\n'
        "\n"
        "[ui]\n"
        "icons = false\n"
    )
    loaded = _load(tmp_path, user_config_path=path)
    assert loaded.model == ["a:x", "b:y"]
    assert loaded.mode == "edit"
    assert loaded.icons is False


def test_save_core_value_appends_core_table_when_missing(tmp_path: Path) -> None:
    from orcha_agent.core.config import save_core_value

    path = tmp_path / "config.toml"
    path.write_text("[ui]\nicons = false\n")
    save_core_value(path, "model", "x:y")
    assert path.read_text() == '[ui]\nicons = false\n\n[core]\nmodel = "x:y"\n'
    assert _load(tmp_path, user_config_path=path).model == "x:y"

def test_ui_theme_and_symbols_parse_with_icon_compatibility(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(
        '[ui]\ntheme = "nord"\nsymbols = "unicode"\nicons = false\n'
    )
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[ui]\nicons = false\n')

    explicit_cfg = _load(tmp_path, user_config_path=explicit)
    legacy_cfg = _load(tmp_path, user_config_path=legacy)

    assert explicit_cfg.theme == "nord"
    assert explicit_cfg.symbols == "unicode"
    assert explicit_cfg.icons is False
    assert legacy_cfg.symbols == "ascii"


def test_ui_theme_and_symbols_have_maintainable_defaults(tmp_path: Path) -> None:
    cfg = _load(tmp_path)

    assert cfg.theme == "dark"
    assert cfg.symbols is None


@pytest.mark.parametrize("shape", ["box", "claude", "borderless"])
def test_ui_composer_accepts_supported_shapes(tmp_path: Path, shape: str) -> None:
    config = tmp_path / "ui.toml"
    config.write_text(f'[ui]\ncomposer = "{shape}"\n', encoding="utf-8")
    assert _load(tmp_path, user_config_path=config).composer == shape


def test_ui_composer_rejects_unknown_shape(tmp_path: Path) -> None:
    config = tmp_path / "ui.toml"
    config.write_text('[ui]\ncomposer = "floating"\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=config)

@pytest.mark.parametrize("value", ['["box"]', '{ shape = "box" }'])
def test_ui_composer_rejects_non_string_values(
    tmp_path: Path,
    value: str,
) -> None:
    config = tmp_path / "ui.toml"
    config.write_text(f"[ui]\ncomposer = {value}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=config)


def test_ui_statusline_defaults_are_maintainable(tmp_path: Path) -> None:
    statusline = _load(tmp_path).statusline
    assert statusline.preset == "default"
    assert statusline.separator == "powerline-thin"
    assert statusline.left is None
    assert statusline.right is None
    assert statusline.transparent is False


def test_ui_statusline_groups_and_style_survive_toml_loading(tmp_path: Path) -> None:
    config = tmp_path / "ui.toml"
    config.write_text(
        """
[ui.statusline]
preset = "full"
separator = "slash"
left = ["model", "path", "plugin.custom"]
right = ["context", "cost"]
transparent = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    statusline = _load(tmp_path, user_config_path=config).statusline
    assert statusline.preset == "full"
    assert statusline.separator == "slash"
    assert statusline.left == ("model", "path", "plugin.custom")
    assert statusline.right == ("context", "cost")
    assert statusline.transparent is True


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('preset = "wide"', "preset"),
        ('separator = "dots"', "separator"),
        ('left = "model"', "left"),
        ('right = ["model", 4]', "right"),
        ('left = [""]', "left"),
        ('left = ["bad name"]', "left"),
        ('transparent = "yes"', "transparent"),
    ],
)
def test_ui_statusline_rejects_invalid_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
    message: str,
) -> None:
    config = tmp_path / "ui.toml"
    config.write_text(f"[ui.statusline]\n{body}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=config)
    assert message in capsys.readouterr().err


def test_persistence_and_memory_store_defaults_preserve_sqlite_behavior(
    tmp_path: Path,
) -> None:
    cfg = _load(tmp_path, env={"HOME": str(tmp_path)})

    expected_db_path = tmp_path / ".local/share/orcha-agent/sessions.db"
    assert cfg.db_path == expected_db_path
    assert cfg.memory == ("AGENTS.md", "CLAUDE.md")
    assert cfg.persistence.backend == "sqlite"
    assert cfg.persistence.replica_path == expected_db_path
    assert cfg.persistence.url is None
    assert cfg.persistence.sync_on_start is True
    assert cfg.persistence.sync_on_exit is True
    assert cfg.memory_store.backend == "files"
    assert cfg.memory_store.workspace is None


def test_turso_uses_a_distinct_replica_path_without_uploading_legacy_sqlite(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[core]\ndb_path = "~/sessions.db"\n\n'
        '[persistence]\nbackend = "turso"\n',
        encoding="utf-8",
    )

    cfg = _load(
        tmp_path,
        env={"HOME": str(tmp_path)},
        user_config_path=config,
    )

    assert cfg.db_path == tmp_path / "sessions.db"
    assert cfg.persistence.replica_path == (
        tmp_path / ".local/share/orcha-agent/turso-replica.db"
    )
    assert cfg.persistence.replica_path != cfg.db_path



def test_turso_persistence_and_structured_memory_parse_from_toml(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[core]
memory = ["PROJECT.md"]
db_path = "~/legacy-sessions.db"

[persistence]
backend = "turso"
replica_path = "~/.cache/orcha/replica.db"
url = "libsql://example.turso.io"
sync_on_start = false
sync_on_exit = true

[memory_store]
backend = "hybrid"
workspace = "orcha-agent"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = _load(
        tmp_path,
        env={"HOME": str(tmp_path)},
        user_config_path=config,
    )

    assert cfg.memory == ("PROJECT.md",)
    assert cfg.db_path == tmp_path / "legacy-sessions.db"
    assert cfg.persistence.backend == "turso"
    assert cfg.persistence.replica_path == tmp_path / ".cache/orcha/replica.db"
    assert cfg.persistence.url == "libsql://example.turso.io"
    assert cfg.persistence.sync_on_start is False
    assert cfg.persistence.sync_on_exit is True
    assert cfg.memory_store.backend == "hybrid"
    assert cfg.memory_store.workspace == "orcha-agent"


def test_persistence_and_memory_store_environment_overrides_toml(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[persistence]
backend = "sqlite"
replica_path = "from-toml.db"
sync_on_start = true
sync_on_exit = false

[memory_store]
backend = "files"
workspace = "from-toml"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = _load(
        tmp_path,
        env={
            "HOME": str(tmp_path),
            "ORCHA_PERSISTENCE_BACKEND": "turso",
            "ORCHA_PERSISTENCE_REPLICA_PATH": "~/env-replica.db",
            "ORCHA_PERSISTENCE_URL": "libsql://env.turso.io",
            "ORCHA_PERSISTENCE_SYNC_ON_START": "off",
            "ORCHA_PERSISTENCE_SYNC_ON_EXIT": "yes",
            "ORCHA_MEMORY_STORE_BACKEND": "turso",
            "ORCHA_MEMORY_STORE_WORKSPACE": "env-memory",
        },
        user_config_path=config,
    )

    assert cfg.persistence.backend == "turso"
    assert cfg.persistence.replica_path == tmp_path / "env-replica.db"
    assert cfg.persistence.url == "libsql://env.turso.io"
    assert cfg.persistence.sync_on_start is False
    assert cfg.persistence.sync_on_exit is True
    assert cfg.memory_store.backend == "turso"
    assert cfg.memory_store.workspace == "env-memory"


def test_legacy_environment_keys_remain_distinct_from_new_store_settings(
    tmp_path: Path,
) -> None:
    cfg = _load(
        tmp_path,
        env={
            "HOME": str(tmp_path),
            "ORCHA_BACKEND": "remote_shell",
            "ORCHA_MEMORY": "ONE.md,TWO.md",
            "ORCHA_DB_PATH": "~/custom-sessions.db",
        },
    )

    assert cfg.backend == "remote_shell"
    assert cfg.memory == ("ONE.md", "TWO.md")
    assert cfg.db_path == tmp_path / "custom-sessions.db"
    assert cfg.persistence.backend == "sqlite"
    assert cfg.persistence.replica_path == cfg.db_path
    assert cfg.memory_store.backend == "files"


def test_trusted_project_store_tables_deep_merge_over_user_config(
    tmp_path: Path,
) -> None:
    user_config = tmp_path / "user.toml"
    project_config = tmp_path / "project.toml"
    user_config.write_text(
        """
[persistence]
backend = "turso"
url = "libsql://user.turso.io"
sync_on_start = false

[memory_store]
backend = "files"
workspace = "user-memory"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    project_config.write_text(
        """
[persistence]
url = "libsql://project.turso.io"
sync_on_exit = false

[memory_store]
backend = "hybrid"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = _load(
        tmp_path,
        argv=("--trust-cwd",),
        user_config_path=user_config,
        project_config_path=project_config,
    )

    assert cfg.persistence.backend == "turso"
    assert cfg.persistence.url == "libsql://project.turso.io"
    assert cfg.persistence.sync_on_start is False
    assert cfg.persistence.sync_on_exit is False
    assert cfg.memory_store.backend == "hybrid"
    assert cfg.memory_store.workspace == "user-memory"


def test_sync_is_a_top_level_command(tmp_path: Path) -> None:
    cfg = _load(tmp_path, argv=("sync",))

    assert cfg.command == "sync"
    assert cfg.login_prefix is None
    assert cfg.login_mode == "auto"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('persistence = "turso"', "persistence must be a TOML table"),
        ('memory_store = "files"', "memory_store must be a TOML table"),
        ('[persistence]\nbackend = "postgres"', "backend"),
        ('[persistence]\nreplica_path = ""', "replica_path"),
        ('[persistence]\nurl = ""', "url"),
        ('[persistence]\nsync_on_start = "false"', "sync_on_start"),
        ('[persistence]\nsync_on_exit = 1', "sync_on_exit"),
        ('[memory_store]\nbackend = "sqlite"', "backend"),
        ('[memory_store]\nworkspace = ""', "workspace"),
        ('[memory_store]\nworkspace = 3', "workspace"),
    ],
)
def test_persistence_and_memory_store_reject_invalid_toml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
    message: str,
) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(body + "\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        _load(tmp_path, user_config_path=config)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "name",
    [
        "ORCHA_PERSISTENCE_SYNC_ON_START",
        "ORCHA_PERSISTENCE_SYNC_ON_EXIT",
    ],
)
def test_persistence_rejects_invalid_environment_booleans(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    with pytest.raises(SystemExit):
        _load(tmp_path, env={name: "sometimes"})
    assert name in capsys.readouterr().err


def test_turso_auth_tokens_are_never_copied_into_config(tmp_path: Path) -> None:
    secret = "secret-that-must-not-be-stored"
    cfg = _load(
        tmp_path,
        env={
            "TURSO_AUTH_TOKEN": secret,
            "ORCHA_TURSO_AUTH_TOKEN": secret,
        },
    )

    assert "auth_token" not in {item.name for item in fields(cfg)}
    assert "auth_token" not in {item.name for item in fields(cfg.persistence)}
    assert secret not in repr(cfg)
