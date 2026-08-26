"""Layered configuration for orcha-agent."""

from __future__ import annotations

import argparse
import os
import tomllib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "anthropic:claude-opus-5"
DEFAULT_MEMORY = ("AGENTS.md", "CLAUDE.md")


@dataclass(frozen=True, slots=True)
class Config:
    """Fully resolved application configuration."""

    model: str | list[str]
    subagent_model: str | list[str]
    summarizer_model: str | list[str]
    mode: str
    backend: str
    memory: tuple[str, ...]
    db_path: Path
    cwd: Path
    resume: str | None
    list_sessions: bool
    strict_plugins: bool
    plugin_dirs: tuple[Path, ...]
    models: dict[str, str | list[str]]
    providers: dict[str, dict[str, Any]]
    plugins: dict[str, Any]
    trust_cwd: bool = False
    model_overridden: bool = False

    def plugin_config(self, name: str) -> Mapping[str, Any]:
        value = self.plugins.get(name, {})
        return value if isinstance(value, dict) else {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orcha", description="Pluggable terminal coding agent")
    parser.add_argument("--model", metavar="PREFIX:MODEL")
    parser.add_argument("--mode")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--resume", metavar="ID")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--strict-plugins", action="store_true")
    parser.add_argument("--plugin-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--trust-cwd",
        action="store_true",
        help="trust project config and plugins in the working directory",
    )
    return parser


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _home_path(value: str | Path, home: Path) -> Path:
    text = str(value)
    if text == "~":
        return home
    if text.startswith("~/"):
        return home / text[2:]
    return Path(text)


def _memory(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
        return tuple(value)
    return DEFAULT_MEMORY


def _model_spec(value: Any, parser: argparse.ArgumentParser) -> str | list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            parser.error("model must be a non-empty prefix:model string")
        return parts[0] if len(parts) == 1 else parts
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item for item in value
    ):
        return list(value)
    parser.error("model must be a prefix:model string or fallback list")


def _trusted_cwd(
    cwd: Path,
    user_values: Mapping[str, Any],
    *,
    home: Path,
    cli_trust: bool,
) -> bool:
    if cli_trust:
        return True
    trust = user_values.get("trust", {})
    dirs = trust.get("dirs", ()) if isinstance(trust, Mapping) else ()
    if isinstance(dirs, str):
        dirs = (dirs,)
    for value in dirs:
        if not isinstance(value, str):
            continue
        trusted = _home_path(value, home).resolve()
        if cwd == trusted or cwd.is_relative_to(trusted):
            return True
    return False


def load_config(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    user_config_path: Path | None = None,
    project_config_path: Path | None = None,
) -> Config:
    """Load defaults, user/project TOML, environment, then CLI overrides."""

    parser = _parser()
    args = parser.parse_args(argv)
    environ = os.environ if env is None else env
    home = Path(environ.get("HOME", str(Path.home())))
    working_dir = Path(args.cwd or cwd or Path.cwd()).resolve()
    user_path = user_config_path or home / ".config/orcha-agent/config.toml"
    project_path = project_config_path or working_dir / ".orcha-agent/config.toml"

    user_values = _read_toml(user_path)
    trust_cwd = _trusted_cwd(
        working_dir,
        user_values,
        home=home,
        cli_trust=args.trust_cwd,
    )
    project_values: dict[str, Any] = {}
    if trust_cwd:
        project_values = _read_toml(project_path)
    elif project_path.is_file() or (working_dir / ".orcha-agent/plugins").is_dir():
        print(
            "Skipping untrusted project config; use --trust-cwd or add cwd to [trust] dirs.",
            file=sys.stderr,
        )
    values = _merge(user_values, project_values)
    core = dict(values.get("core", {}))
    env_to_core = {
        "ORCHA_MODEL": "model",
        "ORCHA_SUBAGENT_MODEL": "subagent_model",
        "ORCHA_SUMMARIZER_MODEL": "summarizer_model",
        "ORCHA_MODE": "mode",
        "ORCHA_BACKEND": "backend",
        "ORCHA_MEMORY": "memory",
        "ORCHA_DB_PATH": "db_path",
        "ORCHA_CWD": "cwd",
    }
    for env_name, key in env_to_core.items():
        if env_name in environ:
            core[key] = environ[env_name]

    if args.model is not None:
        core["model"] = args.model
    if args.mode is not None:
        core["mode"] = args.mode
    if args.cwd is not None:
        core["cwd"] = args.cwd

    model = _model_spec(core.get("model", DEFAULT_MODEL), parser)
    subagent_model = _model_spec(core.get("subagent_model", model), parser)
    summarizer_model = _model_spec(core.get("summarizer_model", model), parser)
    resolved_cwd = _home_path(core.get("cwd", working_dir), home).resolve()
    db_path = _home_path(core.get("db_path", "~/.local/share/orcha-agent/sessions.db"), home)

    models = values.get("models", {})
    providers = values.get("providers", {})
    plugins = values.get("plugins", {})
    if not isinstance(models, dict) or not isinstance(providers, dict) or not isinstance(plugins, dict):
        _parser().error("models, providers, and plugins must be TOML tables")

    plugin_dirs = tuple(_home_path(path, home).resolve() for path in args.plugin_dir)
    return Config(
        model=model,
        subagent_model=subagent_model,
        summarizer_model=summarizer_model,
        mode=str(core.get("mode", "ask")),
        backend=str(core.get("backend", "local_shell")),
        memory=_memory(core.get("memory", DEFAULT_MEMORY)),
        db_path=db_path,
        cwd=resolved_cwd,
        resume=args.resume,
        list_sessions=args.list_sessions,
        strict_plugins=args.strict_plugins,
        plugin_dirs=plugin_dirs,
        models=dict(models),
        providers={key: dict(value) for key, value in providers.items() if isinstance(value, Mapping)},
        plugins=dict(plugins),
        trust_cwd=trust_cwd,
        model_overridden=args.model is not None,
    )
