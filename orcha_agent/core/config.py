"""Layered configuration for orcha-agent."""

from __future__ import annotations

import argparse
import os
import tomllib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "anthropic:claude-opus-5"
DEFAULT_MEMORY = ("AGENTS.md", "CLAUDE.md")

STATUSLINE_PRESETS = frozenset(
    {"default", "minimal", "compact", "full", "nerd", "ascii"}
)
STATUSLINE_SEPARATORS = frozenset(
    {"powerline", "powerline-thin", "slash", "pipe", "block", "none", "ascii"}
)


@dataclass(frozen=True, slots=True)
class StatusLineConfig:
    preset: str = "default"
    separator: str = "powerline-thin"
    left: tuple[str, ...] | None = None
    right: tuple[str, ...] | None = None
    transparent: bool = False


def _statusline_group(
    value: object,
    name: str,
    parser: argparse.ArgumentParser,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        parser.error(f"[ui.statusline] {name} must be a list of segment names")
    if not all(isinstance(item, str) and item.strip() for item in value):
        parser.error(
            f"[ui.statusline] {name} must contain only non-empty segment names"
        )
    if not all(
        item[0].isalnum()
        and all(char.isalnum() or char in "._-" for char in item)
        for item in value
    ):
        parser.error(
            f"[ui.statusline] {name} contains an invalid segment name"
        )
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        parser.error(f"[ui.statusline] {name} must not contain duplicate names")
    return normalized


def _statusline_config(
    value: object,
    parser: argparse.ArgumentParser,
) -> StatusLineConfig:
    if not isinstance(value, Mapping):
        parser.error("[ui.statusline] must be a TOML table")
    preset = value.get("preset", "default")
    if not isinstance(preset, str) or preset not in STATUSLINE_PRESETS:
        parser.error(
            "[ui.statusline] preset must be default, minimal, compact, full, "
            "nerd, or ascii"
        )
    separator = value.get("separator", "powerline-thin")
    if not isinstance(separator, str) or separator not in STATUSLINE_SEPARATORS:
        parser.error(
            "[ui.statusline] separator must be powerline, powerline-thin, "
            "slash, pipe, block, none, or ascii"
        )
    transparent = value.get("transparent", False)
    if not isinstance(transparent, bool):
        parser.error("[ui.statusline] transparent must be true or false")
    return StatusLineConfig(
        preset=preset,
        separator=separator,
        left=_statusline_group(value.get("left"), "left", parser),
        right=_statusline_group(value.get("right"), "right", parser),
        transparent=transparent,
    )




@dataclass(frozen=True, slots=True)
class Config:
    """Fully resolved application configuration."""

    model: str | list[str]
    subagent_model: str | list[str] | None
    summarizer_model: str | list[str] | None
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
    trusted_dirs: tuple[Path, ...] = ()
    trust_all_cwd: bool = False
    command: str = "repl"
    login_prefix: str | None = None
    login_mode: str = "auto"
    gallery_tool: str | None = None
    gallery_state: str | None = None
    gallery_width: int | None = None
    gallery_expanded: bool = False
    gallery_plain: bool = False
    banner: bool = True
    notify: bool = False
    statusbar: bool = True
    icons: bool = True
    thinking: str = "summary"
    theme: str = "dark"
    symbols: str | None = None
    composer: str = "box"
    statusline: StatusLineConfig = field(default_factory=StatusLineConfig)

    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

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
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("repl", help="start the interactive terminal agent")
    login = subcommands.add_parser("login", help="log in to a provider")
    login.add_argument("prefix")
    modes = login.add_mutually_exclusive_group()
    modes.add_argument("--browser", dest="login_mode", action="store_const", const="browser")
    modes.add_argument("--device", dest="login_mode", action="store_const", const="device")
    modes.add_argument("--paste", dest="login_mode", action="store_const", const="paste")
    login.set_defaults(login_mode="auto")
    gallery = subcommands.add_parser(
        "gallery",
        help="render built-in TUI blocks and lifecycle states",
    )
    gallery.add_argument("--tool", metavar="NAME")
    gallery.add_argument(
        "--state",
        choices=("streaming", "progress", "success", "error"),
    )
    gallery.add_argument("--width", type=int, metavar="N")
    gallery.add_argument("--expanded", action="store_true")
    gallery.add_argument("--plain", action="store_true")
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


def normalize_model_spec(value: Any) -> str | list[str]:
    """Normalize one model spec or a comma-separated fallback chain."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if parts:
            return parts[0] if len(parts) == 1 else parts
    elif isinstance(value, list) and value and all(
        isinstance(item, str) and item for item in value
    ):
        return list(value)
    raise ValueError("model must be a prefix:model string or fallback list")


def _model_spec(value: Any, parser: argparse.ArgumentParser) -> str | list[str]:
    try:
        return normalize_model_spec(value)
    except ValueError as exc:
        parser.error(str(exc))


def _trusted_directories(
    user_values: Mapping[str, Any],
    home: Path,
) -> tuple[Path, ...]:
    trust = user_values.get("trust", {})
    dirs = trust.get("dirs", ()) if isinstance(trust, Mapping) else ()
    if isinstance(dirs, str):
        dirs = (dirs,)
    return tuple(
        _home_path(value, home).resolve()
        for value in dirs
        if isinstance(value, str)
    )


def is_trusted_cwd(
    cwd: str | Path,
    trusted_dirs: Sequence[Path],
    *,
    trust_all: bool = False,
) -> bool:
    if trust_all:
        return True
    resolved = Path(cwd).resolve()
    return any(
        resolved == trusted or resolved.is_relative_to(trusted)
        for trusted in trusted_dirs
    )


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
    launch_dir = Path(args.cwd or cwd or Path.cwd()).resolve()
    user_path = user_config_path or home / ".config/orcha-agent/config.toml"
    user_values = _read_toml(user_path)
    trusted_dirs = _trusted_directories(user_values, home)
    user_core = user_values.get("core", {})
    user_cwd = user_core.get("cwd") if isinstance(user_core, Mapping) else None
    trust_target = _home_path(
        args.cwd or environ.get("ORCHA_CWD") or user_cwd or launch_dir,
        home,
    ).resolve()
    project_path = project_config_path or trust_target / ".orcha-agent/config.toml"
    trust_cwd = is_trusted_cwd(
        trust_target,
        trusted_dirs,
        trust_all=args.trust_cwd,
    )
    project_values: dict[str, Any] = {}
    if trust_cwd:
        project_values = _read_toml(project_path)
    elif project_path.is_file() or (trust_target / ".orcha-agent/plugins").is_dir():
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
    subagent_model = (
        _model_spec(core["subagent_model"], parser)
        if "subagent_model" in core
        else None
    )
    summarizer_model = (
        _model_spec(core["summarizer_model"], parser)
        if "summarizer_model" in core
        else None
    )
    resolved_cwd = _home_path(core.get("cwd", trust_target), home).resolve()
    trust_cwd = is_trusted_cwd(
        resolved_cwd,
        trusted_dirs,
        trust_all=args.trust_cwd,
    )
    db_path = _home_path(core.get("db_path", "~/.local/share/orcha-agent/sessions.db"), home)

    models = values.get("models", {})
    providers = values.get("providers", {})
    plugins = values.get("plugins", {})
    ui = values.get("ui", {})
    pricing = values.get("pricing", {})
    if not all(
        isinstance(table, dict)
        for table in (models, providers, plugins, ui, pricing)
    ):
        _parser().error(
            "models, providers, plugins, ui, and pricing must be TOML tables"
        )
    thinking = str(ui.get("thinking", "summary"))
    if thinking not in {"summary", "off", "all"}:
        parser.error("[ui] thinking must be summary, off, or all")
    theme = ui.get("theme", "dark")
    if not isinstance(theme, str) or not theme.strip():
        parser.error("[ui] theme must be a non-empty string")
    explicit_symbols = ui.get("symbols")
    if "symbols" not in ui:
        symbols = None if bool(ui.get("icons", True)) else "ascii"
    elif isinstance(explicit_symbols, str) and explicit_symbols in {
        "unicode",
        "nerd",
        "ascii",
    }:
        symbols = explicit_symbols
    else:
        parser.error("[ui] symbols must be unicode, nerd, or ascii")
    composer = ui.get("composer", "box")
    if not isinstance(composer, str) or composer not in {
        "box",
        "claude",
        "borderless",
    }:
        parser.error("[ui] composer must be box, claude, or borderless")
    statusline = _statusline_config(ui.get("statusline", {}), parser)
    if "banner" in ui and not isinstance(ui["banner"], bool):
        parser.error("[ui] banner must be true or false")
    if "notify" in ui and not isinstance(ui["notify"], bool):
        parser.error("[ui] notify must be true or false")
    banner = ui.get("banner", core.get("banner", True))
    notify = ui.get("notify", False)


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
        command=args.command or "repl",
        login_prefix=getattr(args, "prefix", None),
        login_mode=getattr(args, "login_mode", "auto"),
        gallery_tool=getattr(args, "tool", None),
        gallery_state=getattr(args, "state", None),
        gallery_width=getattr(args, "width", None),
        gallery_expanded=getattr(args, "expanded", False),
        gallery_plain=getattr(args, "plain", False),
        banner=bool(banner),
        notify=bool(notify),
        statusbar=bool(ui.get("statusbar", True)),
        icons=bool(ui.get("icons", True)),
        thinking=thinking,
        theme=theme,
        symbols=symbols,
        composer=composer,
        statusline=statusline,
        pricing={
            str(model_name): {
                str(key): float(value)
                for key, value in price.items()
                if isinstance(value, (int, float))
            }
            for model_name, price in pricing.items()
            if isinstance(price, Mapping)
        },
        resume=args.resume,
        list_sessions=args.list_sessions,
        strict_plugins=args.strict_plugins,
        plugin_dirs=plugin_dirs,
        models=dict(models),
        providers={key: dict(value) for key, value in providers.items() if isinstance(value, Mapping)},
        plugins=dict(plugins),
        trust_cwd=trust_cwd,
        model_overridden=args.model is not None,
        trusted_dirs=trusted_dirs,
        trust_all_cwd=args.trust_cwd,
    )
