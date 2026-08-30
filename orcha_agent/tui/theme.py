"""Concrete themes, JSON parsing, and layered theme discovery."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.styles import Style as PromptToolkitStyle
from rich.theme import Theme as RichTheme

from .symbols import resolve_symbols

COLOR_TOKENS = (
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "selectedBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "statusLineBg",
    "userMessageText",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingMax",
    "bashMode",
    "statusLineSep",
    "statusLineModel",
    "statusLinePath",
    "statusLineGitClean",
    "statusLineGitDirty",
    "statusLineContext",
    "statusLineCost",
    "statusLineSubagents",
)

_THEMES_DIR = Path(__file__).with_name("themes")
_BUILTIN_NAMES = ("dark", "light", "ansi", "dracula", "nord", "gruvbox")
Warn = Callable[[str], None]


def _warn_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _palette_rgb(index: int) -> tuple[int, int, int]:
    base = (
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
        (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
    )
    if index < 16:
        return base[index]
    if index < 232:
        value = index - 16
        steps = (0, 95, 135, 175, 215, 255)
        return (
            steps[value // 36],
            steps[(value % 36) // 6],
            steps[value % 6],
        )
    gray = 8 + (index - 232) * 10
    return gray, gray, gray


def _palette_hex(index: int) -> str:
    return "#{:02x}{:02x}{:02x}".format(*_palette_rgb(index))


def _parse_literal(value: object, *, location: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be a color value")
    if isinstance(value, int):
        if not 0 <= value <= 255:
            raise ValueError(f"{location} palette index must be in 0..255")
        return f"color({value})"
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a color value")
    if value == "":
        return "default"
    if len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
        except ValueError as exc:
            raise ValueError(f"{location} must be a #rrggbb color") from exc
        return value.lower()
    raise ValueError(f"{location} must be #rrggbb, palette 0..255, $var, or empty")


def _resolve_variables(values: Mapping[str, object]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    resolving: list[str] = []

    def resolve(name: str) -> str:
        if name in resolved:
            return resolved[name]
        if name not in values:
            raise ValueError(f"unknown variable ${name}")
        if name in resolving:
            cycle = " -> ".join((*resolving[resolving.index(name):], name))
            raise ValueError(f"theme variable cycle: {cycle}")
        resolving.append(name)
        value = values[name]
        if isinstance(value, str) and value.startswith("$"):
            result = resolve(value[1:])
        else:
            result = _parse_literal(value, location=f"variable {name}")
        resolving.pop()
        resolved[name] = result
        return result

    for key in sorted(values):
        if not isinstance(key, str) or not key:
            raise ValueError("theme variable names must be non-empty strings")
        resolve(key)
    return resolved


def _resolve_color(value: object, variables: Mapping[str, str], token: str) -> str:
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in variables:
            raise ValueError(f"color {token} references unknown variable ${name}")
        return variables[name]
    return _parse_literal(value, location=f"color {token}")


def _prompt_color(value: str) -> str:
    if value == "default":
        return ""
    if value.startswith("color("):
        return _palette_hex(int(value[6:-1]))
    return value


@dataclass(frozen=True, slots=True)
class Theme:
    """Renderer-ready color and symbol surface with Rich/PT adapters."""

    id: str
    name: str
    colors: Mapping[str, str]
    symbols: Mapping[str, Any]
    rich: RichTheme = field(init=False, repr=False, compare=False)
    pt: PromptToolkitStyle = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        colors = dict(self.colors)
        symbols = dict(self.symbols)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "rich", RichTheme(colors, inherit=True))
        object.__setattr__(
            self,
            "pt",
            PromptToolkitStyle.from_dict(
                {token.lower(): _prompt_color(value) for token, value in colors.items()}
            ),
        )

    def color(self, token: str) -> str:
        return self.colors[token]

    def symbol(self, key: str) -> Any:
        return self.symbols[key]


def _read_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"theme {path} must contain a JSON object")
    return value


def load_theme_file(
    path: str | Path,
    *,
    fallback: Theme | None = None,
    warn: Warn = _warn_stderr,
    symbols: str | None = None,
    encoding: str | None = None,
    theme_id: str | None = None,
    _authoritative: bool = False,
) -> Theme:
    """Load one JSON theme, filling optional tokens from the dark theme."""

    source = Path(path)
    data = _read_object(source)
    identifier = theme_id or source.stem
    name = data.get("name", identifier)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"theme {identifier} name must be a non-empty string")
    raw_vars = data.get("vars", {})
    raw_colors = data.get("colors", {})
    raw_symbols = data.get("symbols", {})
    if not isinstance(raw_vars, Mapping):
        raise ValueError(f"theme {identifier} vars must be an object")
    if not isinstance(raw_colors, Mapping):
        raise ValueError(f"theme {identifier} colors must be an object")
    if not isinstance(raw_symbols, Mapping):
        raise ValueError(f"theme {identifier} symbols must be an object")

    variables = _resolve_variables(raw_vars)
    unknown = sorted(set(raw_colors) - set(COLOR_TOKENS))
    if unknown:
        raise ValueError(f"theme {identifier} has unknown color tokens: {', '.join(unknown)}")
    colors = {
        token: _resolve_color(value, variables, token)
        for token, value in raw_colors.items()
    }
    missing = [token for token in COLOR_TOKENS if token not in colors]
    if missing:
        if fallback is None:
            if _authoritative:
                raise ValueError(
                    f"authoritative dark theme missing {len(missing)} color tokens: "
                    + ", ".join(missing)
                )
            fallback = load_theme_file(
                _THEMES_DIR / "dark.json",
                warn=warn,
                encoding=encoding,
                _authoritative=True,
            )
        colors.update({token: fallback.colors[token] for token in missing})
        warn(
            f"Theme '{identifier}' missing {len(missing)} color tokens; "
            f"using dark fallback: {', '.join(missing)}"
        )

    preset = symbols if symbols is not None else raw_symbols.get("preset", "nerd")
    if not isinstance(preset, str):
        raise ValueError(f"theme {identifier} symbol preset must be a string")
    overrides = raw_symbols.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError(f"theme {identifier} symbol overrides must be an object")
    return Theme(
        id=identifier,
        name=name,
        colors=colors,
        symbols=resolve_symbols(
            preset,
            overrides,
            encoding=encoding,
            warn=warn,
        ),
    )


def _load_optional_directory(
    themes: dict[str, Theme],
    directory: Path,
    *,
    fallback: Theme,
    warn: Warn,
    symbols: str | None,
    encoding: str | None,
) -> None:
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        try:
            theme = load_theme_file(
                path,
                fallback=fallback,
                warn=warn,
                symbols=symbols,
                encoding=encoding,
            )
        except Exception as exc:
            warn(f"Skipping theme {path.name}: {exc}")
            continue
        themes[theme.id] = theme


def load_themes(
    *,
    home: Path | None = None,
    cwd: Path | None = None,
    trusted: bool = False,
    warn: Warn = _warn_stderr,
    symbols: str | None = None,
    encoding: str | None = None,
) -> dict[str, Theme]:
    """Discover built-in, user, then trusted-project themes."""

    terminal_encoding = encoding or getattr(sys.stdout, "encoding", None)
    themes: dict[str, Theme] = {}
    dark = load_theme_file(
        _THEMES_DIR / "dark.json",
        warn=warn,
        symbols=symbols,
        encoding=terminal_encoding,
        _authoritative=True,
    )
    themes["dark"] = dark
    for name in _BUILTIN_NAMES[1:]:
        themes[name] = load_theme_file(
            _THEMES_DIR / f"{name}.json",
            fallback=dark,
            warn=warn,
            symbols=symbols,
            encoding=terminal_encoding,
            _authoritative=True,
        )
    user_home = Path.home() if home is None else home
    project_cwd = Path.cwd() if cwd is None else cwd
    _load_optional_directory(
        themes,
        user_home / ".config/orcha-agent/themes",
        fallback=dark,
        warn=warn,
        symbols=symbols,
        encoding=terminal_encoding,
    )
    if trusted:
        _load_optional_directory(
            themes,
            project_cwd / ".orcha-agent/themes",
            fallback=dark,
            warn=warn,
            symbols=symbols,
            encoding=terminal_encoding,
        )
    return themes


def _auto_theme(environ: Mapping[str, str]) -> str:
    value = environ.get("COLORFGBG", "")
    try:
        background = int(value.split(";")[-1])
        if not 0 <= background <= 255:
            return "dark"
    except ValueError:
        return "dark"
    red, green, blue = _palette_rgb(background)
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "light" if luminance >= 0.5 else "dark"


def select_theme(
    themes: Mapping[str, Theme],
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Theme:
    """Select a named theme, resolving auto from the terminal background."""

    selected = _auto_theme(os.environ if environ is None else environ) if name == "auto" else name
    if selected not in themes:
        raise KeyError(selected)
    return themes[selected]


__all__ = [
    "COLOR_TOKENS",
    "Theme",
    "load_theme_file",
    "load_themes",
    "select_theme",
]
