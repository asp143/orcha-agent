from __future__ import annotations

import json
from pathlib import Path

import pytest
from prompt_toolkit.styles import Style
from rich.theme import Theme as RichTheme

from orcha_agent.tui.theme import (
    COLOR_TOKENS,
    Theme,
    load_theme_file,
    load_themes,
    select_theme,
)


def _theme_file(path: Path, *, name: str = "custom", **values: object) -> Path:
    path.write_text(json.dumps({"name": name, **values}), encoding="utf-8")
    return path


def test_theme_parses_all_color_forms_and_exposes_complete_style_adapters(
    tmp_path: Path,
) -> None:
    path = _theme_file(
        tmp_path / "forms.json",
        vars={"base": "#AABBCC", "nested": "$base", "palette": 42},
        colors={
            "accent": "$nested",
            "border": 17,
            "text": "",
            "success": "#010203",
        },
        symbols={
            "preset": "unicode",
            "overrides": {"status.success": "OK"},
        },
    )
    fallback = Theme(
        id="fallback",
        name="Fallback",
        colors={token: "#ffffff" for token in COLOR_TOKENS},
        symbols={},
    )

    theme = load_theme_file(path, fallback=fallback, warn=lambda _message: None)

    assert theme.id == "forms"
    assert theme.name == "custom"
    assert theme.colors["accent"] == "#aabbcc"
    assert theme.colors["border"] == "color(17)"
    assert theme.colors["text"] == "default"
    assert theme.colors["success"] == "#010203"
    assert set(theme.colors) == set(COLOR_TOKENS)
    assert theme.symbols["status.success"] == "OK"
    assert theme.symbols["status.error"]
    assert isinstance(theme.rich, RichTheme)
    assert isinstance(theme.pt, Style)
    assert theme.rich.styles["accent"].color is not None
    assert theme.pt.get_attrs_for_style_str("class:accent").color == "aabbcc"


@pytest.mark.parametrize(
    ("vars", "colors", "match"),
    [
        ({}, {"accent": "$missing"}, "unknown variable"),
        ({"one": "$two", "two": "$one"}, {"accent": "$one"}, "cycle"),
        ({"bad": 256}, {"accent": "$bad"}, "0..255"),
        ({}, {"accent": "blue"}, "color"),
    ],
)
def test_theme_rejects_invalid_color_references(
    tmp_path: Path,
    vars: dict[str, object],
    colors: dict[str, object],
    match: str,
) -> None:
    path = _theme_file(tmp_path / "invalid.json", vars=vars, colors=colors)

    with pytest.raises(ValueError, match=match):
        load_theme_file(path)


def test_missing_tokens_fall_back_with_exactly_one_warning(tmp_path: Path) -> None:
    warnings: list[str] = []
    fallback = Theme(
        id="dark",
        name="Dark",
        colors={token: "#123456" for token in COLOR_TOKENS},
        symbols={},
    )
    path = _theme_file(
        tmp_path / "partial.json",
        colors={"accent": "#abcdef"},
    )

    theme = load_theme_file(path, fallback=fallback, warn=warnings.append)

    assert theme.colors["accent"] == "#abcdef"
    assert theme.colors["border"] == "#123456"
    assert len(warnings) == 1
    assert "partial" in warnings[0]
    assert str(len(COLOR_TOKENS) - 1) in warnings[0]
    assert "border" in warnings[0]


def test_all_packaged_themes_have_complete_color_and_symbol_surfaces() -> None:
    themes = load_themes(home=Path("/nonexistent"), cwd=Path("/nonexistent"))

    assert set(themes) == {"dark", "light", "ansi", "dracula", "nord", "gruvbox"}
    for theme in themes.values():
        assert set(theme.colors) == set(COLOR_TOKENS)
        assert theme.symbols["status.success"]
        assert theme.symbols["icon.model"]


@pytest.mark.parametrize(
    ("colorfgbg", "expected"),
    [("15;0", "dark"), ("0;15", "light"), ("0;255", "light"), (None, "dark"), ("oops", "dark")],
)
def test_auto_theme_uses_background_palette(
    colorfgbg: str | None,
    expected: str,
) -> None:
    themes = load_themes(home=Path("/nonexistent"), cwd=Path("/nonexistent"))
    env = {} if colorfgbg is None else {"COLORFGBG": colorfgbg}

    assert select_theme(themes, "auto", environ=env).id == expected


def test_discovery_precedence_and_bad_optional_theme_isolation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    user_dir = home / ".config/orcha-agent/themes"
    project_dir = cwd / ".orcha-agent/themes"
    user_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    _theme_file(
        user_dir / "dark.json",
        name="User Dark",
        colors={"accent": "#111111"},
    )
    _theme_file(
        project_dir / "dark.json",
        name="Project Dark",
        colors={"accent": "#222222"},
    )
    (project_dir / "broken.json").write_text("{not json", encoding="utf-8")
    warnings: list[str] = []

    untrusted = load_themes(home=home, cwd=cwd, trusted=False, warn=warnings.append)
    trusted = load_themes(home=home, cwd=cwd, trusted=True, warn=warnings.append)

    assert untrusted["dark"].name == "User Dark"
    assert untrusted["dark"].colors["accent"] == "#111111"
    assert trusted["dark"].name == "Project Dark"
    assert trusted["dark"].colors["accent"] == "#222222"
    assert "broken" not in trusted
    assert sum("broken.json" in warning for warning in warnings) == 1
