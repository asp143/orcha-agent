"""Shared, bounded syntax highlighting using the active terminal palette."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import PurePath
from typing import Any

from pygments.lexers import find_lexer_class_for_filename
from pygments.plugin import find_plugin_lexers
from pygments.style import Style as PygmentsStyle
from pygments.styles import get_style_by_name
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Token
from pygments.util import ClassNotFound
from rich.color import Color
from rich.syntax import Syntax
from rich.style import Style
from rich.text import Span
from rich.text import Text

from . import theme_value


@lru_cache(maxsize=512)
def language_from_path(path: str) -> str:
    """Infer syntax from a file path, including omp's :N-M line selectors."""
    name = PurePath(re.sub(r":\d+(?:-\d+)?$", "", path)).name
    aliases = {"Dockerfile": "docker", "Makefile": "make", ".gitignore": "text"}
    if name in aliases:
        return aliases[name]
    # Python's ordinary extensions have one built-in filename candidate. Avoid
    # compiling the entire lexer registry's filename globs for every cold process.
    # Ambiguous extensions (for example .h and .m) still use Pygments' ranking.
    if PurePath(name).suffix in {".py", ".pyw", ".pyi"} and not any(
        fnmatchcase(name, pattern) for lexer in find_plugin_lexers() for pattern in lexer.filenames
    ):
        return "python"
    try:
        lexer = find_lexer_class_for_filename(name)
        return lexer.aliases[0] if lexer is not None else "text"
    except (ClassNotFound, IndexError):
        return "text"


def _hex(value: str) -> str:
    if value in {"", "default"}:
        return ""
    try:
        return Color.parse(value).get_truecolor().hex
    except Exception:
        return ""


_TOKENS = {
    Token: "text",
    Comment: "syntaxComment",
    Keyword: "syntaxKeyword",
    Name.Function: "syntaxFunction",
    Name.Variable: "syntaxVariable",
    Name: "syntaxVariable",
    String: "syntaxString",
    Number: "syntaxNumber",
    Keyword.Type: "syntaxType",
    Name.Class: "syntaxType",
    Operator: "syntaxOperator",
    Punctuation: "syntaxPunctuation",
}


@lru_cache(maxsize=64)
def _style(values: tuple[str, ...], background: str) -> type[PygmentsStyle]:
    rgb = Color.parse(background or "#111827").get_truecolor()
    light = (rgb.red * 0.2126 + rgb.green * 0.7152 + rgb.blue * 0.0722) > 127
    base = get_style_by_name("default" if light else "monokai")
    styles = dict(base.styles)
    styles.update({token: color for token, color in zip(_TOKENS, values) if color})
    return type(
        "TerminalPalette",
        (PygmentsStyle,),
        {
            "styles": styles,
            "background_color": background or base.background_color,
        },
    )


def syntax_style(theme: Any) -> Any:
    # Rich accepts Pygments Style classes at runtime; its annotations only name strings.
    return _style(
        tuple(_hex(str(theme_value(theme, key))) for key in _TOKENS.values()),
        _hex(str(theme_value(theme, "userMessageBg"))),
    )


def highlight(source: str, language: str, theme: Any) -> Text:
    """Lex a complete preview so multiline strings/comments retain their state."""
    highlighted = Syntax(source, language, theme=syntax_style(theme)).highlight(source)
    highlighted.style = (
        Style(color=highlighted.style.color) if isinstance(highlighted.style, Style) else ""
    )
    highlighted.spans = [
        Span(
            span.start,
            span.end,
            Style(color=span.style.color, bold=span.style.bold, italic=span.style.italic),
        )
        if isinstance(span.style, Style)
        else span
        for span in highlighted.spans
    ]
    return highlighted
