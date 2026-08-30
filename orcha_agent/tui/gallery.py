"""Non-interactive renderer gallery for visual TUI development."""

from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console

from .blocks import DEFAULT_RENDERERS
from .frame import Block
from .gallery_fixtures import GALLERY_FIXTURES, GALLERY_STATES, GalleryState
from .gallery_fixtures.blocks import TOOL_GALLERY_FIXTURES
from .theme import Theme, load_themes, select_theme

_MIN_WIDTH = 40
_MAX_WIDTH = 200


def _width(requested: object) -> int:
    fallback = shutil.get_terminal_size((_MAX_WIDTH // 2, 24)).columns
    value = fallback if requested is None else int(requested)
    return max(_MIN_WIDTH, min(_MAX_WIDTH, value))


def _block(renderer: str, state: GalleryState, tool_name: str | None = None) -> Block:
    fixture = (
        TOOL_GALLERY_FIXTURES[tool_name][state]
        if renderer == "tool" and tool_name is not None
        else GALLERY_FIXTURES[renderer][state]
    )
    suffix = f"-{tool_name}" if tool_name is not None else ""
    return Block(
        id=f"gallery-{renderer}{suffix}-{state}",
        kind=renderer,
        state=fixture.state,
        data=deepcopy(fixture.data),
    )


def _renderable(
    renderer: str,
    state: GalleryState,
    *,
    theme: Theme,
    width: int,
    expanded: bool,
    tool_name: str | None = None,
) -> Any:
    return DEFAULT_RENDERERS[renderer](
        _block(renderer, state, tool_name),
        theme,
        width,
        200,
        expanded,
    )


def _console(file: TextIO, *, width: int, plain: bool) -> Console:
    return Console(
        file=file,
        width=width,
        height=500,
        force_terminal=not plain,
        no_color=plain,
        color_system=None if plain else "truecolor",
        legacy_windows=False,
    )


def render_gallery_state(
    renderer: str,
    state: GalleryState,
    *,
    theme: Theme,
    width: int,
    expanded: bool,
    plain: bool,
) -> str:
    """Render one fixture through the production renderer."""

    stream = StringIO()
    console = _console(stream, width=_width(width), plain=plain)
    for tool_name in _tool_names(renderer):
        if tool_name is not None:
            console.print(f"  · {tool_name}", style="dim")
        renderable = _renderable(
            renderer,
            state,
            theme=theme,
            width=_width(width),
            expanded=expanded,
            tool_name=tool_name,
        )
        if renderable is not None:
            console.print(renderable)
    return stream.getvalue()


def _theme(cfg: object, file: TextIO) -> Theme:
    cwd = Path(getattr(cfg, "cwd", Path.cwd()))
    themes = load_themes(
        cwd=cwd,
        trusted=bool(getattr(cfg, "trust_cwd", False)),
        symbols=getattr(cfg, "symbols", None),
        encoding=getattr(file, "encoding", None),
    )
    requested = str(getattr(cfg, "theme", "dark"))
    return select_theme(themes, requested)


def _tool_names(renderer: str) -> tuple[str | None, ...]:
    return tuple(TOOL_GALLERY_FIXTURES) if renderer == "tool" else (None,)


def run_gallery(cfg: object, *, file: TextIO = sys.stdout) -> int:
    """Print selected renderers and lifecycle states to ``file``."""

    known = sorted(DEFAULT_RENDERERS)
    selected = getattr(cfg, "gallery_tool", None)
    if selected is not None and selected not in DEFAULT_RENDERERS:
        print(
            f"Unknown renderer '{selected}'. Known renderers: {', '.join(known)}",
            file=file,
        )
        return 2

    requested_state = getattr(cfg, "gallery_state", None)
    states = (requested_state,) if requested_state is not None else GALLERY_STATES
    renderers = [selected] if selected is not None else known
    width = _width(getattr(cfg, "gallery_width", None))
    expanded = bool(getattr(cfg, "gallery_expanded", False))
    plain = bool(getattr(cfg, "gallery_plain", False))
    theme = _theme(cfg, file)
    console = _console(file, width=width, plain=plain)

    for index, renderer in enumerate(renderers):
        if index:
            console.print()
        console.rule(
            f"[bold]{renderer}[/]",
            style=theme.colors.get("accent", "cyan"),
        )
        for state in states:
            console.print(f"  · {state}", style="dim")
            for tool_name in _tool_names(renderer):
                if tool_name is not None:
                    console.print(f"    · {tool_name}", style="dim")
                renderable = _renderable(
                    renderer,
                    state,
                    theme=theme,
                    width=width,
                    expanded=expanded,
                    tool_name=tool_name,
                )
                if renderable is not None:
                    console.print(renderable)
    return 0


__all__ = ["render_gallery_state", "run_gallery"]
