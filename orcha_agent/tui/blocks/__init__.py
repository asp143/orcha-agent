"""Pure Rich renderers for transcript and HUD blocks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rich import box

from orcha_agent.tui.frame import Block

BlockRenderer = Callable[[Block, Any, int, int, bool], Any]

DEFAULT_THEME: dict[str, Any] = {
    "id": "default",
    "colors": {
        "accent": "cyan",
        "border": "bright_black",
        "dim": "bright_black",
        "error": "red",
        "muted": "bright_black",
        "success": "green",
        "text": "default",
        "thinkingOff": "cyan",
        "thinkingText": "bright_black",
        "toolDiffAdded": "green",
        "toolDiffContext": "bright_black",
        "toolDiffRemoved": "red",
        "toolErrorBg": "grey15",
        "toolOutput": "default",
        "toolPendingBg": "grey11",
        "toolSuccessBg": "grey15",
        "toolTitle": "cyan",
        "userMessageBg": "grey19",
        "userMessageText": "default",
        "warning": "yellow",
    },
    "symbols": {"boxRound": box.ROUNDED},
}


def theme_value(theme: Any, name: str, default: Any = "default") -> Any:
    """Read a token from mapping- or attribute-shaped theme fixtures."""

    if theme is None:
        theme = DEFAULT_THEME
    if isinstance(theme, Mapping):
        colors = theme.get("colors", {})
        if isinstance(colors, Mapping) and name in colors:
            return colors[name]
        return theme.get(name, default)
    colors = getattr(theme, "colors", None)
    if isinstance(colors, Mapping) and name in colors:
        return colors[name]
    return getattr(theme, name, default)


def theme_symbol(theme: Any, name: str, default: Any) -> Any:
    if theme is None:
        theme = DEFAULT_THEME
    symbols = theme.get("symbols", {}) if isinstance(theme, Mapping) else getattr(theme, "symbols", {})
    if isinstance(symbols, Mapping):
        return symbols.get(name, default)
    return getattr(symbols, name, default)


def theme_spinner(
    theme: Any,
    name: str,
    frame: int,
    default: Sequence[str],
) -> str:
    frames = theme_symbol(theme, name, default)
    if not isinstance(frames, (str, Sequence)) or not frames:
        frames = default
    return str(frames[frame % len(frames)])


def theme_id(theme: Any) -> str:
    if theme is None:
        return "default"
    if isinstance(theme, Mapping):
        return str(theme.get("id", theme.get("name", "mapping")))
    return str(getattr(theme, "id", getattr(theme, "name", id(theme))))


class BlockRendererDispatcher:
    """Resolve and memoize renderers without coupling them to the runtime."""

    def __init__(self, renderers: Mapping[str, BlockRenderer] | Sequence[Any]) -> None:
        if isinstance(renderers, Mapping):
            self._renderers = dict(renderers)
        else:
            self._renderers = {entry.kind: entry.render for entry in renderers}
        self._cache: dict[
            tuple[str, str, int],
            dict[tuple[int, int, bool, str], Any],
        ] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def render(
        self,
        block: Block,
        theme: Any,
        width: int,
        budget_rows: int,
        expanded: bool,
    ) -> Any:
        renderer = self._renderers.get(block.kind)
        if renderer is None:
            if block.kind == "raw":
                return block.data.get("renderable", "")
            return block.data.get("text", block.data.get("message", str(block.data)))
        partition = (block.id, block.kind, budget_rows)
        key = (block.revision, width, expanded, theme_id(theme))
        cache = self._cache.setdefault(partition, {})
        if key not in cache:
            cache.clear()
            cache[key] = renderer(block, theme, width, budget_rows, expanded)
        return cache[key]


from .assistant import render as render_assistant
from .banner import render as render_banner
from .diff import render as render_diff
from .hud import render_subagents, render_todo
from .marker import render as render_marker
from .thinking import render as render_thinking
from .tool import render as render_tool
from .user import render as render_user

DEFAULT_RENDERERS: dict[str, BlockRenderer] = {
    "user": render_user,
    "assistant": render_assistant,
    "thinking": render_thinking,
    "tool": render_tool,
    "diff": render_diff,
    "banner": render_banner,
    "marker": render_marker,
    "todo": render_todo,
    "subagents": render_subagents,
}

__all__ = [
    "BlockRenderer",
    "BlockRendererDispatcher",
    "DEFAULT_RENDERERS",
    "DEFAULT_THEME",
    "theme_id",
    "theme_symbol",
    "theme_value",
    "theme_spinner",
]
