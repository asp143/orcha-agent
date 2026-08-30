"""Themed segmented status-line rendering and built-in segment producers."""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from prompt_toolkit.utils import get_cwidth

from .symbols import resolve_symbols


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    token: str
    icon_key: str | None = None


PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "default": (
        ("model", "mode", "path", "git", "context", "cost"),
        ("subagents", "session"),
    ),
    "minimal": (("model", "path"), ("context",)),
    "compact": (("mode", "path", "git"), ("context", "time")),
    "full": (
        ("model", "mode", "path", "git", "session"),
        ("subagents", "tokens", "cache", "cost", "context", "time"),
    ),
    "nerd": (
        ("model", "mode", "path", "git", "session"),
        ("subagents", "tokens", "cache", "cost", "context", "time"),
    ),
    "ascii": (
        ("model", "mode", "path", "git"),
        ("subagents", "context", "cost"),
    ),
}

SEPARATORS = frozenset(
    {"powerline", "powerline-thin", "slash", "pipe", "block", "none", "ascii"}
)

WINDOWS = {
    "codex:gpt-5.6-sol": 272_000,
    "codex:gpt-5.6-luna": 272_000,
    "codex:gpt-5.6-terra": 272_000,
    "codex:gpt-5.5": 272_000,
    "codex:gpt-5.4": 272_000,
    "codex:gpt-5.4-mini": 272_000,
    "codex:gpt-5.3-codex-spark": 128_000,
}

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "codex:gpt-5.6-sol": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-luna": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.6-terra": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.5": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4": {"input": 5, "output": 30, "cache_read": 0.5},
    "codex:gpt-5.4-mini": {"input": 1.5, "output": 6, "cache_read": 0.15},
    "codex:gpt-5.3-codex-spark": {
        "input": 1.5,
        "output": 6,
        "cache_read": 0.15,
    },
    "anthropic:claude-opus-5": {"input": 15, "output": 75, "cache_read": 1.5},
    "anthropic:claude-sonnet-5": {"input": 3, "output": 15, "cache_read": 0.3},
    "anthropic:claude-haiku-4-5": {"input": 0.8, "output": 4, "cache_read": 0.08},
}

_GIT_REFRESH_SECONDS = 2.0
_GIT_TIMEOUT_SECONDS = 1.0
_CONTEXT_BAR_CELLS = 20
_GIT_LOCK = threading.Lock()
_USAGE_TRACKERS: deque[tuple[dict[str, Any], deque[Any]]] = deque(maxlen=16)
_GIT_VOLATILE = (
    "_git_at",
    "_git_text",
    "_git_dirty",
    "_git_refreshing",
    "_git_scope",
)


def reset_git_state(state: dict[str, Any]) -> None:
    generation = int(state.get("_git_generation", 0)) + 1
    for key in _GIT_VOLATILE:
        state.pop(key, None)
    state["_git_generation"] = generation
    state["_git_refreshing"] = False


def _usage_tracker(state: dict[str, Any]) -> deque[Any]:
    for tracked_state, chunks in _USAGE_TRACKERS:
        if tracked_state is state:
            return chunks
    chunks: deque[Any] = deque(maxlen=256)
    _USAGE_TRACKERS.append((state, chunks))
    return chunks


def reset_usage_dedup(state: dict[str, Any]) -> None:
    _usage_tracker(state).clear()


def wrap_segment(value: Segment | str | None) -> Segment | None:
    """Normalize plugin output while retaining the legacy string protocol."""

    if value is None or isinstance(value, Segment):
        return value
    if isinstance(value, str):
        return Segment(value, "text")
    raise TypeError(f"status segment returned {type(value).__name__}, expected Segment, str, or None")


def _state(ctx: Any) -> dict[str, Any]:
    return ctx.plugin_states.setdefault("statusbar", {})


def _spec(ctx: Any) -> tuple[str, int]:
    configured = ctx.cfg.model
    specs = configured if isinstance(configured, list) else [configured]
    return str(specs[0]), max(0, len(specs) - 1)


def _display_model(ctx: Any, spec: str) -> str:
    prefix, separator, model_name = spec.partition(":")
    if not separator:
        return spec
    provider = ctx.registry.providers.get(prefix)
    if provider is None or model_name not in provider.models:
        return spec
    if model_name.startswith("gpt-"):
        parts = model_name.split("-")
        return " ".join([f"GPT-{parts[1]}", *(part.capitalize() for part in parts[2:])])
    return " ".join(part.capitalize() for part in model_name.split("-"))


def _configured_thinking(ctx: Any, spec: str) -> str | None:
    prefix = spec.partition(":")[0]
    provider_config = getattr(ctx.cfg, "providers", {}).get(prefix, {})
    value = provider_config.get("reasoning_effort") or provider_config.get("thinking")
    if isinstance(value, Mapping):
        value = value.get("effort") or value.get("type")
    return str(value) if value else None


def _thinking_level(ctx: Any, spec: str) -> str | None:
    ui_level = getattr(getattr(ctx, "ui", None), "thinking_level", None)
    if isinstance(ui_level, str):
        return None if ui_level == "off" else ui_level
    composer = getattr(ctx, "plugin_states", {}).get("composer", {})
    saved = composer.get("thinking_level") if isinstance(composer, Mapping) else None
    if isinstance(saved, str):
        return None if saved == "off" else saved
    return _configured_thinking(ctx, spec)


def model_segment(ctx: Any) -> Segment:
    spec, fallbacks = _spec(ctx)
    pieces = [_display_model(ctx, spec)]
    thinking = _thinking_level(ctx, spec)
    if thinking:
        pieces.append(thinking)
    value = " · ".join(pieces)
    if fallbacks:
        value = f"{value} +{fallbacks}"
    return Segment(value, "statusLineModel", "icon.model")


def mode_segment(ctx: Any) -> Segment:
    mode = str(ctx.cfg.mode)
    return Segment(mode, "error" if mode == "yolo" else "warning", "icon.mode")


def _console_width(ctx: Any) -> int:
    console = getattr(ctx, "console", None)
    nested = getattr(console, "console", None)
    return int(getattr(console, "width", getattr(nested, "width", 80)))


def path_segment(ctx: Any) -> Segment:
    cwd = Path(ctx.cfg.cwd)
    value = f"{cwd.parent.name}/{cwd.name}" if _console_width(ctx) >= 120 else cwd.name
    return Segment(value, "statusLinePath", "icon.path")


def _parse_git(stdout: str) -> str | None:
    lines = stdout.splitlines()
    if not lines or not lines[0].startswith("## "):
        return None
    head = lines[0][3:]
    if head.startswith("No commits yet on "):
        branch = head.removeprefix("No commits yet on ").split("...", 1)[0]
    elif head.startswith("Initial commit on "):
        branch = head.removeprefix("Initial commit on ").split("...", 1)[0]
    elif head.startswith("HEAD (no branch)"):
        branch = "detached"
    else:
        branch = head.split("...", 1)[0].split(" ", 1)[0]
    staged = 0
    unstaged = 0
    untracked = 0
    for line in lines[1:]:
        if line.startswith("??"):
            untracked += 1
            continue
        if not line or line.startswith("!!"):
            continue
        code = f"{line[:2]:2}"
        staged += code[0] not in {" ", "?", "!"}
        unstaged += code[1] not in {" ", "?", "!"}
    suffix = ""
    if staged:
        suffix += f" *{staged}"
    if unstaged:
        suffix += f" !{unstaged}"
    if untracked:
        suffix += f" ?{untracked}"
    return f"{branch}{suffix}"


def _notify_invalidation(ctx: Any) -> None:
    invalidate = getattr(getattr(ctx, "ui", None), "invalidate", None)
    if callable(invalidate):
        invalidate()


def _refresh_git(
    ctx: Any,
    state: dict[str, Any],
    cwd: Path,
    scope: tuple[str, str],
    generation: int,
) -> None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--branch", "--untracked-files=all"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
        value = _parse_git(result.stdout) if result.returncode == 0 else None
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        value = None
    with _GIT_LOCK:
        current_scope = (
            str(getattr(ctx, "session_id", "")),
            str(Path(ctx.cfg.cwd).resolve()),
        )
        if (
            current_scope != scope
            or state.get("_git_scope") != scope
            or state.get("_git_generation") != generation
        ):
            return
        if value is None:
            state.pop("_git_text", None)
            state.pop("_git_dirty", None)
        else:
            state["_git_text"] = value
            state["_git_dirty"] = any(marker in value for marker in (" *", " !", " ?"))
        state["_git_at"] = monotonic()
        state["_git_refreshing"] = False
    _notify_invalidation(ctx)


def git_segment(ctx: Any) -> Segment | None:
    state = _state(ctx)
    cwd = Path(ctx.cfg.cwd).resolve()
    scope = (str(getattr(ctx, "session_id", "")), str(cwd))
    with _GIT_LOCK:
        if state.get("_git_scope") != scope:
            reset_git_state(state)
            state["_git_scope"] = scope
    now = monotonic()
    cached_at = state.get("_git_at")
    stale = (
        not isinstance(cached_at, (int, float))
        or now < cached_at
        or now - cached_at >= _GIT_REFRESH_SECONDS
    )
    if stale:
        with _GIT_LOCK:
            if not state.get("_git_refreshing"):
                state["_git_refreshing"] = True
                generation = int(state["_git_generation"])
                threading.Thread(
                    target=_refresh_git,
                    args=(ctx, state, cwd, scope, generation),
                    name="orcha-status-git",
                    daemon=True,
                ).start()
    value = state.get("_git_text")
    if not isinstance(value, str) or not value:
        return None
    token = "statusLineGitDirty" if state.get("_git_dirty") else "statusLineGitClean"
    return Segment(value, token, "icon.git")


def session_segment(ctx: Any) -> Segment | None:
    session = getattr(ctx, "session", None)
    get_session = getattr(session, "get", None)
    info = get_session(ctx.session_id) if callable(get_session) else None
    title = getattr(info, "title", None)
    if not title:
        return None
    return Segment(str(title), "text")


def _live_agents(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if not isinstance(value, (list, tuple)):
        return None
    live = {"running", "active", "pending", "working", "starting"}
    count = 0
    for item in value:
        status = item.get("status", "running") if isinstance(item, Mapping) else "running"
        count += str(status).lower() in live
    return count


def subagents_segment(ctx: Any) -> Segment | None:
    ui = getattr(ctx, "ui", None)
    count = _live_agents(getattr(ui, "subagents", None))
    if count is None:
        hud = getattr(ctx, "plugin_states", {}).get("hud", {})
        count = _live_agents(hud.get("subagents") if isinstance(hud, Mapping) else None)
    if count is None:
        transcript = getattr(ctx, "transcript", None)
        frame = getattr(transcript, "frame", None)
        sources = {
            block.source_id
            for block in getattr(frame, "blocks", ())
            if block.source_id
            and block.source_id != "main"
            and getattr(getattr(block, "state", None), "value", None) == "active"
        }
        count = len(sources)
    if not count:
        return None
    return Segment(str(count), "statusLineSubagents", "icon.subagents")


def _quantity(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def tokens_segment(ctx: Any) -> Segment:
    state = _state(ctx)
    value = (
        f"{_quantity(float(state.get('input_tokens', 0)))} in "
        f"{_quantity(float(state.get('output_tokens', 0)))} out"
    )
    return Segment(value, "text", "icon.tokens")


def cache_segment(ctx: Any) -> Segment | None:
    state = _state(ctx)
    if not state.get("cache_known"):
        return None
    reads = _quantity(float(state.get("cache_read_tokens", 0)))
    writes = _quantity(float(state.get("cache_write_tokens", 0)))
    return Segment(f"{reads} read {writes} write", "muted", "icon.tokens")


def _window(ctx: Any, spec: str) -> int | None:
    if spec in WINDOWS:
        return WINDOWS[spec]
    prefix, _, model_name = spec.partition(":")
    lowered = model_name.lower()
    if prefix == "codex":
        if lowered.startswith(("gpt-5.6-", "gpt-5.5", "gpt-5.4")):
            return 272_000
        if "spark" in lowered:
            return 128_000
    if prefix == "anthropic":
        if "opus" in lowered or "sonnet" in lowered:
            return 1_000_000
        if "haiku" in lowered:
            return 200_000
    provider = ctx.registry.providers.get(prefix)
    return None if provider is None else provider.capabilities.max_context


def context_segment(ctx: Any) -> Segment | None:
    spec, _ = _spec(ctx)
    window = _window(ctx, spec)
    if not window:
        return None
    used = int(_state(ctx).get("last_input_tokens", 0))
    percent = min(100.0, max(0.0, used / window * 100))
    return Segment(f"{percent:.1f}%/{_quantity(window)}", "statusLineContext", "icon.context")


def cost_segment(ctx: Any) -> Segment | None:
    spec, _ = _spec(ctx)
    configured = getattr(ctx.cfg, "pricing", {}).get(spec, {})
    price = {**DEFAULT_PRICING.get(spec, {}), **configured}
    if not price:
        return None
    state = _state(ctx)
    inputs = float(state.get("input_tokens", 0))
    outputs = float(state.get("output_tokens", 0))
    reads = float(state.get("cache_read_tokens", 0))
    writes = float(state.get("cache_write_tokens", 0))
    uncached = max(0.0, inputs - reads - writes)
    cost = (
        uncached * float(price.get("input", 0))
        + reads * float(price.get("cache_read", price.get("input", 0)))
        + writes * float(price.get("cache_write", price.get("input", 0)))
        + outputs * float(price.get("output", 0))
    ) / 1_000_000
    return Segment(f"${cost:.2f}", "statusLineCost", "icon.cost")


def time_segment(ctx: Any) -> Segment | None:
    state = _state(ctx)
    started = state.get("_turn_started")
    if isinstance(started, (int, float)):
        elapsed = max(0.0, monotonic() - started)
    else:
        elapsed = state.get("_last_turn_elapsed")
    if not isinstance(elapsed, (int, float)):
        return None
    return Segment(f"{elapsed:.1f}s", "muted", "icon.thinking")


def record_usage(event: Any, state: dict[str, Any]) -> None:
    usage = getattr(event.chunk, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        return
    seen = _usage_tracker(state)
    if any(chunk is event.chunk for chunk in seen):
        return
    seen.append(event.chunk)

    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    details = usage.get("input_token_details", {})
    if not isinstance(details, Mapping):
        details = {}
    read_keys = ("cache_read", "cache_read_input_tokens")
    write_keys = ("cache_creation", "cache_write", "cache_creation_input_tokens")
    cache_read = next((int(details[key]) for key in read_keys if key in details), 0)
    cache_write = next((int(details[key]) for key in write_keys if key in details), 0)
    state["input_tokens"] = int(state.get("input_tokens", 0)) + input_tokens
    state["output_tokens"] = int(state.get("output_tokens", 0)) + output_tokens
    state["cache_read_tokens"] = int(state.get("cache_read_tokens", 0)) + cache_read
    state["cache_write_tokens"] = int(state.get("cache_write_tokens", 0)) + cache_write
    if any(key in details for key in (*read_keys, *write_keys)):
        state["cache_known"] = True
    if getattr(event, "role", "main") == "main":
        state["last_input_tokens"] = input_tokens


def reset_accounting(state: dict[str, Any]) -> None:
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "last_input_tokens",
    ):
        state[name] = 0
    state["cache_known"] = False
    reset_usage_dedup(state)


def record_turn_start(state: dict[str, Any]) -> None:
    state["_turn_started"] = monotonic()


def record_turn_end(state: dict[str, Any]) -> None:
    started = state.pop("_turn_started", None)
    if isinstance(started, (int, float)):
        state["_last_turn_elapsed"] = max(0.0, monotonic() - started)


BUILTIN_SEGMENTS = (
    ("model", model_segment),
    ("mode", mode_segment),
    ("path", path_segment),
    ("git", git_segment),
    ("session", session_segment),
    ("subagents", subagents_segment),
    ("tokens", tokens_segment),
    ("cache", cache_segment),
    ("cost", cost_segment),
    ("context", context_segment),
    ("time", time_segment),
)


def _status_config(ctx: Any) -> Any:
    value = getattr(ctx.cfg, "statusline", None)
    if value is not None:
        return value
    return type(
        "StatusLineDefaults",
        (),
        {
            "preset": "default",
            "separator": "powerline-thin",
            "left": None,
            "right": None,
            "transparent": False,
        },
    )()


def _resolved_names(ctx: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    config = _status_config(ctx)
    preset = config.preset if config.preset in PRESETS else "default"
    preset_left, preset_right = PRESETS[preset]
    left = tuple(config.left) if config.left is not None else preset_left
    right = tuple(config.right) if config.right is not None else preset_right
    if config.left is None and config.right is None:
        builtin_names = {name for name, _render in BUILTIN_SEGMENTS}
        custom = tuple(
            entry.name
            for entry in ctx.registry.status_segments
            if entry.name not in builtin_names
        )
        left = (*left, *custom)
    return left, right


def _registration_map(ctx: Any) -> dict[str, Any]:
    return {entry.name: entry for entry in ctx.registry.status_segments}


def _evaluate(ctx: Any, names: tuple[str, ...]) -> list[tuple[str, Segment]]:
    registrations = _registration_map(ctx)
    values: list[tuple[str, Segment]] = []
    for name in names:
        registration = registrations.get(name)
        if registration is None:
            continue
        try:
            value = wrap_segment(registration.render(ctx))
        except Exception:
            value = Segment(f"!{name}", "error")
        if value is not None:
            values.append((name, value))
    return values


def visible_segments(ctx: Any) -> list[tuple[str, Segment]]:
    """Return effective visible segments in left-to-right display order."""

    if not bool(getattr(ctx.cfg, "statusbar", True)):
        return []
    left, right = _resolved_names(ctx)
    return [*_evaluate(ctx, left), *_evaluate(ctx, right)]


def _terminal_encoding(ctx: Any) -> str | None:
    console = getattr(ctx, "console", None)
    value = getattr(console, "encoding", None)
    if value is None:
        value = getattr(getattr(console, "console", None), "encoding", None)
    return str(value) if value else None


def _ascii_mode(ctx: Any, preset: str) -> bool:
    if preset == "ascii":
        return True
    encoding = _terminal_encoding(ctx)
    if encoding is None:
        return False
    try:
        "✓".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return True
    return False


def _safe_text(value: object, ascii_mode: bool) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.encode("ascii", "replace").decode("ascii") if ascii_mode else text


def _theme_colors(theme: Any) -> Mapping[str, str]:
    colors = theme.get("colors") if isinstance(theme, Mapping) else getattr(theme, "colors", None)
    return colors if isinstance(colors, Mapping) else {}


def _style(theme: Any, token: str, *, transparent: bool) -> str:
    colors = _theme_colors(theme)
    effective = token if token in colors else "text"
    parts = [f"class:{effective.lower()}"]
    background = colors.get("statusLineBg")
    if not transparent and background and background != "default":
        parts.append(f"bg:{background}")
    return " ".join(parts)


def _symbols(theme: Any, ascii_mode: bool) -> Mapping[str, Any]:
    if ascii_mode:
        return resolve_symbols("ascii")
    symbols = theme.get("symbols") if isinstance(theme, Mapping) else getattr(theme, "symbols", None)
    return symbols if isinstance(symbols, Mapping) else resolve_symbols("unicode")


def _separator(name: str, symbols: Mapping[str, Any], ascii_mode: bool) -> str:
    if name == "none":
        return ""
    if name == "ascii" or ascii_mode and name in {"pipe", "block"}:
        return "|"
    if name == "slash":
        return "/"
    key = {
        "powerline": "sep.right",
        "powerline-thin": "sep.thin",
        "pipe": "sep.middle",
        "block": "sep.left",
    }.get(name)
    if key is not None:
        return _safe_text(symbols.get(key, "|"), ascii_mode)
    return "/"


def _segment_fragments(
    segment: Segment,
    theme: Any,
    symbols: Mapping[str, Any],
    *,
    transparent: bool,
    ascii_mode: bool,
) -> list[tuple[str, str]]:
    icon = symbols.get(segment.icon_key, "") if segment.icon_key else ""
    label = f"{icon} {segment.text}" if icon else segment.text
    return [(_style(theme, segment.token, transparent=transparent), f" {_safe_text(label, ascii_mode)} ")]


def _separator_fragments(
    separator: str,
    theme: Any,
    *,
    transparent: bool,
) -> list[tuple[str, str]]:
    if not separator:
        return []
    return [(_style(theme, "statusLineSep", transparent=transparent), f" {separator} ")]


def _join(
    items: list[tuple[str, Segment]],
    theme: Any,
    symbols: Mapping[str, Any],
    separator: str,
    *,
    transparent: bool,
    ascii_mode: bool,
) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    divider = _separator_fragments(separator, theme, transparent=transparent)
    for index, (_name, segment) in enumerate(items):
        if index:
            fragments.extend(divider)
        fragments.extend(
            _segment_fragments(
                segment,
                theme,
                symbols,
                transparent=transparent,
                ascii_mode=ascii_mode,
            )
        )
    return fragments


def _width(fragments: list[tuple[str, str]]) -> int:
    return sum(get_cwidth(text) for _style_name, text in fragments)


def _truncate(fragments: list[tuple[str, str]], width: int) -> list[tuple[str, str]]:
    remaining = max(0, width)
    result: list[tuple[str, str]] = []
    for style, text in fragments:
        if remaining <= 0:
            break
        chars: list[str] = []
        for char in text:
            char_width = max(0, get_cwidth(char))
            if char_width > remaining:
                break
            chars.append(char)
            remaining -= char_width
        if chars:
            result.append((style, "".join(chars)))
    return result


def _context_percent(segment: Segment) -> float:
    try:
        return min(100.0, max(0.0, float(segment.text.split("%", 1)[0])))
    except (ValueError, IndexError):
        return 0.0


def _context_label(segment: Segment) -> str:
    return f"{_context_percent(segment):g}%"


def _gauge_width(segment: Segment) -> int:
    return _CONTEXT_BAR_CELLS + 1 + get_cwidth(_context_label(segment))


def _without_provider(text: str) -> str:
    provider, separator, model = text.partition(":")
    if separator and provider and model:
        return model
    if text.startswith("("):
        closing = text.find(") ")
        if closing > 1 and closing + 2 < len(text):
            return text[closing + 2 :]
    return text


def _truncate_text(text: str, width: int, *, ascii_mode: bool) -> str:
    if width <= 0:
        return ""
    marker = "..." if ascii_mode else "…"
    marker_width = get_cwidth(marker)
    if width <= marker_width:
        return marker[:width]
    remaining = width - marker_width
    chars: list[str] = []
    for char in text:
        char_width = max(0, get_cwidth(char))
        if char_width > remaining:
            break
        chars.append(char)
        remaining -= char_width
    return f"{''.join(chars)}{marker}"


def _shrink_model(
    items: list[tuple[str, Segment]],
    excess: int,
    *,
    ascii_mode: bool,
) -> bool:
    for index, (name, segment) in enumerate(items):
        if name != "model":
            continue
        model_only = _without_provider(segment.text)
        if model_only != segment.text:
            items[index] = (
                name,
                Segment(model_only, segment.token, segment.icon_key),
            )
            return True
        text_width = get_cwidth(_safe_text(segment.text, ascii_mode))
        target_width = max(0, text_width - max(1, excess))
        if target_width <= 0:
            items.pop(index)
        else:
            items[index] = (
                name,
                Segment(
                    _truncate_text(segment.text, target_width, ascii_mode=ascii_mode),
                    segment.token,
                    segment.icon_key,
                ),
            )
        return True
    return False


def _gauge(
    segment: Segment,
    width: int,
    theme: Any,
    *,
    transparent: bool,
    ascii_mode: bool,
) -> list[tuple[str, str]]:
    if width <= 0:
        return []
    percent = _context_percent(segment)
    label = _context_label(segment)
    label_width = get_cwidth(label)
    bar_width = min(_CONTEXT_BAR_CELLS, max(0, width - label_width - 1))
    filled = round(bar_width * percent / 100)
    filled_glyph = "#" if ascii_mode else "━"
    empty_glyph = "-" if ascii_mode else "─"
    token = "success" if percent < 70 else "warning" if percent < 90 else "error"
    active_style = _style(theme, token, transparent=transparent)
    rest_style = _style(theme, "statusLineSep", transparent=transparent)
    return [
        (active_style, filled_glyph * filled),
        (rest_style, empty_glyph * (bar_width - filled)),
        (active_style, f" {label}"),
    ]


def render_statusline(
    ctx: Any,
    theme: Any,
    *,
    width: int | None = None,
    composer_shape: str | None = None,
) -> list[tuple[str, str]]:
    """Render one width-bounded prompt-toolkit formatted-text status row."""

    if not bool(getattr(ctx.cfg, "statusbar", True)):
        return []
    target_width = _console_width(ctx) if width is None else max(0, int(width))
    if target_width <= 0:
        return []
    config = _status_config(ctx)
    preset = config.preset if config.preset in PRESETS else "default"
    ascii_mode = _ascii_mode(ctx, preset)
    symbols = _symbols(theme, ascii_mode)
    separator = _separator(config.separator, symbols, ascii_mode)
    transparent = bool(config.transparent)
    left_names, right_names = _resolved_names(ctx)
    left_items = _evaluate(ctx, left_names)
    right_items = _evaluate(ctx, right_names)
    shape = composer_shape or getattr(ctx.cfg, "composer", "box")
    context: Segment | None = None
    if shape == "box":
        for items in (left_items, right_items):
            for index, (name, value) in enumerate(items):
                if name == "context":
                    context = value
                    items.pop(index)
                    break
            if context is not None:
                break
    if not left_items and not right_items and context is None:
        return []

    show_gauge = context is not None and not transparent
    if show_gauge:
        minimum_gap = _gauge_width(context)
    elif context is not None and left_items and right_items:
        minimum_gap = 1
    else:
        minimum_gap = 0
    while True:
        left = _join(
            left_items,
            theme,
            symbols,
            separator,
            transparent=transparent,
            ascii_mode=ascii_mode,
        )
        right = _join(
            right_items,
            theme,
            symbols,
            separator,
            transparent=transparent,
            ascii_mode=ascii_mode,
        )
        left_width = _width(left)
        right_width = _width(right)
        if left_width + right_width + minimum_gap <= target_width:
            break
        excess = left_width + right_width + minimum_gap - target_width
        if _shrink_model(left_items, excess, ascii_mode=ascii_mode):
            continue
        if _shrink_model(right_items, excess, ascii_mode=ascii_mode):
            continue
        if right_items:
            right_items.pop(0)
        elif left_items:
            left_items.pop()
        else:
            break

    gap = max(0, target_width - left_width - right_width)
    if show_gauge and context is not None:
        gauge_width = min(gap, _gauge_width(context))
        leading = (gap - gauge_width) // 2
        trailing = gap - gauge_width - leading
        gap_style = _style(theme, "statusLineBg", transparent=False)
        middle = []
        if leading:
            middle.append((gap_style, " " * leading))
        middle.extend(
            _gauge(
                context,
                gauge_width,
                theme,
                transparent=False,
                ascii_mode=ascii_mode,
            )
        )
        if trailing:
            middle.append((gap_style, " " * trailing))
    else:
        gap_style = "" if transparent else _style(theme, "statusLineBg", transparent=False)
        middle = [(gap_style, " " * gap)]
    return _truncate([*left, *middle, *right], target_width)


__all__ = [
    "BUILTIN_SEGMENTS",
    "DEFAULT_PRICING",
    "PRESETS",
    "SEPARATORS",
    "Segment",
    "WINDOWS",
    "cache_segment",
    "context_segment",
    "cost_segment",
    "git_segment",
    "mode_segment",
    "model_segment",
    "path_segment",
    "render_statusline",
    "record_turn_end",
    "record_turn_start",
    "record_usage",
    "reset_accounting",
    "session_segment",
    "subagents_segment",
    "time_segment",
    "tokens_segment",
    "visible_segments",
    "wrap_segment",
]
