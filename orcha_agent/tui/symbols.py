"""Terminal-safe symbol presets used by TUI themes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich import box

SYMBOL_KEYS = (
    "boxRound.topLeft",
    "boxRound.topRight",
    "boxRound.bottomLeft",
    "boxRound.bottomRight",
    "boxRound.horizontal",
    "boxRound.vertical",
    "boxSharp.topLeft",
    "boxSharp.topRight",
    "boxSharp.bottomLeft",
    "boxSharp.bottomRight",
    "boxSharp.horizontal",
    "boxSharp.vertical",
    "status.success",
    "status.error",
    "status.pending",
    "status.warning",
    "status.info",
    "tree.branch",
    "tree.last",
    "tree.vertical",
    "tree.space",
    "tree.expanded",
    "tree.collapsed",
    "sep.left",
    "sep.right",
    "sep.middle",
    "sep.thin",
    "sep.dot",
    "spinner.status",
    "spinner.activity",
    "icon.model",
    "icon.mode",
    "icon.path",
    "icon.git",
    "icon.context",
    "icon.tokens",
    "icon.cost",
    "icon.thinking",
    "icon.subagents",
)

_UNICODE = (
    "╭", "╮", "╰", "╯", "─", "│",
    "┌", "┐", "└", "┘", "─", "│",
    "✓", "✗", "○", "!", "i",
    "├─", "└─", "│ ", "  ", "▾", "▸",
    "", "", "│", "·", "•",
    "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "✻✼❉❊✺✹✸✶",
    "◈", "◆", "⌂", "", "◔", "▣", "$", "✦", "◇",
)

_NERD = (
    "╭", "╮", "╰", "╯", "─", "│",
    "┌", "┐", "└", "┘", "─", "│",
    "󰄬", "󰅖", "󰐊", "󰀪", "󰋽",
    "├─", "└─", "│ ", "  ", "", "",
    "", "", "", "│", "",
    "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "✻✼❉❊✺✹✸✶",
    "󰧑", "󰘳", "", "", "󰍛", "󰘚", "󰇭", "󰔟", "󰙅",
)

_ASCII = (
    "+", "+", "+", "+", "-", "|",
    "+", "+", "+", "+", "-", "|",
    "+", "x", "o", "!", "i",
    "+-", "`-", "| ", "  ", "v", ">",
    "[", "]", "|", "|", ".",
    "|/-\\", "|/-\\",
    "M", "!", "/", "G", "C", "T", "$", "?", "A",
)

SYMBOL_PRESETS: dict[str, dict[str, str]] = {
    "unicode": dict(zip(SYMBOL_KEYS, _UNICODE, strict=True)),
    "nerd": dict(zip(SYMBOL_KEYS, _NERD, strict=True)),
    "ascii": dict(zip(SYMBOL_KEYS, _ASCII, strict=True)),
}


def _supports_unicode(encoding: str | None) -> bool:
    if encoding is None:
        return True
    try:
        "✓".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _rich_box(values: Mapping[str, Any], prefix: str, *, ascii: bool) -> box.Box:
    top_left = values[f"{prefix}.topLeft"]
    top_right = values[f"{prefix}.topRight"]
    bottom_left = values[f"{prefix}.bottomLeft"]
    bottom_right = values[f"{prefix}.bottomRight"]
    horizontal = values[f"{prefix}.horizontal"]
    vertical = values[f"{prefix}.vertical"]
    return box.Box(
        "\n".join(
            (
                f"{top_left}{horizontal}{horizontal}{top_right}",
                f"{vertical} {vertical}{vertical}",
                f"{vertical}{horizontal}{horizontal}{vertical}",
                f"{vertical} {vertical}{vertical}",
                f"{vertical}{horizontal}{horizontal}{vertical}",
                f"{vertical}{horizontal}{horizontal}{vertical}",
                f"{vertical} {vertical}{vertical}",
                f"{bottom_left}{horizontal}{horizontal}{bottom_right}",
            )
        ),
        ascii=ascii,
    )


def resolve_symbols(
    preset: str,
    overrides: Mapping[str, object] | None = None,
    *,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Resolve a preset plus validated overrides and Rich box adapters."""

    selected = preset if preset in SYMBOL_PRESETS else ""
    if not selected:
        raise ValueError(f"unknown symbol preset: {preset}")
    if selected != "ascii" and not _supports_unicode(encoding):
        selected = "ascii"
    values: dict[str, Any] = dict(SYMBOL_PRESETS[selected])
    for key, value in (overrides or {}).items():
        if key not in SYMBOL_KEYS:
            raise ValueError(f"unknown symbol key: {key}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"symbol override {key} must be a non-empty string")
        if key.startswith(("boxRound.", "boxSharp.")) and len(value) != 1:
            raise ValueError(f"symbol override {key} must be one character")
        values[key] = value
    ascii_preset = selected == "ascii"
    if any(key.startswith("boxRound.") for key in (overrides or {})):
        values["boxRound"] = _rich_box(values, "boxRound", ascii=ascii_preset)
    else:
        values["boxRound"] = box.ASCII if ascii_preset else box.ROUNDED
    if any(key.startswith("boxSharp.") for key in (overrides or {})):
        values["boxSharp"] = _rich_box(values, "boxSharp", ascii=ascii_preset)
    else:
        values["boxSharp"] = box.ASCII if ascii_preset else box.SQUARE
    values["preset"] = selected
    return values


__all__ = ["SYMBOL_KEYS", "SYMBOL_PRESETS", "resolve_symbols"]
