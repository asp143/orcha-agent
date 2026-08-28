"""Configurable composer key bindings."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_bindings import _parse_key

DEFAULT_BINDINGS: dict[str, tuple[str, ...]] = {
    "submit": ("enter", "c-j"),
    "newline": ("escape enter", "escape c-j"),
    "queue": ("c-q",),
    "dequeue": ("escape up",),
    "toggle_thinking": ("c-t",),
    "cycle_thinking_level": ("s-tab",),
    "expand_tools": ("c-o",),
    "model_picker": ("escape p",),
    "cycle_model": ("c-p",),
    "history_search": ("c-r",),
    "external_editor": ("c-g",),
    "clear_screen": ("c-l",),
    "interrupt": ("c-c",),
    "exit": ("c-d",),
    "tree": ("escape escape",),
}

Warn = Callable[[str], None]


def default_keybindings_path() -> Path:
    return Path(__file__).with_name("keybindings.toml")


def user_keybindings_path() -> Path:
    return Path.home() / ".config/orcha-agent/keybindings.toml"


def _binding_values(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return None


def _canonical(binding: str) -> tuple[str, ...]:
    parts = binding.split()
    if not parts:
        raise ValueError("binding cannot be empty")
    parsed = (_parse_key(part) for part in parts)
    return tuple(str(getattr(key, "value", key)) for key in parsed)


def _valid(binding: str) -> bool:
    try:
        _canonical(binding)
    except ValueError:
        return False
    return True


def _read_bindings(path: Path, warn: Warn) -> list[tuple[str, object]]:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError:
        return []
    except (OSError, tomllib.TOMLDecodeError) as exc:
        warn(f"Cannot load keybindings from {path}: {exc}")
        return []
    table = data.get("bindings", data)
    if not isinstance(table, dict):
        warn(f"Invalid keybindings in {path}: expected a [bindings] table")
        return []
    return [(str(action), value) for action, value in table.items()]


def load_keybindings(
    *,
    user_path: str | Path | None = None,
    registry: Any = None,
    warn: Warn = lambda _message: None,
) -> dict[str, tuple[str, ...]]:
    """Load defaults, plugin defaults, and user overrides with last-wins conflicts."""

    effective = dict(DEFAULT_BINDINGS)
    for action, value in _read_bindings(default_keybindings_path(), warn):
        values = _binding_values(value)
        if values is not None and values and all(_valid(binding) for binding in values):
            effective[action] = values
    definition_order = list(effective)
    if registry is not None:
        for action, registration in registry.keybindings.items():
            values = _binding_values(registration.default)
            if values is None:
                warn(f"Invalid default binding for plugin action {action!r}")
                continue
            valid = tuple(binding for binding in values if _valid(binding))
            if not valid:
                warn(f"Invalid default binding for plugin action {action!r}")
                continue
            effective[action] = valid
            if action in definition_order:
                definition_order.remove(action)
            definition_order.append(action)

    path = Path(user_path) if user_path is not None else user_keybindings_path()
    for action, value in _read_bindings(path, warn):
        if action not in effective:
            warn(f"Unknown keybinding action {action!r}; ignoring it")
            continue
        values = _binding_values(value)
        if values is None:
            warn(f"Invalid binding value for {action}; retaining prior binding")
            continue
        valid: list[str] = []
        for binding in values:
            if _valid(binding):
                valid.append(binding)
            else:
                warn(f"Invalid binding {binding!r} for {action}; retaining prior binding")
        if not valid:
            continue
        effective[action] = tuple(dict.fromkeys(valid))
        definition_order.remove(action)
        definition_order.append(action)

    owners: dict[tuple[str, ...], tuple[str, str]] = {}
    resolved: dict[str, list[str]] = {action: [] for action in definition_order}
    for action in definition_order:
        for binding in effective[action]:
            canonical = _canonical(binding)
            previous = owners.get(canonical)
            if previous is not None:
                previous_action, previous_binding = previous
                if previous_action == action:
                    continue
                resolved[previous_action] = [
                    candidate
                    for candidate in resolved[previous_action]
                    if candidate != previous_binding
                ]
                warn(
                    f"Keybinding conflict for {binding!r}: "
                    f"{previous_action} and {action}; {action} wins"
                )
            owners[canonical] = (action, binding)
            resolved[action].append(binding)
    return {action: tuple(resolved[action]) for action in effective}


def create_key_bindings(
    effective: Mapping[str, tuple[str, ...]],
    handlers: Mapping[str, Callable[[Any], None]],
) -> KeyBindings:
    bindings = KeyBindings()
    for action, values in effective.items():
        handler = handlers.get(action)
        if handler is None:
            continue
        for value in values:
            keys = value.split()
            bindings.add(*keys)(handler)
            if keys == ["escape", "escape"]:
                bindings.add("s-escape")(handler)
    return bindings


__all__ = [
    "DEFAULT_BINDINGS",
    "create_key_bindings",
    "default_keybindings_path",
    "load_keybindings",
    "user_keybindings_path",
]
