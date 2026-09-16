"""Tabbed settings with surgical, atomic user TOML persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from orcha_agent.core.config import user_config_dir

from .select import SelectList
from .hints import key_hints

CATEGORIES = {
    "Appearance": (
        ("composer", ("box", "claude", "borderless", "band", "rail")),
        ("theme", ("dark", "light")),
    ),
    "Status": (
        ("preset", ("minimal", "default", "full", "powerline")),
        ("separator", ("powerline-thin", "powerline", "slash", "pipe", "none", "ascii")),
        ("transparent", (False, True)),
    ),
    "Behaviour": (
        ("mode", ("ask", "yolo")),
        ("auto_compact", (True, False)),
        ("notify", (False, True)),
        ("hyperlinks", (True, False)),
        ("mouse", ("scroll", "full", "off")),
        ("vim", (False, True)),
        ("colorblind", (False, True)),
        ("synchronized_output", (True, False)),
    ),
    "Terminal": (
        ("vim", (False, True)),
        ("hyperlinks", (True, False)),
        ("mouse", ("scroll", "full", "off")),
        ("colorblind", (False, True)),
        ("synchronized_output", (True, False)),
        ("resize", ("preserve", "rebuild")),
    ),
}


def persist_setting(path: Path, section: str, key: str, value: str | bool) -> None:
    """Keep unrelated values/comments intact and replace atomically after validation."""
    original = path.read_text() if path.exists() else ""
    lines = original.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    heading = f"[{section}]"
    header_pattern = re.compile(rf"^\s*\[\s*{re.escape(section)}\s*\]\s*(?:#.*)?$")
    start = next((i for i, line in enumerate(lines) if header_pattern.match(line)), None)
    assignment = f"{key} = {json.dumps(value)}\n"
    if not section:
        end = next((i for i, line in enumerate(lines) if line.lstrip().startswith("[")), len(lines))
        found = next(
            (i for i in range(end) if re.match(rf"\s*{re.escape(key)}\s*=", lines[i])), None
        )
        if found is None:
            lines.insert(end, assignment)
        else:
            lines[found] = assignment
    elif start is None:
        lines.extend(
            ["\n" if original and not original.endswith("\n\n") else "", heading + "\n", assignment]
        )
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
            len(lines),
        )
        found = next(
            (i for i in range(start + 1, end) if re.match(rf"\s*{re.escape(key)}\s*=", lines[i])),
            None,
        )
        if found is None:
            while end > start + 1 and (
                not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")
            ):
                end -= 1
            lines.insert(end, assignment)
        else:
            lines[found] = assignment
    content = "".join(lines)
    tomllib.loads(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".orcha-settings-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _setting_section(path: Path, key: str, *, status: bool, terminal: bool) -> str:
    if key == "mode":
        return "core"
    if terminal:
        return "tui"
    values = tomllib.loads(path.read_text()) if path.exists() else {}
    tui = values.get("tui", {})
    if status:
        return "tui.statusline" if "statusline" in tui else "ui.statusline"
    return "tui" if key in tui else "ui"


class SettingsOverlay(SelectList[str]):
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.category = 0
        self.categories = tuple(CATEGORIES)
        tabs = Window(FormattedTextControl(self._tabs), height=1)
        super().__init__("Settings", (), label=self._label, show_filter=False, prefix=tabs)
        self._load()
        self._body.children.insert(
            -1,
            Window(
                FormattedTextControl(self._description), wrap_lines=True, dont_extend_height=True
            ),
        )
        self.footer_control.text = lambda: key_hints(
            (
                ("↑↓", "move"),
                ("PgUp/PgDn", "page"),
                ("Enter", "change"),
                ("Esc", "close"),
                ("Tab", "category"),
            )
        )

        @self.bindings.add("tab")
        def next_tab(event: Any) -> None:
            self.category = (self.category + 1) % len(self.categories)
            self._load()
            event.app.invalidate()

        @self.bindings.add("s-tab")
        def previous_tab(event: Any) -> None:
            self.category = (self.category - 1) % len(self.categories)
            self._load()
            event.app.invalidate()

    def render_text(self) -> str:
        return (
            "".join(fragment[1] for fragment in self._fragments())
            + "".join(fragment[1] for fragment in self._description())
            + "\n"
            + "".join(
                fragment[1]
                for fragment in key_hints(
                    (
                        ("↑↓", "move"),
                        ("PgUp/PgDn", "page"),
                        ("Enter", "change"),
                        ("Esc", "close"),
                        ("Tab", "category"),
                    )
                )
            )
        )

    def _description(self) -> StyleAndTextTuples:
        descriptions = {
            "composer": "Choose the input border style.",
            "theme": "Choose the terminal colour palette.",
            "mode": "Choose when tools require approval.",
            "auto_compact": "Compact long conversations automatically.",
            "notify": "Notify when a turn finishes while unfocused.",
            "vim": "Use Vim insert and normal modes.",
            "hyperlinks": "Make file paths clickable.",
            "mouse": "Choose terminal mouse behaviour.",
            "colorblind": "Use a colourblind-friendly palette.",
            "synchronized_output": "Reduce flicker during terminal updates.",
            "resize": "Choose how the transcript responds to resizing.",
            "preset": "Choose status line information.",
            "separator": "Choose status segment separators.",
            "transparent": "Use the terminal background.",
        }
        key = self.items[self.index] if self.items else ""
        return [("class:muted", descriptions.get(key, ""))]

    def _tabs(self) -> StyleAndTextTuples:
        return [
            ("class:accent" if i == self.category else "class:muted", f" {name} ")
            for i, name in enumerate(self.categories)
        ]

    def _load(self) -> None:
        self.items = tuple(name for name, _ in CATEGORIES[self.categories[self.category]])
        self.index = 0

    def _setting_value(self, key: str) -> Any:
        config = (
            self.ctx.cfg.statusline if self.categories[self.category] == "Status" else self.ctx.cfg
        )
        if key in dict(CATEGORIES["Terminal"]):
            config = self.ctx.cfg.tui
        return getattr(config, key, True if key == "auto_compact" else "ask")

    def _label(self, key: str) -> str:
        value = self._setting_value(key)
        label = "on" if value is True else "off" if value is False else str(value)
        return f"{key.replace('_', ' '):<18} {label}"

    def _accept(self, value: str | list[str], event: Any) -> None:
        key = str(value)
        options = dict(CATEGORIES[self.categories[self.category]])[key]
        if key == "theme":
            options = tuple(getattr(self.ctx.ui, "themes", {})) or options
        if key == "mode":
            options = tuple(getattr(self.ctx.registry, "modes", {})) or options
        current = self._setting_value(key)
        index = options.index(current) if current in options else -1
        selected = options[(index + 1) % len(options)]
        status = self.categories[self.category] == "Status"
        terminal = key in dict(CATEGORIES["Terminal"])
        path = getattr(self.ctx.cfg, "user_config_path", None) or user_config_dir() / "config.toml"
        try:
            persist_setting(
                Path(path),
                _setting_section(Path(path), key, status=status, terminal=terminal),
                key,
                selected,
            )
            if key == "mode":
                event.app.create_background_task(self.ctx.switch_mode(str(selected)))
            elif status:
                self.ctx.cfg = replace(
                    self.ctx.cfg, statusline=replace(self.ctx.cfg.statusline, **{key: selected})
                )
            elif terminal:
                self.ctx.cfg = replace(
                    self.ctx.cfg, tui=replace(self.ctx.cfg.tui, **{key: selected})
                )
            else:
                self.ctx.cfg = replace(self.ctx.cfg, **{key: selected})
            if key == "auto_compact":
                self.ctx.rebuild_requested = True
            if key == "theme":
                self.ctx.ui.set_theme(selected)
                states = getattr(self.ctx, "plugin_states", None)
                if isinstance(states, dict):
                    states.setdefault("commands_core", {})["theme"] = selected
                    save = getattr(self.ctx, "persist_plugin_states", None)
                    if callable(save):
                        save()
            apply_settings = getattr(self.ctx.ui, "apply_settings", None)
            if callable(apply_settings):
                apply_settings(self.ctx.cfg)
            self._error = None
        except (OSError, ValueError, TypeError) as exc:
            self._error = str(exc)
        event.app.invalidate()


__all__ = ["SettingsOverlay", "persist_setting"]
