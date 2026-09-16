"""Shared safe YAML frontmatter parsing for Markdown extensions."""

from __future__ import annotations

from typing import Any

import yaml


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return metadata and body; malformed metadata raises ValueError."""
    lines = text.removeprefix("\ufeff").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index, line in enumerate(lines[1:], 1):
        if line.strip() in {"---", "..."}:
            try:
                value = yaml.safe_load("".join(lines[1:index]))
            except yaml.YAMLError as exc:
                raise ValueError("Invalid YAML frontmatter") from exc
            if value is None:
                value = {}
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise ValueError("Frontmatter must be a mapping with string keys")
            return value, "".join(lines[index + 1 :]).lstrip("\r\n")
    raise ValueError("Unterminated YAML frontmatter")
