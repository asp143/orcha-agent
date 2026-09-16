"""Bounded, deterministic discovery and progressive rendering of SKILL.md files."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frontmatter import parse_frontmatter

MAX_SKILL_BYTES = 256 * 1024
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def _safe_path(path: Path) -> bool:
    return not any(part == "Credentials" or part.startswith(".env") for part in path.parts)


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path
    globs: tuple[str, ...] = ()
    always_apply: bool = False
    hide: bool = False
    disable_model_invocation: bool = False

    def read(self) -> str:
        """Read the current body, with the same bounds as discovery."""
        return parse_frontmatter(_read(self.path))[1].strip()

    def invocation(self, args: str = "") -> str:
        return (
            f'<skill name="{html.escape(self.name, quote=True)}" '
            f'base-dir="{html.escape(str(self.path.parent), quote=True)}">\n'
            f"{self.read()}\n</skill>" + (f"\n\n{args.strip()}" if args.strip() else "")
        )


def _read(path: Path) -> str:
    if not _safe_path(path) or not _safe_path(path.resolve()):
        raise ValueError("Sensitive skill path is excluded")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SKILL_BYTES + 1)
    if len(raw) > MAX_SKILL_BYTES:
        raise ValueError("Skill exceeds 256 KiB")
    return raw.decode("utf-8")


def parse_skill(path: Path) -> Skill:
    metadata, _body = parse_frontmatter(_read(path))
    name = metadata.get("name", path.parent.name)
    description = metadata.get("description", "")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            "Skill name must contain only letters, numbers, dots, dashes or underscores"
        )
    if not isinstance(description, str):
        raise ValueError("Skill description must be a string")
    globs = metadata.get("globs", [])
    if isinstance(globs, str):
        globs = [value.strip() for value in globs.split(",") if value.strip()]
    if not isinstance(globs, list) or not all(isinstance(value, str) for value in globs):
        raise ValueError("Skill globs must be a string or list of strings")
    for field in ("alwaysApply", "hide", "disableModelInvocation"):
        if field in metadata and not isinstance(metadata[field], bool):
            raise ValueError(f"Skill {field} must be a boolean")
    return Skill(
        name=name,
        description=description.strip(),
        path=path,
        globs=tuple(globs),
        always_apply=metadata.get("alwaysApply", False),
        hide=metadata.get("hide", False),
        disable_model_invocation=metadata.get("disableModelInvocation", False),
    )


def skill_roots(cwd: Path, home: Path, config: Mapping[str, Any]) -> list[Path]:
    """Nearest project wins, native before importers, then user roots."""
    roots: list[Path] = []
    cwd = cwd.resolve()
    importers = [
        directory
        for directory, key in (
            (".claude", "import_claude"),
            (".codex", "import_codex"),
            (".github", "import_github"),
        )
        if config.get(key, True)
    ]
    for directory in (cwd, *cwd.parents):
        roots.append(directory / ".orcha-agent" / "skills")
        if directory != home:
            roots.extend(directory / importer / "skills" for importer in importers)
    roots.append(home / ".config" / "orcha-agent" / "skills")
    roots.extend(home / importer / "skills" for importer in importers if importer != ".github")
    return list(dict.fromkeys(roots))


def discover_skills(
    cwd: Path, home: Path, config: Mapping[str, Any] | None = None
) -> tuple[dict[str, Skill], list[str]]:
    config = config or {}
    skills: dict[str, Skill] = {}
    warnings: list[str] = []
    if not config.get("enabled", True):
        return skills, warnings
    for root in skill_roots(cwd, home, config):
        if not _safe_path(root) or not _safe_path(root.resolve()):
            continue
        try:
            children = sorted(root.iterdir()) if root.is_dir() else []
        except OSError:
            warnings.append(f"Could not list skill root: {root}")
            continue
        for child in children:
            if not _safe_path(child) or not _safe_path(child.resolve()):
                continue
            path = child / "SKILL.md"
            try:
                if not path.is_file():
                    continue
                skill = parse_skill(path)
            except (OSError, UnicodeError, ValueError) as exc:
                warnings.append(f"Skipping skill {path}: {exc}")
                continue
            skills.setdefault(skill.name, skill)
    return skills, warnings


def render_skills(skills: Mapping[str, Skill], *, warnings: list[str] | None = None) -> str:
    entries: list[str] = []
    for skill in sorted(skills.values(), key=lambda item: item.name):
        if skill.hide or skill.disable_model_invocation:
            continue
        if skill.always_apply:
            try:
                entries.append(skill.invocation())
            except (OSError, UnicodeError, ValueError) as exc:
                if warnings is not None:
                    warnings.append(f"Could not load always-apply skill {skill.name}: {exc}")
        else:
            entries.append(
                f'<skill name="{html.escape(skill.name, quote=True)}">'
                f"{html.escape(skill.description)}</skill>"
            )
    if not entries:
        return ""
    return (
        "Available skills: use the skill tool with a name to read its instructions.\n"
        "<skills>\n" + "\n".join(entries) + "\n</skills>"
    )
