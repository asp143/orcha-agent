"""Markdown rule discovery and durable, file-aware model reminders."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.config import get_config

from .frontmatter import parse_frontmatter
from .skill_globs import SkillGlobsMiddleware

MARKER = "orcha_rules"


@dataclass(frozen=True)
class Rule:
    name: str
    body: str
    globs: tuple[str, ...] = ()
    always_apply: bool = False
    conditions: tuple[re.Pattern[str], ...] = ()
    scope: tuple[str, ...] = ("text", "tool")
    interrupt_mode: str | None = None

    def reminder(self) -> SystemMessage:
        return SystemMessage(
            content=f"<system-reminder rule={self.name!r}>\n{self.body}\n</system-reminder>",
            additional_kwargs={MARKER: [self.name]},
        )


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def discover_rules(cwd: Path, home: Path) -> tuple[dict[str, Rule], list[str]]:
    """Native project > native user > imported project > imported user, by name."""
    sources: list[tuple[Path, bool]] = [(cwd / "RULES.md", True)]
    for root in (cwd / ".orcha-agent", home / ".config/orcha-agent"):
        sources.extend((path, False) for path in sorted((root / "rules").glob("*.md")))
        sources.append((root / "RULES.md", True))
    for root in (cwd, home):
        for directory, suffix in ((".claude", "md"), (".cursor", "mdc")):
            sources.extend(
                (path, False) for path in sorted((root / directory / "rules").glob(f"*.{suffix}"))
            )
    rules: dict[str, Rule] = {}
    warnings: list[str] = []
    for path, sticky in sources:
        if not path.is_file() or path.stem in rules:
            continue
        # Never follow rule symlinks or inspect forbidden file/directory names.
        if any(part.startswith(".env") or part == "Credentials" for part in path.parts):
            continue
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            warnings.append(f"Skipping symlink rule: {path.name}")
            continue
        if path.stat().st_size > 1024 * 1024:
            warnings.append(f"Skipping oversized rule: {path.name}")
            continue
        try:
            metadata, body = parse_frontmatter(path.read_text())
            patterns = []
            for condition in _strings(metadata.get("condition")):
                try:
                    patterns.append(re.compile(condition))
                except re.error as exc:
                    warnings.append(f"Rule {path.stem}: invalid condition: {exc}")
            globs = _strings(metadata.get("globs", metadata.get("paths")))
            # Cursor commonly encodes multiple globs in a comma-separated string.
            if isinstance(metadata.get("globs"), str):
                globs = tuple(part.strip() for part in globs[0].split(",") if part.strip())
            mode = metadata.get("interruptMode")
            if mode not in (None, "always", "never", "prose-only", "tool-only"):
                raise ValueError("invalid interruptMode")
            rules[path.stem] = Rule(
                name=path.stem,
                body=body,
                globs=globs,
                always_apply=sticky or metadata.get("alwaysApply") is True,
                conditions=tuple(patterns),
                scope=_strings(metadata.get("scope")) or ("text", "tool"),
                interrupt_mode=mode,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append(f"Could not load rule {path.name}: {exc}")
    return rules, warnings


def matches_paths(rule: Rule, paths: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(path, pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]))
        or ("/" not in pattern and fnmatch.fnmatchcase(Path(path).name, pattern))
        for path in paths
        for pattern in rule.globs
    )


def rulebook(rules: Mapping[str, Rule]) -> str:
    if not rules:
        return ""
    lines = ["Rulebook: use the rule tool with rule://name to read instructions on demand."]
    for rule in rules.values():
        if rule.always_apply:
            lines.append(rule.reminder().text)
        else:
            lines.append(f"- rule://{rule.name} — globs: {', '.join(rule.globs) or '(on demand)'}")
    return "\n".join(lines)


class RulesState(AgentState):
    rule_reminders: NotRequired[dict[str, SystemMessage]]


class RulesMiddleware(AgentMiddleware[RulesState]):
    """Use the skill path normalizer; inject file rules before the next model call."""

    state_schema = RulesState

    def __init__(self, rules: Mapping[str, Rule], cwd: Path) -> None:
        self.rules = rules
        self.pending: dict[str, list[SystemMessage]] = {}
        self.paths = SkillGlobsMiddleware({})
        self.paths.cwd = cwd

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        durable = dict(state.get("rule_reminders", {}))
        for message in messages:
            if isinstance(message, SystemMessage):
                for name in message.additional_kwargs.get(MARKER, []):
                    durable[name] = message
        attached = {
            name
            for message in messages
            if isinstance(message, SystemMessage)
            for name in message.additional_kwargs.get(MARKER, [])
        }
        file_tools = {
            "read_file",
            "write_file",
            "edit_file",
            "read",
            "write",
            "edit",
            "delete",
            "ls",
            "glob",
            "grep",
        }
        calls = {
            call["id"]: call["args"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
            if call["name"] in file_tools
        }
        paths = [
            path
            for message in messages
            if isinstance(message, ToolMessage)
            and message.tool_call_id in calls
            and message.status != "error"
            and not message.text.lstrip().lower().startswith("error")
            for path in self.paths._paths(calls[message.tool_call_id])
        ]
        selected = [
            rule.reminder()
            for rule in self.rules.values()
            if rule.name not in attached and not rule.always_apply and matches_paths(rule, paths)
        ]
        if self.pending:
            config = get_config()
            thread_id = str(config.get("configurable", {}).get("thread_id", ""))
            selected.extend(self.pending.pop(thread_id, []))
        selected.extend(message for name, message in durable.items() if name not in attached)
        for message in selected:
            for name in message.additional_kwargs[MARKER]:
                durable[name] = message
        update: dict[str, Any] = {}
        if selected:
            update["messages"] = selected
        if durable != state.get("rule_reminders", {}):
            update["rule_reminders"] = durable
        return update or None

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        reminders, ordinary = [], []
        for message in request.messages:
            if isinstance(message, SystemMessage) and MARKER in message.additional_kwargs:
                reminders.append(message)
            else:
                ordinary.append(message)
        present = {name for message in reminders for name in message.additional_kwargs[MARKER]}
        durable = getattr(request, "state", {}).get("rule_reminders", {})
        reminders.extend(message for name, message in durable.items() if name not in present)
        if not reminders:
            return await handler(request)
        base = request.system_message
        blocks = list(base.content_blocks) if base is not None else []
        blocks.extend({"type": "text", "text": message.text} for message in reminders)
        system = SystemMessage(content=blocks)
        return await handler(request.override(messages=ordinary, system_message=system))
