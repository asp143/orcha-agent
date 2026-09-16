"""Markdown rule discovery and durable, file-aware model reminders."""

from __future__ import annotations

import fnmatch
import regex
from html import escape
from itertools import islice
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
MAX_RULES = 256
MAX_CONDITIONS = 8
MAX_PATTERN_LENGTH = 512


@dataclass(frozen=True)
class Rule:
    name: str
    body: str
    globs: tuple[str, ...] = ()
    always_apply: bool = False
    conditions: tuple[regex.Pattern[str], ...] = ()
    scope: tuple[str, ...] = ("text", "tool")
    interrupt_mode: str | None = None
    trusted: bool = True

    def reminder(self) -> SystemMessage:
        return SystemMessage(
            content=(
                f'<system-reminder rule="{escape(self.name, quote=True)}" '
                f'trust="{"trusted" if self.trusted else "untrusted"}">\n'
                f"{escape(self.body)}\n</system-reminder>"
            ),
            additional_kwargs={MARKER: [self.name]},
        )


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def discover_rules(
    cwd: Path, home: Path, *, trust_cwd: bool = False
) -> tuple[dict[str, Rule], list[str]]:
    """Trust controls automatic instructions and precedence, never explicit lookup."""
    roots = [
        (cwd / "RULES.md", None, True, trust_cwd),
        (cwd / ".orcha-agent/rules", "*.md", False, trust_cwd),
        (cwd / ".orcha-agent/RULES.md", None, True, trust_cwd),
        (home / ".config/orcha-agent/rules", "*.md", False, True),
        (home / ".config/orcha-agent/RULES.md", None, True, True),
        (cwd / ".claude/rules", "*.md", False, trust_cwd),
        (cwd / ".cursor/rules", "*.mdc", False, trust_cwd),
        (home / ".claude/rules", "*.md", False, True),
        (home / ".cursor/rules", "*.mdc", False, True),
    ]
    if not trust_cwd:
        roots.sort(key=lambda item: not item[3])
    rules: dict[str, Rule] = {}
    warnings: list[str] = []
    for root, pattern, sticky, trusted in roots:
        try:
            sources = (
                [root] if pattern is None else sorted(islice(root.glob(pattern), MAX_RULES + 1))
            )
        except OSError:
            warnings.append(f"Could not list rules: {root}")
            continue
        for path in sources:
            if len(rules) >= MAX_RULES:
                warnings.append(f"Rule discovery limit reached ({MAX_RULES})")
                return rules, warnings
            existing = rules.get(path.stem)
            if existing is not None:
                if existing.trusted and not trusted:
                    warnings.append(
                        f"Skipping untrusted rule {path.name}: name already defined by a trusted rule"
                    )
                continue
            # Never follow rule symlinks or inspect forbidden names.
            if any(part.startswith(".env") or part == "Credentials" for part in path.parts):
                continue
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                warnings.append(f"Skipping symlink rule: {path.name}")
                continue
            try:
                if not path.is_file():
                    continue
                if path.stat().st_size > 1024 * 1024:
                    warnings.append(f"Skipping oversized rule: {path.name}")
                    continue
                metadata, body = parse_frontmatter(path.read_text())
                patterns = []
                conditions = _strings(metadata.get("condition")) if trusted else ()
                if len(conditions) > MAX_CONDITIONS:
                    warnings.append(f"Rule {path.stem}: condition limit is {MAX_CONDITIONS}")
                for condition in conditions[:MAX_CONDITIONS]:
                    if len(condition) > MAX_PATTERN_LENGTH:
                        warnings.append(
                            f"Rule {path.stem}: pattern length limit is {MAX_PATTERN_LENGTH}"
                        )
                        continue
                    try:
                        patterns.append(regex.compile(condition))
                    except regex.error as exc:
                        warnings.append(f"Rule {path.stem}: invalid condition: {exc}")
                globs = _strings(metadata.get("globs", metadata.get("paths")))
                if isinstance(metadata.get("globs"), str):
                    globs = tuple(part.strip() for part in globs[0].split(",") if part.strip())
                mode = metadata.get("interruptMode")
                if mode not in (None, "always", "never", "prose-only", "tool-only"):
                    raise ValueError("invalid interruptMode")
                rules[path.stem] = Rule(
                    name=path.stem,
                    body=body,
                    globs=globs,
                    always_apply=trusted and (sticky or metadata.get("alwaysApply") is True),
                    conditions=tuple(patterns),
                    scope=_strings(metadata.get("scope")) or ("text", "tool"),
                    interrupt_mode=mode,
                    trusted=trusted,
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
        if rule.trusted and rule.always_apply:
            lines.append(rule.reminder().text)
        else:
            lines.append(
                f"- rule://{escape(rule.name)} — globs: {escape(', '.join(rule.globs)) or '(on demand)'}"
            )
    return "\n".join(lines)


class RulesState(AgentState):
    rule_reminders: NotRequired[dict[str, SystemMessage]]


class RulesMiddleware(AgentMiddleware[RulesState]):
    """Use the skill path normalizer; inject file rules before the next model call."""

    state_schema = RulesState

    def __init__(self, rules: Mapping[str, Rule], cwd: Path) -> None:
        self.rules = rules
        self.pending: dict[str, list[SystemMessage]] = {}
        self.interrupt_pending: dict[str, Any] = {}
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
            if rule.trusted
            and rule.name not in attached
            and not rule.always_apply
            and matches_paths(rule, paths)
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

    async def _with_reminders(self, request: Any, handler: Any) -> Any:
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

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        from .stream_rules import (
            ModelStreamMonitor,
            _model_monitor,
            _stream_host,
            install_model_callback,
        )

        host = _stream_host.get()
        if host is None:
            return await self._with_reminders(request, handler)
        config = get_config()
        # Subagent graphs own their own retry boundaries; never rewind a task.
        namespace = str(config.get("configurable", {}).get("checkpoint_ns", ""))
        if "|" in namespace:
            return await self._with_reminders(request, handler)
        thread = str(config.get("configurable", {}).get("thread_id", ""))
        self.interrupt_pending.pop(thread, None)
        monitor = ModelStreamMonitor(host)
        install_model_callback(request.model)
        token = _model_monitor.set(monitor)
        try:
            result = await self._with_reminders(request, handler)
            if monitor.pending is not None:
                raise monitor.pending
            return result
        finally:
            _model_monitor.reset(token)
            if monitor.pending is not None:
                self.interrupt_pending[thread] = monitor.pending

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        config = get_config()
        thread = str(config.get("configurable", {}).get("thread_id", ""))
        pending = self.interrupt_pending.get(thread)
        if pending is not None:
            raise pending
