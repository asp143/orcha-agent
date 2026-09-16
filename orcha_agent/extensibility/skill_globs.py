"""Attach trusted file-matching skills once per user turn, with durable markers."""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .skills import Skill

_MARKER = "orcha_skill_globs"
_FILE_TOOLS = {"read_file", "write_file", "edit_file", "delete", "ls", "glob", "grep"}


def _turn_start(messages: list[Any]) -> int:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        0,
    )


class SkillGlobsMiddleware(AgentMiddleware):
    """Persist reminders in message state and present them as system context."""

    def __init__(self, skills: Mapping[str, Skill], *, enabled: bool = True) -> None:
        self.skills = skills
        self.enabled = enabled
        self.cwd = Path.cwd()

    def _paths(self, arguments: Mapping[str, Any]) -> list[str]:
        paths: list[str] = []
        for key in ("file_path", "path", "file", "paths"):
            value = arguments.get(key)
            values = value if isinstance(value, list) else [value]
            for raw in values:
                if not isinstance(raw, str):
                    continue
                path = Path(raw)
                # Deepagents' filesystem tools use workspace-virtual absolute paths.
                if path.is_absolute() and path.is_relative_to(self.cwd):
                    raw = path.relative_to(self.cwd).as_posix()
                else:
                    raw = raw.lstrip("/")
                normalized = PurePosixPath(raw)
                if ".." not in normalized.parts:
                    paths.append(normalized.as_posix())
        return paths

    def _matches(self, skill: Skill, paths: list[str]) -> bool:
        return any(
            fnmatch.fnmatchcase(path, pattern)
            or (pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]))
            or ("/" not in pattern and fnmatch.fnmatchcase(PurePosixPath(path).name, pattern))
            for pattern in skill.globs
            for path in paths
        )

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        messages = state.get("messages", [])
        turn = messages[_turn_start(messages) :]
        attached: set[str] = set()
        calls: dict[str, Mapping[str, Any]] = {}
        paths: list[str] = []
        for message in turn:
            if isinstance(message, SystemMessage):
                attached.update(message.additional_kwargs.get(_MARKER, []))
            elif isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call["name"] in _FILE_TOOLS and call["id"] is not None:
                        calls[call["id"]] = call["args"]
            elif isinstance(message, ToolMessage) and message.tool_call_id in calls:
                if message.status != "error" and not message.text.lstrip().lower().startswith(
                    "error"
                ):
                    paths.extend(self._paths(calls[message.tool_call_id]))
        selected = [
            skill
            for skill in self.skills.values()
            if skill.name not in attached
            and skill.trusted
            and not skill.disable_model_invocation
            and not skill.hide
            and not skill.always_apply
            and self._matches(skill, paths)
        ]
        if not selected:
            return None

        def render() -> tuple[list[str], list[str]]:
            names, bodies = [], []
            for skill in selected:
                try:
                    bodies.append(skill.invocation())
                    names.append(skill.name)
                except (OSError, UnicodeError, ValueError):
                    continue
            return names, bodies

        names, bodies = await asyncio.to_thread(render)
        if not names:
            return None
        return {
            "messages": [
                SystemMessage(
                    content="<system-reminder>\nSkills matching files used this turn:\n"
                    + "\n".join(bodies)
                    + "\n</system-reminder>",
                    additional_kwargs={_MARKER: names},
                )
            ]
        }

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        messages = request.messages
        start = _turn_start(messages)
        reminders = [
            message
            for message in messages[start:]
            if isinstance(message, SystemMessage) and _MARKER in message.additional_kwargs
        ]
        ordinary = [
            message
            for message in messages
            if not (isinstance(message, SystemMessage) and _MARKER in message.additional_kwargs)
        ]
        if not reminders and len(ordinary) == len(messages):
            return await handler(request)
        base = request.system_message
        blocks = list(base.content_blocks) if base is not None else []
        blocks.extend({"type": "text", "text": message.text} for message in reminders)
        system = (
            base.model_copy(update={"content": blocks})
            if base is not None
            else SystemMessage(content=blocks)
        )
        return await handler(request.override(messages=ordinary, system_message=system))
