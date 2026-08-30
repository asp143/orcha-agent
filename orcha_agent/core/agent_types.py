"""Declarative agent roles used by the orchestration registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentType:
    name: str
    description: str
    system_prompt: str
    tools: set[str] | None
    model_role: str
    spawns: bool
    output_schema: dict[str, Any] | None
    blocking: bool = False


_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings", "overall", "explanation"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "title",
                    "body",
                    "priority",
                    "confidence",
                    "file",
                    "line_start",
                    "line_end",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {"enum": ["P0", "P1", "P2", "P3"]},
                    "confidence": {"type": "number"},
                    "file": {"type": "string"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                },
            },
        },
        "overall": {"enum": ["correct", "incorrect"]},
        "explanation": {"type": "string"},
    },
}


def builtin_agent_types() -> dict[str, AgentType]:
    """Return fresh built-in role definitions."""

    values = (
        AgentType(
            name="task",
            description="General-purpose coding worker",
            system_prompt="Complete the assigned task and finish by calling yield.",
            tools=None,
            model_role="task",
            spawns=True,
            output_schema=None,
        ),
        AgentType(
            name="scout",
            description="Read-only codebase exploration",
            system_prompt="Investigate the assigned question read-only and finish by calling yield.",
            tools={"ls", "read_file", "glob", "grep"},
            model_role="scout",
            spawns=False,
            output_schema=None,
        ),
        AgentType(
            name="reviewer",
            description="Evidence-backed code review",
            system_prompt="Review only the assigned change and yield the requested findings schema.",
            tools={"ls", "read_file", "glob", "grep"},
            model_role="reviewer",
            spawns=False,
            output_schema=_REVIEW_SCHEMA,
        ),
        AgentType(
            name="advisor",
            description="Persistent session watchdog",
            system_prompt="Review the transcript delta and call advise exactly once.",
            tools={"read_file", "grep", "glob", "advise"},
            model_role="advisor",
            spawns=False,
            output_schema=None,
        ),
    )
    return {value.name: value for value in values}


__all__ = ["AgentType", "builtin_agent_types"]
