"""Resolve actual targets and shell state when generating approval interrupts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

from .common import PathPolicy
from .shell import shell_approval_description


def approval_configs(
    interrupts: Mapping[str, Any],
    policy: PathPolicy,
    tools: Mapping[str, Any],
) -> dict[str, Any]:
    def describe(call: Any, state: Any, runtime: Any) -> str:
        name, args = call["name"], call["args"]
        if name in {"bash", "bash_jobs"} and name in tools:
            config: RunnableConfig
            try:
                config = get_config()
            except RuntimeError:
                config = {}
            return shell_approval_description(tools[name], config, args)
        paths = []
        supplied = args.get("path", args.get("file_path"))
        if isinstance(supplied, str):
            paths.append(supplied)
        if isinstance(args.get("patch"), str):
            from .hashline import _sections

            try:
                for section in _sections(args["patch"]):
                    paths.append(section.path)
                    paths.extend(row[3:].strip() for row in section.rows if row.startswith("MV "))
            except ValueError as exc:
                return f"Invalid hashline patch: {exc}"
        targets = []
        for path in dict.fromkeys(paths):
            try:
                targets.append(str(policy.resolve(path)))
            except (OSError, ValueError) as exc:
                targets.append(f"BLOCKED: {exc}")
        return f"{name} resolved target(s):\n" + "\n".join(targets)

    result = dict(interrupts)
    for name, value in interrupts.items():
        if value and name in {"write", "edit", "delete", "bash", "bash_jobs"}:
            decision_config: dict[str, Any] = (
                dict(value)
                if isinstance(value, Mapping)
                else {
                    "allowed_decisions": ["approve", "edit", "reject", "respond"],
                }
            )
            decision_config["description"] = describe
            result[name] = decision_config
    return result
