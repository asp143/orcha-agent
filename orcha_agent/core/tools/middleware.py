"""Translate deepagents offload pointers at the native model boundary."""

from __future__ import annotations

import re
from typing import Any

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def _rewrite_text(content: str) -> str:
    match = re.match(
        r"Tool result too large, the result of this tool call [^\n]+ was saved in the filesystem at this path: ([^\n]+)\n",
        content,
    )
    marker = "Here is a preview showing the head and tail of the result"
    if match is None or marker not in content:
        return content
    path = match[1]
    return (
        f"Tool result offloaded to {path}.\n"
        f"Use read(path={path + ':1+100'!r}); continue with path:N+K (1-based lines).\n\n"
        + marker
        + content.split(marker, 1)[1]
    )


def rewrite_result(value: Any) -> Any:
    if isinstance(value, ToolMessage):
        if isinstance(value.content, str):
            content: Any = _rewrite_text(value.content)
        else:
            content = [
                {**item, "text": _rewrite_text(item["text"])}
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
                else item
                for item in value.content
            ]
        return value.model_copy(update={"content": content}) if content != value.content else value
    if isinstance(value, Command) and isinstance(value.update, dict) and "messages" in value.update:
        return Command(
            graph=value.graph,
            update={
                **value.update,
                "messages": [rewrite_result(message) for message in value.update["messages"]],
            },
            resume=value.resume,
            goto=value.goto,
        )
    return value


class NativeOutputMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return rewrite_result(handler(request))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return rewrite_result(await handler(request))

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        # Also covers summarization's overflow retry, which has no before_model hook.
        return handler(request.override(messages=[rewrite_result(m) for m in request.messages]))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(
            request.override(messages=[rewrite_result(m) for m in request.messages])
        )


class NativeFilesystemMiddleware(FilesystemMiddleware):
    @property
    def name(self) -> str:
        return "FilesystemMiddleware"

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return rewrite_result(super().wrap_tool_call(request, handler))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return rewrite_result(await super().awrap_tool_call(request, handler))
