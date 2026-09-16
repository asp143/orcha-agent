"""Turn-local prose keywords without changing saved modes or model settings."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

_KEYWORDS = ("ultrathink", "orchestrate", "plan")
_BOUNDARY = re.compile(
    r"(?<![\w./\\-])(?<!::)(ultrathink|orchestrate|plan)(?![\w/\\-])(?!\.[\w-])(?!\()"
)
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)\b(?:[^<>\"']|\"[^\"]*\"|'[^']*')*>")
_NOTICES = {
    "ultrathink": "Use the highest supported reasoning effort for this turn. Carefully verify assumptions and conclusions.",
    "orchestrate": "Fan out independent work through the task tool when it is available; coordinate and integrate the results.",
    "plan": "This turn is read-only planning. Inspect and propose a plan; do not modify files, run commands, or delegate writes.",
}


def prose(text: str) -> str:
    """Mask fenced/inline code and balanced XML/HTML sections before matching."""
    lines = text.splitlines(keepends=True)
    fence: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            lines[index] = "\n" if line.endswith("\n") else " "
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= fence[1]
                and not line[marker.end() :].strip()
            ):
                fence = None
        elif marker:
            fence = (marker[1][0], len(marker[1]))
            lines[index] = "\n" if line.endswith("\n") else " "
    text = "".join(lines)
    # Exact-length backtick runs delimit inline code; unmatched runs are prose.
    runs = list(re.finditer(r"`+", text))
    masked = list(text)
    index = 0
    while index < len(runs):
        start = runs[index]
        end = next(
            (j for j in range(index + 1, len(runs)) if len(runs[j][0]) == len(start[0])), None
        )
        if end is None:
            index += 1
            continue
        masked[start.start() : runs[end].end()] = " " * (runs[end].end() - start.start())
        index = end + 1
    text = "".join(masked)
    text = re.sub(r"<!--[\s\S]*?(?:-->|$)", " ", text)
    masked = list(text)
    stack: list[tuple[str, int]] = []
    for tag in _TAG.finditer(text):
        masked[tag.start() : tag.end()] = " " * len(tag[0])
        name = tag[1].lower()
        if tag[0].startswith("</"):
            opening = next((j for j in range(len(stack) - 1, -1, -1) if stack[j][0] == name), None)
            if opening is not None:
                start = stack[opening][1]
                masked[start : tag.end()] = " " * (tag.end() - start)
                del stack[opening:]
        elif not tag[0].endswith("/>"):
            stack.append((name, tag.start()))
    return "".join(masked)


def keywords(text: str) -> frozenset[str]:
    if not any(word in text for word in _KEYWORDS):
        return frozenset()
    return frozenset(match[1] for match in _BOUNDARY.finditer(prose(text)))


def turn_keywords(messages: list[Any]) -> frozenset[str]:
    latest = next((item for item in reversed(messages) if isinstance(item, HumanMessage)), None)
    return keywords(latest.text) if latest is not None else frozenset()


def tool_name(tool: Any) -> str:
    return str(tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", ""))


class MagicKeywordsMiddleware(AgentMiddleware):
    def __init__(self, ctx: Any = None) -> None:
        self.ctx = ctx

    def plan_tools(self) -> set[str] | None:
        registry = getattr(self.ctx, "registry", None)
        registration = registry.modes.get("plan") if registry is not None else None
        if registration is None:
            return None
        spec = getattr(registration, "spec", registration)
        allowed = getattr(spec, "allowed_tools", None)
        if allowed is None:
            return None
        names = set(allowed)
        if "read_file" in names:
            names.add("read")
        return names

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        active = set(turn_keywords(request.messages))
        allowed = self.plan_tools()
        if allowed is None:
            active.discard("plan")
        if not active:
            return await handler(request)
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message is not None
            else []
        )
        blocks.append(
            {
                "type": "text",
                "text": "<system-reminder>\n"
                + "\n".join(_NOTICES[word] for word in _KEYWORDS if word in active)
                + "\n</system-reminder>",
            }
        )
        overrides: dict[str, Any] = {"system_message": SystemMessage(content=blocks)}
        if "plan" in active and allowed is not None:
            overrides["tools"] = [tool for tool in request.tools if tool_name(tool) in allowed]
        if "ultrathink" in active:
            settings = dict(request.model_settings)
            cfg = getattr(self.ctx, "cfg", None)
            spec = getattr(cfg, "model", "")
            spec = spec[0] if isinstance(spec, list) and spec else spec
            aliases = getattr(cfg, "models", {})
            seen: set[str] = set()
            while isinstance(spec, str) and spec in aliases and spec not in seen:
                seen.add(spec)
                spec = aliases[spec]
            prefix = spec.partition(":")[0] if isinstance(spec, str) else ""
            # Temporary command/subagent models can differ from the saved main model.
            # Inspect adapter ancestry without importing optional providers.
            adapters = {
                "langchain_anthropic": "anthropic",
                "langchain_openai": "openai",
                "langchain_google_genai": "google",
                "langchain_ollama": "ollama",
            }
            for cls in type(request.model).__mro__:
                adapter = cls.__module__.partition(".")[0]
                if adapter in adapters:
                    prefix = adapters[adapter]
                    break
            registry = getattr(self.ctx, "registry", None)
            provider = registry.providers.get(prefix) if registry is not None else None
            profile = getattr(request.model, "profile", None) or {}
            levels = profile.get("reasoning_effort_levels", ())
            highest = next(
                (level for level in ("max", "xhigh", "high", "medium", "low") if level in levels),
                None,
            )
            if (
                provider is not None
                and provider.capabilities.thinking
                and profile.get("reasoning_output") is not False
            ):
                if prefix in {"openai", "codex"}:
                    if highest is not None:
                        settings["reasoning"] = {"effort": highest, "summary": "auto"}
                elif prefix == "anthropic":
                    if highest is not None:
                        settings["reasoning_effort"] = highest
                    if "xhigh" in levels:
                        settings["thinking"] = {"type": "adaptive"}
                    elif profile.get("reasoning_output") is True:
                        max_tokens = (
                            settings.get("max_tokens")
                            or getattr(request.model, "max_tokens", None)
                            or 4096
                        )
                        if max_tokens > 1024:
                            settings["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": max_tokens - 1,
                            }
                            settings["temperature"] = 1
                elif prefix == "google":
                    model_name = str(
                        getattr(request.model, "model", "")
                        or getattr(request.model, "model_name", "")
                        or str(spec).partition(":")[2]
                    )
                    if "gemini-2.5-pro" in model_name or "gemini-2.5-flash" in model_name:
                        settings["thinking_level"] = None
                        settings["reasoning_effort"] = None
                        settings["thinking_budget"] = (
                            32768 if "gemini-2.5-pro" in model_name else 24576
                        )
                    elif re.search(r"gemini-[3-9](?:[.-]|$)", model_name):
                        settings["thinking_level"] = "high"
                        settings["thinking_budget"] = None
            overrides["model_settings"] = settings
        return await handler(request.override(**overrides))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        allowed = self.plan_tools()
        if (
            allowed is not None
            and "plan" in turn_keywords(request.state.get("messages", []))
            and request.tool_call["name"] not in allowed
        ):
            return ToolMessage(
                content="Blocked: this turn is in read-only plan mode.",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return await handler(request)
