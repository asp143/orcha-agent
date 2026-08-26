"""Minimal third-party orcha plugin loaded from a plugin directory."""

from __future__ import annotations

from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="hello", version="1.0.0", priority=100)


def hello(name: str = "world") -> str:
    """Return a greeting."""

    return f"Hello, {name}!"


async def hello_command(ctx: Any, args: str) -> None:
    ctx.console.print(hello(args.strip() or "world"))


def render_hello(event: Any) -> str:
    return str(getattr(event, "result", event))


def register(api: PluginAPI) -> None:
    api.add_tool(hello)
    api.add_command("hello", hello_command, help="Greet a name")
    api.add_renderer("hello", render_hello, priority=50)
    api.system_prompt_fragment("The hello tool provides friendly greetings.")
