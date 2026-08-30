"""Default native transcript block renderers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.tui.blocks import DEFAULT_RENDERERS
from orcha_agent.tui.frame import Block

PLUGIN = PluginSpec(name="render_default", version="1.0.0")


def _is_subagent(role: str) -> bool:
    return role == "subagent" or role.startswith("subagent:")


async def _thinking_command(api: PluginAPI, ctx: Any, args: str) -> None:
    value = args.strip()
    modes = {"on": "summary", "off": "off"}
    if value not in modes:
        ctx.console.error("Usage: /thinking on|off")
        return

    mode = modes[value]
    api.state["thinking"] = mode
    ctx.plugin_states.setdefault("provider_anthropic", {})["thinking"] = mode
    ctx.persist_plugin_states()
    await ctx.rebuild()
    ctx.console.print(f"Thinking: {value}")


def _thinking_renderer(api: PluginAPI):
    configured = str(api.config.get("thinking", "summary"))
    renderer = DEFAULT_RENDERERS["thinking"]

    def render(
        block: Block,
        theme: Any,
        width: int,
        budget_rows: int,
        expanded: bool,
    ) -> Any:
        mode = str(api.state.get("thinking", configured))
        role = str(block.data.get("role", "main"))
        visible = mode != "off" and (not _is_subagent(role) or mode == "all")
        value = replace(block, data={**block.data, "visible": visible})
        return renderer(value, theme, width, budget_rows, expanded)

    return render


def register(api: PluginAPI) -> None:
    api.add_command(
        "thinking",
        lambda ctx, args: _thinking_command(api, ctx, args),
        help="Toggle thinking display: /thinking on|off",
    )
    for kind, renderer in DEFAULT_RENDERERS.items():
        api.add_block_renderer(
            kind,
            _thinking_renderer(api) if kind == "thinking" else renderer,
        )
