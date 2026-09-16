"""Native tool migration preserves explicitly named third-party subagent tools."""

from pathlib import Path

import pytest
from langchain_core.tools import tool
from test_native_boundaries import capture_kwargs, plugin
from test_native_integration import config, kernel

from orcha_agent.core.agent import build_agent
from orcha_agent.core.events import AppExit
from orcha_agent.core.session import SessionStore


@pytest.mark.asyncio
async def test_subagent_resolves_registered_custom_string_alongside_native_alias(tmp_path: Path):
    cfg = config(tmp_path)
    registry, bus = kernel(cfg, [])
    api = plugin(registry, bus)

    @tool
    def project_probe(value: str) -> str:
        """Return a project-specific result."""
        return f"project:{value}"

    api.add_tool(project_probe)
    api.add_subagent(
        {
            "name": "project-reader",
            "description": "Read the project and use its custom probe.",
            "system_prompt": "Inspect the project with the tools provided.",
            "tools": ["read_file", "project_probe"],
        }
    )
    captured = capture_kwargs(bus)
    try:
        with SessionStore(cfg.db_path) as session:
            await build_agent(registry, cfg, session, bus)
        spec = next(spec for spec in captured["subagents"] if spec["name"] == "project-reader")
        selected = {item.name: item for item in spec["tools"]}
        assert set(selected) == {"read", "project_probe"}
        assert selected["project_probe"] is registry.tools["project_probe"]
        assert selected["project_probe"].invoke({"value": "ready"}) == "project:ready"
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_subagent_unknown_string_tool_fails_explicitly(tmp_path: Path):
    cfg = config(tmp_path)
    registry, bus = kernel(cfg, [])
    plugin(registry, bus).add_subagent(
        {
            "name": "missing-tool-reader",
            "description": "Requests an unavailable project tool.",
            "system_prompt": "Inspect the project.",
            "tools": ["read_file", "unregistered_project_probe"],
        }
    )
    try:
        with SessionStore(cfg.db_path) as session:
            with pytest.raises(
                ValueError, match="Unknown subagent tool: unregistered_project_probe"
            ):
                await build_agent(registry, cfg, session, bus)
    finally:
        await bus.emit(AppExit())
