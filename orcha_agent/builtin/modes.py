"""Built-in approval and tool-access modes."""

from __future__ import annotations

from orcha_agent.core.plugin import ModeSpec, PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="modes", version="1.0.0")


def register(api: PluginAPI) -> None:
    api.add_mode(
        "ask",
        ModeSpec(
            description="Ask before destructive actions",
            interrupt_on={
                "write_file": True,
                "edit_file": True,
                "delete": True,
                "execute": True,
            },
            allowed_tools=None,
        ),
    )
    api.add_mode(
        "edit",
        ModeSpec(
            description="Allow file changes but ask before execution",
            interrupt_on={"execute": True},
            allowed_tools=None,
        ),
    )
    api.add_mode(
        "yolo",
        ModeSpec(
            description="Allow all actions",
            interrupt_on={},
            allowed_tools=None,
        ),
    )
    api.add_mode(
        "plan",
        ModeSpec(
            description="Read-only planning",
            interrupt_on={},
            allowed_tools={"ls", "read_file", "glob", "grep"},
        ),
    )
