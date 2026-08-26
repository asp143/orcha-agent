"""Local filesystem and shell backend plugin."""

from __future__ import annotations

from typing import Any

from deepagents.backends import LocalShellBackend

from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="filesystem", version="1.0.0")


def _local_shell(config: Any) -> LocalShellBackend:
    return LocalShellBackend(root_dir=config.cwd)


def register(api: PluginAPI) -> None:
    api.add_backend("local_shell", _local_shell)
