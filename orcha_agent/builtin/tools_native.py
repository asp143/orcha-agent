"""Native local tools, registered through the same API as third-party tools."""

from __future__ import annotations

from pathlib import Path

from orcha_agent.core.events import AppExit
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.core.tools.filesystem import create_filesystem_tools
from orcha_agent.core.tools.common import DEFAULT_DENY, PathPolicy
from orcha_agent.core.tools.middleware import NativeOutputMiddleware
from orcha_agent.core.tools.search import create_search_tools
from orcha_agent.core.tools.shell import close_shell_tools, create_shell_tools

PLUGIN = PluginSpec(name="tools_native", version="1.0.0", requires=("filesystem",))


def register(api: PluginAPI) -> None:
    if not api.config.get("native", True):
        return
    cwd = Path(api.config.get("cwd", Path.cwd()))
    policy = PathPolicy(
        cwd, api.config.get("allowed_roots", ()), api.config.get("deny", DEFAULT_DENY)
    )
    shell_tools = create_shell_tools(
        cwd,
        policy=policy,
        shell_env_passthrough=api.config.get("shell_env_passthrough", ()),
    )
    filesystem_tools = create_filesystem_tools(
        cwd,
        str(api.config.get("edit_format", "replace")),
        policy=policy,
        max_read_bytes=api.config.get("max_read_bytes", 64 * 1024 * 1024),
        read_summary=api.config.get("read_summary", False),
    )
    for native_tool in [*filesystem_tools, *create_search_tools(cwd, policy=policy), *shell_tools]:
        api.add_tool(native_tool)
    api.add_middleware(NativeOutputMiddleware())

    async def close(_event: AppExit) -> None:
        close_shell_tools(shell_tools)

    api.on(AppExit, close)
    api.system_prompt_fragment(
        "Use read for file contents (path:N-M or path:N+K), grep for content search, "
        "glob for file discovery and ls for directories. Read before edit; use write "
        "to create files. Use bash for builds, tests and other commands. Its working "
        "directory and exported environment persist within the session. Background "
        "commands return a job_id for bash_jobs. Tool results include recovery "
        "instructions when clipped."
    )
