"""Prompt behavior for repository memory files."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from orcha_agent.core.memory_store import MemoryDocument, MemoryScope
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="memory", version="1.0.0")

_MEMORY_PROMPT = (
    "Repository memory files and stored memories are authoritative working context. "
    "Follow their instructions throughout the session, and when instructions differ, "
    "prefer the memory closest to the path being worked on. Never save, change, or "
    "delete memory unless the user explicitly asks you to. Never store credentials, "
    "API keys, access tokens, passwords, or private keys in memory."
)


def memory_tools(host: Any) -> tuple[StructuredTool, ...]:
    """Return structured-memory tools when the selected store supports them."""

    store = getattr(getattr(host, "session", None), "structured_memory", None)
    settings = getattr(getattr(host, "cfg", None), "memory_store", None)
    workspace = getattr(settings, "workspace", None)
    if store is None:
        return ()

    async def list_memories() -> list[dict[str, object]]:
        """List stored memory metadata without returning memory contents."""

        return [
            {
                "id": document.id,
                "scope": str(document.scope),
                "workspace": document.workspace,
                "path": str(document.path) if document.path is not None else None,
                "revision": document.revision,
            }
            for document in store.all()
            if document.scope is MemoryScope.GLOBAL or document.workspace == workspace
        ]

    async def read_memory(name: str) -> list[dict[str, object]]:
        """Read all visible stored-memory definitions with one logical name."""

        return [
            {
                "id": document.id,
                "scope": str(document.scope),
                "path": str(document.path) if document.path is not None else None,
                "revision": document.revision,
                "content": document.content,
            }
            for document in store.all()
            if document.id == name
            and (document.scope is MemoryScope.GLOBAL or document.workspace == workspace)
        ]

    async def save_memory(
        name: str,
        content: str,
        scope: str = "workspace",
        path: str | None = None,
    ) -> dict[str, object]:
        """Save user-requested durable memory; never call without explicit permission."""

        if scope == "global":
            if path is not None:
                raise ValueError("Global memory cannot have a path")
            document = MemoryDocument.global_document(name, content)
        elif scope == "workspace":
            if not isinstance(workspace, str) or not workspace:
                raise ValueError("A structured-memory workspace is required")
            if path is not None:
                raise ValueError("Workspace memory cannot have a path")
            document = MemoryDocument.workspace_document(name, content, workspace)
        elif scope == "path":
            if not isinstance(workspace, str) or not workspace or not path:
                raise ValueError("Path memory requires a workspace and path")
            document = MemoryDocument.path_document(name, content, workspace, path)
        else:
            raise ValueError("Memory scope must be global, workspace, or path")
        current = store.get(
            name,
            scope=document.scope,
            workspace=document.workspace,
            path=document.path,
            include_deleted=True,
        )
        saved = store.save(
            document,
            expected_revision=0 if current is None else current.revision,
        )
        rebuild = getattr(host, "request_rebuild", None)
        if callable(rebuild):
            rebuild()
        return {
            "id": saved.id,
            "scope": str(saved.scope),
            "revision": saved.revision,
        }

    return (
        StructuredTool.from_function(
            coroutine=list_memories,
            name="list_memories",
            description="List durable structured memories without exposing their contents.",
        ),
        StructuredTool.from_function(
            coroutine=read_memory,
            name="read_memory",
            description="Read visible durable memories with one exact logical name.",
        ),
        StructuredTool.from_function(
            coroutine=save_memory,
            name="save_memory",
            description=(
                "Save a durable global, workspace, or path memory only when the user "
                "explicitly asks you to remember or save it. Never store credentials."
            ),
        ),
    )


def register(api: PluginAPI) -> None:
    api.system_prompt_fragment(_MEMORY_PROMPT, priority=50)
