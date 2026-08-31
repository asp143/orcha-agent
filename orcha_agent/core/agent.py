"""Build the compiled deepagents graph from plugin registrations."""

from __future__ import annotations

import logging
from html import escape
from collections.abc import Iterable, Mapping
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    create_deep_agent,
)
import deepagents.graph as deepagents_graph
from langchain.agents.middleware import ModelFallbackMiddleware, TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from deepagents.middleware.summarization import create_summarization_middleware

from .config import Config
from .events import AgentBuildAfter, AgentBuildBefore, EventBus
from .models import ModelResolver
from .registry import Registry
from .session import SessionStore

DEFAULT_SYSTEM_PROMPT = "You are a careful terminal coding agent. Use tools deliberately and report concrete results."
FILESYSTEM_TOOL_NAMES = {
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",
}
logger = logging.getLogger(__name__)



def _memory_sources(cfg: Config) -> list[str]:
    sources: list[str] = []
    root = cfg.cwd.resolve()
    for name in cfg.memory:
        path = Path(name)
        host_path = (path if path.is_absolute() else root / path).resolve()
        if not host_path.is_file():
            continue
        try:
            relative = host_path.relative_to(root)
        except ValueError:
            logger.warning("Skipping memory source outside workspace: %s", host_path)
            continue
        sources.append(f"/{relative.as_posix()}")
    return sources


def _structured_memory_prompt(cfg: Config, session: SessionStore) -> str:
    store = getattr(session, "structured_memory", None)
    settings = getattr(cfg, "memory_store", None)
    workspace = getattr(settings, "workspace", None)
    if store is None or not isinstance(workspace, str) or not workspace:
        return ""

    documents = list(store.resolve(workspace=workspace))
    path_documents = [
        document
        for document in store.all(include_deleted=True)
        if str(document.scope) == "path" and document.workspace == workspace
    ]
    documents.extend(document for document in path_documents if not document.deleted)
    suppressions = [document for document in path_documents if document.deleted]
    if not documents and not suppressions:
        return ""

    def precedence(document: Any) -> tuple[int, int, str, str]:
        scope = str(document.scope)
        path = str(document.path or "")
        rank = {"global": 0, "workspace": 1, "path": 2}.get(scope, 3)
        depth = len([part for part in path.split("/") if part])
        return rank, depth, path, document.id

    lines = [
        "<stored_memories>",
        "These are user-approved durable memories. Apply a path memory only when "
        "working in that path or its descendants; sibling paths do not inherit it. A "
        "memory-suppression disables the named less-specific memory in that path and "
        "its descendants. More specific path memories override workspace memories, "
        "which override global memories. Current user instructions and closer "
        "repository memory files take precedence.",
    ]
    for document in sorted(documents, key=precedence):
        attributes = [
            f'scope="{escape(str(document.scope), quote=True)}"',
            f'name="{escape(document.id, quote=True)}"',
        ]
        if document.path is not None:
            attributes.append(f'path="{escape(str(document.path), quote=True)}"')
        lines.append(f"<memory {' '.join(attributes)}>")
        # Stored content is model-written: escape it so it cannot forge
        # </memory> framing or sibling tags inside the system prompt.
        lines.append(escape(document.content))
        lines.append("</memory>")
    for document in sorted(suppressions, key=precedence):
        lines.append(
            "<memory-suppression "
            f'name="{escape(document.id, quote=True)}" '
            f'path="{escape(str(document.path), quote=True)}" />'
        )
    lines.append("</stored_memories>")
    return "\n".join(lines)



def _configured_model_specs(
    spec: str | list[str],
    aliases: Mapping[str, str | list[str]],
    seen: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    if isinstance(spec, list):
        return tuple(
            resolved
            for item in spec
            for resolved in _configured_model_specs(item, aliases, seen)
        )
    target = aliases.get(spec)
    if target is None or spec in seen:
        return (spec,)
    return _configured_model_specs(target, aliases, seen | {spec})


def _general_purpose_enabled(registry: Registry, cfg: Config) -> bool:
    for spec in _configured_model_specs(cfg.model, cfg.models):
        prefix, separator, _ = spec.partition(":")
        if not separator:
            continue
        provider = registry.providers.get(prefix)
        harness = None if provider is None else provider.harness
        general = (
            harness.get("general_purpose_subagent")
            if isinstance(harness, Mapping)
            else getattr(harness, "general_purpose_subagent", None)
        )
        enabled = (
            general.get("enabled")
            if isinstance(general, Mapping)
            else getattr(general, "enabled", None)
        )
        if enabled is False:
            return False
    return True


def _subagents(
    registry: Registry,
    resolver: ModelResolver,
    default_model: Any,
    filesystem: FilesystemMiddleware | None,
    include_general_purpose: bool,
) -> list[Any]:
    configured: list[Any] = []
    for entry in registry.subagents:
        spec = entry.spec
        if filesystem is not None and (
            not isinstance(spec, dict)
            or "runnable" in spec
            or "graph_id" in spec
        ):
            continue
        model = (
            default_model
            if entry.model is None
            else resolver.resolve(entry.model, f"subagent:{entry.name}")
        )
        if isinstance(spec, dict):
            configured_spec = {**spec, "model": model}
            if filesystem is not None:
                configured_spec["middleware"] = [
                    *(spec.get("middleware") or ()),
                    filesystem,
                ]
            configured.append(configured_spec)
        else:
            configured.append(spec)

    if include_general_purpose and not any(
        isinstance(spec, dict) and spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"]
        for spec in configured
    ):
        general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "model": default_model}
        if filesystem is not None:
            general_purpose["middleware"] = [filesystem]
        configured.insert(0, general_purpose)
    return configured


def _create_graph(kwargs: dict[str, Any], *, exclude_general_purpose: bool) -> Any:
    if not exclude_general_purpose:
        return create_deep_agent(**kwargs)
    resolve_profile = deepagents_graph._harness_profile_for_model

    def without_general_purpose(model: Any, spec: str | None) -> Any:
        profile = resolve_profile(model, spec)
        return dataclass_replace(
            profile,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )

    deepagents_graph._harness_profile_for_model = without_general_purpose
    try:
        return create_deep_agent(**kwargs)
    finally:
        deepagents_graph._harness_profile_for_model = resolve_profile


async def build_agent(
    registry: Registry,
    cfg: Config,
    session: SessionStore,
    bus: EventBus,
    *,
    always_allowed: Iterable[str] = (),
    extra_tools: Iterable[Any] = (),
    system_prompt: str | None = None,
    exclude_general_purpose: bool = False,
    tool_scope: set[str] | None = None,
) -> Any:
    """Resolve plugin contributions and compile one deepagents graph."""

    if cfg.mode not in registry.modes:
        available = ", ".join(sorted(registry.modes))
        raise ValueError(f"unknown mode {cfg.mode!r}; available modes: {available}")
    if cfg.backend not in registry.backends:
        available = ", ".join(sorted(registry.backends))
        raise ValueError(f"unknown backend {cfg.backend!r}; available backends: {available}")

    resolver = ModelResolver(registry, cfg)
    main_models = resolver.resolve_chain(cfg.model, "main")
    roles = {"main": main_models[0]}
    subagent_model = (
        None
        if exclude_general_purpose
        else resolver.resolve(cfg.subagent_model or cfg.model, "subagent")
    )
    roles["summarizer"] = resolver.resolve(
        cfg.summarizer_model or cfg.model, "summarizer"
    )
    backend = registry.backends[cfg.backend].factory(cfg)
    mode = registry.modes[cfg.mode]
    allowed = set(always_allowed)
    interrupts = {name: value for name, value in mode.interrupt_on.items() if name not in allowed}
    middleware = [entry.middleware for entry in registry.middleware]
    middleware.append(TodoListMiddleware())
    filesystem: FilesystemMiddleware | None = None
    if mode.allowed_tools is not None or tool_scope is not None:
        filesystem_tools = (
            set(FILESYSTEM_TOOL_NAMES)
            if mode.allowed_tools is None
            else set(mode.allowed_tools) & FILESYSTEM_TOOL_NAMES
        )
        if tool_scope is not None:
            filesystem_tools &= tool_scope
        constructor_tools = filesystem_tools | {"read_file"}
        filesystem = FilesystemMiddleware(
            backend=backend,
            tools=sorted(constructor_tools),
        )
        # deepagents 0.7.9 requires read_file during construction even when a
        # custom mode allows no filesystem tools. Remove that synthetic tool
        # afterward so the effective allowlist remains exact.
        if "read_file" not in filesystem_tools:
            filesystem._enabled_tools = frozenset(filesystem_tools)
            filesystem.tools = [
                tool for tool in filesystem.tools if tool.name in filesystem_tools
            ]
        middleware.append(filesystem)
    tools = [
        tool
        for name, tool in registry.tools.items()
        if (mode.allowed_tools is None or name in mode.allowed_tools)
        and (tool_scope is None or name in tool_scope)
    ]
    tools.extend(
        tool
        for tool in extra_tools
        if tool_scope is None or getattr(tool, "name", None) in tool_scope
    )
    if len(main_models) > 1:
        middleware.append(ModelFallbackMiddleware(*main_models[1:]))
    middleware.append(create_summarization_middleware(roles["summarizer"], backend))

    prompt = "\n\n".join(
        value
        for value in (
            system_prompt,
            *(fragment.text for fragment in registry.prompt_fragments),
            _structured_memory_prompt(cfg, session),
        )
        if value
    )
    kwargs: dict[str, Any] = {
        "model": roles["main"],
        "tools": tools,
        "middleware": middleware,
        "subagents": (
            []
            if exclude_general_purpose
            else _subagents(
                registry,
                resolver,
                subagent_model,
                filesystem,
                _general_purpose_enabled(registry, cfg),
            )
        ),
        "backend": backend,
        "memory": (
            []
            if getattr(getattr(cfg, "memory_store", None), "backend", "files") == "turso"
            else _memory_sources(cfg)
        ),
        "interrupt_on": interrupts,
        "system_prompt": prompt or DEFAULT_SYSTEM_PROMPT,
        "checkpointer": session.saver,
    }
    await bus.emit(AgentBuildBefore(kwargs))
    graph = _create_graph(
        kwargs, exclude_general_purpose=exclude_general_purpose
    )
    await bus.emit(AgentBuildAfter(graph))
    return graph
