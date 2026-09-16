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
from deepagents.backends import CompositeBackend, LocalShellBackend
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from .compaction import CompactionMiddleware, Compactor

from .config import Config
from .events import AgentBuildAfter, AgentBuildBefore, EventBus
from .models import ModelResolver
from .model_fallback import ModelFallbackMiddleware
from .registry import Registry
from .session import SessionStore
from .tools.approvals import approval_configs
from .tools.backend import GuardedLocalShellBackend
from .tools.common import PathPolicy
from .tools.middleware import NativeFilesystemMiddleware, NativeOutputMiddleware

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
NATIVE_TOOL_ALIASES = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "execute": "bash",
}


def _native_names(names: Iterable[str]) -> set[str]:
    return {NATIVE_TOOL_ALIASES.get(name, name) for name in names}


def _native_interrupts(interrupts: Mapping[str, Any]) -> dict[str, Any]:
    mapped = {NATIVE_TOOL_ALIASES.get(name, name): value for name, value in interrupts.items()}
    if mapped.get("bash"):
        mapped.setdefault("bash_jobs", mapped["bash"])
    return mapped


def _filesystem_middleware(backend: Any, names: set[str], native: bool = False) -> FilesystemMiddleware:
    middleware_type = NativeFilesystemMiddleware if native else FilesystemMiddleware
    filesystem = middleware_type(backend=backend, tools=sorted(names | {"read_file"}))
    # deepagents 0.7.9 requires read_file at construction, even when all of its
    # file tools are replaced. Keep its overflow middleware with the exact scope.
    filesystem._enabled_tools = frozenset(names)
    filesystem.tools = [tool for tool in filesystem.tools if tool.name in names]
    return filesystem



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
    native_tools: list[Any] | None = None,
    policy: PathPolicy | None = None,
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
            scoped_filesystem = filesystem
            if native_tools is not None:
                if isinstance(spec.get("interrupt_on"), Mapping):
                    configured_spec["interrupt_on"] = _native_interrupts(spec["interrupt_on"])
                    if policy is not None:
                        configured_spec["interrupt_on"] = approval_configs(
                            configured_spec["interrupt_on"], policy, registry.tools,
                        )
                requested = spec.get("tools")
                if requested is None:
                    configured_spec["tools"] = native_tools
                else:
                    requested_names = _native_names(
                        item if isinstance(item, str) else item.name for item in requested
                    )
                    configured_spec["tools"] = [
                        item for item in requested if not isinstance(item, str)
                        and item.name not in {tool.name for tool in native_tools}
                    ] + [tool for tool in native_tools if tool.name in requested_names]
                    selected_names = {tool.name for tool in configured_spec["tools"]}
                    filesystem_names = {tool.name for tool in filesystem.tools} if filesystem else set()
                    for requested_name in requested_names - selected_names - filesystem_names:
                        if requested_name not in registry.tools:
                            raise ValueError(f"Unknown subagent tool: {requested_name}")
                        configured_spec["tools"].append(registry.tools[requested_name])
                    if filesystem is not None:
                        scoped_filesystem = _filesystem_middleware(
                            filesystem.backend,
                            {tool.name for tool in filesystem.tools} & requested_names,
                            native=True,
                        )
            if filesystem is not None:
                configured_spec["middleware"] = [
                    *(spec.get("middleware") or ()),
                    scoped_filesystem,
                    *([NativeOutputMiddleware()] if native_tools is not None else []),
                ]
            configured.append(configured_spec)
        else:
            configured.append(spec)

    if include_general_purpose and not any(
        isinstance(spec, dict) and spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"]
        for spec in configured
    ):
        general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "model": default_model}
        if native_tools is not None:
            general_purpose["tools"] = native_tools
        if filesystem is not None:
            general_purpose["middleware"] = [filesystem]
            if native_tools is not None:
                general_purpose["middleware"].append(NativeOutputMiddleware())
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
        cfg.summarizer_model or cfg.model_roles.get("summarizer") or cfg.model, "summarizer"
    )
    backend = registry.backends[cfg.backend].factory(cfg)
    mode = registry.modes[cfg.mode]
    native = (
        cfg.tools.native
        and cfg.backend == "local_shell"
        and {"read", "write", "edit", "bash"}.issubset(registry.tools)
    )
    policy = PathPolicy(cfg.cwd, cfg.tools.allowed_roots, cfg.tools.deny) if native else None
    if native:
        if isinstance(backend, LocalShellBackend):
            assert policy is not None
            backend = GuardedLocalShellBackend(policy, cfg.tools.max_read_bytes)
            backend = CompositeBackend(
                default=backend,
                routes={},
                artifacts_root=str(cfg.cwd / ".orcha" / "artifacts"),
            )
        mode = dataclass_replace(
            mode,
            interrupt_on=_native_interrupts(mode.interrupt_on),
            allowed_tools=(
                None if mode.allowed_tools is None else _native_names(mode.allowed_tools)
            ),
        )
        if tool_scope is not None:
            tool_scope = _native_names(tool_scope)
    allowed = _native_names(always_allowed) if native else set(always_allowed)
    interrupts = {name: value for name, value in mode.interrupt_on.items() if name not in allowed}
    if policy is not None:
        interrupts = approval_configs(interrupts, policy, registry.tools)
    middleware = [entry.middleware for entry in registry.middleware
                  if native or entry.plugin != "tools_native"]
    middleware.append(TodoListMiddleware())
    filesystem: FilesystemMiddleware | None = None
    if native or mode.allowed_tools is not None or tool_scope is not None:
        filesystem_tools = (
            set(FILESYSTEM_TOOL_NAMES)
            if mode.allowed_tools is None
            else set(mode.allowed_tools) & FILESYSTEM_TOOL_NAMES
        )
        if tool_scope is not None:
            filesystem_tools &= tool_scope
        if native:
            filesystem_tools &= {"delete"}
        filesystem = _filesystem_middleware(backend, filesystem_tools, native=native)
        middleware.append(filesystem)
    tools = [
        tool
        for name, tool in registry.tools.items()
        if (mode.allowed_tools is None or name in mode.allowed_tools)
        and (tool_scope is None or name in tool_scope)
        and (native or registry._tool_owners.get(name) != "tools_native")
    ]
    tools.extend(
        tool
        for tool in extra_tools
        if tool_scope is None or getattr(tool, "name", None) in tool_scope
    )
    if len(main_models) > 1:
        middleware.append(ModelFallbackMiddleware(*main_models[1:]))
    from .catalog import get_model
    from .models import expand_model_spec
    catalog_model = get_model(expand_model_spec(cfg.model, cfg)[0], cfg)
    compactor = Compactor(
        roles["summarizer"], cfg.compaction,
        (catalog_model.context_window or 128_000) if catalog_model else 128_000,
    )
    middleware.append(CompactionMiddleware(compactor))

    prompt = "\n\n".join(
        value
        for value in (
            system_prompt,
            *(fragment.text for fragment in registry.prompt_fragments
              if native or fragment.plugin != "tools_native"),
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
                [tool for tool in tools if tool.name in {
                    "read", "write", "edit", "bash", "bash_jobs", "grep", "glob", "ls"
                }] if native else None,
                policy,
            )
        ),
        "backend": backend,
        "memory": (
            []
            if getattr(getattr(cfg, "memory_store", None), "backend", "files") == "turso"
            else [str(cfg.cwd / path.lstrip("/")) for path in _memory_sources(cfg)]
            if native else _memory_sources(cfg)
        ),
        "interrupt_on": interrupts,
        "system_prompt": prompt or DEFAULT_SYSTEM_PROMPT,
        "checkpointer": session.saver,
    }
    effective_scope = tool_scope
    if mode.allowed_tools is not None:
        effective_scope = set(mode.allowed_tools)
        if tool_scope is not None:
            effective_scope &= tool_scope
    await bus.emit(AgentBuildBefore(
        kwargs,
        tool_scope=effective_scope,
        always_allowed=frozenset(allowed),
        mode_interrupt_on=dict(mode.interrupt_on),
    ))
    compactor.attach_bus(bus)
    graph = _create_graph(
        kwargs, exclude_general_purpose=exclude_general_purpose
    )
    await bus.emit(AgentBuildAfter(graph))
    from .usage_store import UsageCallback

    if hasattr(graph, "with_config"):
        graph = graph.with_config(callbacks=[UsageCallback(session, cfg, bus=bus)])
    return graph
