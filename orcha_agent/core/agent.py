"""Build the compiled deepagents graph from plugin registrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from deepagents.middleware.summarization import create_summarization_middleware

from .config import Config
from .events import AgentBuildAfter, AgentBuildBefore, EventBus
from .models import ModelResolver
from .registry import Registry
from .session import SessionStore

DEFAULT_SYSTEM_PROMPT = "You are a careful terminal coding agent. Use tools deliberately and report concrete results."


def _memory_sources(cfg: Config) -> list[str]:
    sources: list[str] = []
    for name in cfg.memory:
        path = Path(name)
        host_path = path if path.is_absolute() else cfg.cwd / path
        if host_path.is_file():
            sources.append(name)
    return sources


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


async def build_agent(
    registry: Registry,
    cfg: Config,
    session: SessionStore,
    bus: EventBus,
    *,
    always_allowed: Iterable[str] = (),
) -> Any:
    """Resolve plugin contributions and compile one deepagents graph."""

    if cfg.mode not in registry.modes:
        available = ", ".join(sorted(registry.modes))
        raise ValueError(f"unknown mode {cfg.mode!r}; available modes: {available}")
    if cfg.backend not in registry.backends:
        available = ", ".join(sorted(registry.backends))
        raise ValueError(f"unknown backend {cfg.backend!r}; available backends: {available}")

    resolver = ModelResolver(registry, cfg)
    roles = resolver.resolve_roles()
    backend = registry.backends[cfg.backend].factory(cfg)
    mode = registry.modes[cfg.mode]
    allowed = set(always_allowed)
    interrupts = {name: value for name, value in mode.interrupt_on.items() if name not in allowed}
    middleware = [entry.middleware for entry in registry.middleware]
    filesystem: FilesystemMiddleware | None = None
    if mode.allowed_tools is not None:
        filesystem = FilesystemMiddleware(
            backend=backend,
            tools=sorted(mode.allowed_tools),
        )
        middleware.append(filesystem)
    middleware.append(create_summarization_middleware(roles["summarizer"], backend))

    prompt = "\n\n".join(fragment.text for fragment in registry.prompt_fragments)
    kwargs: dict[str, Any] = {
        "model": roles["main"],
        "tools": list(registry.tools.values()),
        "middleware": middleware,
        "subagents": _subagents(
            registry,
            resolver,
            roles["subagent"],
            filesystem,
            _general_purpose_enabled(registry, cfg),
        ),
        "backend": backend,
        "memory": _memory_sources(cfg),
        "interrupt_on": interrupts,
        "system_prompt": prompt or DEFAULT_SYSTEM_PROMPT,
        "checkpointer": session.saver,
    }
    await bus.emit(AgentBuildBefore(kwargs))
    graph = create_deep_agent(**kwargs)
    await bus.emit(AgentBuildAfter(graph))
    return graph
