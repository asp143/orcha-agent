"""Build the compiled deepagents graph from plugin registrations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware

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


def _subagents(registry: Registry, resolver: ModelResolver) -> list[Any]:
    configured: list[Any] = []
    for entry in registry.subagents:
        spec = entry.spec
        if entry.model is None:
            configured.append(spec)
            continue
        model = resolver.resolve(entry.model, f"subagent:{entry.name}")
        if isinstance(spec, dict):
            configured.append({**spec, "model": model})
        else:
            configured.append(spec)
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
    if mode.allowed_tools is not None:
        middleware.append(
            FilesystemMiddleware(backend=backend, tools=sorted(mode.allowed_tools))
        )

    prompt = "\n\n".join(fragment.text for fragment in registry.prompt_fragments)
    kwargs: dict[str, Any] = {
        "model": roles["main"],
        "tools": list(registry.tools.values()),
        "middleware": middleware,
        "subagents": _subagents(registry, resolver),
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
