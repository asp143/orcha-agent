from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

CommandHandler: TypeAlias = Callable[[Any, str], Awaitable[None]]
ProviderFactory: TypeAlias = Callable[[str, Mapping[str, Any]], Any]
BackendFactory: TypeAlias = Callable[[Any], Any]
RendererMatch: TypeAlias = str | Callable[[Any], bool]
Renderer: TypeAlias = Callable[[Any], Any | None]
AvailabilityCheck: TypeAlias = Callable[[], str | None]


@dataclass(frozen=True, slots=True)
class CommandRegistration:
    plugin: str
    handler: CommandHandler
    help: str


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    plugin: str
    factory: ProviderFactory
    capabilities: Any
    env_keys: tuple[str, ...]
    harness: Any | None
    available: AvailabilityCheck
    foreign_block_types: frozenset[str]


@dataclass(frozen=True, slots=True)
class BackendRegistration:
    plugin: str
    factory: BackendFactory


@dataclass(frozen=True, slots=True)
class MiddlewareRegistration:
    plugin: str
    priority: int
    middleware: Any
    name: str


@dataclass(frozen=True, slots=True)
class RendererRegistration:
    plugin: str
    priority: int
    match: RendererMatch
    render: Renderer
    name: str


@dataclass(frozen=True, slots=True)
class SubagentRegistration:
    plugin: str
    priority: int
    spec: Any
    model: str | None
    name: str


@dataclass(frozen=True, slots=True)
class PromptFragment:
    plugin: str
    priority: int
    text: str


def _value_name(value: Any) -> str:
    if isinstance(value, Mapping):
        name = value.get("name")
    else:
        name = getattr(value, "name", None) or getattr(value, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(value).__name__


def _match_name(match: RendererMatch) -> str:
    return match if isinstance(match, str) else _value_name(match)


class Registry:
    """All contributions registered by plugins.

    Tools and modes are exposed as their raw values because they are consumed
    directly while building an agent. Other named contributions retain their
    registration metadata.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.commands: dict[str, CommandRegistration] = {}
        self.providers: dict[str, ProviderRegistration] = {}
        self.backends: dict[str, BackendRegistration] = {}
        self.modes: dict[str, Any] = {}

        self.middleware: list[MiddlewareRegistration] = []
        self.renderers: list[RendererRegistration] = []
        self.subagents: list[SubagentRegistration] = []
        self.prompt_fragments: list[PromptFragment] = []

        self._tool_owners: dict[str, str] = {}
        self._command_owners: dict[str, str] = {}
        self._provider_owners: dict[str, str] = {}
        self._backend_owners: dict[str, str] = {}
        self._mode_owners: dict[str, str] = {}
        self._middleware_owners: dict[str, str] = {}
        self._renderer_owners: dict[str, str] = {}
        self._subagent_owners: dict[str, str] = {}

    @staticmethod
    def _claim(
        kind: str,
        name: str,
        plugin: str,
        owners: dict[str, str],
        *,
        replace: bool,
    ) -> None:
        if not name:
            raise ValueError(f"{kind} name cannot be empty")
        owner = owners.get(name)
        if owner is not None and not replace:
            raise ValueError(
                f"{kind} {name!r} is already registered by plugin {owner!r}; "
                f"plugin {plugin!r} must pass replace=True to replace it"
            )
        owners[name] = plugin

    def _add_tool(
        self,
        plugin: str,
        tool: Any,
        *,
        replace: bool = False,
    ) -> None:
        name = _value_name(tool)
        self._claim("tool", name, plugin, self._tool_owners, replace=replace)
        self.tools[name] = tool

    def _add_command(
        self,
        plugin: str,
        name: str,
        handler: CommandHandler,
        help: str,
        *,
        replace: bool = False,
    ) -> None:
        self._claim("command", name, plugin, self._command_owners, replace=replace)
        self.commands[name] = CommandRegistration(plugin=plugin, handler=handler, help=help)

    def _add_provider(
        self,
        plugin: str,
        prefix: str,
        factory: ProviderFactory,
        *,
        capabilities: Any,
        env_keys: Sequence[str] = (),
        harness: Any | None = None,
        foreign_block_types: Sequence[str] = (),
        available: AvailabilityCheck,
        replace: bool = False,
    ) -> None:
        self._claim("provider", prefix, plugin, self._provider_owners, replace=replace)
        self.providers[prefix] = ProviderRegistration(
            plugin=plugin,
            factory=factory,
            capabilities=capabilities,
            env_keys=tuple(env_keys),
            harness=harness,
            available=available,
            foreign_block_types=frozenset(foreign_block_types),
        )

    def _add_backend(
        self,
        plugin: str,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> None:
        self._claim("backend", name, plugin, self._backend_owners, replace=replace)
        self.backends[name] = BackendRegistration(plugin=plugin, factory=factory)

    def _add_mode(
        self,
        plugin: str,
        name: str,
        spec: Any,
        *,
        replace: bool = False,
    ) -> None:
        self._claim("mode", name, plugin, self._mode_owners, replace=replace)
        self.modes[name] = spec

    def _add_middleware(
        self,
        plugin: str,
        middleware: Any,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        name = _value_name(middleware)
        self._claim(
            "middleware",
            name,
            plugin,
            self._middleware_owners,
            replace=replace,
        )
        if replace:
            self.middleware[:] = [entry for entry in self.middleware if entry.name != name]
        self.middleware.append(
            MiddlewareRegistration(
                plugin=plugin,
                priority=priority,
                middleware=middleware,
                name=name,
            )
        )
        self.middleware.sort(key=lambda entry: (entry.priority, entry.plugin, entry.name))

    def _add_renderer(
        self,
        plugin: str,
        match: RendererMatch,
        render: Renderer,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        name = _match_name(match)
        self._claim(
            "renderer",
            name,
            plugin,
            self._renderer_owners,
            replace=replace,
        )
        if replace:
            self.renderers[:] = [entry for entry in self.renderers if entry.name != name]
        self.renderers.append(
            RendererRegistration(
                plugin=plugin,
                priority=priority,
                match=match,
                render=render,
                name=name,
            )
        )
        self.renderers.sort(key=lambda entry: (entry.priority, entry.plugin, entry.name))

    def _add_subagent(
        self,
        plugin: str,
        spec: Any,
        *,
        model: str | None = None,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        name = _value_name(spec)
        self._claim(
            "subagent",
            name,
            plugin,
            self._subagent_owners,
            replace=replace,
        )
        if replace:
            self.subagents[:] = [entry for entry in self.subagents if entry.name != name]
        self.subagents.append(
            SubagentRegistration(
                plugin=plugin,
                priority=priority,
                spec=spec,
                model=model,
                name=name,
            )
        )
        self.subagents.sort(key=lambda entry: (entry.priority, entry.plugin, entry.name))

    def _add_prompt_fragment(
        self,
        plugin: str,
        text: str,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        if replace:
            self.prompt_fragments[:] = [entry for entry in self.prompt_fragments if entry.text != text]
        self.prompt_fragments.append(PromptFragment(plugin=plugin, priority=priority, text=text))
        self.prompt_fragments.sort(key=lambda entry: (entry.priority, entry.plugin, entry.text))


__all__ = [
    "AvailabilityCheck",
    "BackendFactory",
    "BackendRegistration",
    "CommandHandler",
    "CommandRegistration",
    "MiddlewareRegistration",
    "PromptFragment",
    "ProviderFactory",
    "ProviderRegistration",
    "Registry",
    "Renderer",
    "RendererMatch",
    "RendererRegistration",
    "SubagentRegistration",
]
