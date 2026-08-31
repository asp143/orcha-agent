from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from .agent_types import AgentType, builtin_agent_types

CommandHandler: TypeAlias = Callable[[Any, str], Awaitable[None]]
ProviderFactory: TypeAlias = Callable[[str, Mapping[str, Any]], Any]
BackendFactory: TypeAlias = Callable[[Any], Any]
RendererMatch: TypeAlias = str | Callable[[Any], bool]
Renderer: TypeAlias = Callable[[Any], Any | None]
BlockRenderer: TypeAlias = Callable[..., Any]
AvailabilityCheck: TypeAlias = Callable[[], str | None]
CompleterFunction: TypeAlias = Callable[[Any], Any]
KeybindingHandler: TypeAlias = Callable[[Any, Any], Any]
OverlayFactory: TypeAlias = Callable[..., Any]
CORE_KEY_ACTIONS = frozenset(
    {
        "submit",
        "newline",
        "queue",
        "dequeue",
        "toggle_thinking",
        "cycle_thinking_level",
        "expand_tools",
        "model_picker",
        "cycle_model",
        "history_search",
        "external_editor",
        "clear_screen",
        "interrupt",
        "exit",
        "tree",
    }
)


@dataclass(frozen=True, slots=True)
class AuthRegistration:
    plugin: str
    flow: Any


@dataclass(frozen=True, slots=True)
class CommandRegistration:
    plugin: str
    handler: CommandHandler
    help: str


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    plugin: str
    models: tuple[str, ...]
    default_model: str | None
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
class BlockRendererRegistration:
    plugin: str
    priority: int
    kind: str
    render: BlockRenderer


@dataclass(frozen=True, slots=True)
class StatusSegmentRegistration:
    name: str
    plugin: str
    priority: int
    render: Callable[[Any], Any | None]


@dataclass(frozen=True, slots=True)
class CompleterRegistration:
    plugin: str
    priority: int
    trigger: str
    fn: CompleterFunction


@dataclass(frozen=True, slots=True)
class KeybindingRegistration:
    plugin: str
    priority: int
    action: str
    handler: KeybindingHandler
    default: str | Sequence[str]


@dataclass(frozen=True, slots=True)
class OverlayRegistration:
    plugin: str
    priority: int
    name: str
    factory: OverlayFactory


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
        self.auth: dict[str, AuthRegistration] = {}
        self.commands: dict[str, CommandRegistration] = {}
        self.providers: dict[str, ProviderRegistration] = {}
        self.backends: dict[str, BackendRegistration] = {}
        self.modes: dict[str, Any] = {}

        self.middleware: list[MiddlewareRegistration] = []
        self.renderers: list[RendererRegistration] = []
        self.block_renderers: list[BlockRendererRegistration] = []
        self.status_segments: list[StatusSegmentRegistration] = []
        self.completers: list[CompleterRegistration] = []
        self.keybindings: dict[str, KeybindingRegistration] = {}
        self.subagents: list[SubagentRegistration] = []
        self.agent_types: dict[str, AgentType] = builtin_agent_types()
        self.overlays: dict[str, OverlayRegistration] = {}
        self.prompt_fragments: list[PromptFragment] = []

        self._tool_owners: dict[str, str] = {}
        self._auth_owners: dict[str, str] = {}
        self._command_owners: dict[str, str] = {}
        self._provider_owners: dict[str, str] = {}
        self._backend_owners: dict[str, str] = {}
        self._mode_owners: dict[str, str] = {}
        self._middleware_owners: dict[str, str] = {}
        self._status_segment_owners: dict[str, str] = {}
        self._completer_owners: dict[str, str] = {}
        self._keybinding_owners: dict[str, str] = {
            action: "<core>" for action in CORE_KEY_ACTIONS
        }
        self._renderer_owners: dict[object, str] = {}
        self._block_renderer_owners: dict[str, str] = {}
        self._subagent_owners: dict[str, str] = {}
        self._agent_type_owners: dict[str, str] = {
            name: "<core>" for name in self.agent_types
        }
        self._overlay_owners: dict[str, str] = {}

    @staticmethod
    def _claim(
        kind: str,
        name: object,
        plugin: str,
        owners: dict[object, str],
        *,
        replace: bool,
    ) -> None:
        if isinstance(name, str) and not name:
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

    def _add_auth(
        self,
        plugin: str,
        prefix: str,
        flow: Any,
        *,
        replace: bool = False,
    ) -> None:
        self._claim("auth", prefix, plugin, self._auth_owners, replace=replace)
        self.auth[prefix] = AuthRegistration(plugin=plugin, flow=flow)


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
        models: Sequence[str] = (),
        default_model: str | None = None,
        env_keys: Sequence[str] = (),
        harness: Any | None = None,
        foreign_block_types: Sequence[str] = (),
        available: AvailabilityCheck,
        replace: bool = False,
    ) -> None:
        self._claim("provider", prefix, plugin, self._provider_owners, replace=replace)
        self.providers[prefix] = ProviderRegistration(
            plugin=plugin,
            models=tuple(models),
            default_model=default_model,
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
        owner_key: object = match if callable(match) else name
        self._claim(
            "renderer",
            owner_key,
            plugin,
            self._renderer_owners,
            replace=replace,
        )
        if replace:
            self.renderers[:] = [
                entry
                for entry in self.renderers
                if not (
                    entry.match is match
                    if callable(match)
                    else entry.name == name
                )
            ]
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

    def _add_block_renderer(
        self,
        plugin: str,
        kind: str,
        render: BlockRenderer,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._claim(
            "block renderer",
            kind,
            plugin,
            self._block_renderer_owners,
            replace=replace,
        )
        if replace:
            self.block_renderers[:] = [
                entry for entry in self.block_renderers if entry.kind != kind
            ]
        self.block_renderers.append(
            BlockRendererRegistration(
                plugin=plugin,
                priority=priority,
                kind=kind,
                render=render,
            )
        )
        self.block_renderers.sort(
            key=lambda entry: (entry.priority, entry.plugin, entry.kind)
        )

    def _add_agent_type(
        self,
        plugin: str,
        spec: AgentType,
        *,
        replace: bool = False,
    ) -> None:
        self._claim(
            "agent type",
            spec.name,
            plugin,
            self._agent_type_owners,
            replace=replace,
        )
        self.agent_types[spec.name] = spec

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

    def _add_status_segment(
        self,
        plugin: str,
        name: str,
        render: Callable[[Any], Any | None],
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._claim(
            "status segment",
            name,
            plugin,
            self._status_segment_owners,
            replace=replace,
        )
        if replace:
            self.status_segments[:] = [
                entry for entry in self.status_segments if entry.name != name
            ]
        self.status_segments.append(
            StatusSegmentRegistration(
                name=name,
                plugin=plugin,
                priority=priority,
                render=render,
            )
        )
        self.status_segments.sort(key=lambda entry: (entry.priority, entry.name))

    def _add_completer(
        self,
        plugin: str,
        trigger: str,
        fn: CompleterFunction,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._claim(
            "completer",
            trigger,
            plugin,
            self._completer_owners,
            replace=replace,
        )
        if replace:
            self.completers[:] = [
                entry for entry in self.completers if entry.trigger != trigger
            ]
        self.completers.append(
            CompleterRegistration(
                plugin=plugin,
                priority=priority,
                trigger=trigger,
                fn=fn,
            )
        )
        self.completers.sort(
            key=lambda entry: (entry.priority, entry.plugin, entry.trigger)
        )

    def _add_keybinding(
        self,
        plugin: str,
        action: str,
        handler: KeybindingHandler,
        default: str | Sequence[str],
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._claim(
            "keybinding",
            action,
            plugin,
            self._keybinding_owners,
            replace=replace,
        )
        self.keybindings[action] = KeybindingRegistration(
            plugin=plugin,
            priority=priority,
            action=action,
            handler=handler,
            default=default,
        )

    def _add_overlay(
        self,
        plugin: str,
        name: str,
        factory: OverlayFactory,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._claim(
            "overlay",
            name,
            plugin,
            self._overlay_owners,
            replace=replace,
        )
        self.overlays[name] = OverlayRegistration(
            plugin=plugin,
            priority=priority,
            name=name,
            factory=factory,
        )

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
    "AuthRegistration",
    "BackendFactory",
    "BlockRenderer",
    "BlockRendererRegistration",
    "BackendRegistration",
    "CommandHandler",
    "CommandRegistration",
    "CompleterFunction",
    "CompleterRegistration",
    "KeybindingHandler",
    "KeybindingRegistration",
    "MiddlewareRegistration",
    "PromptFragment",
    "ProviderFactory",
    "ProviderRegistration",
    "Registry",
    "Renderer",
    "OverlayFactory",
    "OverlayRegistration",
    "RendererMatch",
    "RendererRegistration",
    "StatusSegmentRegistration",
    "SubagentRegistration",
]
