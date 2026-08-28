from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .auth import AuthFlow
    from .events import Event, EventBus
    from .registry import BlockRenderer, Registry, Renderer, RendererMatch


@dataclass(frozen=True, slots=True)
class PluginSpec:
    name: str
    version: str = "0"
    requires: Sequence[str] = ()
    priority: int = 100


@dataclass(frozen=True, slots=True)
class ModeSpec:
    description: str
    interrupt_on: dict[str, bool]
    allowed_tools: set[str] | None


@dataclass(frozen=True, slots=True)
class ProviderCaps:
    tool_calling: bool
    streaming: bool
    thinking: bool
    structured_output: bool
    max_context: int | None


class Handled:
    """A handler result that prevents later handlers from running."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Resolved(Handled):
    resume_value: Any


CommandHandler: TypeAlias = Callable[[Any, str], Awaitable[None]]
ProviderFactory: TypeAlias = Callable[[str, Mapping[str, Any]], Any]
BackendFactory: TypeAlias = Callable[[Any], Any]
EventHandler: TypeAlias = Callable[[Any], Awaitable[Handled | None]]
AvailabilityCheck: TypeAlias = Callable[[], str | None]


def _available() -> str | None:
    return None


class PluginAPI:
    """The only mutation surface exposed to plugin registration functions."""

    def __init__(
        self,
        *,
        name: str,
        config: Mapping[str, Any],
        state: dict[str, Any],
        registry: Registry,
        bus: EventBus,
        request_rebuild: Callable[[], None],
    ) -> None:
        self.name = name
        self.config = config
        self.state = state
        self._registry = registry
        self._bus = bus
        self._request_rebuild = request_rebuild

    def add_auth(
        self,
        prefix: str,
        flow: AuthFlow,
        *,
        replace: bool = False,
    ) -> None:
        self._registry._add_auth(
            self.name,
            prefix,
            flow,
            replace=replace,
        )


    def add_tool(self, tool: Any, *, replace: bool = False) -> None:
        self._registry._add_tool(self.name, tool, replace=replace)

    def add_middleware(
        self,
        middleware: Any,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_middleware(
            self.name,
            middleware,
            priority=priority,
            replace=replace,
        )

    def add_command(
        self,
        name: str,
        handler: CommandHandler,
        help: str,
        *,
        replace: bool = False,
    ) -> None:
        self._registry._add_command(
            self.name,
            name,
            handler,
            help,
            replace=replace,
        )

    def add_renderer(
        self,
        match: RendererMatch,
        render: Renderer,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_renderer(
            self.name,
            match,
            render,
            priority=priority,
            replace=replace,
        )

    def add_block_renderer(
        self,
        kind: str,
        render: BlockRenderer,
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_block_renderer(
            self.name,
            kind,
            render,
            priority=priority,
            replace=replace,
        )

    def add_status_segment(
        self,
        name: str,
        render: Callable[[Any], Any | None],
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_status_segment(
            self.name,
            name,
            render,
            priority=priority,
            replace=replace,
        )

    def add_completer(
        self,
        trigger: str,
        fn: Callable[[Any], Any],
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_completer(
            self.name,
            trigger,
            fn,
            priority=priority,
            replace=replace,
        )

    def add_keybinding(
        self,
        action: str,
        handler: Callable[[Any, Any], Any],
        default: str | Sequence[str] = "",
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_keybinding(
            self.name,
            action,
            handler,
            default,
            priority=priority,
            replace=replace,
        )

    def add_overlay(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        self._registry._add_overlay(
            self.name,
            name,
            factory,
            priority=priority,
            replace=replace,
        )


    def add_provider(
        self,
        prefix: str,
        factory: ProviderFactory,
        *,
        capabilities: ProviderCaps,
        models: Sequence[str] = (),
        default_model: str | None = None,
        env_keys: Sequence[str] = (),
        harness: Any | None = None,
        available: AvailabilityCheck = _available,
        foreign_block_types: Sequence[str] = (),
        replace: bool = False,
    ) -> None:
        self._registry._add_provider(
            self.name,
            prefix,
            factory,
            capabilities=capabilities,
            env_keys=env_keys,
            models=models,
            default_model=default_model,
            harness=harness,
            available=available,
            foreign_block_types=foreign_block_types,
            replace=replace,
        )

    def add_backend(
        self,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> None:
        self._registry._add_backend(self.name, name, factory, replace=replace)

    def add_subagent(
        self,
        spec: Any,
        *,
        model: str | None = None,
        replace: bool = False,
    ) -> None:
        self._registry._add_subagent(
            self.name,
            spec,
            model=model,
            priority=100,
            replace=replace,
        )

    def add_mode(
        self,
        name: str,
        spec: ModeSpec,
        *,
        replace: bool = False,
    ) -> None:
        self._registry._add_mode(self.name, name, spec, replace=replace)

    def on(
        self,
        event_type: type[Event],
        handler: EventHandler,
        *,
        priority: int = 100,
    ) -> None:
        self._bus.on(event_type, handler, plugin=self.name, priority=priority)

    def system_prompt_fragment(
        self,
        text: str,
        *,
        priority: int = 100,
    ) -> None:
        self._registry._add_prompt_fragment(self.name, text, priority=priority)

    def request_rebuild(self) -> None:
        self._request_rebuild()


__all__ = [
    "Handled",
    "ModeSpec",
    "PluginAPI",
    "PluginSpec",
    "ProviderCaps",
    "Resolved",
]
