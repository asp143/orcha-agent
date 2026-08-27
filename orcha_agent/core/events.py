from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from .plugin import Handled


@dataclass(slots=True)
class Event:
    """Base type for events emitted by the application kernel."""


@dataclass(slots=True)
class AppStart(Event):
    ctx: Any


@dataclass(slots=True)
class AgentBuildBefore(Event):
    kwargs: dict[str, Any]


@dataclass(slots=True)
class AgentBuildAfter(Event):
    graph: Any


@dataclass(slots=True)
class TurnStart(Event):
    thread_id: str
    text: str


@dataclass(slots=True)
class ModelChunk(Event):
    chunk: Any
    role: str
    model_name: str | None = None
    source_id: str | None = None


@dataclass(slots=True)
class ToolCallStart(Event):
    name: str
    args: dict[str, Any]
    id: str
    source_id: str | None = None


@dataclass(slots=True)
class ToolCallEnd(Event):
    name: str
    id: str
    result: Any


@dataclass(slots=True)
class InterruptRaised(Event):
    payload: dict[str, Any]


@dataclass(slots=True)
class TurnEnd(Event):
    thread_id: str


@dataclass(slots=True)
class SessionSwitch(Event):
    old: str | None
    new: str | None


@dataclass(slots=True)
class ModelSwitch(Event):
    old: str
    new: str


@dataclass(slots=True)
class AppExit(Event):
    pass


EventHandler: TypeAlias = Callable[[Any], Awaitable[Handled | None]]


@dataclass(frozen=True, slots=True)
class EventHandlerRegistration:
    event_type: type[Event]
    handler: EventHandler
    plugin: str
    priority: int
    name: str


def _handler_name(handler: EventHandler) -> str:
    name = getattr(handler, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(handler).__name__


class EventBus:
    def __init__(self) -> None:
        self.handlers: list[EventHandlerRegistration] = []

    def on(
        self,
        event_type: type[Event],
        handler: EventHandler,
        *,
        plugin: str = "<anonymous>",
        priority: int = 100,
        replace: bool = False,
    ) -> None:
        name = _handler_name(handler)
        if replace:
            self.handlers[:] = [
                entry
                for entry in self.handlers
                if not (entry.event_type is event_type and entry.name == name)
            ]
        self.handlers.append(
            EventHandlerRegistration(
                event_type=event_type,
                handler=handler,
                plugin=plugin,
                priority=priority,
                name=name,
            )
        )
        self.handlers.sort(
            key=lambda entry: (
                entry.priority,
                entry.plugin,
                entry.name,
                entry.event_type.__name__,
            )
        )

    async def emit(self, event: Event) -> Handled | None:
        for registration in tuple(self.handlers):
            if not isinstance(event, registration.event_type):
                continue
            result = await registration.handler(event)
            if isinstance(result, Handled):
                return result
        return None


__all__ = [
    "AgentBuildAfter",
    "AgentBuildBefore",
    "AppExit",
    "AppStart",
    "Event",
    "EventBus",
    "EventHandlerRegistration",
    "InterruptRaised",
    "ModelChunk",
    "ModelSwitch",
    "SessionSwitch",
    "ToolCallEnd",
    "ToolCallStart",
    "TurnEnd",
    "TurnStart",
]
