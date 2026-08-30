"""Compatibility re-exports for the split terminal application."""

from orcha_agent.core.agent import build_agent
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.models import strip_foreign_blocks
from orcha_agent.core.session import SessionStore

from .console import ConsoleOutput
from .context import AppContext, EventBusView, RegistryView, _stored_model
from .runtime import (
    ApplicationRuntime,
    UIFacade,
    _bindings,
    _bottom_toolbar,
    _history_path,
    dispatch_command,
    run_app,
)
from .turn import (
    _ModelLabelBuffer,
    _ToolCallBuffer,
    _message_event,
    _render,
    _run_cancellable_turn,
    _run_turn,
    _updates_event,
)


__all__ = [
    "AppContext",
    "ApplicationRuntime",
    "EventBusView",
    "RegistryView",
    "UIFacade",
    "_ModelLabelBuffer",
    "_ToolCallBuffer",
    "_bindings",
    "_bottom_toolbar",
    "_history_path",
    "_message_event",
    "_render",
    "_run_cancellable_turn",
    "_run_turn",
    "_stored_model",
    "_updates_event",
    "dispatch_command",
    "run_app",
]
