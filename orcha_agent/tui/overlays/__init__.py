"""Interactive prompt-toolkit overlays and first-party factories."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .approval import ApprovalOverlay
    from .ask import AskOverlay
    from .base import Anchor, Overlay
    from .help import HelpOverlay, KeyBindingsCard, KeyBindingsOverlay
    from .hub import HubOverlay
    from .history import HistoryOverlay
    from .model import ModelOverlay
    from .select import SelectList
    from .session import SessionOverlay
    from .settings import SettingsOverlay
    from .theme import ThemeOverlay
    from .tree import TreeOverlay

_OVERLAY_MODULES = {
    "ApprovalOverlay": "approval",
    "AskOverlay": "ask",
    "Anchor": "base",
    "Overlay": "base",
    "HelpOverlay": "help",
    "KeyBindingsCard": "help",
    "KeyBindingsOverlay": "help",
    "HubOverlay": "hub",
    "HistoryOverlay": "history",
    "ModelOverlay": "model",
    "SelectList": "select",
    "SessionOverlay": "session",
    "SettingsOverlay": "settings",
    "ThemeOverlay": "theme",
    "TreeOverlay": "tree",
}


def __getattr__(name: str) -> Any:
    """Load only the requested overlay, keeping unrelated imports out of startup."""

    module = _OVERLAY_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def register_builtin_overlays(registry: Any) -> None:
    """Claim the first-party overlay names before third-party plugins load."""

    factories = {
        "settings": lambda ctx, **_payload: __getattr__("SettingsOverlay")(ctx),
        "model": lambda ctx, **_payload: __getattr__("ModelOverlay")(ctx),
        "session": lambda ctx, **_payload: __getattr__("SessionOverlay")(ctx),
        "hub": lambda ctx, **_payload: __getattr__("HubOverlay")(ctx),
        "tree": lambda ctx, **_payload: __getattr__("TreeOverlay")(ctx),
        "theme": lambda ctx, **_payload: __getattr__("ThemeOverlay")(ctx),
        "approval": lambda _ctx, action=None, **payload: __getattr__("ApprovalOverlay")(
            action, **payload
        ),
        "ask": lambda _ctx, questions, **_payload: __getattr__("AskOverlay")(questions),
        "history": lambda ctx, **_payload: __getattr__("HistoryOverlay")(ctx),
        "help": lambda ctx, **_payload: __getattr__("HelpOverlay")(ctx),
    }
    for name, factory in factories.items():
        registry._add_overlay("<core>", name, factory)


__all__ = [
    "Anchor",
    "ApprovalOverlay",
    "AskOverlay",
    "HelpOverlay",
    "HubOverlay",
    "KeyBindingsCard",
    "KeyBindingsOverlay",
    "HistoryOverlay",
    "ModelOverlay",
    "Overlay",
    "SelectList",
    "SessionOverlay",
    "SettingsOverlay",
    "ThemeOverlay",
    "TreeOverlay",
    "register_builtin_overlays",
]
