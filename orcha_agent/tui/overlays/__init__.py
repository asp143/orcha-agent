"""Interactive prompt-toolkit overlays and first-party factories."""

from __future__ import annotations

from typing import Any

from .approval import ApprovalOverlay
from .ask import AskOverlay
from .base import Anchor, Overlay
from .help import HelpOverlay
from .history import HistoryOverlay
from .model import ModelOverlay
from .select import SelectList
from .session import SessionOverlay
from .theme import ThemeOverlay
from .tree import TreeOverlay


def register_builtin_overlays(registry: Any) -> None:
    """Claim the first-party overlay names before third-party plugins load."""

    factories = {
        "model": lambda ctx, **_payload: ModelOverlay(ctx),
        "session": lambda ctx, **_payload: SessionOverlay(ctx),
        "tree": lambda ctx, **_payload: TreeOverlay(ctx),
        "theme": lambda ctx, **_payload: ThemeOverlay(ctx),
        "approval": lambda _ctx, action=None, **payload: ApprovalOverlay(action, **payload),
        "ask": lambda _ctx, questions, **_payload: AskOverlay(questions),
        "history": lambda ctx, **_payload: HistoryOverlay(ctx),
        "help": lambda ctx, **_payload: HelpOverlay(ctx),
    }
    for name, factory in factories.items():
        registry._add_overlay("<core>", name, factory)


__all__ = [
    "Anchor",
    "ApprovalOverlay",
    "AskOverlay",
    "HelpOverlay",
    "HistoryOverlay",
    "ModelOverlay",
    "Overlay",
    "SelectList",
    "SessionOverlay",
    "ThemeOverlay",
    "TreeOverlay",
    "register_builtin_overlays",
]
