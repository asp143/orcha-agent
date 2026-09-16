"""Theme picker with live preview and cancellation rollback."""

from __future__ import annotations

from typing import Any

from orcha_agent.tui.theme import _PORTED_NAMES

from .select import SelectList


class ThemeOverlay(SelectList[str]):
    def __init__(self, ctx: Any) -> None:
        themes = dict(getattr(ctx.ui, "themes", {}))
        previous = str(getattr(getattr(ctx.ui, "theme", None), "id", "dark"))

        previous = _PORTED_NAMES.get(previous, previous)

        def preview(name: str | None) -> None:
            if name is not None:
                ctx.ui.set_theme(name)

        def cancel() -> None:
            ctx.ui.set_theme(previous)

        def persist(name: str | list[str]) -> str:
            selected = str(name)
            ctx.ui.set_theme(selected)
            ctx.plugin_states.setdefault("commands_core", {})["theme"] = selected
            callback = getattr(ctx, "persist_plugin_states", None)
            if callback is not None:
                callback()
            return selected

        super().__init__(
            "Themes",
            sorted(name for name in themes if not name.endswith(("-legacy", "-omp"))),
            label=lambda name: f"{name}{' (current)' if name == previous else ''}",
            empty_text="No themes available",
            show_filter=False,
            on_change=preview,
            on_accept=persist,
            on_cancel=cancel,
        )


__all__ = ["ThemeOverlay"]
