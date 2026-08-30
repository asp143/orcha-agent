"""Model selection overlay."""

from __future__ import annotations

from typing import Any

from .select import SelectList


class ModelOverlay(SelectList[str]):
    def __init__(self, ctx: Any) -> None:
        current = getattr(getattr(ctx, "cfg", None), "model", "")
        current_models = {current} if isinstance(current, str) else set(current)
        labels: dict[str, str] = {}
        models: list[str] = []
        for provider_name, provider in sorted(ctx.registry.providers.items()):
            try:
                reason = provider.available()
            except Exception as exc:
                reason = str(exc)
            available = reason is None
            glyph = "●" if available else "○"
            for model_name in provider.models:
                spec = f"{provider_name}:{model_name}"
                marker = " *" if spec in current_models else ""
                suffix = "" if available else f" — {reason}"
                labels[spec] = f"{glyph} {provider_name}  {model_name}{marker}{suffix}"
                models.append(spec)
        super().__init__(
            "Models",
            models,
            label=labels.__getitem__,
            empty_text="No models registered",
            on_accept=ctx.switch_model,
        )


__all__ = ["ModelOverlay"]
