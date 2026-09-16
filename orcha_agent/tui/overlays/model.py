"""Model selection overlay."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.formatted_text import StyleAndTextTuples

from .select import SelectList
from orcha_agent.tui.statusline import _window, _quantity
from orcha_agent.core.usage import DEFAULT_PRICING


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
                window = _window(ctx, spec)
                context = f"  {_quantity(window)} ctx" if window else ""
                price = {
                    **DEFAULT_PRICING.get(spec, {}),
                    **getattr(ctx.cfg, "pricing", {}).get(spec, {}),
                }
                cost = (
                    f"  ${price['input']:g}/${price['output']:g} /M"
                    if "input" in price and "output" in price
                    else ""
                )
                labels[spec] = (
                    f"{glyph} {provider_name}  {model_name}{marker}{context}{cost}{suffix}"
                )
                models.append(spec)
        super().__init__(
            "Models",
            models,
            label=labels.__getitem__,
            empty_text="No models registered",
            on_accept=ctx.switch_model,
        )

    def _fragments(self) -> StyleAndTextTuples:
        filtered = self._filtered_pairs()
        if not filtered:
            return super()._fragments()
        fragments: StyleAndTextTuples = []
        if self._error is not None:
            fragments.append(("class:error", f"  {self._error}\n"))
        previous = None
        for visible, (_original, spec) in enumerate(filtered):
            provider = spec.partition(":")[0]
            if provider != previous:
                fragments.append(("class:overlay.section", f" {provider}\n"))
                previous = provider
            current = visible == self.index
            if current:
                fragments.append(("[SetCursorPosition]", ""))
            marker = "›" if current else " "
            style = "class:overlay.selection" if current else "class:overlay.item"
            fragments.append((style, f" {marker} {self.label(spec)}\n"))
        return fragments


__all__ = ["ModelOverlay"]
