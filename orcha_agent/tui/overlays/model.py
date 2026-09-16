"""Model selection overlay."""

from __future__ import annotations

from typing import Any

from .select import SelectList
from orcha_agent.tui.statusline import _quantity
from orcha_agent.core.catalog import get_catalog
from orcha_agent.core.model_roles import MODEL_ROLES
from orcha_agent.core.usage import DEFAULT_PRICING


class ModelOverlay(SelectList[str]):
    def __init__(self, ctx: Any, *, browse: bool = False) -> None:
        current = getattr(getattr(ctx, "cfg", None), "model", "")
        current_models = {current} if isinstance(current, str) else set(current)
        catalog = get_catalog(ctx.cfg)
        labels: dict[str, str] = {}
        models: list[str] = []
        for provider_name, provider in sorted(ctx.registry.providers.items()):
            try:
                reason = provider.available()
            except Exception as exc:
                reason = str(exc)
            available = reason is None
            glyph = "●" if available else "○"
            names = (
                sorted(
                    {
                        *provider.models,
                        *(info.id for info in catalog.values() if info.provider == provider_name),
                    }
                )
                if browse
                else provider.models
            )
            for model_name in names:
                spec = f"{provider_name}:{model_name}"
                marker = " *" if spec in current_models else ""
                suffix = "" if available else f" — {reason}"
                info = catalog.get(spec)
                window = (
                    info.context_window
                    if info
                    else getattr(getattr(provider, "capabilities", None), "max_context", None)
                )
                context = f"  {_quantity(window)} ctx" if window else ""
                price = {
                    **DEFAULT_PRICING.get(spec, {}),
                    **(info.cost if info else {}),
                    **getattr(ctx.cfg, "pricing", {}).get(spec, {}),
                }
                cost = (
                    f"  ${price['input']:g}/${price['output']:g} /M"
                    if "input" in price and "output" in price
                    else ""
                )
                capabilities = "  thinking" if info and info.thinking else ""
                capabilities += "  vision" if info and info.vision else ""
                labels[spec] = (
                    f"{glyph} {provider_name}  {model_name}{marker}{context}{cost}{capabilities}{suffix}"
                )
                models.append(spec)
        for role in MODEL_ROLES:
            spec = "@" + role
            labels[spec] = (
                f"{spec}  {getattr(ctx.cfg, 'model_roles', {}).get(role, 'main (inherited)')}"
            )
            models.append(spec)
        if not browse:
            models.append("__browse__")
            labels["__browse__"] = "Browse catalog… (search context, cost, thinking and vision)"

        async def accept(spec: Any) -> Any:
            if spec == "__browse__":
                self.resolve(None)
                return await ctx.ui.show("model", browse=True)
            return await ctx.switch_model(spec)

        super().__init__(
            "Model catalog" if browse else "Models",
            models,
            label=labels.__getitem__,
            group=self._model_group,
            empty_text="No models registered",
            on_accept=accept,
        )

    @staticmethod
    def _model_group(spec: str) -> str:
        if spec.startswith("@"):
            return "Roles"
        if spec == "__browse__":
            return "Catalog"
        return spec.partition(":")[0]


__all__ = ["ModelOverlay"]
