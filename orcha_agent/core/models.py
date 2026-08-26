"""Model specification resolution and provider-safe message history cleanup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.runnables import Runnable
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .config import Config
from .registry import ProviderRegistration, Registry

ModelSpec: TypeAlias = str | list[str]
ResolvedModel: TypeAlias = Runnable[Any, Any]


class ModelResolver:
    """Resolve configured model specs through the provider registry."""

    def __init__(self, registry: Registry, config: Config) -> None:
        self._registry = registry
        self._config = config
        self._register_harness_profiles()

    def _register_harness_profiles(self) -> None:
        """Keep the beta deepagents profile API isolated to this module."""
        profiles = [
            (prefix, registration.harness)
            for prefix, registration in sorted(self._registry.providers.items())
            if registration.harness is not None
        ]
        if not profiles:
            return

        from deepagents import register_harness_profile

        for prefix, profile in profiles:
            register_harness_profile(prefix, profile)

    def resolve(self, spec: ModelSpec, role: str) -> ResolvedModel:
        """Resolve one model or an invocation-time fallback chain."""
        expanded = self._expand_aliases(spec, role=role)
        if not expanded:
            raise ValueError(f"Model fallback chain for role {role!r} cannot be empty")
        if len(expanded) == 1:
            return self._resolve_one(expanded[0], role)

        resolved: list[ResolvedModel] = []
        unavailable: list[str] = []
        for model_spec in expanded:
            try:
                resolved.append(self._resolve_one(model_spec, role))
            except RuntimeError as exc:
                unavailable.append(str(exc))

        if not resolved:
            details = "; ".join(unavailable)
            raise RuntimeError(
                f"No model in the fallback chain for role {role!r} is available: {details}"
            )
        if len(resolved) == 1:
            return resolved[0]
        return resolved[0].with_fallbacks(resolved[1:])

    def resolve_roles(self) -> dict[str, ResolvedModel]:
        """Construct independent model objects for each built-in role."""
        return {
            "main": self.resolve(self._config.model, "main"),
            "subagent": self.resolve(self._config.subagent_model, "subagent"),
            "summarizer": self.resolve(self._config.summarizer_model, "summarizer"),
        }

    def _expand_aliases(
        self,
        spec: ModelSpec,
        *,
        role: str,
        aliases: tuple[str, ...] = (),
    ) -> list[str]:
        if isinstance(spec, list):
            expanded: list[str] = []
            for entry in spec:
                expanded.extend(self._expand_aliases(entry, role=role, aliases=aliases))
            return expanded
        if not isinstance(spec, str) or not spec:
            raise ValueError(
                f"Model specification for role {role!r} must be a non-empty string "
                "or fallback list"
            )

        target = self._config.models.get(spec)
        if target is None:
            return [spec]
        if spec in aliases:
            cycle = " -> ".join((*aliases, spec))
            raise ValueError(f"Model alias cycle for role {role!r}: {cycle}")
        return self._expand_aliases(target, role=role, aliases=(*aliases, spec))

    def _resolve_one(self, spec: str, role: str) -> ResolvedModel:
        prefix, separator, model_name = spec.partition(":")
        if not separator or not prefix or not model_name:
            raise ValueError(
                f"Invalid model specification {spec!r} for role {role!r}; "
                "expected '<provider>:<model>'"
            )

        registration = self._registry.providers.get(prefix)
        if registration is None:
            registered = ", ".join(sorted(self._registry.providers)) or "(none)"
            raise ValueError(
                f"Unknown model provider prefix {prefix!r} for role {role!r}; "
                f"registered prefixes: {registered}"
            )

        hint = self._availability_hint(prefix, registration, role)
        if hint:
            raise RuntimeError(
                f"Model provider {prefix!r} is unavailable for role {role!r}. {hint}"
            )

        provider_config = dict(self._config.providers.get(prefix, {}))
        if not registration.capabilities.thinking:
            provider_config.pop("thinking", None)

        try:
            return registration.factory(model_name, provider_config)
        except Exception as exc:
            raise RuntimeError(
                f"Could not construct model {spec!r} for role {role!r} "
                f"with provider {prefix!r}: {exc}"
            ) from exc

    @staticmethod
    def _availability_hint(
        prefix: str,
        registration: ProviderRegistration,
        role: str,
    ) -> str | None:
        try:
            return registration.available()
        except Exception as exc:
            raise RuntimeError(
                f"Could not check availability of model provider {prefix!r} "
                f"for role {role!r}: {exc}"
            ) from exc


def strip_foreign_blocks(
    graph: Any,
    thread_config: Any,
    foreign_types: set[str] | frozenset[str],
) -> None:
    """Replace stored history after removing provider-private AI content blocks."""
    state = graph.get_state(thread_config)
    messages = getattr(state, "values", {}).get("messages", ())
    if not messages:
        return

    private_types = frozenset(foreign_types)
    replacement: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    for message in messages:
        if not isinstance(message, AIMessage):
            replacement.append(message)
            continue

        content = message.content
        if isinstance(content, list):
            content = [
                dict(block) if isinstance(block, Mapping) else block
                for block in content
                if not (
                    isinstance(block, Mapping)
                    and block.get("type") in private_types
                )
            ]
        replacement.append(message.model_copy(update={"content": content}))

    graph.update_state(thread_config, {"messages": replacement})


__all__ = ["ModelResolver", "ModelSpec", "ResolvedModel", "strip_foreign_blocks"]
