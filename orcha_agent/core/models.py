"""Model specification resolution and provider-safe message history cleanup."""

from __future__ import annotations
import os

from collections.abc import Iterator, Mapping
from typing import Any, TypeAlias

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .config import Config, DEFAULT_MODEL
from .registry import ProviderRegistration, Registry

ModelSpec: TypeAlias = str | list[str]
ResolvedModel: TypeAlias = BaseChatModel
_REGISTERED_HARNESS_PROFILES: set[tuple[str, int]] = set()


MODEL_ROLES = (
    "main",
    "subagent",
    "summarizer",
    "smol",
    "slow",
    "plan",
    "vision",
    "task",
    "commit",
    "advisor",
)
EFFORTS = {"off", "none", "minimal", "low", "medium", "high", "xhigh", "max"}


def expand_model_spec(spec: ModelSpec, config: Config, seen: tuple[str, ...] = ()) -> list[str]:
    """Expand roles and aliases without constructing providers or resolving secrets."""
    if isinstance(spec, list):
        return [value for entry in spec for value in expand_model_spec(entry, config, seen)]
    if not isinstance(spec, str) or not spec:
        raise ValueError("Model specification must be a non-empty string or fallback list")
    if spec in seen:
        raise ValueError("Model alias cycle: " + " -> ".join((*seen, spec)))
    if spec.startswith("@"):
        role, separator, effort = spec[1:].partition(":")
        if separator and effort not in EFFORTS:
            raise ValueError(f"Unknown reasoning effort: {effort}")
        if role not in MODEL_ROLES and role not in getattr(config, "model_roles", {}):
            raise ValueError(f"Unknown model role: {role}")
        target = getattr(config, "model_roles", {}).get(role) or getattr(config, "models", {}).get(
            role
        )
        if target is None:
            target = (
                config.subagent_model
                if role in {"task", "subagent"}
                else config.summarizer_model
                if role == "summarizer"
                else None
            ) or (
                config.model
                if not (isinstance(config.model, str) and config.model.startswith("@"))
                else getattr(config, "model_role_default", None) or DEFAULT_MODEL
            )
        values = expand_model_spec(target, config, (*seen, spec))
        if not separator:
            return values
        return [
            value
            if value.partition(":")[0] in {"ollama", "langchain"}
            else f"{value.rpartition(':')[0] if value.count(':') > 1 and value.rpartition(':')[2] in EFFORTS else value}:{effort}"
            for value in values
        ]
    target = getattr(config, "models", {}).get(spec)
    return expand_model_spec(target, config, (*seen, spec)) if target is not None else [spec]


def role_fallback_notices(spec: ModelSpec, config: Config) -> list[str]:
    """Explain an implicit role fallback once at the user action boundary."""
    notices: dict[str, None] = {}
    visited: set[str] = set()

    def visit(value: ModelSpec) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if value in visited:
            return
        visited.add(value)
        if value.startswith("@"):
            role = value[1:].partition(":")[0]
            configured = getattr(config, "model_roles", {}).get(role) or getattr(
                config, "models", {}
            ).get(role)
            if configured is None:
                if role in {"task", "subagent"}:
                    configured = config.subagent_model
                elif role == "summarizer":
                    configured = config.summarizer_model
            if configured is not None:
                visit(configured)
            elif role in MODEL_ROLES and role != "main":
                notices[f"role {role} is not configured, using main"] = None
        else:
            alias = getattr(config, "models", {}).get(value)
            if alias is not None:
                visit(alias)

    visit(spec)
    return list(notices)


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
            key = (prefix, id(profile))
            if key in _REGISTERED_HARNESS_PROFILES:
                continue
            register_harness_profile(prefix, profile)
            _REGISTERED_HARNESS_PROFILES.add(key)

    def resolve(self, spec: ModelSpec, role: str) -> BaseChatModel:
        """Resolve the primary available chat model for a role."""
        return next(self._iter_available(spec, role))

    def resolve_chain(self, spec: ModelSpec, role: str) -> list[BaseChatModel]:
        """Resolve available models in fallback order without wrapping them."""
        return list(self._iter_available(spec, role))

    def _iter_available(self, spec: ModelSpec, role: str) -> Iterator[BaseChatModel]:
        expanded = self._expand_aliases(spec, role=role)
        if not expanded:
            raise ValueError(f"Model fallback chain for role {role!r} cannot be empty")

        resolved = False
        unavailable: list[str] = []
        for model_spec in expanded:
            try:
                model = self._resolve_one(model_spec, role)
            except RuntimeError as exc:
                unavailable.append(str(exc))
                continue
            resolved = True
            yield model

        if not resolved:
            details = "; ".join(unavailable)
            raise RuntimeError(
                f"No model in the fallback chain for role {role!r} is available: {details}"
            )

    def resolve_roles(self) -> dict[str, ResolvedModel]:
        """Construct independent model objects for each built-in role."""
        return {
            "main": self.resolve(self._config.model, "main"),
            "subagent": self.resolve(
                self._config.subagent_model
                or self._config.model_roles.get("subagent")
                or self._config.model_roles.get("task")
                or self._config.model,
                "subagent",
            ),
            "summarizer": self.resolve(
                self._config.summarizer_model
                or self._config.model_roles.get("summarizer")
                or self._config.model,
                "summarizer",
            ),
        }

    def _expand_aliases(
        self,
        spec: ModelSpec,
        *,
        role: str,
        aliases: tuple[str, ...] = (),
    ) -> list[str]:
        return expand_model_spec(spec, self._config, aliases)

    def _resolve_one(self, spec: str, role: str) -> BaseChatModel:
        prefix, separator, model_name = spec.partition(":")
        if not separator or not prefix or not model_name:
            raise ValueError(
                f"Invalid model specification {spec!r} for role {role!r}; "
                "expected '<provider>:<model>'"
            )

        effort = None
        if (
            prefix not in {"ollama", "langchain"}
            and ":" in model_name
            and model_name.rpartition(":")[2] in EFFORTS
        ):
            model_name, _, effort = model_name.rpartition(":")
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
        from .catalog import provider_api_key

        environment_ready = any(os.environ.get(key) for key in registration.env_keys)
        api_key = None if environment_ready else provider_api_key(prefix, self._config)
        if (
            registration.env_keys
            and not api_key
            and not any(os.environ.get(key) for key in registration.env_keys)
        ):
            accepted = ", ".join(registration.env_keys)
            raise RuntimeError(f"Model provider {prefix!r} requires one of: {accepted}")

        provider_config = dict(self._config.providers.get(prefix, {}))
        if api_key:
            provider_config["api_key"] = api_key
        if effort and registration.capabilities.thinking:
            provider_config["reasoning_effort"] = "none" if effort == "off" else effort
        if not registration.capabilities.thinking:
            provider_config.pop("thinking", None)

        try:
            model = registration.factory(model_name, provider_config)
        except Exception as exc:
            detail = str(exc).replace(api_key, "[redacted]") if api_key else str(exc)
            raise RuntimeError(
                f"Could not construct model {spec!r} for role {role!r} "
                f"with provider {prefix!r}: {detail}"
            ) from None
        if not isinstance(model, BaseChatModel):
            raise RuntimeError(
                f"Provider {prefix!r} returned {type(model).__name__}, expected BaseChatModel"
            )
        model.metadata = {
            **(model.metadata or {}),
            "orcha_model": f"{prefix}:{model_name}",
            "orcha_role": role,
        }
        return model

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


def filter_foreign_blocks(
    messages: Any,
    foreign_types: set[str] | frozenset[str],
) -> list[Any]:
    """Copy messages after removing provider-private AI content blocks."""
    private_types = frozenset(foreign_types)
    filtered: list[Any] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            filtered.append(message)
            continue

        content = message.content
        if isinstance(content, list):
            content = [
                dict(block) if isinstance(block, Mapping) else block
                for block in content
                if not (isinstance(block, Mapping) and block.get("type") in private_types)
            ]
        additional_kwargs = {
            key: value
            for key, value in message.additional_kwargs.items()
            if key not in private_types
        }
        response_metadata = {
            key: value
            for key, value in message.response_metadata.items()
            if key not in private_types
        }
        filtered.append(
            message.model_copy(
                update={
                    "content": content,
                    "additional_kwargs": additional_kwargs,
                    "response_metadata": response_metadata,
                }
            )
        )
    return filtered


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

    replacement: list[Any] = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *filter_foreign_blocks(messages, foreign_types),
    ]

    graph.update_state(
        thread_config,
        {"messages": replacement},
        as_node="model",
    )


__all__ = [
    "ModelResolver",
    "ModelSpec",
    "ResolvedModel",
    "filter_foreign_blocks",
    "strip_foreign_blocks",
]
