"""Lazy bundled model metadata with trusted, mtime-cached YAML overrides."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    id: str
    context_window: int = 0
    max_tokens: int = 0
    cost: dict[str, float] = field(default_factory=dict)
    thinking: bool = False
    vision: bool = False
    tool_calls: bool = True


class _Loader(yaml.SafeLoader):
    pass


def _command(loader: Any, node: Any) -> str:
    return "!" + str(loader.construct_scalar(node))


_Loader.add_constructor("!command", _command)


def _paths(config: Any = None) -> tuple[Path, ...]:
    user = getattr(config, "user_config_path", None)
    root = Path(user).parent if user else Path.home() / ".config/orcha-agent"
    paths = [root / "models.yml"]
    if getattr(config, "trust_cwd", False):
        paths.append(Path(config.cwd) / ".orcha-agent/models.yml")
    return tuple(paths)


@lru_cache(maxsize=16)
def _overrides(stamps: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, _, _size in stamps:
        data = yaml.load(Path(name).read_text(), Loader=_Loader) or {}
        if not isinstance(data, dict):
            raise ValueError("models.yml must contain a mapping")
        for provider, values in data.get("providers", data).items():
            if not isinstance(values, dict):
                raise ValueError(f"Model provider {provider} must contain a mapping")
            previous = result.setdefault(provider, {})
            for key, value in values.items():
                if key in {"models", "model_overrides"} and isinstance(value, dict):
                    merged = dict(previous.get(key, {}))
                    for model_id, patch in value.items():
                        prior = merged.get(model_id, {})
                        merged[model_id] = {**prior, **patch}
                        if "cost" in prior or "cost" in patch:
                            merged[model_id]["cost"] = {
                                **prior.get("cost", {}),
                                **patch.get("cost", {}),
                            }
                    previous[key] = merged
                else:
                    previous[key] = value
    return result


def provider_overrides(config: Any = None) -> dict[str, Any]:
    if config is None:
        return {}
    stamps = []
    for path in _paths(config):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
    return _overrides(tuple(stamps))


@lru_cache(maxsize=1)
def _bundled() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("models.json").read_text())


@lru_cache(maxsize=16)
def _merged(serialized: str) -> dict[str, ModelInfo]:
    values = {key: {**value, "cost": dict(value["cost"])} for key, value in _bundled().items()}
    for provider, options in json.loads(serialized).items():
        for section in ("models", "model_overrides"):
            models = options.get(section, {})
            if isinstance(models, list):
                models = {item["id"]: item for item in models}
            for name, override in models.items():
                key = f"{provider}:{name}"
                prior = values.get(key, {"provider": provider, "id": name})
                values[key] = {
                    **prior,
                    **override,
                    "provider": provider,
                    "id": name,
                    "cost": {**prior.get("cost", {}), **override.get("cost", {})},
                }
    for value in values.values():
        for key in ("context_window", "max_tokens"):
            number = value.get(key, 0)
            if not isinstance(number, int) or isinstance(number, bool) or number < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        for key in ("thinking", "vision", "tool_calls"):
            if key in value and not isinstance(value[key], bool):
                raise ValueError(f"{key} must be a boolean")
        for number in value.get("cost", {}).values():
            if not isinstance(number, (float, int)) or isinstance(number, bool) or number < 0:
                raise ValueError("Model costs must be non-negative numbers")
    fields = ModelInfo.__dataclass_fields__
    return {
        key: ModelInfo(**{k: v for k, v in value.items() if k in fields})
        for key, value in values.items()
    }


def get_catalog(config: Any = None) -> dict[str, ModelInfo]:
    return _merged(json.dumps(provider_overrides(config), sort_keys=True))


def get_model(spec: str, config: Any = None) -> ModelInfo | None:
    if (
        spec.partition(":")[0] not in {"ollama", "langchain"}
        and spec.count(":") > 1
        and spec.rpartition(":")[2]
        in {
            "off",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }
    ):
        spec = spec.rpartition(":")[0]
    model = get_catalog(config).get(spec)
    if model is None and spec.startswith("langchain:"):
        return ModelInfo(provider="langchain", id=spec[len("langchain:") :])
    return model


def provider_api_key(provider: str, config: Any = None) -> str | None:
    """Resolve credentials only when constructing a provider, never while browsing."""
    value = provider_overrides(config).get(provider, {}).get("api_key")
    if not value:
        return None
    if not isinstance(value, str):
        raise ValueError("api_key must be an environment name or !command")
    if value.startswith("!"):
        try:
            result = subprocess.run(
                value[1:], shell=True, capture_output=True, text=True, timeout=10, check=True
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError("Model API key command failed") from exc
        return result.stdout.strip()
    return os.environ.get(value)
