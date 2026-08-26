"""Plugin discovery and registration."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import logging
import pkgutil
import sys
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .events import EventBus
from .plugin import PluginAPI, PluginSpec
from .registry import Registry

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "orcha_agent.plugins"
_BUILTIN_PACKAGE = "orcha_agent.builtin"


@dataclass(frozen=True, slots=True)
class PluginRecord:
    """The result of discovering and attempting to load one plugin."""

    name: str
    version: str
    source: str
    status: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    spec: PluginSpec
    source: str
    register: Callable[[PluginAPI], Any] | None
    failure: Exception | None = None


def _candidate(
    loaded: Any,
    *,
    default_name: str,
    source: str,
) -> _Candidate:
    """Normalize an imported module or loaded entry point into a plugin."""

    try:
        metadata = getattr(loaded, "PLUGIN", None)
        spec = PluginSpec(name=default_name) if metadata is None else metadata
        if not isinstance(spec, PluginSpec):
            raise TypeError(f"PLUGIN in {source} must be a PluginSpec")

        register = getattr(loaded, "register", None)
        if register is None and callable(loaded):
            register = loaded
        if not callable(register):
            raise TypeError(f"plugin {spec.name} has no callable register(api)")
        return _Candidate(spec=spec, source=source, register=register)
    except Exception as exc:
        return _Candidate(
            spec=PluginSpec(name=default_name),
            source=source,
            register=None,
            failure=exc,
        )


def _failed_candidate(
    *,
    default_name: str,
    source: str,
    failure: Exception,
) -> _Candidate:
    return _Candidate(
        spec=PluginSpec(name=default_name),
        source=source,
        register=None,
        failure=failure,
    )


def _discover_builtins() -> list[_Candidate]:
    try:
        package = importlib.import_module(_BUILTIN_PACKAGE)
    except Exception as exc:
        return [
            _failed_candidate(
                default_name="builtin",
                source=f"builtin:{_BUILTIN_PACKAGE}",
                failure=exc,
            )
        ]

    candidates: list[_Candidate] = []
    module_infos = sorted(
        pkgutil.iter_modules(package.__path__, prefix=f"{package.__name__}."),
        key=lambda info: info.name,
    )
    for module_info in module_infos:
        module_name = module_info.name
        source = f"builtin:{module_name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            candidates.append(
                _failed_candidate(
                    default_name=module_name.rsplit(".", maxsplit=1)[-1],
                    source=source,
                    failure=exc,
                )
            )
            continue
        candidates.append(
            _candidate(
                module,
                default_name=module_name.rsplit(".", maxsplit=1)[-1],
                source=source,
            )
        )
    return candidates


def _entry_point_source(entry_point: Any) -> str:
    value = getattr(entry_point, "value", None)
    return f"entry-point:{value or entry_point.name}"


def _discover_entry_points() -> list[_Candidate]:
    try:
        entry_points = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as exc:
        return [
            _failed_candidate(
                default_name="entry-points",
                source=f"entry-point-group:{_ENTRY_POINT_GROUP}",
                failure=exc,
            )
        ]

    candidates: list[_Candidate] = []
    for entry_point in sorted(
        entry_points,
        key=lambda item: (item.name, str(getattr(item, "value", ""))),
    ):
        source = _entry_point_source(entry_point)
        try:
            loaded = entry_point.load()
        except Exception as exc:
            candidates.append(
                _failed_candidate(
                    default_name=entry_point.name,
                    source=source,
                    failure=exc,
                )
            )
            continue
        candidates.append(
            _candidate(loaded, default_name=entry_point.name, source=source)
        )
    return candidates


def _module_name_for_path(path: Path) -> str:
    """Return a stable, collision-resistant import name for a plugin file."""

    normalized_stem = "".join(
        character if character.isalnum() else "_" for character in path.stem
    )
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return f"_orcha_agent_plugin_{normalized_stem}_{digest}"


def _load_file_module(path: Path) -> Any:
    module_name = _module_name_for_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(module_name, missing)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _plugin_directories(cfg: Config) -> Iterable[Path]:
    yield Path.home() / ".config" / "orcha-agent" / "plugins"
    yield cfg.cwd / ".orcha-agent" / "plugins"
    yield from cfg.plugin_dirs


def _discover_directories(cfg: Config) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    seen_directories: set[Path] = set()
    seen_files: set[Path] = set()

    for configured_directory in _plugin_directories(cfg):
        directory = Path(configured_directory).expanduser().resolve()
        if directory in seen_directories:
            continue
        seen_directories.add(directory)
        if not directory.is_dir():
            continue

        for discovered_path in sorted(directory.glob("*.py"), key=lambda path: path.name):
            path = discovered_path.resolve()
            if path in seen_files:
                continue
            seen_files.add(path)
            source = str(path)
            try:
                module = _load_file_module(path)
            except Exception as exc:
                candidates.append(
                    _failed_candidate(
                        default_name=path.stem,
                        source=source,
                        failure=exc,
                    )
                )
                continue
            candidates.append(
                _candidate(module, default_name=path.stem, source=source)
            )
    return candidates


def _disabled_plugins(cfg: Config) -> set[str]:
    disabled = cfg.plugins.get("disabled", ())
    if isinstance(disabled, str):
        return {disabled}
    try:
        return {name for name in disabled if isinstance(name, str)}
    except TypeError:
        return set()


def _requirements(spec: PluginSpec) -> tuple[str, ...]:
    if isinstance(spec.requires, str):
        return (spec.requires,)
    return tuple(requirement for requirement in spec.requires if isinstance(requirement, str))


def _snapshot_registry(registry: Registry) -> dict[str, dict[Any, Any] | list[Any]]:
    return {
        name: value.copy()
        for name, value in vars(registry).items()
        if isinstance(value, (dict, list))
    }


def _restore_registry(
    registry: Registry,
    snapshot: dict[str, dict[Any, Any] | list[Any]],
) -> None:
    for name, saved in snapshot.items():
        current = getattr(registry, name)
        if isinstance(current, dict) and isinstance(saved, dict):
            current.clear()
            current.update(saved)
        elif isinstance(current, list) and isinstance(saved, list):
            current[:] = saved


def load_plugins(
    registry: Registry,
    bus: EventBus,
    cfg: Config,
    state_by_plugin: dict[str, dict[str, Any]] | None = None,
    request_rebuild: Callable[[], None] = lambda: None,
) -> list[PluginRecord]:
    """Discover plugins, register them in deterministic order, and report status."""

    candidates = [
        *_discover_builtins(),
        *_discover_entry_points(),
        *_discover_directories(cfg),
    ]
    candidates.sort(key=lambda candidate: (candidate.spec.priority, candidate.spec.name))

    disabled = _disabled_plugins(cfg)
    states = {} if state_by_plugin is None else state_by_plugin
    loaded_names: set[str] = set()
    records: list[PluginRecord] = []

    for candidate in candidates:
        spec = candidate.spec

        if candidate.failure is not None:
            logger.error("plugin %s failed: %s", spec.name, candidate.failure)
            records.append(
                PluginRecord(spec.name, spec.version, candidate.source, "failed")
            )
            if cfg.strict_plugins:
                raise candidate.failure.with_traceback(candidate.failure.__traceback__)
            continue

        if spec.name in disabled:
            logger.info("plugin %s disabled", spec.name)
            records.append(
                PluginRecord(spec.name, spec.version, candidate.source, "disabled")
            )
            continue

        missing = sorted(set(_requirements(spec)) - loaded_names)
        if missing:
            logger.info(
                "plugin %s skipped: missing requirements: %s",
                spec.name,
                ", ".join(missing),
            )
            records.append(
                PluginRecord(spec.name, spec.version, candidate.source, "skipped")
            )
            continue

        state_existed = spec.name in states
        state = states.setdefault(spec.name, {})
        state_snapshot = deepcopy(state)
        registry_snapshot = _snapshot_registry(registry)
        handlers_snapshot = bus.handlers.copy()
        api = PluginAPI(
            name=spec.name,
            config=cfg.plugin_config(spec.name),
            state=state,
            registry=registry,
            bus=bus,
            request_rebuild=request_rebuild,
        )
        try:
            assert candidate.register is not None
            candidate.register(api)
        except Exception as exc:
            _restore_registry(registry, registry_snapshot)
            bus.handlers[:] = handlers_snapshot
            if state_existed:
                state.clear()
                state.update(state_snapshot)
            else:
                states.pop(spec.name, None)
            logger.error("plugin %s failed: %s", spec.name, exc)
            records.append(
                PluginRecord(spec.name, spec.version, candidate.source, "failed")
            )
            if cfg.strict_plugins:
                raise
            continue

        loaded_names.add(spec.name)
        records.append(
            PluginRecord(spec.name, spec.version, candidate.source, "loaded")
        )

    return records


__all__ = ["PluginRecord", "load_plugins"]
