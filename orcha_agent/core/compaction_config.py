"""Lightweight compaction configuration; no provider imports."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    enabled: bool = True
    threshold_tokens: int | None = None
    threshold_ratio: float = 0.8
    keep_recent_tokens: int = 20_000
    reserve_tokens: int | None = None
    supersede_reads: bool = True
    drop_useless: bool = True
    method_order: tuple[str, ...] = ("summary", "handoff", "shake")
    speculative: bool = True
    idle_seconds: float = 60.0

    def threshold(self, window: int) -> int:
        reserve = (
            self.reserve_tokens
            if self.reserve_tokens is not None
            else max(16_384, int(window * 0.15))
        )
        target = (
            self.threshold_tokens
            if self.threshold_tokens is not None
            else int(window * self.threshold_ratio)
        )
        return max(1, min(target, window - reserve))


def compaction_config(raw: Any) -> CompactionConfig:
    if not isinstance(raw, dict):
        raise ValueError("[compaction] must be a table")
    values = dict(raw)
    for key in ("enabled", "supersede_reads", "drop_useless", "speculative"):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"[compaction] {key} must be a boolean")
    for key in ("threshold_tokens", "keep_recent_tokens", "reserve_tokens"):
        value = values.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "keep_recent_tokens" else 1)
        ):
            raise ValueError(f"[compaction] {key} must be a positive integer")
    ratio = values.get("threshold_ratio", 0.8)
    if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or not 0 < ratio <= 1:
        raise ValueError("[compaction] threshold_ratio must be between 0 and 1")
    idle = values.get("idle_seconds", 60.0)
    if isinstance(idle, bool) or not isinstance(idle, (float, int)) or idle <= 0:
        raise ValueError("[compaction] idle_seconds must be positive")
    methods = values.get("method_order", ["summary", "handoff", "shake"])
    if (
        not isinstance(methods, (list, tuple))
        or not methods
        or any(
            not isinstance(method, str) or method not in {"summary", "handoff", "shake"}
            for method in methods
        )
    ):
        raise ValueError("[compaction] method_order must contain summary, handoff, or shake")
    values["method_order"] = tuple(methods)
    unknown = values.keys() - CompactionConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown compaction settings: {', '.join(sorted(unknown))}")
    return CompactionConfig(**values)
