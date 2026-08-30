from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from time import perf_counter_ns
from typing import Any, TypeVar

SCHEMA_VERSION = 1
RESULTS_DIR = Path(__file__).resolve().parent / "results"
_PERCENTILE_METHOD = "nearest-rank"
_GIT_STATE_SCOPE = ("pyproject.toml", "uv.lock", "orcha_agent/**", "benchmarks/**")
_GIT_STATE_EXCLUSIONS = ("**/.env*", "**/Credentials", "**/Credentials/**")
_GIT_STATE_PATHSPECS = (
    ":(top)pyproject.toml",
    ":(top)uv.lock",
    ":(top)orcha_agent/**",
    ":(top)benchmarks/**",
    ":(top,exclude,icase)**/.env*",
    ":(top,exclude,icase)**/Credentials",
    ":(top,exclude,icase)**/Credentials/**",
)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RunConfig:
    repetitions: int
    startup_runs: int
    quick: bool = False


def percentile(samples: list[int | float], fraction: float) -> int | float:
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), math.ceil(len(ordered) * fraction)))
    return ordered[rank - 1]


def measurement(samples: Iterable[int | float], unit: str) -> dict[str, Any]:
    raw = list(samples)
    if not raw:
        raise ValueError("measurement requires at least one sample")
    return {
        "unit": unit,
        "samples": raw,
        "median": statistics.median(raw),
        "p50": percentile(raw, 0.50),
        "p95": percentile(raw, 0.95),
        "percentile_method": _PERCENTILE_METHOD,
    }


def timed_calls(call: Callable[[], T], repetitions: int) -> tuple[list[float], T]:
    samples: list[float] = []
    result: T | None = None
    for _ in range(repetitions):
        started = perf_counter_ns()
        result = call()
        samples.append((perf_counter_ns() - started) / 1_000_000_000)
    if result is None:
        raise ValueError("timed_calls requires at least one repetition")
    return samples, result


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_capture(arguments: list[str]) -> bytes | None:
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return process.stdout


def _sensitive_untracked_path(path: bytes) -> bool:
    parts = Path(os.fsdecode(path)).parts
    return any(
        (normalized := part.casefold()) == "credentials" or normalized.startswith(".env")
        for part in parts
    )


def _add_hash_part(digest: Any, label: bytes, value: bytes) -> None:
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _untracked_content_hash(path: Path) -> bytes | None:
    try:
        if path.is_symlink():
            return hashlib.sha256(os.fsencode(os.readlink(path))).digest()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.digest()
    except OSError:
        return None


def _git_metadata() -> dict[str, Any]:
    commit_raw = _git_capture(["rev-parse", "HEAD"])
    status = _git_capture(
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *_GIT_STATE_PATHSPECS,
        ]
    )
    tracked_diff = _git_capture(
        [
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
            *_GIT_STATE_PATHSPECS,
        ]
    )
    untracked_raw = _git_capture(
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *_GIT_STATE_PATHSPECS,
        ]
    )
    available = all(
        value is not None for value in (commit_raw, status, tracked_diff, untracked_raw)
    )
    if not available:
        return {
            "git_metadata_available": False,
            "git_commit": None,
            "git_dirty": False,
            "git_state_sha256": None,
            "git_state_complete": False,
            "git_state_omitted_sensitive_files": 0,
            "git_state_scope": _GIT_STATE_SCOPE,
            "git_state_exclusions": _GIT_STATE_EXCLUSIONS,
            "git_state_scope_description": (
                "Reproducibility-relevant paths only; sensitive pathspec exclusions "
                "are applied before Git reads status or diff content."
            ),
        }

    assert commit_raw is not None
    assert status is not None
    assert tracked_diff is not None
    assert untracked_raw is not None
    digest = hashlib.sha256()
    _add_hash_part(digest, b"commit", commit_raw.strip())
    _add_hash_part(digest, b"status", status)
    _add_hash_part(digest, b"tracked-diff", tracked_diff)
    omitted = 0
    complete = True
    untracked_paths = sorted(path for path in untracked_raw.split(b"\0") if path)
    for raw_path in untracked_paths:
        _add_hash_part(digest, b"untracked-path", raw_path)
        if _sensitive_untracked_path(raw_path):
            omitted += 1
            complete = False
            _add_hash_part(digest, b"untracked-content", b"sensitive-content-omitted")
            continue
        content_hash = _untracked_content_hash(repository_root() / os.fsdecode(raw_path))
        if content_hash is None:
            complete = False
            _add_hash_part(digest, b"untracked-content", b"unavailable")
        else:
            _add_hash_part(digest, b"untracked-content", content_hash)

    return {
        "git_metadata_available": True,
        "git_commit": commit_raw.decode("ascii", errors="replace").strip() or None,
        "git_dirty": bool(status),
        "git_state_sha256": digest.hexdigest(),
        "git_state_complete": complete,
        "git_state_omitted_sensitive_files": omitted,
        "git_state_scope": _GIT_STATE_SCOPE,
        "git_state_exclusions": _GIT_STATE_EXCLUSIONS,
        "git_state_scope_description": (
            "Reproducibility-relevant paths only; sensitive pathspec exclusions "
            "are applied before Git reads status or diff content."
        ),
    }


@cache
def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        **_git_metadata(),
    }


def result_document(name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": name,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment_metadata(),
        "cases": cases,
    }


def write_result(result: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['benchmark']}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def database_file_bytes(path: Path) -> dict[str, int]:
    paths = {
        "database": path,
        "wal": path.with_name(f"{path.name}-wal"),
        "shm": path.with_name(f"{path.name}-shm"),
    }
    return {name: item.stat().st_size if item.exists() else 0 for name, item in paths.items()}


def _format_value(value: int | float, unit: str) -> str:
    if unit == "bytes":
        return f"{float(value) / (1024 * 1024):.2f} MiB"
    if unit == "seconds":
        return f"{float(value):.6f} s"
    if unit == "seconds_per_mib":
        return f"{float(value):.6f} s/MiB"
    return f"{float(value):.3f}"


def render_table(result: dict[str, Any], *, plain: bool = False) -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for case in result["cases"]:
        for name, values in case["measurements"].items():
            unit = str(values["unit"])
            rows.append(
                (
                    str(case["name"]),
                    str(name),
                    _format_value(values["median"], unit),
                    _format_value(values["p50"], unit),
                    _format_value(values["p95"], unit),
                    f"{len(values['samples'])} {unit}",
                )
            )

    if not plain:
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title=str(result["benchmark"]))
            for column in ("Case", "Metric", "Median", "P50", "P95", "Samples"):
                table.add_column(column)
            for row in rows:
                table.add_row(*row)
            Console().print(table)
            return
        except ImportError:
            pass

    print(f"\n{result['benchmark']}")
    print("case | metric | median | p50 | p95 | samples")
    print("-" * 80)
    for row in rows:
        print(" | ".join(row))
