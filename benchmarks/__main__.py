from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from .common import RESULTS_DIR, RunConfig, render_table, write_result

BENCHMARKS: dict[str, tuple[str, str]] = {
    "startup": ("benchmarks.startup", "run"),
    "streaming": ("benchmarks.streaming", "run"),
    "ledger": ("benchmarks.persistence", "run_ledger"),
    "turn_capture": ("benchmarks.persistence", "run_turn_capture"),
    "history_load": ("benchmarks.persistence", "run_history_load"),
    "session_overlay_load": (
        "benchmarks.persistence",
        "run_session_overlay_load",
    ),
}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Run reproducible orcha-agent performance workloads.",
    )
    parser.add_argument(
        "benchmarks",
        nargs="*",
        choices=tuple(BENCHMARKS),
        metavar="BENCHMARK",
        help="benchmark(s) to run; defaults to the full suite",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_integer,
        default=None,
        help="timed repetitions per case (default: 20, or 1 with --quick)",
    )
    parser.add_argument(
        "--startup-runs",
        type=_positive_integer,
        default=None,
        help="fresh subprocesses per startup command (default: 20, or 1 with --quick)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"JSON output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use development-scale fixtures while retaining every benchmark type",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print plain tables instead of Rich tables",
    )
    return parser


def _load_runner(module_name: str, function_name: str) -> Callable[[RunConfig], dict[str, Any]]:
    module = importlib.import_module(module_name)
    runner = getattr(module, function_name)
    if not callable(runner):
        raise TypeError(f"{module_name}.{function_name} is not callable")
    return cast(Callable[[RunConfig], dict[str, Any]], runner)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    selected = list(dict.fromkeys(arguments.benchmarks or BENCHMARKS))
    config = RunConfig(
        repetitions=arguments.repetitions or (1 if arguments.quick else 20),
        startup_runs=arguments.startup_runs or (1 if arguments.quick else 20),
        quick=bool(arguments.quick),
    )

    written: list[Path] = []
    for name in selected:
        module_name, function_name = BENCHMARKS[name]
        result = _load_runner(module_name, function_name)(config)
        path = write_result(result, arguments.results_dir)
        render_table(result, plain=bool(arguments.plain))
        written.append(path)

    print("\nResults:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
