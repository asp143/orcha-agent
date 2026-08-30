from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from time import perf_counter_ns
from typing import Any

from .common import RunConfig, measurement, result_document


def _orcha_command(*arguments: str) -> list[str]:
    executable = shutil.which("orcha")
    if executable is not None:
        return [executable, *arguments]
    return [sys.executable, "-m", "orcha_agent", *arguments]


def _rss_bytes(maximum_resident_set: int) -> int:
    if platform.system() == "Darwin":
        return maximum_resident_set
    return maximum_resident_set * 1024


def _run_posix(command: list[str]) -> tuple[float, int]:
    started = perf_counter_ns()
    child = os.fork()
    if child == 0:
        try:
            descriptor = os.open(os.devnull, os.O_RDWR)
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
            if descriptor > 2:
                os.close(descriptor)
            os.execv(command[0], command)
        except BaseException:
            os._exit(127)
    _, status, usage = os.wait4(child, 0)
    elapsed = (perf_counter_ns() - started) / 1_000_000_000
    return_code = os.waitstatus_to_exitcode(status)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed, _rss_bytes(int(usage.ru_maxrss))


def _run_fallback(command: list[str]) -> tuple[float, None]:
    started = perf_counter_ns()
    subprocess.run(
        command,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return (perf_counter_ns() - started) / 1_000_000_000, None


def _run_once(command: list[str]) -> tuple[float, int | None]:
    if hasattr(os, "fork") and hasattr(os, "wait4"):
        return _run_posix(command)
    return _run_fallback(command)


def run(config: RunConfig) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    commands = {
        "help": _orcha_command("--help"),
        "gallery_plain": _orcha_command("gallery", "--plain"),
    }
    for name, command in commands.items():
        wall_samples: list[float] = []
        rss_samples: list[int] = []
        for _ in range(config.startup_runs):
            wall, rss = _run_once(command)
            wall_samples.append(wall)
            if rss is not None:
                rss_samples.append(rss)
        measurements = {"wall": measurement(wall_samples, "seconds")}
        if rss_samples:
            measurements["peak_rss"] = measurement(rss_samples, "bytes")
        cases.append(
            {
                "name": name,
                "parameters": {
                    "command": command,
                    "subprocesses": config.startup_runs,
                    "peak_rss_available": bool(rss_samples),
                },
                "measurements": measurements,
            }
        )
    return result_document("startup", cases)
