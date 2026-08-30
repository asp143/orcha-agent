from __future__ import annotations

import asyncio
from collections.abc import Iterator
from io import StringIO
from time import perf_counter_ns, process_time_ns
from typing import Any

from langchain_core.messages import AIMessageChunk
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.markdown import Markdown

from orcha_agent.core.events import ModelChunk
from orcha_agent.tui.frame import Frame
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.transcript import Transcript

from .common import RunConfig, measurement, result_document

MIB = 1024 * 1024
DEFAULT_PAYLOAD_BYTES = (100 * 1024, MIB)
DEFAULT_CHUNK_BYTES = (1, 10, 100)


class _SizedDummyOutput(DummyOutput):
    def __init__(self, *, columns: int, rows: int) -> None:
        self._size = Size(rows=rows, columns=columns)

    def get_size(self) -> Size:
        return self._size


def _payload(size: int) -> str:
    seed = "## Streaming benchmark\n\nDeterministic assistant text for layout and accumulation.\n"
    repetitions = (size // len(seed)) + 1
    value = (seed * repetitions)[:size]
    if len(value.encode("ascii")) != size:
        raise AssertionError("streaming fixture must contain exactly the requested ASCII bytes")
    return value


async def _stream_once(payload: str, chunk_bytes: int) -> tuple[float, int]:
    frame = Frame()
    transcript = Transcript(frame)
    started = process_time_ns()
    for offset in range(0, len(payload), chunk_bytes):
        fragment = payload[offset : offset + chunk_bytes]
        await transcript.handle(
            ModelChunk(
                chunk=AIMessageChunk(content=fragment),
                role="main",
                source_id="main",
            )
        )
    elapsed = (process_time_ns() - started) / 1_000_000_000
    assistant = next(block for block in frame.blocks if block.kind == "assistant")
    return elapsed, assistant.revision


async def _noop_submit(_: str) -> None:
    return None


async def _viewport_case(
    payload: str,
    repetitions: int,
    *,
    columns: int = 100,
    rows: int = 36,
) -> tuple[list[float], list[int]]:
    stream = StringIO()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            _noop_submit,
            input=pipe,
            output=_SizedDummyOutput(columns=columns, rows=rows),
            console=Console(file=stream, force_terminal=False),
        )
        block = runtime.frame.add("assistant", {"text": "", "role": "main"})
        try:
            paint_samples: list[float] = []
            for _ in range(repetitions):
                block.update(text=payload)
                started = perf_counter_ns()
                runtime._viewport_text()
                paint_samples.append((perf_counter_ns() - started) / 1_000_000_000)

            layout_samples: list[int] = []
            layout_calls = 0
            original: Any = getattr(Markdown, "__rich_console__")

            def counted_layout(markdown: Any, console: Any, options: Any) -> Iterator[Any]:
                nonlocal layout_calls
                layout_calls += 1
                yield from original(markdown, console, options)

            setattr(Markdown, "__rich_console__", counted_layout)
            try:
                for _ in range(repetitions):
                    before = layout_calls
                    block.update(text=payload)
                    runtime._viewport_text()
                    layout_samples.append(layout_calls - before)
            finally:
                setattr(Markdown, "__rich_console__", original)
            return paint_samples, layout_samples
        finally:
            await runtime.scheduler.aclose()


async def _run(config: RunConfig) -> dict[str, Any]:
    payload_sizes = (1024, 4096) if config.quick else DEFAULT_PAYLOAD_BYTES
    chunk_sizes = DEFAULT_CHUNK_BYTES
    cases: list[dict[str, Any]] = []

    for payload_bytes in payload_sizes:
        payload = _payload(payload_bytes)
        for chunk_bytes in chunk_sizes:
            cpu_samples: list[float] = []
            revision_samples: list[int] = []
            for _ in range(config.repetitions):
                cpu_seconds, revisions = await _stream_once(payload, chunk_bytes)
                cpu_samples.append(cpu_seconds)
                revision_samples.append(revisions)
            mib = payload_bytes / MIB
            cases.append(
                {
                    "name": f"transcript_{payload_bytes}_bytes_{chunk_bytes}_byte_chunks",
                    "parameters": {
                        "payload_bytes": payload_bytes,
                        "chunk_bytes": chunk_bytes,
                        "event_construction_timed": True,
                        "clock": "time.process_time_ns",
                    },
                    "measurements": {
                        "cpu": measurement(cpu_samples, "seconds"),
                        "cpu_per_mib": measurement(
                            (sample / mib for sample in cpu_samples),
                            "seconds_per_mib",
                        ),
                        "frame_revisions": measurement(revision_samples, "revisions"),
                    },
                }
            )

        paint_samples, layout_samples = await _viewport_case(
            payload,
            config.repetitions,
        )
        cases.append(
            {
                "name": f"viewport_{payload_bytes}_bytes",
                "parameters": {
                    "payload_bytes": payload_bytes,
                    "columns": 100,
                    "rows": 36,
                    "paint_scope": "ApplicationRuntime._viewport_text",
                    "displayed_revisions": config.repetitions,
                },
                "measurements": {
                    "viewport_paint": measurement(paint_samples, "seconds"),
                    "rich_layouts_per_revision": measurement(
                        layout_samples, "layouts_per_revision"
                    ),
                },
            }
        )

    return result_document("streaming", cases)


def run(config: RunConfig) -> dict[str, Any]:
    return asyncio.run(_run(config))
