from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from orcha_agent.tui.runtime import ApplicationRuntime


type Predicate = Callable[[], bool]
type WaitUntil = Callable[[Predicate], Awaitable[None]]
type WaitForRender = Callable[[ApplicationRuntime, Predicate], Awaitable[None]]


async def _wait_until(predicate: Predicate, *, timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _wait_for_render(
    runtime: ApplicationRuntime,
    predicate: Predicate,
    *,
    timeout: float = 1.0,
) -> None:
    rendered = asyncio.Event()

    def after_render(_application: object) -> None:
        if predicate():
            rendered.set()

    runtime.application.after_render += after_render
    try:
        if predicate():
            return
        runtime.application.invalidate()
        await asyncio.wait_for(rendered.wait(), timeout=timeout)
    finally:
        runtime.application.after_render -= after_render


@pytest.fixture
def wait_until() -> WaitUntil:
    return _wait_until


@pytest.fixture
def wait_for_render() -> WaitForRender:
    return _wait_for_render
