from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

import pytest

from orcha_agent.tui.runtime import ApplicationRuntime


type Predicate = Callable[[], bool]


class WaitUntil(Protocol):
    def __call__(
        self,
        predicate: Predicate,
        *,
        timeout: float = 1.0,
    ) -> Awaitable[None]: ...


type WaitForRender = Callable[[ApplicationRuntime, Predicate], Awaitable[None]]


async def _wait_until(predicate: Predicate, *, timeout: float = 1.0) -> None:
    async def poll() -> None:
        failed_checks = 0
        while not predicate():
            failed_checks += 1
            await asyncio.sleep(0 if failed_checks <= 50 else 0.001)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _wait_for_render(
    runtime: ApplicationRuntime,
    predicate: Predicate,
    *,
    timeout: float = 1.0,
) -> None:
    rendered = asyncio.Event()
    # The predicate must change as an observable result of rendering; otherwise,
    # after_render must re-invalidate while it remains false.

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
