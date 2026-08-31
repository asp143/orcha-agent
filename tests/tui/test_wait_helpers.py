from __future__ import annotations

import asyncio

import pytest

from conftest import _wait_until


@pytest.mark.asyncio
async def test_wait_until_backs_off_after_fifty_failed_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    checks = 0

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def succeeds_after_fifty_two_failures() -> bool:
        nonlocal checks
        checks += 1
        return checks > 52

    monkeypatch.setattr(asyncio, "sleep", record_sleep)

    await _wait_until(succeeds_after_fifty_two_failures)

    assert sleeps == [0] * 50 + [0.001] * 2
