import asyncio
from collections.abc import Awaitable, Callable

import pytest

from orcha_agent.core.events import EventBus, InterruptRaised, TurnStart
from orcha_agent.core.plugin import Handled, PluginAPI, Resolved
from orcha_agent.core.registry import Registry


def _api(name: str, registry: Registry, bus: EventBus) -> PluginAPI:
    return PluginAPI(
        name=name,
        config={},
        state={},
        registry=registry,
        bus=bus,
        request_rebuild=lambda: None,
    )


@pytest.mark.asyncio
async def test_emit_awaits_each_handler_in_priority_then_plugin_name_order() -> None:
    registry = Registry()
    bus = EventBus()
    observed: list[str] = []

    def recording_handler(label: str) -> Callable[[TurnStart], Awaitable[None]]:
        async def handler(event: TurnStart) -> None:
            observed.append(f"{label}:start:{event.text}")
            await asyncio.sleep(0)
            observed.append(f"{label}:end:{event.thread_id}")

        return handler

    _api("zeta", registry, bus).on(TurnStart, recording_handler("zeta"), priority=50)
    _api("early", registry, bus).on(TurnStart, recording_handler("early"), priority=10)
    _api("alpha", registry, bus).on(TurnStart, recording_handler("alpha"), priority=50)

    result = await bus.emit(TurnStart(thread_id="thread-1", text="hello"))

    assert result is None
    assert observed == [
        "early:start:hello",
        "early:end:thread-1",
        "alpha:start:hello",
        "alpha:end:thread-1",
        "zeta:start:hello",
        "zeta:end:thread-1",
    ]


@pytest.mark.asyncio
async def test_handled_return_stops_later_handlers_and_is_returned_to_the_emitter() -> None:
    registry = Registry()
    bus = EventBus()
    observed: list[str] = []
    handled = Handled()

    async def handles(event: TurnStart) -> Handled:
        observed.append(event.text)
        return handled

    async def must_not_run(event: TurnStart) -> None:
        observed.append(f"late:{event.text}")

    _api("handler", registry, bus).on(TurnStart, handles, priority=10)
    _api("late", registry, bus).on(TurnStart, must_not_run, priority=20)

    result = await bus.emit(TurnStart(thread_id="thread-1", text="stop"))

    assert result is handled
    assert observed == ["stop"]


@pytest.mark.asyncio
async def test_resolved_interrupt_stops_fallback_approval_and_preserves_resume_value() -> None:
    registry = Registry()
    bus = EventBus()
    observed: list[str] = []
    resolved = Resolved({"decisions": [{"type": "approve"}]})

    async def auto_approve(event: InterruptRaised) -> Resolved:
        observed.append(event.payload["question"])
        await asyncio.sleep(0)
        return resolved

    async def fallback_prompt(event: InterruptRaised) -> None:
        observed.append(f"prompted:{event.payload['question']}")

    _api("auto", registry, bus).on(InterruptRaised, auto_approve, priority=10)
    _api("fallback", registry, bus).on(InterruptRaised, fallback_prompt, priority=1000)

    result = await bus.emit(InterruptRaised(payload={"question": "write file?"}))

    assert result is resolved
    assert result.resume_value == {"decisions": [{"type": "approve"}]}
    assert observed == ["write file?"]
