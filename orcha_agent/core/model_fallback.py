"""Keep rule interrupts out of provider fallback while retaining upstream policy."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware as _ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from orcha_agent.extensibility.stream_rules import (
    StreamInterrupt,
    _model_monitor,
    install_model_callback,
)


def _prepare_attempt(request: ModelRequest[Any]) -> None:
    monitor = _model_monitor.get()
    if monitor is not None:
        # A genuine provider failure can switch models after partial streaming.
        # Give each attempt its callback and preserve the final-message fallback
        # when the replacement provider emits no chunks.
        monitor.inspected = False
        install_model_callback(request.model)


class _RuleBubbleUp(GraphBubbleUp):
    """Private escape through fallback; never allowed to reach the graph runtime."""

    def __init__(self, interrupt: StreamInterrupt) -> None:
        self.interrupt = interrupt


class ModelFallbackMiddleware(_ModelFallbackMiddleware):
    """Fallback for provider failures, never for deliberate rule interruptions.

    Upstream preserves GraphBubbleUp exceptions and otherwise catches everything.
    Translate only while inside that middleware, then restore the original error:
    LangGraph must see a regular failed task so it discards rejected model writes.
    """

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        def guarded(current: ModelRequest[Any]) -> ModelResponse[Any]:
            try:
                _prepare_attempt(current)
                return handler(current)
            except StreamInterrupt as exc:
                raise _RuleBubbleUp(exc) from exc

        try:
            return super().wrap_model_call(request, guarded)
        except _RuleBubbleUp as exc:
            raise exc.interrupt from None

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        async def guarded(current: ModelRequest[Any]) -> ModelResponse[Any]:
            try:
                _prepare_attempt(current)
                return await handler(current)
            except StreamInterrupt as exc:
                raise _RuleBubbleUp(exc) from exc

        try:
            return await super().awrap_model_call(request, guarded)
        except _RuleBubbleUp as exc:
            raise exc.interrupt from None
