"""Keep rule interrupts out of provider fallback while retaining upstream policy."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware as _ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from orcha_agent.extensibility.stream_rules import StreamInterrupt


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
                return await handler(current)
            except StreamInterrupt as exc:
                raise _RuleBubbleUp(exc) from exc

        try:
            return await super().awrap_model_call(request, guarded)
        except _RuleBubbleUp as exc:
            raise exc.interrupt from None
