"""Register turn-local magic keyword controls through the plugin API."""

from orcha_agent.core.events import AppStart
from orcha_agent.core.plugin import PluginAPI, PluginSpec
from orcha_agent.extensibility.magic_keywords import MagicKeywordsMiddleware

PLUGIN = PluginSpec(name="magic_keywords", version="1.0.0")


def register(api: PluginAPI) -> None:
    middleware = MagicKeywordsMiddleware()

    async def start(event: AppStart) -> None:
        middleware.ctx = event.ctx

    api.on(AppStart, start)
    api.add_middleware(middleware, priority=40)
