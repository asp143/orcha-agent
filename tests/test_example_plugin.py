from examples.plugins import hello

from orcha_agent.core.events import EventBus, ToolCallEnd, ToolCallStart
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


def test_hello_renderer_matches_only_completed_hello_tool_calls() -> None:
    registry = Registry()
    hello.register(
        PluginAPI(
            name=hello.PLUGIN.name,
            registry=registry,
            bus=EventBus(),
            config={},
            state={},
            request_rebuild=lambda: None,
        )
    )
    registration = next(
        entry for entry in registry.renderers if entry.plugin == hello.PLUGIN.name
    )

    assert callable(registration.match)
    assert registration.match(
        ToolCallEnd(name="hello", id="hello-1", result="Hello, Ada!")
    )
    assert not registration.match(
        ToolCallStart(name="hello", args={"name": "Ada"}, id="hello-1")
    )
    assert not registration.match(
        ToolCallEnd(name="write_file", id="write-1", result="ok")
    )
