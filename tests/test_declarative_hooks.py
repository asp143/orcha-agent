from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import ToolMessage

from orcha_agent.builtin.hooks import register
from orcha_agent.core.config import HookConfig, load_config
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.extensibility.hooks import HooksMiddleware, matches, run_hook


def test_user_and_trusted_project_hooks_are_additive(tmp_path: Path) -> None:
    user, project = tmp_path / "user.toml", tmp_path / "project.toml"
    user.write_text('[[hooks]]\nevent="session_start"\ncommand="true"\n')
    project.write_text('[[hooks]]\nevent="turn_start"\ncommand="false"\n')

    def load(args):
        return load_config(
            args,
            env={"HOME": str(tmp_path)},
            cwd=tmp_path,
            user_config_path=user,
            project_config_path=project,
        )

    assert [h.scope for h in load([]).hooks] == ["user"]
    assert [h.scope for h in load(["--trust-cwd"]).hooks] == ["user", "project"]


@pytest.mark.asyncio
async def test_commands_receive_json_and_exit_semantics(tmp_path: Path) -> None:
    result = await run_hook(HookConfig("turn_start", command="cat"), {"text": "hello"}, tmp_path)
    assert result.code == 0 and "hello" in result.output
    result = await run_hook(
        HookConfig("turn_start", command="echo denied >&2; exit 2"), {}, tmp_path
    )
    assert result.code == 2 and result.error.strip() == "denied"
    result = await run_hook(
        HookConfig("turn_start", command="sleep 10", timeout=0.01), {}, tmp_path
    )
    assert result.code == 1 and "timed out" in result.error


@pytest.mark.asyncio
async def test_python_hook_supports_async_and_timeout(tmp_path: Path) -> None:
    (tmp_path / "hookfn.py").write_text(
        'async def run(payload):\n return {"args": {"value": payload["value"] + 1}}\n'
    )
    result = await run_hook(HookConfig("turn_start", python="hookfn:run"), {"value": 2}, tmp_path)
    assert result.code == 0 and "3" in result.output


def test_matchers() -> None:
    assert matches(HookConfig("tool_call_before", matcher="write*"), {"name": "write_file"})
    assert not matches(HookConfig("tool_call_before", matcher="write*"), {"name": "read"})
    assert matches(HookConfig("turn_start", matcher="secret"), {"text": "a secret"})
    assert matches(
        HookConfig("tool_call_before", matcher="regex:secret"), {"args": {"text": "secret"}}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,command,blocking,expected",
    [
        ("write", "echo denied >&2; exit 2", True, "blocked"),
        ("read", "echo denied >&2; exit 2", True, "blocked"),
        ("write", """echo '{"args":{"path":"new"}}' """, True, "new"),
        ("read", """echo '{"args":{"path":"new"}}' """, True, "old"),
        ("write", "echo denied >&2; exit 2", False, "old"),
    ],
)
async def test_real_tool_boundary(
    tmp_path: Path, name: str, command: str, blocking: bool, expected: str
) -> None:
    bus, registry = EventBus(), Registry()
    api = PluginAPI(
        name="hooks", config={}, state={}, registry=registry, bus=bus, request_rebuild=lambda: None
    )
    register(api)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=tmp_path,
            hooks=(HookConfig("tool_call_before", command=command, blocking=blocking),),
        ),
        console=Mock(),
    )
    await bus.emit(AppStart(ctx))
    call = {"name": name, "id": "x", "args": {"path": "old"}}
    request = SimpleNamespace(tool_call=call, override=lambda **kwargs: SimpleNamespace(**kwargs))
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="x"))
    result = await HooksMiddleware(api.emit).awrap_tool_call(request, handler)
    if expected == "blocked":
        handler.assert_not_called()
        assert result.status == "error" and result.content == "denied"
    else:
        assert handler.call_args.args[0].tool_call["args"]["path"] == expected
    from orcha_agent.core.events import AppExit

    await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_output_limit(tmp_path: Path) -> None:
    result = await run_hook(HookConfig("turn_start", command="yes", timeout=2), {}, tmp_path)
    assert result.code == 1 and "exceeded" in result.error


def test_exec_tools_are_not_rewritable() -> None:
    from orcha_agent.extensibility.hooks import WRITE_TOOLS

    assert not {"bash", "execute"} & WRITE_TOOLS


@pytest.mark.asyncio
async def test_automatic_compaction_extended_response_and_resumed_threads() -> None:
    from deepagents.middleware.summarization import SummarizationState
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain.agents.middleware.types import ExtendedModelResponse
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from orcha_agent.core.events import Compaction

    class CompactOnce(AgentMiddleware):
        state_schema = SummarizationState

        async def awrap_model_call(self, request, handler):
            response = await handler(request)
            if request.state.get("_summarization_event"):
                return response
            return ExtendedModelResponse(
                model_response=response,
                command=Command(
                    update={
                        "_summarization_event": {
                            "cutoff_index": 1,
                            "summary_message": HumanMessage("Keep the user's constraint"),
                            "file_path": "/history.md",
                        }
                    }
                ),
            )

    events = []

    async def emit(event):
        events.append(event)

    graph = create_agent(
        FakeListChatModel(responses=["ok"]),
        middleware=[HooksMiddleware(emit), CompactOnce()],
        checkpointer=InMemorySaver(),
    )
    for thread in ("first", "second", "first"):
        await graph.ainvoke(
            {"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": thread}}
        )
    assert [
        (event.session_id, event.summary) for event in events if isinstance(event, Compaction)
    ] == [
        ("first", "Keep the user's constraint"),
        ("second", "Keep the user's constraint"),
    ]
