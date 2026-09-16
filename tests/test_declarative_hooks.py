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
    result = await run_hook(
        HookConfig("turn_start", command="cat"),
        {"text": "hello"},
        tmp_path,
        user_config_dir=tmp_path,
    )
    assert result.code == 0 and "hello" in result.output
    result = await run_hook(
        HookConfig("turn_start", command="echo denied >&2; exit 2"),
        {},
        tmp_path,
        user_config_dir=tmp_path,
    )
    assert result.code == 2 and result.error.strip() == "denied"
    result = await run_hook(
        HookConfig("turn_start", command="sleep 10", timeout=0.01),
        {},
        tmp_path,
        user_config_dir=tmp_path,
    )
    assert result.code == 1 and "timed out" in result.error


@pytest.mark.asyncio
async def test_python_hook_supports_async_and_timeout(tmp_path: Path) -> None:
    (tmp_path / "hooks").mkdir()
    (tmp_path / "hooks" / "hookfn.py").write_text(
        'async def run(payload):\n return {"args": {"value": payload["value"] + 1}}\n'
    )
    result = await run_hook(
        HookConfig("turn_start", python="hookfn:run"),
        {"value": 2},
        tmp_path,
        user_config_dir=tmp_path,
    )
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
        ("write", "echo ordinary informational output", True, "old"),
        ("write", "echo '[1, 2]'", True, "old"),
        ("write", "echo '{malformed'", True, "failed"),
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
            user_config_path=tmp_path / "config.toml",
            hooks=(HookConfig("tool_call_before", command=command, blocking=blocking),),
        ),
        console=Mock(),
    )
    await bus.emit(AppStart(ctx))
    call = {"name": name, "id": "x", "args": {"path": "old"}}
    request = SimpleNamespace(tool_call=call, override=lambda **kwargs: SimpleNamespace(**kwargs))
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="x"))
    result = await HooksMiddleware(api.emit).awrap_tool_call(request, handler)
    if expected in {"blocked", "failed"}:
        handler.assert_not_called()
        assert result.status == "error"
        assert result.content == (
            "denied" if expected == "blocked" else "Hook failed; tool execution blocked"
        )
    else:
        assert handler.call_args.args[0].tool_call["args"]["path"] == expected
    from orcha_agent.core.events import AppExit

    await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_output_limit(tmp_path: Path) -> None:
    result = await run_hook(
        HookConfig("turn_start", command="yes", timeout=2), {}, tmp_path, user_config_dir=tmp_path
    )
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


@pytest.mark.asyncio
async def test_python_hook_never_imports_repository_modules(tmp_path: Path, monkeypatch) -> None:
    repo, user = tmp_path / "repo", tmp_path / "user"
    repo.mkdir()
    user.mkdir()
    (user / "hooks").mkdir()
    (repo / "trusted_hook.py").write_text('def run(payload): return "repository executed"')
    monkeypatch.setenv("PYTHONPATH", str(repo))
    hook = HookConfig("turn_start", python="trusted_hook:run", env_passthrough=("PYTHONPATH",))
    missing = await run_hook(hook, {}, repo, user_config_dir=user)
    assert missing.code != 0 and "ModuleNotFoundError" in missing.error
    (user / "hooks" / "trusted_hook.py").write_text('def run(payload): return "user executed"')
    trusted = await run_hook(hook, {}, repo, user_config_dir=user)
    assert trusted.code == 0 and "user executed" in trusted.output
    assert "repository executed" not in trusted.output


@pytest.mark.asyncio
async def test_hook_shell_scope_selects_working_directory(tmp_path: Path) -> None:
    repo, user = tmp_path / "repo", tmp_path / "user"
    repo.mkdir()
    user.mkdir()
    (repo / "origin").write_text("repo")
    (user / "origin").write_text("user")
    for scope in ("user", "project"):
        result = await run_hook(
            HookConfig("turn_start", command="cat origin", scope=scope),
            {},
            repo,
            user_config_dir=user,
        )
        assert result.code == 0
        assert result.output == ("user" if scope == "user" else "repo")


@pytest.mark.asyncio
async def test_hooks_drop_secrets_even_when_explicitly_allowed(tmp_path: Path, monkeypatch) -> None:
    import json
    import shlex
    import sys

    secrets = (
        "OPENAI_API_KEY",
        "CUSTOM_TOKEN",
        "APP_SECRET_VALUE",
        "DB_PASSWORD",
        "PROVIDER_LOGIN",
    )
    for key in (*secrets, "ORDINARY_HOOK_VALUE", "ORDINARY_SHELL_VALUE", "NOT_ALLOWED"):
        monkeypatch.setenv(key, "sentinel")
    command = shlex.join(
        [sys.executable, "-I", "-c", "import os,json; print(json.dumps(dict(os.environ)))"]
    )
    result = await run_hook(
        HookConfig(
            "turn_start", command=command, env_passthrough=(*secrets, "ORDINARY_HOOK_VALUE")
        ),
        {},
        tmp_path,
        user_config_dir=tmp_path,
        shell_env_passthrough=("ORDINARY_SHELL_VALUE",),
        provider_env_keys=("PROVIDER_LOGIN",),
    )
    assert result.code == 0
    environment = json.loads(result.output)
    assert not set(secrets) & environment.keys()
    assert "NOT_ALLOWED" not in environment
    assert environment["ORDINARY_HOOK_VALUE"] == environment["ORDINARY_SHELL_VALUE"] == "sentinel"


@pytest.mark.asyncio
async def test_hook_result_payload_is_bounded_utf8_text(tmp_path: Path) -> None:
    import json

    result = await run_hook(
        HookConfig("tool_call_after", command="cat"),
        {
            "event": "tool_call_after",
            "name": "read",
            "id": "call",
            "result": ToolMessage(content="界" * 20_000, tool_call_id="call"),
        },
        tmp_path,
        user_config_dir=tmp_path,
    )
    payload = json.loads(result.output)
    assert len(payload["result"].encode()) <= 20_000
    assert payload["result"] == "界" * (20_000 // 3)
    assert payload["name"] == "read" and payload["id"] == "call"


def test_hook_env_passthrough_config(tmp_path: Path) -> None:
    user = tmp_path / "config.toml"
    user.write_text(
        '[[hooks]]\nevent="turn_start"\ncommand="true"\nenv_passthrough=["HOOK_COLOR"]\n'
    )
    cfg = load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=user)
    assert cfg.hooks[0].env_passthrough == ("HOOK_COLOR",)
    user.write_text('[[hooks]]\nevent="turn_start"\ncommand="true"\nenv_passthrough="ALL"\n')
    with pytest.raises(SystemExit):
        load_config([], env={"HOME": str(tmp_path)}, cwd=tmp_path, user_config_path=user)
