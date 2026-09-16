from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from orcha_agent.builtin import filesystem, modes, tools_native
from orcha_agent.core.agent import build_agent
from orcha_agent.core.config import Config, ToolsConfig, load_config
from orcha_agent.core.events import AgentBuildBefore, AppExit, EventBus
from orcha_agent.core.plugin import PluginAPI, ProviderCaps
from orcha_agent.core.registry import Registry
from orcha_agent.core.session import SessionStore


class FakeToolsModel(GenericFakeChatModel):
    disable_streaming: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolsModel:
        return self


def config(tmp_path: Path, mode: str = "yolo") -> Config:
    return Config(
        model="fake:main",
        subagent_model=None,
        summarizer_model=None,
        mode=mode,
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "session.db",
        cwd=tmp_path,
        resume=None,
        list_sessions=False,
        strict_plugins=True,
        plugin_dirs=(),
        models={},
        providers={},
        plugins={},
    )


def kernel(cfg: Config, calls: list[tuple[str, dict[str, Any]]]):
    registry, bus = Registry(), EventBus()
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"call-{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index, (name, args) in enumerate(calls)
    ] + [AIMessage(content="done")]
    model = FakeToolsModel(messages=iter(script))
    for module in (filesystem, modes, tools_native):
        api = PluginAPI(
            name=module.PLUGIN.name,
            config={
                "cwd": cfg.cwd,
                "native": cfg.tools.native,
                "edit_format": cfg.tools.edit_format,
            },
            state={},
            registry=registry,
            bus=bus,
            request_rebuild=lambda: None,
        )
        module.register(api)
    api.add_provider(
        "fake",
        lambda *_: model,
        capabilities=ProviderCaps(
            tool_calling=True,
            streaming=False,
            thinking=False,
            structured_output=False,
            max_context=100_000,
        ),
    )
    return registry, bus


@pytest.mark.asyncio
async def test_fake_agent_calls_every_native_tool(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    calls = [
        ("write", {"path": "nested/sample.txt", "content": "alpha\nbeta\n"}),
        ("read", {"path": "nested/sample.txt:1-2"}),
        ("edit", {"path": "nested/sample.txt", "old_string": "beta", "new_string": "gamma"}),
        ("grep", {"pattern": "gam.*", "path": "nested"}),
        ("glob", {"pattern": "**/*.txt"}),
        ("ls", {"path": "nested"}),
        ("bash", {"command": "export ORCHA_NATIVE_TEST=value; cd nested"}),
        ("bash", {"command": 'printf "%s:%s" "$ORCHA_NATIVE_TEST" "${PWD##*/}"'}),
        ("bash_jobs", {"action": "list"}),
    ]
    registry, bus = kernel(cfg, calls)
    captured: dict[str, Any] = {}

    async def capture(event: AgentBuildBefore) -> None:
        captured.update(event.kwargs)

    bus.on(AgentBuildBefore, capture, plugin="test")
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Exercise the tools."}]},
                {"configurable": {"thread_id": "native"}, "recursion_limit": 40},
            )
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(messages) == len(calls)
        assert all(m.status == "success" for m in messages), messages
        assert "value:nested" in messages[-2].content
        assert (tmp_path / "nested/sample.txt").read_text() == "alpha\ngamma\n"
        assert "@@" in messages[2].content
        from deepagents.middleware.filesystem import FilesystemMiddleware

        fs = next(m for m in captured["middleware"] if isinstance(m, FilesystemMiddleware))
        assert {t.name for t in fs.tools} == {"delete"}
        assert {t.name for t in captured["tools"]} == {
            "read",
            "write",
            "edit",
            "bash",
            "bash_jobs",
            "grep",
            "glob",
            "ls",
        }
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
async def test_native_ask_approval_prevents_write_until_resumed(tmp_path: Path) -> None:
    cfg = config(tmp_path, "ask")
    registry, bus = kernel(cfg, [("write", {"path": "approved.txt", "content": "yes"})])
    try:
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            thread = {"configurable": {"thread_id": "approval"}}
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "write"}]}, thread
            )
            assert result["__interrupt__"]
            assert not (tmp_path / "approved.txt").exists()
            await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), thread)
            assert (tmp_path / "approved.txt").read_text() == "yes"
    finally:
        await bus.emit(AppExit())


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_tool_scope_maps_legacy_read_and_preserves_fallback(
    tmp_path: Path, native: bool
) -> None:
    cfg = replace(config(tmp_path, "plan"), tools=ToolsConfig(native=native))
    registry, bus = kernel(cfg, [])
    captured: dict[str, Any] = {}

    async def capture(event: AgentBuildBefore) -> None:
        captured.update(event.kwargs)

    bus.on(AgentBuildBefore, capture, plugin="test")
    try:
        with SessionStore(cfg.db_path) as session:
            await build_agent(
                registry,
                cfg,
                session,
                bus,
                exclude_general_purpose=True,
                tool_scope={"read_file", "grep"},
            )
        all_tools = {t.name for t in captured["tools"]}
        for middleware in captured["middleware"]:
            all_tools.update(t.name for t in getattr(middleware, "tools", ()))
        assert ({"read", "grep"} if native else {"read_file", "grep"}) <= all_tools
        assert not all_tools & {"edit", "write", "bash", "bash_jobs", "delete", "execute"}
        assert ("read_file" in all_tools) is not native
    finally:
        await bus.emit(AppExit())


@pytest.mark.parametrize("table", ['native = "yes"', 'edit_format = "invalid"'])
def test_tools_config_rejects_invalid_settings(tmp_path: Path, table: str) -> None:
    settings = tmp_path / "config.toml"
    settings.write_text("[tools]\n" + table)
    with pytest.raises(SystemExit):
        load_config([], env={}, cwd=tmp_path, user_config_path=settings)


def test_tools_config_defaults_and_fallback(tmp_path: Path) -> None:
    settings = tmp_path / "config.toml"
    cfg = load_config([], env={}, cwd=tmp_path, user_config_path=settings)
    assert cfg.tools == ToolsConfig(native=True, edit_format="replace")
    settings.write_text('[tools]\nnative = false\nedit_format = "hashline"\n')
    cfg = load_config([], env={}, cwd=tmp_path, user_config_path=settings)
    assert cfg.tools == ToolsConfig(native=False, edit_format="hashline")


@pytest.mark.asyncio
async def test_hashline_format_works_through_agent_runtime(tmp_path: Path) -> None:
    from orcha_agent.core.tools.hashline import file_hash

    content = "one\ntwo\n"
    (tmp_path / "sample.txt").write_text(content)
    cfg = replace(config(tmp_path), tools=ToolsConfig(edit_format="hashline"))
    registry, bus = kernel(
        cfg,
        [
            ("read", {"path": "sample.txt"}),
            ("edit", {"patch": f"[sample.txt#{file_hash(content)}]\nPUT 2:\n+changed"}),
        ],
    )
    try:
        schema = registry.tools["edit"].tool_call_schema.model_json_schema()
        assert set(schema["properties"]) == {"patch"}
        with SessionStore(cfg.db_path) as session:
            graph = await build_agent(registry, cfg, session, bus, exclude_general_purpose=True)
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": "Update line two."}]},
                {"configurable": {"thread_id": "hashline"}},
            )
        results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert all(m.status == "success" for m in results), results
        assert results[0].content.startswith("[sample.txt#")
        assert (tmp_path / "sample.txt").read_text() == "one\nchanged\n"
    finally:
        await bus.emit(AppExit())
