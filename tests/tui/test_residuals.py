from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from orcha_agent.builtin import provider_anthropic
from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.complete import PathIndex
from orcha_agent.tui.runtime import ApplicationRuntime, UIFacade


@pytest.mark.asyncio
async def test_shift_tab_maps_anthropic_levels_to_valid_provider_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_anthropic

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **options: options)
    registry = Registry()
    provider_state: dict[str, object] = {}
    provider_anthropic.register(
        PluginAPI(
            name="provider_anthropic",
            config={"_ui_thinking": "summary"},
            state=provider_state,
            registry=registry,
            bus=EventBus(),
            request_rebuild=lambda: None,
        )
    )
    plugin_states = {
        "composer": {"thinking_level": "max"},
        "provider_anthropic": provider_state,
    }
    rebuilt: list[dict[str, object]] = []
    rebuilt_event = asyncio.Event()
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            cwd=tmp_path,
            model="anthropic:claude-opus-5",
            models={},
            providers={"anthropic": {"reasoning_effort": "max"}},
            thinking="summary",
        ),
        registry=registry,
        plugin_states=plugin_states,
        persist_plugin_states=lambda: None,
        ui=UIFacade(),
        bus=EventBus(),
        _bus=EventBus(),
    )

    async def rebuild() -> None:
        rebuilt.append(
            registry.providers["anthropic"].factory(
                "claude-opus-5",
                ctx.cfg.providers["anthropic"],
            )
        )
        rebuilt_event.set()

    ctx.rebuild = rebuild
    ctx.switch_model = lambda _model: asyncio.sleep(0)

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            ctx=ctx,
            registry=registry,
            input=pipe,
            output=DummyOutput(),
        )
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)

        pipe.send_bytes(b"\x1b[Z")
        await asyncio.wait_for(rebuilt_event.wait(), 1)
        rebuilt_event.clear()
        assert runtime.thinking_level == "off"
        assert "reasoning_effort" not in rebuilt[-1]
        assert "thinking" not in rebuilt[-1]

        pipe.send_bytes(b"\x1b[Z")
        await asyncio.wait_for(rebuilt_event.wait(), 1)
        assert runtime.thinking_level == "low"
        assert rebuilt[-1]["reasoning_effort"] == "low"
        assert rebuilt[-1]["thinking"] == {
            "type": "adaptive",
            "display": "summarized",
        }

        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, 1)


def test_path_index_prunes_ignored_directories_before_scanning_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ignored = tmp_path / "a-ignored"
    ignored.mkdir()
    for index in range(20):
        nested = ignored / f"nested-{index}"
        nested.mkdir()
        (nested / "noise.py").write_text("", encoding="utf-8")
    visible = tmp_path / "z-visible"
    visible.mkdir()
    (visible / "answer.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("a-ignored/\n", encoding="utf-8")

    scanned: list[Path] = []
    real_scandir = os.scandir

    def recording_scandir(path: str | os.PathLike[str]):
        scanned.append(Path(path).resolve())
        return real_scandir(path)

    monkeypatch.setattr("orcha_agent.tui.complete.os.scandir", recording_scandir)

    paths = PathIndex(tmp_path, cap=3).paths()

    assert "z-visible/answer.py" in paths
    assert ignored.resolve() not in scanned
    assert not any(ignored.resolve() in path.parents for path in scanned)
