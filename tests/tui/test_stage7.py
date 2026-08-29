from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from orcha_agent.builtin.banner import build_welcome, choose_tip
from orcha_agent.core.events import InterruptRaised, ToolCallEnd, ToolCallStart, TurnEnd, TurnStart
from orcha_agent.tui.blocks.welcome import render as render_welcome
from orcha_agent.tui.frame import Block, BlockState, Frame, FrameScheduler
from orcha_agent.tui.notify import DesktopNotifier
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.statusline import subagents_segment
from orcha_agent.tui.turn import _updates_event
from orcha_agent.tui.title import TerminalTitle


class _Output(DummyOutput):
    def __init__(self) -> None:
        super().__init__()
        self.titles: list[str] = []
        self.raw: list[str] = []

    def set_title(self, title: str) -> None:
        self.titles.append(title)

    def write_raw(self, data: str) -> None:
        self.raw.append(data)


class _Rng:
    def __init__(self, index: int) -> None:
        self.index = index
        self.seen: list[str] = []

    def choice(self, values: list[str]) -> str:
        self.seen = values
        return values[self.index]


def _plain(renderable: Any, width: int) -> str:
    stream = StringIO()
    Console(file=stream, width=width, force_terminal=False).print(renderable)
    return stream.getvalue()


def test_new_tips_have_four_times_the_deterministic_selection_weight() -> None:
    rng = _Rng(0)
    assert choose_tip(["[NEW] fresh", "steady"], rng=rng) == "fresh"
    assert rng.seen == ["fresh", "fresh", "fresh", "fresh", "steady"]


def test_welcome_has_fixed_slots_static_gradient_and_width_cap(tmp_path: Path) -> None:
    sessions = [
        SimpleNamespace(thread_id=str(i), title=f"session {i}", created="2026-08-29T00:00:00+00:00")
        for i in range(2)
    ]
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            model="openai:gpt-5",
            mode="ask",
            cwd=tmp_path,
            trust_cwd=True,
            symbols="unicode",
        ),
        session=SimpleNamespace(list=lambda: sessions),
        session_id="current",
        plugins=[SimpleNamespace(name="one")],
        registry=SimpleNamespace(providers={"openai": object()}),
    )

    first = build_welcome(ctx, rng=_Rng(0), tips=["one"])
    second = build_welcome(ctx, rng=_Rng(0), tips=["one"])

    assert len(first["sessions"]) == len(first["hints"]) == 4
    assert first["logo_styles"] == second["logo_styles"]
    text = _plain(
        render_welcome(Block("welcome", "welcome", BlockState.ACTIVE, data=first), None, 140, 100, False),
        140,
    )
    assert max(map(len, text.splitlines())) <= 100


def test_welcome_narrow_ascii_is_readable_and_fixed_height(tmp_path: Path) -> None:
    base = {
        "model": "gpt-5",
        "mode": "ask",
        "cwd": str(tmp_path),
        "tip": "Use /help",
        "sessions": ["", "", "", ""],
        "hints": ["Trusted", "", "", ""],
        "ascii": True,
        "logo": ["ORCHA"],
        "logo_styles": [[None] * 5],
    }
    full = {**base, "sessions": ["one", "two", "three", "four"]}
    empty_text = _plain(render_welcome(Block("a", "welcome", data=base), None, 36, 100, False), 36)
    full_text = _plain(render_welcome(Block("b", "welcome", data=full), None, 36, 100, False), 36)
    assert empty_text.isascii()
    assert len(empty_text.splitlines()) == len(full_text.splitlines())
    assert "ORCHA" in empty_text and "Use /help" in empty_text


@pytest.mark.asyncio
async def test_scheduler_stops_when_idle_and_restarts_for_later_spinner() -> None:
    frame = Frame()
    scheduler = FrameScheduler(frame, commit=lambda _blocks: None, invalidate=lambda: None)
    idle = scheduler.start_spinner()
    await asyncio.wait_for(idle, timeout=0.25)
    assert idle.done()

    active = frame.add("thinking")
    restarted = scheduler.start_spinner()
    await asyncio.sleep(scheduler.SPINNER_INTERVAL * 1.5)
    assert restarted is not idle and not restarted.done()
    active.settle()
    await asyncio.wait_for(restarted, timeout=0.25)
    await scheduler.aclose()


@pytest.mark.asyncio
async def test_notification_obeys_idle_threshold_and_safe_fallback() -> None:
    now = [0.0]
    output = _Output()
    commands: list[list[str]] = []

    def missing(_name: str) -> None:
        return None

    notifier = DesktopNotifier(
        enabled=True,
        output=output,
        clock=lambda: now[0],
        which=missing,
        spawn=lambda command: commands.append(command),
        run_terminal=lambda callback: callback(),
    )
    now[0] = 5.0
    assert await notifier.notify("Orcha", "Turn complete") is False
    now[0] = 5.001
    assert await notifier.notify("Orcha", "Turn complete") is True
    assert commands == []
    assert output.raw == ["\x1b]9;Turn complete\x07"]

    notifier.record_keypress()
    now[0] = 9.0
    assert await notifier.notify("Orcha", "Turn complete") is False


@pytest.mark.asyncio
async def test_notification_prefers_notify_send_without_duplicate_fallback() -> None:
    output = _Output()
    commands: list[list[str]] = []
    notifier = DesktopNotifier(
        enabled=True,
        output=output,
        clock=lambda: 10.0,
        which=lambda _name: "/usr/bin/notify-send",
        spawn=lambda command: commands.append(command),
        run_terminal=lambda callback: callback(),
    )
    notifier._last_keypress = 0.0
    assert await notifier.notify("Orcha", "Approval required") is True
    assert commands == [["/usr/bin/notify-send", "Orcha", "Approval required"]]
    assert output.raw == []


def test_terminal_title_transitions_dedupe_and_ascii_safety() -> None:
    output = _Output()
    title = TerminalTitle(output, unicode=False)
    title.set_session("unsafe\x07 title")
    title.set_turn(True, spinner="✻")
    title.set_approval(True)
    title.set_approval(False)
    title.set_turn(False)
    title.set_turn(False)

    assert output.titles == [
        "orcha - unsafe title",
        "* orcha - unsafe title",
        "[wait] orcha - unsafe title",
        "* orcha - unsafe title",
        "orcha - unsafe title",
    ]
    assert all(value.isascii() for value in output.titles)


@pytest.mark.asyncio
async def test_runtime_hud_tracks_todos_queue_and_real_subagent_lifecycle() -> None:
    output = _Output()
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        output=output,
        ctx=SimpleNamespace(
            cfg=SimpleNamespace(
                cwd=Path.cwd(),
                notify=False,
                symbols="ascii",
                statusbar=True,
            ),
            session=SimpleNamespace(get=lambda _session: SimpleNamespace(title="work")),
            session_id="session",
            plugin_states={},
        ),
    )
    runtime.set_todos([{"content": f"todo {i}", "status": "pending"} for i in range(12)])
    runtime.queue.extend(["queued one", "queued two"])
    await runtime.handle_presentation(ToolCallStart("task", {"description": "worker"}, "call-1"))

    text = runtime._hud_text().value
    assert "todo 5" in text and "todo 6" not in text
    assert "worker" in text and "queued one" in text
    assert len(runtime.ui.subagents) == 1
    assert subagents_segment(runtime.ctx).text == "1"

    await runtime.handle_presentation(ToolCallEnd("task", "call-1", "done"))
    assert runtime.ui.subagents == []
    assert "Subagents" not in runtime._hud_text().value
    await runtime.scheduler.aclose()



@pytest.mark.asyncio
async def test_hud_clips_each_section_to_eight_rendered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _Output()
    output.get_size = lambda: SimpleNamespace(rows=40, columns=16)
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        output=output,
        ctx=SimpleNamespace(
            cfg=SimpleNamespace(
                cwd=Path.cwd(),
                notify=False,
                symbols="ascii",
                statusbar=True,
            ),
            session=SimpleNamespace(get=lambda _session: SimpleNamespace(title="work")),
            session_id="session",
            plugin_states={},
        ),
    )
    runtime.set_todos(
        [{"content": "a very long todo label " * 20, "status": "pending"}]
    )
    monkeypatch.setattr(
        runtime,
        "_capture_block",
        lambda *_args, **_kwargs: "\n".join(
            f"visual row {index}" for index in range(12)
        ),
    )

    rendered = runtime._hud_text().value

    assert len(rendered.splitlines()) <= 8
    assert runtime._hud_height() == len(rendered.splitlines())
    await runtime.scheduler.aclose()



@pytest.mark.asyncio
async def test_startup_warnings_are_replayed_only_after_welcome(tmp_path: Path) -> None:
    keys = tmp_path / "keys.toml"
    keys.write_text(
        '[bindings]\nsubmit = "c-x"\nqueue = "c-x"\n',
        encoding="utf-8",
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            keybindings_path=keys,
            input=pipe,
            output=_Output(),
        )
        assert runtime.frame.blocks == []

        runtime.transcript.append_welcome({}, immediate=False)
        runtime.flush_early_notifications()

        assert [block.kind for block in runtime.frame.blocks] == [
            "welcome",
            "banner",
        ]
        await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_live_hud_changes_invalidate_cached_sections_and_spinner_frames() -> None:
    theme = {
        "id": "hud-cache",
        "colors": {"accent": "cyan"},
        "symbols": {
            "spinner.status": ("A", "B"),
            "sep.thin": "|",
            "status.pending": "o",
        },
    }
    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        output=_Output(),
        status=lambda: "",
        theme=theme,
        ctx=SimpleNamespace(
            cfg=SimpleNamespace(cwd=Path.cwd(), notify=False, symbols="ascii"),
            session=SimpleNamespace(get=lambda _session: SimpleNamespace(title="work")),
            session_id="session",
            plugin_states={},
        ),
    )
    runtime.set_todos([{"content": "old todo", "status": "pending"}])
    runtime.queue.append("old prompt")
    await runtime.handle_presentation(
        ToolCallStart("task", {"description": "worker one"}, "call-1")
    )
    first = runtime._hud_text().value
    assert "old todo" in first
    assert "old prompt" in first
    assert "⣾ call-1: worker one" in first

    runtime.set_todos([{"content": "new todo", "status": "pending"}])
    runtime.queue.clear()
    runtime.queue.append("new prompt")
    await runtime.handle_presentation(
        ToolCallStart("task", {"description": "worker two"}, "call-2")
    )
    runtime._spinner_tick(1)
    second = runtime._hud_text().value

    assert "new todo" in second and "old todo" not in second
    assert "new prompt" in second and "old prompt" not in second
    assert "worker one" in second and "⣾ call-2: worker two" in second
    await runtime.scheduler.aclose()


@pytest.mark.asyncio
async def test_runtime_notification_triggers_cover_turn_end_and_approval() -> None:
    messages: list[str] = []

    class Recorder:
        async def notify(self, _title: str, message: str) -> bool:
            messages.append(message)
            return True

    runtime = ApplicationRuntime(
        lambda _text: asyncio.sleep(0),
        output=_Output(),
        status=lambda: "",
        ctx=SimpleNamespace(
            cfg=SimpleNamespace(cwd=Path.cwd(), notify=True, symbols="ascii"),
            session=SimpleNamespace(get=lambda _session: SimpleNamespace(title="work")),
            session_id="session",
            plugin_states={},
        ),
    )
    runtime.notifier = Recorder()

    await runtime.handle_presentation(TurnEnd("thread"))
    await runtime.handle_presentation(InterruptRaised({"action_requests": []}))

    assert messages == ["Turn complete", "Approval required"]
    await runtime.scheduler.aclose()



@pytest.mark.asyncio
async def test_streamed_todo_state_updates_the_hud_before_turn_completion() -> None:
    seen: list[list[dict[str, str]]] = []
    ctx = SimpleNamespace(
        ui=SimpleNamespace(set_todos=lambda todos: seen.append(todos)),
    )

    result = await _updates_event(
        ctx,
        {"tools": {"todos": [{"content": "ship", "status": "in_progress"}]}},
        set(),
        set(),
    )

    assert result is None
    assert seen == [[{"content": "ship", "status": "in_progress"}]]

@pytest.mark.asyncio
async def test_runtime_tracks_actual_keypress_and_turn_title_headlessly() -> None:
    now = [0.0]
    output = _Output()
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _text: asyncio.sleep(0),
            input=pipe,
            output=output,
            status=lambda: "",
            ctx=SimpleNamespace(
                cfg=SimpleNamespace(
                    cwd=Path.cwd(),
                    notify=False,
                    symbols="ascii",
                    statusbar=True,
                ),
                session=SimpleNamespace(get=lambda _session: SimpleNamespace(title="headless")),
                session_id="session",
                plugin_states={},
            ),
        )
        runtime.notifier._clock = lambda: now[0]
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0.02)
        now[0] = 7.0
        pipe.send_text("x")
        await asyncio.sleep(0.02)
        assert runtime.notifier.last_keypress == 7.0

        await runtime.handle_presentation(TurnStart("thread", "hello"))
        await runtime.handle_presentation(TurnEnd("thread"))
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(task, timeout=1)

    assert "* orcha - headless" in output.titles
    assert output.titles[-1] == "orcha - headless"
