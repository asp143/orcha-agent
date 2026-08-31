from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.builtin import banner
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.frame import Frame
from orcha_agent.tui.transcript import Transcript


class RecordingTranscript(Transcript):
    def __init__(self) -> None:
        super().__init__(Frame())
        self.commits: list[list[str]] = []


def _context(tmp_path: Path, *, enabled: bool = True, symbols: str = "unicode") -> Any:
    transcript = RecordingTranscript()
    return SimpleNamespace(
        cfg=SimpleNamespace(
            model="anthropic:claude-opus-5",
            mode="ask",
            cwd=tmp_path,
            banner=enabled,
            symbols=symbols,
            trust_cwd=True,
        ),
        console=SimpleNamespace(console=SimpleNamespace(encoding="utf-8")),
        transcript=transcript,
        session=SimpleNamespace(list=lambda: []),
        session_id="current",
        plugins=[],
        registry=SimpleNamespace(
            providers={
                "anthropic": SimpleNamespace(
                    available=lambda: None,
                    env_keys=(),
                )
            },
            auth={},
        ),
    )


async def _start(ctx: Any) -> None:
    bus = EventBus()
    api = PluginAPI(
        name="banner",
        config={},
        state={},
        registry=Registry(),
        bus=bus,
        request_rebuild=lambda: None,
    )
    banner.register(api)
    await bus.emit(AppStart(ctx=ctx))


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCHA_NO_BANNER", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.mark.asyncio
async def test_welcome_is_the_first_block_and_stays_visible(tmp_path: Path) -> None:
    ctx = _context(tmp_path)

    await _start(ctx)

    assert [block.kind for block in ctx.transcript.frame.blocks] == ["welcome"]
    # Settled, not committed: it must stay in the bottom-anchored viewport on
    # first load and retire with the first real commit.
    assert ctx.transcript.frame.blocks[0].state.value == "settled"
    assert len(ctx.transcript.frame.blocks[0].data["sessions"]) == 4
    assert len(ctx.transcript.frame.blocks[0].data["hints"]) == 4

def test_welcome_provider_hint_requires_actual_readiness(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    provider = ctx.registry.providers["anthropic"]

    provider.available = lambda: "optional package missing"
    assert "provider unavailable" in banner.build_welcome(ctx)["hints"][2]

    provider.available = lambda: None
    provider.env_keys = ("ORCHA_TEST_MISSING_KEY",)
    assert "provider unavailable" in banner.build_welcome(ctx)["hints"][2]

    provider.env_keys = ()
    ctx.registry.auth["anthropic"] = SimpleNamespace(
        flow=SimpleNamespace(status=lambda: "not logged in")
    )
    assert "provider unavailable" in banner.build_welcome(ctx)["hints"][2]

    ctx.registry.auth.clear()
    assert "provider ready" in banner.build_welcome(ctx)["hints"][2]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["ui", "legacy", "environment"])
async def test_welcome_disable_and_backcompat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    ctx = _context(tmp_path, enabled=source == "environment")
    if source == "environment":
        monkeypatch.setenv("ORCHA_NO_BANNER", "1")

    await _start(ctx)

    assert ctx.transcript.frame.blocks == []


@pytest.mark.asyncio
async def test_banner_never_writes_directly_when_application_transcript_exists(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.console.print = lambda *_args, **_kwargs: pytest.fail("direct terminal write")

    await _start(ctx)

    assert ctx.transcript.frame.blocks[0].kind == "welcome"


@pytest.mark.asyncio
async def test_no_color_produces_ascii_safe_welcome_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    ctx = _context(tmp_path)

    await _start(ctx)

    block = ctx.transcript.frame.blocks[0]
    assert block.data["ascii"] is True
    assert "".join(block.data["logo"]).isascii()


@pytest.mark.asyncio
async def test_welcome_stays_in_viewport_until_first_real_commit(tmp_path) -> None:
    from orcha_agent.tui.frame import BlockState
    from orcha_agent.tui.transcript import Transcript

    transcript = Transcript()
    block = transcript.append_welcome({"logo": ["X"], "model": "m"})
    # Visible on first load: settled in the frame, NOT committed to scrollback.
    assert block.state is BlockState.SETTLED
    assert [b.kind for b in transcript.frame.blocks] == ["welcome"]

    # The first real commit retires the welcome ahead of the new content.
    transcript.print("hello")
    ready_states = {b.kind: b.state for b in transcript.frame.blocks}
    assert ready_states["welcome"] is BlockState.COMMITTED
