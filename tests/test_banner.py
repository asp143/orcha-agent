from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.builtin import banner
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry


WIDE_ART = [
    "  ██████╗ ██████╗  ██████╗██╗  ██╗ █████╗",
    " ██╔═══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗",
    " ██║   ██║██████╔╝██║     ███████║███████║",
    " ██║   ██║██╔══██╗██║     ██╔══██║██╔══██║",
    " ╚██████╔╝██║  ██║╚██████╗██║  ██║██║  ██║",
    "  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝",
]
BOX_AND_BLOCK_CHARS = set("█╔╗╚╝║═┌┐└┘│─")


class FakeConsoleOutput:
    """Minimal ConsoleOutput double with deterministic terminal capabilities."""

    def __init__(self, *, width: int, encoding: str) -> None:
        self.console = SimpleNamespace(
            width=width,
            encoding=encoding,
            no_color=False,
        )
        self._chunks: list[str] = []

    def print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        **_: Any,
    ) -> None:
        self._chunks.append(sep.join(str(item) for item in objects) + end)

    @property
    def text(self) -> str:
        return "".join(self._chunks)


@pytest.fixture(autouse=True)
def _clean_banner_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCHA_NO_BANNER", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)


async def _dispatch_start(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str | None = "anthropic:claude-opus-5",
    mode: str = "ask",
    banner_enabled: bool = True,
    width: int = 120,
    encoding: str = "utf-8",
) -> tuple[FakeConsoleOutput, list[str]]:
    home = Path("/home/tester")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    version_requests: list[str] = []

    def fake_version(distribution: str) -> str:
        version_requests.append(distribution)
        return "3.4.5"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    bus = EventBus()
    banner.register(
        PluginAPI(
            name="banner",
            config={},
            state={},
            registry=Registry(),
            bus=bus,
            request_rebuild=lambda: None,
        )
    )
    output = FakeConsoleOutput(width=width, encoding=encoding)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            model=model,
            mode=mode,
            cwd=home / "src/orcha",
            banner=banner_enabled,
        ),
        console=output,
    )

    await bus.emit(AppStart(ctx=ctx))

    return output, version_requests


@pytest.mark.asyncio
async def test_wide_startup_banner_matches_the_v3_spec_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, version_requests = await _dispatch_start(monkeypatch)

    assert output.text.splitlines() == [
        *WIDE_ART,
        "        pluggable coding agent · v3.4.5",
        "  model: anthropic:claude-opus-5   mode: ask   cwd: ~/src/orcha",
        "  /help for commands",
    ]
    assert version_requests == ["orcha-agent"]


@pytest.mark.asyncio
async def test_wide_banner_explains_how_to_select_a_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = await _dispatch_start(monkeypatch, model=None, mode="plan")

    assert (
        "  model: (none) — /model or /login codex   mode: plan   cwd: ~/src/orcha"
        in output.text.splitlines()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("banner_enabled", "no_banner_env"),
    [(False, None), (True, "1")],
    ids=["core-config-false", "environment"],
)
async def test_banner_can_be_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    banner_enabled: bool,
    no_banner_env: str | None,
) -> None:
    if no_banner_env is not None:
        monkeypatch.setenv("ORCHA_NO_BANNER", no_banner_env)

    output, _ = await _dispatch_start(
        monkeypatch,
        banner_enabled=banner_enabled,
    )

    assert output.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "encoding", "no_color"),
    [
        (49, "utf-8", False),
        (120, "ascii", False),
        (120, "utf-8", True),
    ],
    ids=["narrow", "non-utf", "no-color"],
)
async def test_incompatible_terminals_get_a_plain_banner(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    encoding: str,
    no_color: bool,
) -> None:
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")

    output, _ = await _dispatch_start(
        monkeypatch,
        width=width,
        encoding=encoding,
    )
    text = output.text

    assert not BOX_AND_BLOCK_CHARS.intersection(text)
    assert "pluggable coding agent" in text
    assert "v3.4.5" in text
    assert "anthropic:claude-opus-5" in text
    assert "ask" in text
    assert "~/src/orcha" in text
    assert "/help for commands" in text
    if encoding == "ascii":
        assert text.isascii()
