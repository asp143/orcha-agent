from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from orcha_agent.core.config import load_config, save_core_value
from orcha_agent.tui.errors import humanize_error
from orcha_agent.tui.frame import BlockState
from orcha_agent.tui.gallery_fixtures.runtime import runtime_examples
from orcha_agent.tui.runtime import ApplicationRuntime
from orcha_agent.tui.theme import load_themes
from orcha_agent.tui.transcript import Transcript


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ValueError("The usage limit has been reached; plan: pro; resets at: 9040"),
            "Codex usage limit reached (Pro plan) · resets in 2h 14m",
        ),
        (ValueError("Session expired"), "Session expired · run /login codex"),
        (ValueError("Rate limit reached; retry in 12 seconds"), "Rate limited · retrying in 12s"),
        (ConnectionError("broken pipe"), "Network error · check connection"),
        (ValueError("Provider declined request"), "Provider declined request"),
    ],
)
def test_provider_errors_are_human_readable(error, expected):
    assert humanize_error(error, now=1000) == expected


@pytest.mark.parametrize("override", ["ORCHA_CONFIG_DIR", "XDG_CONFIG_HOME"])
def test_config_override_keeps_settings_writes_isolated(tmp_path, override):
    env = {"HOME": str(tmp_path / "home"), override: str(tmp_path / "isolated")}
    cfg = load_config([], env=env, cwd=tmp_path)
    expected = (
        tmp_path
        / "isolated"
        / ("config.toml" if override == "ORCHA_CONFIG_DIR" else "orcha-agent/config.toml")
    )
    assert cfg.user_config_path == expected
    save_core_value(expected, "mode", "edit")
    assert load_config([], env=env, cwd=tmp_path).mode == "edit"
    assert not (tmp_path / "home/.config/orcha-agent/config.toml").exists()


def test_gallery_accepts_theme_after_subcommand(tmp_path):
    cfg = load_config(["gallery", "--theme", "light"], env={"HOME": str(tmp_path)}, cwd=tmp_path)
    assert cfg.theme == "light"


def test_startup_warnings_precede_single_welcome_without_committing_it():
    transcript = Transcript()
    first = transcript.append_welcome({"model": "test"})
    assert transcript.append_welcome({"model": "test"}) is first
    warning = transcript.append_banner("6 skills skipped · Ctrl+O for details", level="warning")
    assert transcript.frame.blocks == [warning, first]
    assert transcript.frame.commit_ready() == []
    transcript.append_banner("6 skills skipped · Ctrl+O for details", level="warning")
    assert warning.data["count"] == 2
    transcript.release_startup()
    assert transcript.frame.commit_ready() == [warning, first]
    assert first.state == BlockState.COMMITTED


@pytest.mark.asyncio
async def test_loop_failure_is_a_pinned_error_without_enter_prompt(tmp_path):
    async def submit(_text):
        raise AssertionError("No provider turn is permitted")

    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, input=pipe, output=DummyOutput())
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        running = asyncio.create_task(runtime.run())
        for _ in range(100):
            if runtime.application.is_running:
                break
            await asyncio.sleep(0.001)
        loop.call_exception_handler({"exception": ValueError("background failure")})
        error = runtime.transcript._pinned_error
        assert error is not None
        assert error.data["message"] == "background failure"
        assert "ValueError" in Path(error.data["log_path"]).read_text()
        assert Path(error.data["log_path"]).stat().st_mode & 0o777 == 0o600
        pipe.send_bytes(b"\x04")
        await asyncio.wait_for(running, 2)
        assert loop.get_exception_handler() is previous
        Path(error.data["log_path"]).unlink()


@pytest.mark.parametrize("width", [80, 120])
def test_runtime_polish_golden(width, update_goldens, tmp_path):
    theme = load_themes(home=tmp_path, symbols="unicode")["dark"]
    stream = StringIO()
    console = Console(
        file=stream, width=width, force_terminal=True, color_system="truecolor", no_color=False
    )
    for title, panel in runtime_examples(theme, width):
        console.print(title)
        console.print(panel)
    golden = Path(__file__).with_name("golden") / f"runtime-polish.{width}.ansi"
    if update_goldens:
        golden.write_text(stream.getvalue())
    assert golden.read_text() == stream.getvalue()


@pytest.mark.asyncio
async def test_provider_listing_never_displays_login_email():
    from orcha_agent.builtin.commands_core import _providers

    stream = StringIO()
    console = Console(file=stream, width=120)
    ctx = SimpleNamespace(
        registry=SimpleNamespace(
            auth={
                "codex": SimpleNamespace(
                    flow=SimpleNamespace(status=lambda: "logged in as private@example.com")
                )
            },
            providers={
                "codex": SimpleNamespace(
                    available=lambda: None,
                    capabilities=SimpleNamespace(
                        tool_calling=True, streaming=True, thinking=True, structured_output=True
                    ),
                    models=[],
                    env_keys=[],
                )
            },
        ),
        console=SimpleNamespace(console=console, print=console.print),
    )
    await _providers(ctx, "")
    assert "logged in" in stream.getvalue()
    assert "private@example.com" not in stream.getvalue()
    assert "╭" in stream.getvalue()


def test_plain_text_goldens_have_no_escape_bytes():
    directory = Path(__file__).with_name("golden")
    escaped = [
        path.name for path in sorted(directory.glob("*.txt")) if b"\x1b" in path.read_bytes()
    ]
    assert not escaped, f"ANSI goldens must use .ansi extensions: {escaped}"
