from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from orcha_agent import __main__ as entrypoint
from orcha_agent.core.config import load_config
from orcha_agent.tui.blocks import DEFAULT_RENDERERS
from orcha_agent.tui.gallery import render_gallery_state, run_gallery
from orcha_agent.tui.gallery_fixtures import GALLERY_FIXTURES, GALLERY_STATES
from orcha_agent.tui.theme import load_themes


def test_every_renderer_state_gallery_output_is_non_empty(tmp_path: Path) -> None:
    theme = load_themes(home=tmp_path, symbols="unicode")["dark"]

    assert set(GALLERY_FIXTURES) == set(DEFAULT_RENDERERS)
    for renderer in DEFAULT_RENDERERS:
        assert set(GALLERY_FIXTURES[renderer]) == set(GALLERY_STATES)
        for state in GALLERY_STATES:
            output = render_gallery_state(
                renderer,
                state,
                theme=theme,
                width=100,
                expanded=False,
                plain=True,
            )
            assert output.strip(), f"{renderer}/{state} rendered nothing"


def test_gallery_cli_parses_filters_and_plain_output(tmp_path: Path) -> None:
    cfg = load_config(
        [
            "gallery",
            "--tool",
            "tool",
            "--state",
            "error",
            "--width",
            "72",
            "--expanded",
            "--plain",
        ],
        env={"HOME": str(tmp_path)},
        cwd=tmp_path,
        user_config_path=tmp_path / "missing.toml",
        project_config_path=tmp_path / "missing-project.toml",
    )

    assert cfg.command == "gallery"
    assert cfg.gallery_tool == "tool"
    assert cfg.gallery_state == "error"
    assert cfg.gallery_width == 72
    assert cfg.gallery_expanded is True
    assert cfg.gallery_plain is True

    output = StringIO()
    assert run_gallery(cfg, file=output) == 0
    rendered = output.getvalue()
    assert "tool" in rendered
    assert "error" in rendered
    assert "streaming" not in rendered
    assert "\x1b[" not in rendered


def test_gallery_rejects_an_unknown_renderer_without_partial_output(tmp_path: Path) -> None:
    output = StringIO()
    cfg = SimpleNamespace(
        gallery_tool="missing",
        gallery_state=None,
        gallery_width=80,
        gallery_expanded=False,
        gallery_plain=True,
        theme="dark",
        symbols="unicode",
        cwd=tmp_path,
        trust_cwd=False,
    )

    assert run_gallery(cfg, file=output) == 2
    assert output.getvalue().startswith("Unknown renderer 'missing'. Known renderers:")


def test_gallery_entrypoint_skips_project_dotenv_and_interactive_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(command="gallery", trust_cwd=True)
    calls: list[object] = []

    monkeypatch.setattr(entrypoint, "load_config", lambda: cfg)
    monkeypatch.setattr(entrypoint, "run_gallery", lambda value: calls.append(value) or 0)
    monkeypatch.setattr(
        entrypoint,
        "load_dotenv",
        lambda *_args, **_kwargs: pytest.fail("gallery must not load project dotenv"),
    )

    async def fail_run_app(_cfg: object) -> int:
        pytest.fail("gallery must not start the interactive app")

    monkeypatch.setattr(entrypoint, "run_app", fail_run_app)

    with pytest.raises(SystemExit) as raised:
        entrypoint.main()

    assert raised.value.code == 0
    assert calls == [cfg]
