from dataclasses import replace
from io import StringIO
from pathlib import Path

from rich.console import Console

from orcha_agent.tui.blocks.syntax import highlight, language_from_path, syntax_style
from orcha_agent.tui.blocks.terminal import terminal_rows
from orcha_agent.tui.blocks.tool import render
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.theme import ThemeWatcher, load_themes, theme_from_background


def test_language_selectors_and_unknown_files():
    assert language_from_path("src/example.py:10-20") == "python"
    assert language_from_path("Dockerfile") == "docker"
    assert language_from_path("notes.unrecognizable") == "text"


def test_palette_changes_keyword_and_multiline_string(tmp_path: Path):
    dark = load_themes(home=tmp_path)["dark"]
    theme = replace(dark, colors={**dark.colors, "syntaxKeyword": "#010203"})
    value = highlight('return """first\nsecond"""', "python", theme)
    console = Console()
    assert value.get_style_at_offset(console, 0).color.triplet == (1, 2, 3)
    assert (
        value.get_style_at_offset(console, 10).color == value.get_style_at_offset(console, 17).color
    )
    assert syntax_style(theme) is syntax_style(theme)
    assert syntax_style(theme) is not syntax_style(dark)


def test_shell_replays_cursor_erase_and_ansi():
    rows = terminal_rows("cursor-test", "progress 10%\r\x1b[2K\x1b[32mdone\x1b[0m\nnext", 40)
    assert [row.plain for row in rows] == ["done", "next"]
    assert rows[0].get_style_at_offset(Console(), 0).color.name == "green"
    bright = terminal_rows("bright-test", "\x1b[91mred", 40)
    assert bright[0].get_style_at_offset(Console(), 0).color.name == "bright_red"
    assert [row.plain for row in terminal_rows("cursor-test", "new", 40)] == ["new"]


def test_shell_incremental_and_resize():
    assert terminal_rows("resize-test", "abc", 10)[0].plain == "abc"
    assert terminal_rows("resize-test", "abcdef", 10)[0].plain == "abcdef"
    assert [row.plain for row in terminal_rows("resize-test", "abcdef", 3)] == ["abc", "def"]


def test_hyperlink_setting_and_line_selector(tmp_path: Path):
    theme = load_themes(home=tmp_path)["dark"]
    block = Block(
        id="link",
        kind="tool",
        state=BlockState.SETTLED,
        data={
            "name": "read",
            "args": {"path": "demo.py:10-20"},
            "cwd": str(tmp_path),
            "result": "return 1",
        },
    )
    for enabled in (True, False):
        output = StringIO()
        Console(file=output, force_terminal=True, width=80).print(
            render(block, replace(theme, hyperlinks=enabled), 80, 20, False)
        )
        assert ("file://" in output.getvalue()) is enabled
        if enabled:
            assert "#L10" in output.getvalue()


def test_background_reply_and_debounced_theme_watcher(tmp_path: Path):
    assert theme_from_background("\x1b]11;rgb:ffff/ffff/ffff\x07") == "light"
    assert theme_from_background("\x1b]11;rgb:00/00/00\x1b\\") == "dark"
    assert theme_from_background("not a reply") is None
    watcher = ThemeWatcher(tmp_path)
    (tmp_path / "custom.json").write_text("{}")
    assert not watcher.changed(1)
    assert not watcher.changed(1.1)
    assert watcher.changed(1.3)
    assert not watcher.changed(2)


def test_incremental_markdown_matches_rich_and_reuses_stable_blocks():
    from rich.markdown import Markdown
    from orcha_agent.tui.blocks.markdown import StreamingMarkdown, _LAYOUTS

    markup = "# Heading\n\n> quote\n> - nested\n\n- first\n  - nested\n- second\n\n|a|b|\n|-|-|\n|1|2|\n\n---\n\n```python\nreturn 1\n```\n\ntail"

    def capture(cls, text):
        stream = StringIO()
        Console(file=stream, width=60, force_terminal=True).print(cls(text))
        return stream.getvalue()

    assert capture(StreamingMarkdown, markup) == capture(Markdown, markup)
    existing = set(_LAYOUTS)
    assert capture(StreamingMarkdown, markup + " grows") == capture(Markdown, markup + " grows")
    assert existing <= set(_LAYOUTS)
    assert len(set(_LAYOUTS) - existing) == 1


def test_image_protocols_validate_data_and_fallback():
    from orcha_agent.tui.blocks.image import image_protocol, render as render_image
    from orcha_agent.tui.gallery_fixtures.blocks import GALLERY_FIXTURES

    block = Block(
        id="image",
        kind="image",
        state=BlockState.SETTLED,
        data=GALLERY_FIXTURES["image"]["success"].data,
    )
    assert image_protocol(block, {"TERM": "xterm-kitty"}).startswith("\x1b_Ga=T,f=100,q=2,m=0;")
    assert image_protocol(block, {"TERM_PROGRAM": "iTerm.app"}).startswith("\x1b]1337;File=")
    assert image_protocol(block, {"TERM": "xterm-256color"}) == ""
    assert render_image(block, None, 80, 20, False).plain.startswith("[Image · PNG")
    malformed = replace(block, data={"image": {"data": "\x1b[2J", "mime_type": "image/png"}})
    assert image_protocol(malformed, {"TERM": "xterm-kitty"}) == ""


def test_image_and_syntax_palette_goldens(tmp_path: Path, update_goldens: bool):
    from orcha_agent.tui.blocks.image import render as render_image
    from orcha_agent.tui.gallery_fixtures.blocks import GALLERY_FIXTURES

    stream = StringIO()
    console = Console(file=stream, width=60, force_terminal=True, color_system="truecolor")
    for name, theme in load_themes(home=tmp_path).items():
        console.print(name)
        console.print(highlight('def greet(name): return "Hello"  # greeting', "python", theme))
    fixture = GALLERY_FIXTURES["image"]["success"]
    console.print(
        render_image(
            Block(id="image", kind="image", state=fixture.state, data=fixture.data),
            None,
            60,
            20,
            False,
        )
    )
    from orcha_agent.tui.blocks.hud import render_todo

    fixture = GALLERY_FIXTURES["todo"]["progress"]
    console.print(
        render_todo(
            Block(id="todo-animation", kind="todo", state=fixture.state, data=fixture.data),
            None,
            60,
            20,
            False,
        )
    )
    colorblind = load_themes(home=tmp_path, symbols="colorblind")["dark"]
    console.print(highlight("return True", "python", colorblind))
    console.print(
        render_todo(
            Block(id="colorblind-todo", kind="todo", state=fixture.state, data=fixture.data),
            colorblind,
            60,
            20,
            False,
        )
    )
    actual = stream.getvalue().replace("\x1b", "<ESC>")
    path = Path(__file__).with_name("golden") / "syntax-palettes-and-image.txt"
    if update_goldens:
        path.write_text(actual)
    assert path.read_text() == actual


def test_requested_theme_families_and_colorblind_preset(tmp_path: Path):
    themes = load_themes(home=tmp_path, symbols="colorblind")
    assert {
        "catppuccin-latte",
        "catppuccin-frappe",
        "catppuccin-macchiato",
        "catppuccin-mocha",
        "dark-tokyo-night",
        "light-tokyo-night",
        "tokyo-night-storm",
        "rose-pine",
        "rose-pine-moon",
        "rose-pine-dawn",
        "kanagawa",
        "everforest",
    } <= themes.keys()
    assert themes["dark"].color("success") == "#56b4e9"
    assert themes["dark"].color("error") == "#e69f00"
    assert themes["dark"].symbol("status.success") != themes["dark"].symbol("status.error")


def test_live_bash_remains_running_with_partial_result():
    from orcha_agent.tui.blocks.tool import _state

    block = Block(
        id="live-bash",
        kind="tool",
        state=BlockState.ACTIVE,
        data={
            "name": "bash",
            "args": {"command": "build"},
            "result": {"stdout": "working"},
            "leading_spacer": False,
        },
    )
    assert _state(block) == "running"
    card = render(block, None, 100, 3, False)
    assert "Ctrl+O" in card.plain
    assert "✔" not in card.plain
