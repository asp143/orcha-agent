from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.markdown import Markdown

from orcha_agent.tui.blocks.diff import render as render_diff
from orcha_agent.tui.blocks.markdown import StreamingMarkdown, _PARSED
from orcha_agent.tui.blocks.terminal import clear_terminal_cache, terminal_rows, _REPLAYS
from orcha_agent.tui.blocks.tool import render
from orcha_agent.tui.frame import Block, BlockState
from orcha_agent.tui.theme import load_themes


def test_terminal_coalesces_styles_and_skips_untouched_cells():
    rows = terminal_rows("runs", "plain text " * 20 + "\x1b[32mgreen text\x1b[0m", 400)
    assert len(rows) == 1
    assert len(rows[0].spans) == 1
    stream = StringIO()
    Console(file=stream, force_terminal=True, color_system="truecolor", width=400).print(rows[0])
    assert len(stream.getvalue()) < len(rows[0].plain) * 2
    assert len(_REPLAYS["runs"].screen.buffer) == 1


def test_terminal_cache_clear_resets_reused_ids():
    terminal_rows("session", "old", 80)
    old = _REPLAYS["session"]
    clear_terminal_cache()
    terminal_rows("session", "old new", 80)
    assert _REPLAYS["session"] is not old


def test_tab_diff_keyword_spans_align_and_polarity_has_background(tmp_path):
    theme = load_themes(home=tmp_path)["dark"]
    block = Block(
        id="tabs",
        kind="diff",
        state=BlockState.SETTLED,
        data={
            "path": "example.py",
            "text": "@@ -1,2 +1,2 @@\n-\treturn 'old'\n+\treturn 'new'\n \treturn 'context'",
        },
    )
    value = render_diff(block, theme, 80, 20, False)
    console = Console()
    for line in value.split("\n")[1:]:
        start = line.plain.index("return")
        keyword = line.get_style_at_offset(console, start).color
        assert all(
            line.get_style_at_offset(console, i).color == keyword for i in range(start, start + 6)
        )
        assert line.get_style_at_offset(console, start + 8).color != keyword
        if line.plain.startswith(("-", "+")):
            assert line.get_style_at_offset(console, start).bgcolor is not None


def test_expand_and_collapse_hint_is_dim_footer():
    block = Block(
        id="footer",
        kind="tool",
        state=BlockState.SETTLED,
        data={
            "name": "bash",
            "args": {"command": "test"},
            "result": "one\ntwo\nthree",
            "leading_spacer": False,
        },
    )
    for expanded, action in ((False, "Expand"), (True, "Collapse")):
        value = render(block, None, 80, 4, expanded)
        assert action not in value.plain.splitlines()[0]
        footer = value.split("\n")[-1]
        assert f"Ctrl+O: {action}" in footer.plain
        assert footer.get_style_at_offset(Console(), footer.plain.index("Ctrl")).dim


@pytest.mark.parametrize(
    "tail",
    [
        "tail grows",
        "- item\n\n- another",
        "> quote\n> more",
        "```python\nreturn 1\n```",
        "|a|b|\n|-|-|\n|1|2|",
        "heading\n---",
        "[ref]\n\n[ref]: https://example.com",
    ],
)
def test_incremental_markdown_boundaries_match_full_parse(tail):
    _PARSED.clear()
    prefix = "# Stable\n\nSettled paragraph.\n\n"
    for length in range(1, len(tail) + 1):
        text = prefix + tail[:length]

        def capture(cls):
            stream = StringIO()
            Console(file=stream, width=60, color_system="truecolor", force_terminal=True).print(
                cls(text, hyperlinks=False)
            )
            return stream.getvalue()

        assert capture(StreamingMarkdown) == capture(Markdown)


def test_incremental_parser_only_receives_mutable_tail(monkeypatch):
    import rich.markdown

    _PARSED.clear()
    source = "# stable\n\nsettled\n\ntail[index]"
    StreamingMarkdown(source)
    parser = rich.markdown.MarkdownIt.parse
    parsed = []

    def record(self, source, *args, **kwargs):
        parsed.append(source)
        return parser(self, source, *args, **kwargs)

    monkeypatch.setattr(rich.markdown.MarkdownIt, "parse", record)
    StreamingMarkdown(source + " grows")
    assert parsed == ["tail[index] grows"]


@pytest.mark.parametrize(
    "action,result,expected",
    [
        (
            "list",
            {"jobs": [{"id": "one", "status": "running", "command": "build"}]},
            "one · running · build",
        ),
        ("read", {"stdout": "\x1b[32mready\x1b[0m", "exit_code": 0}, "ready"),
        ("kill", {"text": "Stopped"}, "Stopped"),
    ],
)
def test_bash_jobs_actions(action, result, expected):
    block = Block(
        id="job-" + action,
        kind="tool",
        state=BlockState.SETTLED,
        data={
            "name": "bash_jobs",
            "args": {"action": action, "job_id": "one"},
            "result": result,
            "leading_spacer": False,
        },
    )
    value = render(block, None, 80, 20, False)
    assert "Bash jobs" in value.plain
    assert expected in value.plain


def test_bash_jobs_gallery_golden(update_goldens, monkeypatch):
    from orcha_agent.tui.gallery_fixtures.blocks import TOOL_GALLERY_FIXTURES

    monkeypatch.delenv("NO_COLOR", raising=False)
    fixture = TOOL_GALLERY_FIXTURES["bash_jobs"]["success"]
    value = render(
        Block(id="jobs", kind="tool", state=fixture.state, data=fixture.data), None, 80, 20, False
    )
    stream = StringIO()
    Console(file=stream, width=80, force_terminal=True, color_system="truecolor").print(value)
    actual = stream.getvalue().replace("\x1b", "<ESC>")
    golden = Path(__file__).with_name("golden") / "bash-jobs.80.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
