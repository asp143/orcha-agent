from __future__ import annotations

from io import StringIO

import pytest
from langchain_core.messages import ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from orcha_agent.tui.blocks import BlockRendererDispatcher
from orcha_agent.tui.blocks.assistant import render as render_assistant
from orcha_agent.tui.blocks.banner import render as render_banner
from orcha_agent.tui.blocks.diff import render as render_diff
from orcha_agent.tui.blocks.hud import render_subagents, render_todo
from orcha_agent.tui.blocks.marker import render as render_marker
from orcha_agent.tui.blocks.thinking import SPINNER_FRAMES, render as render_thinking
from orcha_agent.tui.blocks.tool import render as render_tool
from orcha_agent.tui.blocks.user import render as render_user
from orcha_agent.tui.frame import Block, BlockState


THEME = {
    "id": "test-dark",
    "colors": {
        "text": "white",
        "muted": "bright_black",
        "dim": "bright_black",
        "userMessageBg": "grey19",
        "userMessageText": "white",
        "thinkingText": "bright_black",
        "thinkingOff": "cyan",
        "toolTitle": "cyan",
        "toolOutput": "white",
        "toolPendingBg": "grey11",
        "toolSuccessBg": "grey15",
        "toolErrorBg": "grey15",
        "toolDiffAdded": "green",
        "toolDiffRemoved": "red",
        "toolDiffContext": "bright_black",
        "error": "red",
        "warning": "yellow",
        "accent": "cyan",
    },
}


def block(kind: str, **data: object) -> Block:
    return Block(id=f"{kind}-1", kind=kind, data=dict(data))


def plain(renderable: object, width: int = 80) -> str:
    if renderable is None:
        return ""
    output = StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    ).print(renderable)
    return output.getvalue()


def test_user_bubble_is_full_width_and_supports_lite_markup() -> None:
    rendered = render_user(
        block("user", text="Use **bold** and `code`", queued=True),
        THEME,
        40,
        20,
        False,
    )
    output = plain(rendered, 40)

    assert "Use bold and code" in output
    assert max(map(len, output.splitlines())) == 40
    assert any("bold" in str(span.style) for span in rendered.renderable.spans)
    assert any("dim" in str(span.style) for span in rendered.renderable.spans)


def test_assistant_returns_rich_markdown_for_accumulated_text() -> None:
    rendered = render_assistant(
        block("assistant", text="# Result\n\n- one\n- two"),
        THEME,
        80,
        20,
        False,
    )

    assert isinstance(rendered, Markdown)
    assert "Result" in plain(rendered)
    assert "one" in plain(rendered)


def test_hidden_thinking_has_deterministic_pulse_and_rate() -> None:
    rendered = render_thinking(
        block(
            "thinking",
            text="private plan",
            visible=False,
            spinner_frame=2,
            reasoning_tokens=120,
            tokens_per_second=24.0,
        ),
        THEME,
        80,
        20,
        False,
    )
    output = plain(rendered)

    assert output == f"{SPINNER_FRAMES[2]} 120 tokens · 24.0 tok/s\n"
    assert "private plan" not in output


def test_visible_thinking_renders_italic_markdown() -> None:
    rendered = render_thinking(
        block("thinking", text="**Check** constraints", visible=True),
        THEME,
        80,
        20,
        False,
    )

    assert isinstance(rendered, Markdown)
    assert "Check constraints" in plain(rendered)


@pytest.mark.parametrize(
    ("rows", "expected", "forbidden"),
    [
        (3, "$ pytest -q", "output line"),
        (2, "╭─ Bash · pytest -q · 1.2s", "output line"),
        (1, "✔ Bash · pytest -q · 1.2s", "output line"),
        (0, "", "Bash"),
    ],
)
def test_tool_degrades_with_observable_row_budget(
    rows: int,
    expected: str,
    forbidden: str | None,
) -> None:
    rendered = render_tool(
        block(
            "tool",
            name="execute",
            args={"command": "pytest -q"},
            result={"stdout": "output line", "exit_code": 0},
            elapsed=1.25,
        ),
        THEME,
        80,
        rows,
        False,
    )
    output = plain(rendered)

    assert expected in output
    if forbidden is not None:
        assert forbidden not in output


def test_bash_preview_keeps_ten_output_tail_lines_until_expanded() -> None:
    output = "\n".join(f"line {index}" for index in range(25))
    value = block(
        "tool",
        name="execute",
        args={"command": "python build.py"},
        result={"stdout": output, "exit_code": 3},
    )

    collapsed = plain(render_tool(value, THEME, 5000, 100, False), 5000)
    expanded = plain(render_tool(value, THEME, 5000, 100, True), 5000)

    assert "line 14" not in collapsed
    assert "line 15" in collapsed
    assert "showing 10 of 25" in collapsed
    assert "ctrl+o to expand" in collapsed
    assert "line 24" in expanded
    assert "Exit: 3" in expanded
    assert "ctrl+o to expand" not in expanded


def test_grouped_read_files_share_one_card() -> None:
    rendered = render_tool(
        block(
            "tool",
            name="read_file",
            calls=[
                {"args": {"path": "a.py"}, "result": "alpha"},
                {"args": {"path": "b.py"}, "result": "beta"},
            ],
        ),
        THEME,
        80,
        20,
        False,
    )
    output = plain(rendered)

    assert "• Read (2)" in output
    assert "a.py" in output
    assert "b.py" in output


def test_diff_visualizes_indent_and_inverse_word_changes() -> None:
    rendered = render_diff(
        block(
            "diff",
            text="@@ -1,2 +1,2 @@\n-  old value\n+\tnew value\n context",
        ),
        THEME,
        100,
        20,
        False,
    )
    output = plain(rendered, 100)

    assert "-  1│··old value" in output
    assert "+  1│→new value" in output
    assert "   2│context" in output
    assert any("reverse" in str(span.style) for span in rendered.spans)


def test_streaming_diff_suppresses_trailing_unbalanced_removals() -> None:
    value = block("diff", text="@@ -3,2 +3 @@\n context\n-old\n-unfinished")
    value.state = BlockState.ACTIVE

    output = plain(render_diff(value, THEME, 80, 20, False))

    assert "context" in output
    assert "old" not in output
    assert "unfinished" not in output


def test_banner_caps_error_at_eight_lines() -> None:
    rendered = render_banner(
        block("banner", message="\n".join(f"line {i}" for i in range(12)), level="error"),
        THEME,
        80,
        20,
        False,
    )
    output = plain(rendered)

    assert "line 6" in output
    assert "line 7" not in output
    assert "…" in output
    assert "Error" in output


def test_marker_uses_compact_clear_and_branch_labels() -> None:
    assert "⊟ compacted" in plain(render_marker(block("marker", reason="compact"), THEME, 80, 20, False))
    assert "⊠ cleared" in plain(render_marker(block("marker", reason="clear"), THEME, 80, 20, False))
    assert "⎇ branched to child" in plain(
        render_marker(block("marker", reason="branch", new="child"), THEME, 80, 20, False)
    )


def test_hud_is_hidden_when_empty_and_capped_at_eight_rows() -> None:
    assert render_todo(block("todo", items=[]), THEME, 80, 20, False) is None
    assert render_subagents(block("subagents", agents=[]), THEME, 80, 20, False) is None

    todo = plain(
        render_todo(
            block("todo", items=[{"text": f"task {i}", "done": i == 0} for i in range(12)]),
            THEME,
            80,
            20,
            False,
        )
    )
    agents = plain(
        render_subagents(
            block("subagents", agents=[{"name": f"agent-{i}", "status": "running"} for i in range(12)]),
            THEME,
            80,
            20,
            False,
        )
    )

    assert len(todo.splitlines()) <= 8
    assert "task 5" in todo
    assert "task 6" not in todo
    assert len(agents.splitlines()) <= 8
    assert "agent-8" in agents
    assert "agent-11" in agents
    assert "agent-7" not in agents


def test_dispatcher_memoizes_by_revision_width_expansion_theme_and_budget() -> None:
    calls: list[tuple[int, int, bool]] = []

    def renderer(value: Block, _theme: object, width: int, rows: int, expanded: bool) -> str:
        calls.append((width, rows, expanded))
        return f"{value.data['text']}:{width}:{rows}:{expanded}"

    dispatcher = BlockRendererDispatcher({"assistant": renderer})
    value = block("assistant", text="hello")

    assert dispatcher.render(value, THEME, 80, 3, False) == "hello:80:3:False"
    assert dispatcher.render(value, THEME, 80, 3, False) == "hello:80:3:False"
    assert len(calls) == 1

    assert dispatcher.render(value, THEME, 80, 2, False) == "hello:80:2:False"
    value.update(text="updated")
    assert dispatcher.render(value, THEME, 80, 3, False) == "updated:80:3:False"
    assert len(calls) == 3

    dispatcher.evict([value])
    assert not dispatcher._cache


def test_dispatcher_falls_back_to_raw_output_when_renderer_raises() -> None:
    def broken(*_args: object) -> object:
        raise RuntimeError("broken renderer")

    dispatcher = BlockRendererDispatcher({"tool": broken})
    value = block("tool", result={"stdout": "raw tool output"})

    assert dispatcher.render(value, THEME, 80, 3, False) == "raw tool output"


def test_nonzero_execute_artifact_renders_error_state() -> None:
    result = ToolMessage(
        content="command failed",
        name="execute",
        tool_call_id="execute-1",
        artifact={"exit_code": 7},
        status="success",
    )

    output = plain(
        render_tool(
            block(
                "tool",
                name="execute",
                args={"command": "false"},
                result=result,
            ),
            THEME,
            80,
            3,
            False,
        )
    )

    assert "$ false" in output
    assert len(output.splitlines()) == 3


def test_full_tool_card_fits_allocated_rows() -> None:
    rendered = render_tool(
        block(
            "tool",
            name="read_file",
            args={"path": "a.py"},
            result="one line",
        ),
        THEME,
        80,
        3,
        False,
    )
    output = plain(rendered)

    assert len(output.splitlines()) == 3
    assert output.splitlines()[-1].startswith("╰")


def test_grouped_reads_and_diffs_share_collapsed_preview_caps() -> None:
    grouped = block(
        "tool",
        name="read_file",
        calls=[
            {
                "args": {"path": f"{index}.py"},
                "result": "x" * 4100 if index == 0 else f"line {index}",
            }
            for index in range(25)
        ],
    )
    diff = block(
        "tool",
        name="edit_file",
        args={"file_path": "demo.py"},
        result={
            "diff": "@@ -1,25 +1,25 @@\n"
            + "\n".join(f" context {index}" for index in range(25))
        },
    )

    grouped_collapsed = plain(render_tool(grouped, THEME, 5000, 100, False), 5000)
    grouped_expanded = plain(render_tool(grouped, THEME, 5000, 100, True), 5000)
    diff_collapsed = plain(render_tool(diff, THEME, 100, 100, False), 100)
    diff_expanded = plain(render_tool(diff, THEME, 100, 100, True), 100)

    assert "x" * 4001 not in grouped_collapsed
    assert "20.py" in grouped_collapsed
    assert "24.py" in grouped_expanded
    assert "context 20" in diff_collapsed
    assert "context 24" in diff_expanded


def test_subagent_assistant_is_dim_and_indented_two_columns() -> None:
    rendered = render_assistant(
        block("assistant", text="subagent answer", subagent=True),
        THEME,
        80,
        20,
        False,
    )

    assert isinstance(rendered, Padding)
    assert (rendered.top, rendered.right, rendered.bottom, rendered.left) == (
        0,
        2,
        0,
        2,
    )
    assert isinstance(rendered.renderable, Markdown)
    assert "dim" in str(rendered.renderable.style)
    assert plain(rendered).startswith("  subagent answer")
