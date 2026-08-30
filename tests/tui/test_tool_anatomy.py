from __future__ import annotations

from io import StringIO

from rich.console import Console

from orcha_agent.tui.blocks import DEFAULT_THEME
from orcha_agent.tui.blocks.tool import render
from orcha_agent.tui.frame import Block, BlockState


def _block(name: str, *, active: bool = False, **data: object) -> Block:
    return Block(
        id=f"tool-{name}",
        kind="tool",
        state=BlockState.ACTIVE if active else BlockState.SETTLED,
        data={"name": name, **data},
    )


def _plain(block: Block, *, width: int = 80, rows: int = 80, expanded: bool = False) -> str:
    output = StringIO()
    Console(file=output, width=width, force_terminal=False, color_system=None).print(
        render(block, DEFAULT_THEME, width, rows, expanded)
    )
    return output.getvalue()


def test_read_call_result_and_group_match_omp_anatomy() -> None:
    pending = _plain(
        _block("read_file", active=True, args={"path": "src/app.py", "offset": 4, "limit": 8})
    )
    result = _plain(
        _block(
            "read_file",
            args={"path": "src/app.py", "offset": 4},
            result="\n".join(f"line {index}" for index in range(1, 16)),
        )
    )
    grouped = _plain(
        _block(
            "read_file",
            calls=[
                {"args": {"path": f"src/{name}.py"}, "result": "one\ntwo\nthree\nfour"}
                for name in ("a", "b", "c")
            ],
        )
    )

    assert "⏳ Read: src/app.py:4-11" in pending
    assert "• Read src/app.py:4-18" in result
    assert " 4│line 1" in result
    assert "… 3 more lines ⟦Ctrl+O: Expand⟧" in result
    assert "• Read (3)" in grouped
    assert "├─ src/a.py" in grouped and "└─ src/c.py" in grouped


def test_tool_headers_stay_single_line_and_shorten_paths_at_common_widths() -> None:
    cwd = "/workspace/project"
    basename = "renderer_output.py"
    long_path = f"{cwd}/{'nested/' * 12}{basename}"
    assert len(long_path) >= 120

    for width in (60, 80, 120):
        output = _plain(
            _block(
                "read_file",
                cwd=cwd,
                args={"path": long_path},
                result="content",
            ),
            width=width,
        )
        lines = output.splitlines()
        assert len(lines) == 4
        assert len(lines[1]) == width
        assert basename in lines[1]
        assert ("…" in lines[1]) is (width < 120)
        assert cwd not in lines[1]


def test_tool_header_flattens_control_lines_and_shortens_home() -> None:
    from pathlib import Path

    home_path = Path.home() / "projects" / "demo.py"
    output = _plain(
        _block(
            "read_file",
            args={"path": f"{home_path}\r\nunexpected"},
            result="content",
        ),
        width=120,
    )

    assert "\r" not in output
    assert "~/projects/demo.py unexpected" in output
    assert len(output.splitlines()) == 4


def test_write_streaming_and_result_use_tail_and_line_count() -> None:
    content = "\n".join(f"line {index}" for index in range(20))
    streaming = _plain(
        _block("write_file", active=True, args={"path": "out.py", "content": content})
    )
    result = _plain(
        _block("write_file", args={"path": "out.py", "content": content}, result="ok")
    )

    assert "Write: out.py" in streaming
    assert "line 0" not in streaming and "line 19" in streaming
    assert "(streaming)" in streaming
    assert "✎ Write: out.py (20 lines)" in result
    assert "line 5" in result and "line 6" not in result


def test_edit_header_gutters_counts_and_collapsed_hint() -> None:
    diff = "@@ -313,2 +313,2 @@\n context\n-old value\n+new value"
    output = _plain(
        _block("edit_file", args={"path": "demo.py"}, result={"diff": diff})
    )

    assert "Edit: demo.py:313 ⟦+1/-1⟧" in output
    assert " 313│context" in output
    assert "-314│old value" in output
    assert "+314│new value" in output


def test_bash_has_command_output_sections_and_footer() -> None:
    output = _plain(
        _block(
            "execute",
            args={"command": "printf ok", "timeout": 120},
            result={"stdout": "ok", "exit_code": 1, "wall_time": 1.2},
        )
    )

    assert "execute" not in output.casefold()
    assert "$ printf ok" in output
    assert "├─── Output " in output
    assert "⟦Wall: 1.2s | Exit: 1 | Timeout: 120s⟧" in output


def test_grep_glob_and_web_search_are_inline_without_frames() -> None:
    grep = _plain(
        _block(
            "grep",
            args={"pattern": "needle", "path": "src"},
            result={"matches": ["a.py:1:needle", "b.py:2:needle", "b.py:4:needle"]},
        )
    )
    glob = _plain(_block("glob", args={"pattern": "*.py"}, result=["a.py", "b.py"]))
    web = _plain(_block("web_search", args={"query": "orcha"}, result=["one", "two"]))

    assert grep.startswith("🔍 Grep: needle  3 matches · 2 files · in src")
    assert "╭" not in grep and "├─" in grep and "└─" in grep
    assert "Glob: *.py" in glob and "╭" not in glob
    assert "Web Search: orcha" in web and "╭" not in web


def test_generic_args_hint_and_degradation_rows() -> None:
    value = _block(
        "custom",
        active=True,
        args={"alpha": 1, "beta": "two"},
        result="\n".join(f"line {index}" for index in range(10)),
        status="running",
        elapsed=4.0,
    )
    full = _plain(value, rows=8)
    folded = _plain(value, rows=2)
    single = _plain(value, rows=1)

    assert "└─ alpha=1 beta=two" in full
    assert len(folded.splitlines()) == 2 and folded.startswith("╭─ Custom · 4.0s")
    assert folded.splitlines()[-1] == "╰"
    assert single == "⣾ Custom · 4.0s\n"
    assert render(value, DEFAULT_THEME, 80, 0, False) is None


def test_task_and_todo_cards_match_omp_headers_and_rows() -> None:
    task = _plain(
        _block(
            "task",
            result={
                "agents": [
                    {"id": "a", "description": "inspect", "status": "success", "requests": 4, "tokens": 120, "cost": 0.02, "elapsed": 12},
                    {"id": "b", "description": "test", "status": "error", "requests": 2, "tokens": 80, "cost": 0.01, "elapsed": 9},
                ],
                "requests": 6,
                "elapsed": 21,
            },
        )
    )
    todo = _plain(
        _block(
            "todo",
            args={"items": [{"text": "done", "done": True}, {"text": "next"}]},
            result="ok",
        )
    )

    assert "⇶ Task · 2 agents" in task
    assert "✔ a: inspect" in task and "✘ b: test" in task
    assert "⟦1 succeeded · 1 failed · 6 req · 21s⟧" in task
    assert "☑ Todo · 2 tasks" in todo
    assert "☑ done" in todo and "☐ next" in todo
