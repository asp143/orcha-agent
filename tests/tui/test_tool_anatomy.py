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

    assert "⏳ Read: src/app.py:5-12" in pending
    assert "• Read src/app.py:5-19" in result
    assert " 5│line 1" in result
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
        assert "…" in lines[1]
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


def test_read_uses_deepagents_line_numbers_for_gutter_range_and_more_hint() -> None:
    numbered = "\n".join(f"{line:>3}  source {line}" for line in range(41, 56))
    collapsed = _plain(
        _block(
            "read_file",
            args={"path": "src/numbered.py", "offset": 40},
            result=numbered,
        )
    )
    expanded = _plain(
        _block(
            "read_file",
            args={"path": "src/numbered.py", "offset": 40},
            result=numbered,
        ),
        expanded=True,
    )

    assert "• Read src/numbered.py:41-55" in collapsed
    assert " 41│source 41" in collapsed
    assert " 52│source 52" in collapsed
    assert "source 53" not in collapsed
    assert "… 3 more lines ⟦Ctrl+O: Expand⟧" in collapsed
    assert " 55│source 55" in expanded
    assert "more lines" not in expanded


def test_ls_renders_an_inline_tree_with_directory_suffix_and_row_limits() -> None:
    entries = [
        {"path": f"entry-{index}", "type": "directory" if index in {0, 9} else "file"}
        for index in range(30)
    ]
    value = _block("ls", args={"path": "src"}, result={"entries": entries, "count": 30})

    collapsed = _plain(value)
    expanded = _plain(value, expanded=True)

    assert collapsed.startswith("\n📂 Ls: src  30 items")
    assert "├─ entry-0/" in collapsed
    assert "entry-7" in collapsed and "entry-8" not in collapsed
    assert "… 22 more items ⟦Ctrl+O: Expand⟧" in collapsed
    assert "entry-23" in expanded and "entry-24" not in expanded
    assert "… 6 more items ⟦Ctrl+O: Expand⟧" in expanded
    assert "╭" not in collapsed


def test_glob_and_grep_use_authoritative_counts_and_specific_empty_summaries() -> None:
    glob = _plain(
        _block(
            "glob",
            args={"pattern": "**/*.py"},
            result={"matches": [f"src/{index}.py" for index in range(9)], "count": 12},
        )
    )
    grep = _plain(
        _block(
            "grep",
            args={"pattern": "needle", "path": "src"},
            result={
                "matches": [
                    {"path": "a.py", "line": 2, "text": "needle one"},
                    {"path": "a.py", "line": 8, "text": "needle two"},
                    {"path": "b.py", "line": 3, "text": "needle three"},
                ],
                "match_count": 17,
                "file_count": 5,
            },
        )
    )
    formatted_grep = _plain(
        _block(
            "grep",
            args={"pattern": "needle"},
            result="a.py:\n  2: needle one\n  8: needle two\nb.py:\n  3: needle three",
        )
    )

    assert "Glob: **/*.py  12 items" in glob
    assert "… 4 more items ⟦Ctrl+O: Expand⟧" in glob
    assert "Grep: needle  17 matches · 5 files · in src" in grep
    assert "… 14 more matches ⟦Ctrl+O: Expand⟧" in grep
    assert "Grep: needle  3 matches · 2 files" in formatted_grep
    assert _plain(_block("glob", args={"pattern": "*.none"}, result="")).strip() == "⚠ No files found"
    assert _plain(_block("grep", args={"pattern": "none"}, result="")).strip() == "⚠ No matches found"


def test_grep_treats_only_the_deepagents_no_match_sentinel_as_empty() -> None:
    exact = _plain(_block("grep", args={"pattern": "needle"}, result="No matches found"))
    padded = _plain(_block("grep", args={"pattern": "needle"}, result=" \n No matches found \n\t"))
    real_match = _plain(
        _block(
            "grep",
            args={"pattern": "needle"},
            result="No matches found.txt:\n  7: No matches found in this line",
        )
    )

    assert exact.strip() == "⚠ No matches found"
    assert padded.strip() == "⚠ No matches found"
    assert "Grep: needle  1 matches · 1 files" in real_match
    assert "No matches found.txt:7:No matches found in this line" in real_match


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

    assert grep.lstrip().startswith("🔍 Grep: needle  3 matches · 2 files · in src")
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
    assert len(folded.strip().splitlines()) == 2
    assert folded.lstrip().startswith("╭─ Custom · Elapsed 4s")
    assert folded.splitlines()[-1] == "╰"
    assert single.strip() == "⣾ Custom · Elapsed 4s"
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
