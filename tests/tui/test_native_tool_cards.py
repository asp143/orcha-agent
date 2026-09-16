from __future__ import annotations

from io import StringIO
from pathlib import Path

from langchain_core.messages import ToolMessage
from rich.console import Console

from orcha_agent.tui.blocks import DEFAULT_THEME
from orcha_agent.tui.blocks.tool import render
from orcha_agent.tui.frame import Block, BlockState


def card(name: str, args: dict, content: str, *, artifact=None, status="success") -> str:
    block = Block(
        id=name,
        kind="tool",
        state=BlockState.SETTLED,
        data={
            "name": name,
            "args": args,
            "duration": 0.012,
            "result": ToolMessage(
                content=content, name=name, tool_call_id=name, artifact=artifact, status=status
            ),
        },
    )
    output = StringIO()
    Console(file=output, width=80, color_system=None).print(
        render(block, DEFAULT_THEME, 80, 80, False)
    )
    return output.getvalue()


def test_native_cards_golden(update_goldens: bool) -> None:
    cards = [
        card(
            "read",
            {"path": "src/app.py:40+3"},
            '    40  first\n    41  second\n    42  third\n[notice] {"message": "Use src/app.py:43 to continue."}',
        ),
        card(
            "read", {"path": "src/app.py:40+3"}, "[src/app.py#AB12]\n40:first\n41:second\n42:third"
        ),
        card(
            "edit",
            {"path": "src/app.py", "old_string": "first", "new_string": "changed"},
            "--- src/app.py\n+++ src/app.py\n@@ -40 +40 @@\n-first\n+changed",
        ),
        card(
            "edit",
            {"patch": "[src/app.py#AB12]\nPUT 40:\n+changed"},
            "--- src/app.py\n+++ src/app.py\n@@ -40 +40 @@\n-first\n+changed",
        ),
        card(
            "write", {"path": "new.txt", "content": "hello\nworld\n"}, "Wrote 12 bytes (2 lines)."
        ),
        card(
            "bash",
            {"command": "printf 'hello'", "timeout": 300},
            "hello\nExit code: 0",
            artifact={"raw_output": "\x1b[32mhello\x1b[0m", "exit_code": 0},
        ),
        card(
            "grep",
            {"pattern": "hello", "context": 1},
            "src/app.py:\n  2- before\n  3: hello\n  4- after",
        ),
        card("glob", {"pattern": "**/*.py"}, "src/\n  app.py\n  other.py"),
        card("ls", {"path": "src"}, "app.py  20 bytes  2026-09-16 10:00:00"),
        card(
            "edit",
            {"path": "src/app.py"},
            "Error: Read the current file before editing.",
            status="error",
        ),
    ]
    output = "\n".join(cards)
    assert "40│first" in cards[0] and "src/app.py:40-42" in cards[0]
    assert ":40+3:40-42" not in cards[0]
    assert "40│first" in cards[1] and "[src/app.py#AB12]" in cards[1]
    assert "src/app.py:40" in cards[3]
    assert "1 matches · 1 files" in cards[6]
    assert "src/app.py:3:hello" in cards[6]
    assert "Read the current file" in cards[-1]
    golden = Path(__file__).with_name("golden") / "native-tools-80.txt"
    if update_goldens:
        golden.write_text(output)
    assert output == golden.read_text()


def test_native_shell_truncation_notice_survives_raw_artifact() -> None:
    output = card(
        "bash",
        {"command": "produce-output"},
        "hello\n[Output limited: 200 of 300 lines; read output:201+100.]\nExit code: 0",
        artifact={"raw_output": "hello\n", "exit_code": 0},
    )
    assert "Output limited" in output and "read output:201+100" in output


def test_native_shell_replay_preserves_notices_without_literal_escape_sequences() -> None:
    output = card(
        "bash",
        {"command": "progress"},
        "done\n[Output limited: read output:201+100.]\nCommand timed out",
        artifact={"raw_output": "working\r\x1b[2K\x1b[32mdone\x1b[0m", "exit_code": 1},
    )
    assert "done" in output and "working" not in output
    assert "Output limited" in output and "Command timed out" in output
    assert "\x1b" not in output


def test_grouped_native_reads_keep_one_selector_and_syntax_preview() -> None:
    from orcha_agent.tui.blocks.tool import _read_rows

    block = Block(
        id="grouped-native",
        kind="tool",
        data={
            "name": "read",
            "calls": [
                {
                    "args": {"path": "app.py:40+2"},
                    "result": "    40  def main():\n    41      return 1",
                },
                {
                    "args": {"path": "other.py:5-6"},
                    "result": "     5  def other():\n     6      return 2",
                },
            ],
        },
    )
    title, rows = _read_rows(block, {}, expanded=False, theme=DEFAULT_THEME)
    assert title == "• Read (2)"
    assert rows[0] == "├─ app.py:40-41"
    assert rows[3] == "└─ other.py:5-6"
    assert getattr(rows[1], "spans", ())


def test_native_read_home_path_and_collapse() -> None:
    output = card(
        "read",
        {"path": str(Path.home() / "sample.py:1+30")},
        "\n".join(f"{number:6}  content" for number in range(1, 31)),
    )
    assert "~/sample.py:1-30" in output
    assert "18 more lines" in output


def test_native_read_gutter_preserves_lf_addressing() -> None:
    from orcha_agent.tui.blocks.tool import _read_source_rows

    rows, first, last = _read_source_rows("     7  first\u2028still first\n     8  second", {})
    assert rows == [("7", "first\u2028still first"), ("8", "second")]
    assert (first, last) == (7, 8)


def test_native_diff_capture_accepts_host_absolute_paths(tmp_path: Path) -> None:
    from orcha_agent.core.events import ToolCallStart
    from orcha_agent.tui.turn import _FileDiffCapture

    path = tmp_path / "example.txt"
    path.write_text("old\n")
    capture = _FileDiffCapture(tmp_path)
    capture.start(ToolCallStart(name="edit", args={"path": str(path)}, id="native-edit"))
    path.write_text("new\n")
    result = capture.finish(ToolMessage(content="changed", tool_call_id="native-edit"))
    assert result.artifact["data"]["before"] == "old\n"
    assert result.artifact["data"]["after"] == "new\n"
