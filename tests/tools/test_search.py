from pathlib import Path
import os

import pytest

from orcha_agent.core.tools import search


def invoke(root: Path, name: str, **kwargs: object) -> str:
    tool = next(item for item in search.create_search_tools(root) if item.name == name)
    return str(tool.invoke(kwargs))


@pytest.mark.parametrize("native", [True, False])
def test_regex_case_context_and_multiline(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    (tmp_path / "a.py").write_text("before\nHELLO 42\nworld\nafter\n")
    result = invoke(tmp_path, "grep", pattern=r"hello \d+", case=False, context=1)
    assert "a.py:" in result and "2: HELLO 42" in result
    assert "1- before" in result and "3- world" in result
    result = invoke(tmp_path, "grep", pattern=r"42\nworld")
    assert "2: HELLO 42" in result and "3: world" in result
    result = invoke(tmp_path, "grep", pattern=r"(?<=HELLO )42")
    assert "2: HELLO 42" in result


def test_grep_pagination_and_per_file_cap(tmp_path):
    for index in range(23):
        (tmp_path / f"{index:02}.txt").write_text("match\n" * 24)
    page = invoke(tmp_path, "grep", pattern="match")
    assert "19.txt:" in page and "20.txt:" not in page
    assert "skip=20" in page and "4 omitted" in page
    assert "21: match" not in page
    page2 = invoke(tmp_path, "grep", pattern="match", skip=20)
    assert "20.txt:" in page2 and "22.txt:" in page2 and "19.txt:" not in page2
    single = invoke(tmp_path, "grep", pattern="match", path="00.txt")
    assert "24: match" in single and "truncated" not in single


def test_single_file_cap_and_internal_cap(tmp_path):
    (tmp_path / "a").write_text("hit\n" * 2100)
    result = invoke(tmp_path, "grep", pattern="hit", path="a")
    assert "200: hit" in result and "201: hit" not in result
    assert "Internal limit reached (2000" in result
    assert "1800 omitted" in result


def test_search_ignores_hidden_nested_gitignore_and_binary(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n!keep.log\n")
    for name in ["a.txt", "omit.log", "keep.log", ".hidden"]:
        (tmp_path / name).write_text("needle")
    (tmp_path / "binary").write_bytes(b"needle\0")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "a.txt").write_text("needle")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".gitignore").write_text("local.txt\n")
    (tmp_path / "sub" / "local.txt").write_text("needle")
    result = invoke(tmp_path, "grep", pattern="needle")
    assert "a.txt:" in result and "keep.log:" in result
    for omitted in ["omit.log:", ".hidden:", "binary:", "build/a.txt:", "local.txt:"]:
        assert omitted not in result
    result = invoke(tmp_path, "glob", pattern="**/*", include_hidden=True)
    assert ".hidden" in result and "omit.log" not in result
    assert "local.txt" not in result


def test_grep_scan_cap_and_glob_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "MAX_FILE_BYTES", 12)
    (tmp_path / "a.txt").write_text("needle\n" + "x" * 100 + "tail")
    (tmp_path / "b.py").write_text("needle")
    result = invoke(tmp_path, "grep", pattern="needle", glob="**/*.txt")
    assert "a.txt:" in result and "b.py:" not in result
    assert "scanned first 12 bytes" in result and "remaining bytes omitted" in result


def test_glob_newest_first_cap_and_grouping(tmp_path):
    (tmp_path / "src").mkdir()
    for index, name in enumerate(["src/old.py", "root.py", "src/new.py"]):
        file = tmp_path / name
        file.write_text("")
        os.utime(file, (index + 1, index + 1))
    result = invoke(tmp_path, "glob", pattern="**/*.py", limit=2)
    assert "src/\n  new.py" in result and "./\n  root.py" in result
    assert result.index("new.py") < result.index("root.py")
    assert "old.py" not in result
    assert "Showing 2 of 3" in result and "limit=3" in result


def test_glob_timeout_returns_partial(tmp_path, monkeypatch):
    file = tmp_path / "a.py"
    file.write_text("")
    clock = [0.0]
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])

    def files(*args):
        yield file
        clock[0] = 6.0

    monkeypatch.setattr(search, "_files", files)
    result = invoke(tmp_path, "glob", pattern="*.py")
    assert "a.py" in result and "timed out after 5 s" in result
    assert "unvisited paths omitted (count unknown)" in result


def test_validation_and_empty_results(tmp_path):
    assert "invalid regex" in invoke(tmp_path, "grep", pattern="[")
    assert "Error:" in invoke(tmp_path, "grep", pattern="x", limit=0)
    assert "Error:" in invoke(tmp_path, "glob", pattern="*", limit=0)
    assert "does not exist" in invoke(tmp_path, "grep", pattern="x", path="absent")
    assert "No matches" in invoke(tmp_path, "grep", pattern="x")
    assert "No matching files" in invoke(tmp_path, "glob", pattern="*")


def test_recursive_patterns_and_anchored_ignores(tmp_path):
    (tmp_path / "src" / "nested").mkdir(parents=True)
    for name in ["src/a.py", "src/nested/b.py", "top.py"]:
        (tmp_path / name).write_text("hit")
    (tmp_path / ".gitignore").write_text("/top.py\n")
    (tmp_path / "src" / "top.py").write_text("hit")
    result = invoke(tmp_path, "glob", pattern="src/**/*.py")
    assert "a.py" in result and "b.py" in result and "top.py" in result
    assert "./" not in result
    result = invoke(tmp_path, "glob", pattern="src/*.py")
    assert "a.py" in result and "b.py" not in result


@pytest.mark.parametrize("native", [True, False])
def test_empty_and_eof_regex_do_not_index_past_lines(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    (tmp_path / "empty").write_text("")
    (tmp_path / "line").write_text("one\n")
    result = invoke(tmp_path, "grep", pattern="$")
    assert "1: one" in result


def test_error_tool_message_status(tmp_path):
    for tool in search.create_search_tools(tmp_path):
        args = {"pattern": "["} if tool.name == "grep" else {"pattern": "*", "limit": 0}
        result = tool.invoke({"type": "tool_call", "name": tool.name, "id": "bad", "args": args})
        assert result.status == "error"


@pytest.mark.parametrize("native", [True, False])
def test_single_file_cursor_reaches_beyond_internal_cap(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    (tmp_path / "hits").write_text("".join(f"hit {i}\n" for i in range(2100)))
    result = invoke(tmp_path, "grep", pattern="hit", path="hits", skip=200, limit=50)
    assert "201: hit 200" in result and "250: hit 249" in result
    assert "251: hit 250" not in result and "skip=250" in result
    result = invoke(tmp_path, "grep", pattern="hit", path="hits", skip=2000)
    assert "2001: hit 2000" in result and "2100: hit 2099" in result


@pytest.mark.parametrize("native", [True, False])
def test_huge_multiline_match_obeys_internal_line_cap(monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    import re

    text = "start\n" + "middle\n" * 3000 + "end"
    pattern = r"start[\s\S]*end"
    result = search._match_lines(pattern, re.compile(pattern), text, True)
    assert len(result) == search.INTERNAL_MATCH_CAP
    assert result[-1] == search.INTERNAL_MATCH_CAP


@pytest.mark.parametrize("native", [True, False])
def test_gitignore_escaped_leaders_and_trailing_spaces(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    (tmp_path / ".gitignore").write_text(
        "\\!literal\n\\#literal\nspace\\ \ntrimmed   \nliteral\\*.txt\n#comment\n"
    )
    ignored = ["!literal", "#literal", "space ", "trimmed", "literal*.txt"]
    visible = ["space", "trimmed   ", "literalABC.txt", "kept"]
    for name in ignored + visible:
        (tmp_path / name).write_text("needle")
    result = invoke(tmp_path, "grep", pattern="needle")
    for name in ignored:
        assert f"{name}:" not in result
    for name in visible:
        assert f"{name}:" in result


@pytest.mark.parametrize("native", [True, False])
def test_grep_line_numbers_ignore_unicode_separators(tmp_path, monkeypatch, native):
    if not native:
        monkeypatch.setattr(search.shutil, "which", lambda _: None)
    (tmp_path / "text").write_text("first\u2028same\nneedle\n")
    result = invoke(tmp_path, "grep", pattern="needle")
    assert "2: needle" in result


def test_gitignore_escaped_space_followed_by_ignored_spaces(tmp_path):
    (tmp_path / ".gitignore").write_text("space\\   \n\\#keep\n!\\#keep\n")
    for name in ["space ", "space   ", "#keep"]:
        (tmp_path / name).write_text("needle")
    result = invoke(tmp_path, "grep", pattern="needle")
    assert "space :" not in result
    assert "space   :" in result and "#keep:" in result
