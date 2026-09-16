from pathlib import Path
import stat

import pytest

from orcha_agent.core.tools.filesystem import FilesystemTools, create_filesystem_tools
from orcha_agent.core.tools.fuzzy import replace


@pytest.fixture
def fs(tmp_path: Path) -> FilesystemTools:
    (tmp_path / "sample.txt").write_text("".join(f"line {i}\n" for i in range(1, 1001)))
    return FilesystemTools(tmp_path)


@pytest.mark.parametrize(
    ("selector", "first", "last"),
    [("5", 5, 204), ("5-8", 5, 8), ("5+3", 5, 7), ("-3", 998, 1000), ("5-6,960-961", 5, 961)],
)
def test_selectors(fs: FilesystemTools, selector: str, first: int, last: int):
    result = fs.read("sample.txt:" + selector)
    assert f"{first:6}  line {first}" in result
    assert f"{last:6}  line {last}" in result
    assert "total_lines" in result


def test_raw_and_default(fs: FilesystemTools):
    assert fs.read("sample.txt:raw:1-2").startswith("line 1\nline 2\n")
    result = fs.read("sample.txt")
    assert "Showing lines 1-200 of 1000" in result
    assert "sample.txt:201" in result


def test_repeat_and_changes(fs: FilesystemTools):
    fs.read("sample.txt", thread="one", turn="a")
    assert "Unchanged" in fs.read("sample.txt", thread="one", turn="a")
    assert "will not change" in fs.read("sample.txt", thread="one", turn="a")
    assert "Unchanged" not in fs.read("sample.txt", thread="one", turn="b")
    assert "Unchanged" not in fs.read("sample.txt", thread="two", turn="a")
    (fs.cwd / "sample.txt").write_text("new")
    assert "new" in fs.read("sample.txt", thread="one", turn="a")


def test_edit_guard_thread_and_external_change(fs: FilesystemTools):
    assert fs.edit("sample.txt", "line 1\n", "new\n").startswith("Error:")
    fs.read("sample.txt", thread="one")
    assert fs.edit("sample.txt", "line 1\n", "new\n", thread="two").startswith("Error:")
    (fs.cwd / "sample.txt").write_text("changed")
    assert fs.edit("sample.txt", "changed", "new", thread="one").startswith("Error:")


@pytest.mark.parametrize(
    ("actual", "old"),
    [
        ("“hi”", '"hi"'),
        ("‘hi’", "'hi'"),
        ("a—b", "a-b"),
        ("a–b", "a-b"),
        ("a\u00a0b", "a b"),
        ("a    b", "a b"),
        ("    first\n        second", "first\n  second"),
    ],
)
def test_fuzzy_normalization(actual: str, old: str):
    assert replace(actual, old, "replacement") == "replacement"


def test_occurrences():
    with pytest.raises(ValueError, match="Found 2 occurrences") as error:
        replace("one\none\n", "one", "two")
    assert "line 1" in str(error.value) and "line 2" in str(error.value)
    with pytest.raises(ValueError, match="Found 0 occurrences"):
        replace("one", "something unrelated", "two")
    assert replace("one\none", "one", "two", True) == "two\ntwo"


def test_transaction_and_eol_mode(fs: FilesystemTools):
    path = fs.cwd / "code.py"
    path.write_bytes(b"a = 1\r\nb = 2\r\n")
    path.chmod(0o751)
    fs.read("code.py")
    result = fs.edit(
        "code.py", edits=[{"old": "a = 1", "new": "a = 3"}, {"old": "missing", "new": "no"}]
    )
    assert result.startswith("Error:")
    assert path.read_bytes() == b"a = 1\r\nb = 2\r\n"
    result = fs.edit("code.py", edits=[{"old": "a = 1\nb = 2", "new": "a = 3\nb = 4"}])
    assert "-a = 1" in result and "+a = 3" in result
    assert path.read_bytes() == b"a = 3\r\nb = 4\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o751


def test_write_ls_images_and_errors(fs: FilesystemTools):
    assert "5 bytes" in fs.write("sub/new.txt", "hello")
    assert "+hello" in fs.write("sub/new.txt", "hello again")
    assert "new.txt\t11 bytes" in fs.read("sub")
    assert "omitted_entries" in fs.ls(".", limit=1)
    (fs.cwd / "image.png").write_bytes(b"\xff")
    assert "Image file:" in fs.read("image.png")
    assert fs.read("missing").startswith("Error:")
    assert fs.read("sample.txt:5-2").startswith("Error:")
    assert "outside" in fs.read("sample.txt:2000")


def test_elision(fs: FilesystemTools):
    (fs.cwd / "long").write_text("\n".join(str(i) + "x" * 250 for i in range(200)))
    result = fs.read("long")
    assert "middle lines" in result and "omitted_lines" in result
    (fs.cwd / "huge").write_text("x" * 30_000)
    assert "omitted_bytes" in fs.read("huge")


def test_summary(fs: FilesystemTools):
    (fs.cwd / "large.py").write_text(
        "def first():\n" + "    pass\n" * 500 + "def second():\n    pass\n"
    )
    result = fs.read("large.py")
    assert "Declaration outline" in result and "def second" in result
    assert "Declaration outline" not in fs.read("large.py:1-2")


def test_factory_schema_and_error_status(tmp_path: Path):
    tools = {tool.name: tool for tool in create_filesystem_tools(tmp_path)}
    assert set(tools) == {"read", "edit", "write", "ls"}
    assert "runtime" not in tools["read"].tool_call_schema.model_json_schema()["properties"]
    result = tools["ls"].invoke(
        {"type": "tool_call", "id": "test", "name": "ls", "args": {"path": "missing"}}
    )
    assert result.status == "error"


def test_mixed_eol_preserves_untouched_bytes(fs: FilesystemTools):
    path = fs.cwd / "mixed.txt"
    path.write_bytes(b"first\r\nsecond\nthird\r\n")
    fs.read("mixed.txt")
    assert not fs.edit("mixed.txt", "second", "changed").startswith("Error:")
    assert path.read_bytes() == b"first\r\nchanged\nthird\r\n"


def test_fuzzy_ambiguous_and_near_matches():
    with pytest.raises(ValueError, match="Found 2 occurrences"):
        replace("a   b\na\u00a0b", "a b", "new")
    assert (
        replace("a long enough sentence with a typo.", "a long enough sentence with a typx.", "new")
        == "new"
    )
    with pytest.raises(ValueError, match="Found 0 occurrences"):
        replace("short", "shirt", "new")


def test_normalized_substring_preserves_surrounding_text():
    assert (
        replace('prefix curly "quotes" suffix', "curly “quotes”", "fixed") == "prefix fixed suffix"
    )
    assert replace("prefix a\u00a0   b suffix", "a b", "fixed") == "prefix fixed suffix"
    with pytest.raises(ValueError, match="Found 2 occurrences"):
        replace("a “quote” and a “quote”", 'a "quote"', "fixed")


def test_no_match_shows_closest_candidate():
    with pytest.raises(ValueError, match="Candidate lines") as error:
        replace(
            "unrelated\nconst value = computeSomething();\nother",
            "const value = computeDifferent();",
            "fixed",
        )
    assert "line 2:" in str(error.value)


def test_overlapping_fuzzy_replace_all_rejected():
    with pytest.raises(ValueError, match="overlap"):
        replace("“a”\n“a”\n“a”", '"a"\n"a"', "fixed", True)


def test_close_high_confidence_candidates_remain_ambiguous():
    old = "const sufficientlyLongVariableName = getValue();"
    actual = "const sufficientlyLongVariableName = getValue( );\nconst sufficientlyLongVariableName = getValue(1);"
    with pytest.raises(ValueError, match="Found 2 occurrences"):
        replace(actual, old, "fixed")


@pytest.mark.parametrize("character", list("‐‑‒–—―−"))
def test_all_typographic_dashes(character: str):
    assert replace(f"prefix a{character}b suffix", "a-b", "new") == "prefix new suffix"


@pytest.mark.parametrize("character", list("“”„‟«»"))
def test_all_double_quote_forms(character: str):
    assert (
        replace(f"prefix {character}word{character} suffix", '"word"', "new") == "prefix new suffix"
    )


def test_dominant_high_confidence_candidate():
    target = "a" * 50
    content = "a" * 49 + "b\n" + "a" * 44 + "cccccc"
    assert replace(content, target, "selected") == "selected\n" + "a" * 44 + "cccccc"


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\v", "\f", "\r"])
def test_read_addresses_lf_only(fs: FilesystemTools, separator: str):
    (fs.cwd / "odd").write_bytes(f"first{separator}still first\nsecond\r\n".encode())
    assert fs.read("odd:1-1").startswith(f"     1  first{separator}still first")
    assert fs.read("odd:2-2").startswith("     2  second\n")
    assert "outside" in fs.read("odd:3-3")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", []),
        ("\n", [""]),
        ("a\n", ["a"]),
        ("a\r\n", ["a"]),
        ("a\r", ["a\r"]),
        ("a\n\n", ["a", ""]),
    ],
)
def test_lf_line_boundaries(content: str, expected: list[str]):
    from orcha_agent.core.tools.common import split_text_lines

    assert split_text_lines(content) == expected


@pytest.mark.parametrize("raw", [False, True])
def test_clipped_read_recovers_original_multibyte_bytes(fs: FilesystemTools, raw: bool):
    import json
    import subprocess

    path = fs.cwd / "giant 'quoted' $.txt"
    data = ("é🙂" * 6000 + "\r\n").encode()
    path.write_bytes(data)
    result = fs.read(path.name + (":raw" if raw else ""))
    metadata = [
        json.loads(row.removeprefix("[notice] "))
        for row in result.split("\n")
        if row.startswith("[notice] ")
    ]
    recovery = next(item for item in metadata if "recovery_command" in item)
    ranges = recovery["byte_ranges"]
    displayed = result.split("\n", 1)[0]
    shown_source = displayed if raw else displayed[8:]
    assert ranges == [[len(shown_source.encode()), len(data)]]
    omitted = data[len(shown_source.encode()) :]
    assert recovery["omitted_bytes"] == len(omitted)
    completed = subprocess.run(
        recovery["recovery_command"], shell=True, capture_output=True, check=True
    )
    assert completed.stdout == omitted


def test_clipping_retains_middle_elision_notice(fs: FilesystemTools):
    import json
    import subprocess

    data = ("é" * 12000 + "\n" + "\n".join(str(i) for i in range(1, 200)) + "\n").encode()
    (fs.cwd / "long").write_bytes(data)
    result = fs.read("long")
    metadata = [
        json.loads(row.removeprefix("[notice] "))
        for row in result.split("\n")
        if row.startswith("[notice] ")
    ]
    assert any(item.get("omitted_lines") == 115 for item in metadata)
    recovery = next(item for item in metadata if "recovery_command" in item)
    omitted = b"".join(data[start:end] for start, end in recovery["byte_ranges"])
    assert recovery["omitted_bytes"] == len(omitted)
    completed = subprocess.run(
        recovery["recovery_command"], shell=True, capture_output=True, check=True
    )
    assert completed.stdout == omitted
