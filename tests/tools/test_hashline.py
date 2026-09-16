import re
from pathlib import Path

import pytest

from orcha_agent.core.tools.filesystem import FilesystemTools, create_filesystem_tools
from orcha_agent.core.tools.hashline import HashlineTools, file_hash


def setup(tmp_path: Path, text: str = "one\ntwo\nthree\n"):
    file = tmp_path / "a.txt"
    file.write_bytes(text.encode())
    service = HashlineTools(FilesystemTools(tmp_path))
    header = service.read("a.txt").splitlines()[0]
    return service, file, header


@pytest.mark.parametrize("text,tag", [("a \n b\t\r\nc", "80BA"), ("hello\n", "5BF9"), ("", "5D05")])
def test_reference_hash_vectors(text, tag):
    assert file_hash(text) == tag


def test_read_format_selectors_and_repeat_guard(tmp_path):
    service, _, header = setup(tmp_path)
    assert re.fullmatch(r"\[a.txt#[0-9A-F]{4}\]", header)
    assert "1:one" in service.read("a.txt")
    assert "Unchanged" in service.read("a.txt")
    assert "2:two\n3:three" in service.read("a.txt:2-3")


def test_put_cut_insert_transaction_and_eol_mode(tmp_path):
    service, file, header = setup(tmp_path, "one\r\ntwo\r\nthree\r\nfour\r\n")
    file.chmod(0o751)
    result = service.edit(header + "\nPUT 2:\n+TWO\nCUT 3\nPUT >$:\n+five\nPUT <1:\n+zero")
    assert not result.startswith("Error:")
    assert file.read_bytes() == b"zero\r\none\r\nTWO\r\nfour\r\nfive\r\n"
    assert file.stat().st_mode & 0o777 == 0o751
    assert "+TWO" in result and "-two" in result


def test_unknown_foreign_and_stale_tag_recovery(tmp_path):
    service, file, header = setup(tmp_path)
    result = service.edit(header + "\nPUT 2:\n+TWO", thread="another")
    assert "not from this session" in result
    file.write_text("one\nchanged\nthree\n")
    result = service.edit(header + "\nPUT 2:\n+TWO")
    assert "file changed between read and edit" in result and "2:changed" in result
    fresh = service.read("a.txt", turn="next").splitlines()[0]
    assert not service.edit(fresh + "\nPUT 2:\n+TWO").startswith("Error:")


def test_unique_safe_stale_anchor_relocation(tmp_path):
    service, file, header = setup(tmp_path)
    file.write_text("inserted\none\ntwo\nthree\n")
    result = service.edit(header + "\nPUT 2:\n+TWO")
    assert "Recovered stale snapshot" in result
    assert file.read_text() == "inserted\none\nTWO\nthree\n"


def test_relocation_rejects_ambiguous_lines_and_stale_remove(tmp_path):
    service, file, header = setup(tmp_path, "same\nsame\nlast\n")
    file.write_text("new\nsame\nsame\nlast\n")
    assert "Error:" in service.edit(header + "\nCUT 1")
    assert "Error:" in service.edit(header + "\nREM")
    assert file.exists()


def test_seen_lines_and_validation_prevent_partial_changes(tmp_path):
    service, file, _ = setup(tmp_path)
    fresh = HashlineTools(FilesystemTools(tmp_path))
    header = fresh.read("a.txt:1-1").splitlines()[0]
    assert "never displayed lines 2" in fresh.edit(header + "\nPUT 2:\n+TWO")
    fresh.read("a.txt:2-2")
    assert not fresh.edit(header + "\nPUT 2:\n+TWO").startswith("Error:")
    header = service.read("a.txt", turn="new").splitlines()[0]
    before = file.read_bytes()
    for patch in ["\nPUT 1:\n+ONE\nCUT 99", "\nPUT 1.=2:\n+x\nCUT 2", "\nPUT 1:"]:
        assert "Error:" in service.edit(header + patch)
        assert file.read_bytes() == before


def test_move_remove_and_multifile_prevalidation(tmp_path):
    service, file, header = setup(tmp_path)
    result = service.edit(header + "\nMV nested/moved.txt")
    moved = tmp_path / "nested/moved.txt"
    assert not file.exists() and moved.read_text() == "one\ntwo\nthree\n"
    assert "#" in result
    second = tmp_path / "second.txt"
    second.write_text("second\n")
    first_header = result.splitlines()[0]
    second_header = service.read("second.txt").splitlines()[0]
    invalid = first_header + "\nREM\n" + second_header + "\nCUT 100"
    assert "Error:" in service.edit(invalid)
    assert moved.exists()
    assert "Removed" in service.edit(first_header + "\nREM")
    assert not moved.exists()


def test_hash_collision_does_not_bypass_digest_guard(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.file_hash", lambda _: "AAAA")
    service, file, header = setup(tmp_path)
    file.write_text("changed\ntwo\nthree\n")
    assert "file changed" in service.edit(header + "\nPUT 1:\n+wrong")
    assert file.read_text().startswith("changed")


def test_factory_keeps_default_replace_and_optional_hashline(tmp_path):
    default = next(tool for tool in create_filesystem_tools(tmp_path) if tool.name == "edit")
    alternate = next(
        tool for tool in create_filesystem_tools(tmp_path, "hashline") if tool.name == "edit"
    )
    assert "old_string" in default.args and "patch" not in default.args
    assert "patch" in alternate.args and "old_string" not in alternate.args


def test_edit_keeps_unread_lines_guarded(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("one\ntwo\nthree\n")
    service = HashlineTools(FilesystemTools(tmp_path))
    header = service.read("a.txt:1-1").splitlines()[0]
    result = service.edit(header + "\nPUT 1:\n+ONE")
    new_header = result.splitlines()[0]
    assert "never displayed lines 3" in service.edit(new_header + "\nCUT 3")


def test_multifile_io_failure_rolls_back_prior_write(tmp_path, monkeypatch):
    from orcha_agent.core.tools import hashline

    service, file, header = setup(tmp_path)
    second = tmp_path / "b.txt"
    second.write_text("second\n")
    other_header = service.read("b.txt").splitlines()[0]
    original_write = hashline.atomic_write
    failed = [False]

    def fail_once(path, content, **kwargs):
        if path == second and not failed[0]:
            failed[0] = True
            raise OSError("simulated write failure")
        original_write(path, content, **kwargs)

    monkeypatch.setattr(hashline, "atomic_write", fail_once)
    result = service.edit(header + "\nPUT 1:\n+ONE\n" + other_header + "\nPUT 1:\n+SECOND")
    assert "simulated write failure" in result
    assert file.read_text() == "one\ntwo\nthree\n"
    assert second.read_text() == "second\n"


def test_move_preserves_mixed_eols_and_unicode_separators_exactly(tmp_path):
    text = "first\r\nsecond\nthird\u2028part\vlast\r"
    service, file, header = setup(tmp_path, text)
    assert not service.edit(header + "\nMV moved.txt").startswith("Error:")
    assert not file.exists()
    assert (tmp_path / "moved.txt").read_bytes() == text.encode()


def test_mixed_eol_edits_preserve_untouched_bytes(tmp_path):
    service, file, header = setup(tmp_path, "one\r\ntwo\nthree\r\nfour")
    result = service.edit(header + "\nPUT 2:\n+TWO\nPUT 4:\n+FOUR")
    assert not result.startswith("Error:"), result
    assert file.read_bytes() == b"one\r\nTWO\nthree\r\nFOUR"


def test_unicode_separators_are_payload_not_line_anchors(tmp_path):
    service, file, header = setup(tmp_path, "one\u2028part\vother\ntwo\n")
    assert "1:one\u2028part\vother\n2:two" in service.read("a.txt:1-2", turn="next")
    result = service.edit(header + "\nPUT 2:\n+new\u2028part\vend")
    assert not result.startswith("Error:"), result
    assert file.read_bytes() == "one\u2028part\vother\nnew\u2028part\vend\n".encode()


@pytest.mark.parametrize(
    "source,expected",
    [("one", "one\nlast"), ("one\n", "one\nlast\n"), ("one\r\n", "one\r\nlast\r\n"), ("", "last")],
)
def test_append_preserves_terminal_newline_policy(tmp_path, source, expected):
    service, file, header = setup(tmp_path, source)
    assert not service.edit(header + "\nPUT >$:\n+last").startswith("Error:")
    assert file.read_bytes() == expected.encode()


def test_snapshot_eviction_requires_reread_and_recovers_same_turn(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.MAX_SNAPSHOTS", 1)
    service, file, header = setup(tmp_path)
    (tmp_path / "other").write_text("other\n")
    service.read("other")
    assert len(service.snapshots) == 1
    result = service.edit(header + "\nPUT 1:\n+ONE")
    assert "expired from the snapshot cache" in result and "Re-read" in result
    result = service.read("a.txt")
    assert "1:one" in result and "Unchanged" not in result
    assert not service.edit(header + "\nPUT 1:\n+ONE").startswith("Error:")
    assert file.read_text().startswith("ONE")


def test_snapshot_byte_budget_releases_older_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.SNAPSHOT_BYTE_BUDGET", 15)
    service, _, _ = setup(tmp_path)
    (tmp_path / "other").write_text("some content\n")
    service.read("other")
    assert len(service.snapshots) == 1


def test_colliding_tags_keep_versions_and_reject_ambiguity(tmp_path):
    service, file, header = setup(tmp_path)
    file.write_text("one \ntwo\nthree\n")
    assert service.read("a.txt").splitlines()[0] == header
    assert len(service.snapshots) == 2
    assert all(len(key) == 4 for key in service.snapshots)
    before = file.read_bytes()
    assert "Ambiguous snapshot tag collision" in service.edit(header + "\nPUT 2:\n+TWO")
    assert file.read_bytes() == before


def test_hashline_policy_and_read_cap(tmp_path):
    service, file, header = setup(tmp_path)
    assert service.edit(header + "\nMV ../outside").startswith("Error:")
    assert file.exists()
    denied = tmp_path / ".env.fixture"
    denied.touch()
    assert service.read(str(denied)).startswith("Error:")
    tiny = HashlineTools(FilesystemTools(tmp_path, max_read_bytes=2))
    assert "max_read_bytes" in tiny.read("a.txt")
    service.service.max_read_bytes = 2
    assert "max_read_bytes" in service.edit(header + "\nPUT 1:\n+ONE")


def test_collision_remains_rejected_after_snapshot_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.MAX_SNAPSHOTS", 1)
    service, file, header = setup(tmp_path)
    file.write_text("one \ntwo\nthree\n")
    assert service.read("a.txt").splitlines()[0] == header
    assert len(service.snapshots) == 1
    assert "Ambiguous snapshot tag collision" in service.edit(header + "\nPUT 2:\n+TWO")


def test_eviction_before_collision_cannot_reassign_old_tag(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.MAX_SNAPSHOTS", 1)
    service, file, header = setup(tmp_path)
    (tmp_path / "other").write_text("other\n")
    service.read("other")
    assert not any(key[1] == file for key in service.snapshots)
    file.write_text("one \ntwo \nthree\n")
    assert service.read("a.txt").splitlines()[0] == header
    before = file.read_bytes()
    assert "Ambiguous snapshot tag collision" in service.edit(header + "\nPUT 2:\n+TWO")
    assert file.read_bytes() == before


def test_evicted_natural_hash_collision_rejects_old_header(tmp_path, monkeypatch):
    monkeypatch.setattr("orcha_agent.core.tools.hashline.MAX_SNAPSHOTS", 1)
    old = "item 232\noriginal 232\n"
    new = "item 338\noriginal 338\n"
    assert file_hash(old) == file_hash(new) == "122D"
    service, file, header = setup(tmp_path, old)
    (tmp_path / "other").write_text("other\n")
    service.read("other")
    file.write_text(new)
    assert service.read("a.txt").splitlines()[0] == header
    assert "Ambiguous snapshot tag collision" in service.edit(header + "\nPUT 2:\n+wrong version")
    assert file.read_text() == new
