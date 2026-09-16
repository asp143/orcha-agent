from pathlib import Path

import pytest

from orcha_agent.extensibility.commands import discover_commands


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize(
    ("owner", "root_name", "link_name"),
    [
        ("project", ".claude/commands", ".claude"),
        ("project", ".orcha-agent/commands", ".orcha-agent"),
        ("project", ".orcha-agent/commands", ".orcha-agent/commands"),
        ("home", ".claude/commands", ".claude"),
        ("home", ".config/orcha-agent/commands", ".config/orcha-agent"),
        ("home", ".config/orcha-agent/commands", ".config"),
    ],
)
def test_escaped_command_root_is_rejected_before_glob(
    tmp_path, monkeypatch, owner, root_name, link_name, trusted
):
    cwd, home, outside = tmp_path / "repo", tmp_path / "home", tmp_path / "outside"
    cwd.mkdir()
    home.mkdir()
    scope = cwd if owner == "project" else home
    destination = outside / Path(root_name).relative_to(link_name)
    destination.mkdir(parents=True)
    (destination / "leak.md").write_text("Outside instructions")
    link = scope / link_name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    escaped_root = scope / root_name
    original_glob = Path.glob

    def guarded_glob(path, pattern, *args, **kwargs):
        assert path != escaped_root, "Escaped command root must not be enumerated"
        return original_glob(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    assert discover_commands(cwd, home=home, trust_cwd=trusted) == {}


def test_command_root_symlink_inside_project_scope_still_works(tmp_path):
    cwd = tmp_path / "repo"
    commands = cwd / "shared/commands"
    commands.mkdir(parents=True)
    (commands / "review.md").write_text("Review local changes")
    (cwd / ".claude").symlink_to(commands.parent, target_is_directory=True)
    found = discover_commands(cwd, home=tmp_path / "home")
    assert found["review"].body == "Review local changes"
    assert not found["review"].trusted
