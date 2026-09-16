from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from orcha_agent.core.config import load_config
from orcha_agent.core.tools.backend import GuardedLocalShellBackend
from orcha_agent.core.tools.common import PathPolicy, atomic_write, ensure_artifacts


def test_containment_and_secret_aliases_are_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ordinary.txt").write_text("inert fixture")
    (workspace / "external").symlink_to(outside, target_is_directory=True)
    (workspace / "alias").symlink_to(workspace / "credentials.json")
    policy = PathPolicy(workspace)
    for path in (
        "../outside/ordinary.txt",
        outside / "ordinary.txt",
        "external/ordinary.txt",
        "alias",
    ):
        with pytest.raises(PermissionError):
            policy.open_read(path)
    for path in (
        ".env",
        ".env.test",
        "Credentials/a",
        "tls.pem",
        "key.key",
        "a.p12",
        "b.pfx",
        ".ssh/id_rsa",
        "secrets.yaml",
        "credentials.json",
        "serviceAccountKey.json",
    ):
        assert not policy.permits(path), path
    assert policy.permits("id_example.py")
    assert PathPolicy(workspace, deny=()).permits(".env")  # No content is opened.


def test_explicit_roots_and_both_spellings_use_the_same_policy(tmp_path: Path) -> None:
    root, allowed = tmp_path / "workspace", tmp_path / "allowed"
    root.mkdir()
    allowed.mkdir()
    (allowed / "allowed.txt").write_text("ok")
    policy = PathPolicy(root, allowed_roots=[allowed])
    assert policy.read_bytes(allowed / "allowed.txt", 100) == b"ok"
    (root / "linked").symlink_to(allowed, target_is_directory=True)
    assert policy.read_bytes("linked/allowed.txt", 100) == b"ok"
    (root / ".env.alias").symlink_to(allowed / "allowed.txt")
    with pytest.raises(PermissionError):
        policy.open_read(".env.alias")
    atomic_write(allowed / "created.txt", "new", policy=policy)
    assert policy.read_bytes(allowed / "created.txt", 100) == b"new"


@pytest.mark.parametrize("ancestor", ["secrets.d", "Credentials"])
def test_deny_patterns_ignore_ancestors_of_allowed_roots(tmp_path: Path, ancestor: str) -> None:
    # Policy-only assertions: no contents beneath these fixture paths are read.
    root = tmp_path / ancestor / "workspace"
    allowed = tmp_path / ancestor / "shared"
    policy = PathPolicy(root, allowed_roots=[allowed])
    assert policy.resolve("ordinary.txt") == root / "ordinary.txt"
    assert policy.resolve(allowed / "ordinary.txt") == allowed / "ordinary.txt"
    assert not policy.permits("secrets.json")
    assert not policy.permits("Credentials/ordinary.txt")
    assert not policy.permits(allowed / "credentials.json")
    assert not policy.permits(tmp_path / "outside.txt")


@pytest.mark.parametrize("canonical_cwd", [False, True])
def test_symlinked_workspace_spellings_share_policy(tmp_path: Path, canonical_cwd: bool) -> None:
    actual = tmp_path / "real" / "workspace"
    actual.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(actual.parent, target_is_directory=True)
    spelled_root = alias / "workspace"
    policy = PathPolicy(actual if canonical_cwd else spelled_root)
    (actual / "ordinary.txt").write_text("inside")
    assert policy.read_bytes("ordinary.txt", 100) == b"inside"
    assert policy.read_bytes(spelled_root / "ordinary.txt", 100) == b"inside"
    atomic_write(spelled_root / "created.txt", "new", policy=policy)
    assert (actual / "created.txt").read_text() == "new"
    (actual / ".env.alias").symlink_to(actual / "ordinary.txt")
    assert not policy.permits(spelled_root / ".env.alias")
    outside = tmp_path / "outside"
    outside.mkdir()
    (actual / "escape").symlink_to(outside, target_is_directory=True)
    assert not policy.permits(spelled_root / "escape" / "ordinary.txt")
    assert not policy.permits(spelled_root / ".." / "outside.txt")


@pytest.mark.parametrize("operation", ["read", "write"])
def test_ancestor_symlink_swap_cannot_redirect_io(
    tmp_path: Path, monkeypatch, operation: str
) -> None:
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    (workspace / "sub").mkdir(parents=True)
    outside.mkdir()
    (workspace / "sub/file").write_text("inside")
    (outside / "file").write_text("outside")
    policy = PathPolicy(workspace)
    original_open = os.open
    swapped = False

    def swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "sub" and not swapped:
            swapped = True
            (workspace / "sub").rename(workspace / "parked")
            (workspace / "sub").symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(OSError):
        if operation == "read":
            policy.read_bytes("sub/file", 100)
        else:
            atomic_write(workspace / "sub/file", "changed", policy=policy)
    assert swapped
    assert (outside / "file").read_text() == "outside"


def test_backend_delete_confines_targets_and_protects_descendants(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    policy = PathPolicy(root)
    backend = GuardedLocalShellBackend(policy)
    assert backend.virtual_mode is True
    (tmp_path / "outside").write_text("outside")
    assert backend.delete(str(tmp_path / "outside")).error
    assert backend.delete(str(root)).error
    protected = root / "protected"
    protected.mkdir()
    (protected / "credentials.json").touch()
    (protected / "ordinary").write_text("keep")
    assert backend.delete(str(protected)).error
    assert (protected / "ordinary").exists()
    safe = root / "safe"
    safe.mkdir()
    (safe / "nested").write_text("remove")
    assert backend.delete(str(safe)).error is None
    assert not safe.exists()
    (root / "linked").symlink_to(tmp_path / "outside")
    assert backend.delete(str(root / "linked")).error
    assert (tmp_path / "outside").exists()


def test_backend_offload_is_private_and_artifact_symlinks_are_rejected(tmp_path: Path) -> None:
    backend = GuardedLocalShellBackend(PathPolicy(tmp_path))
    target = tmp_path / ".orcha/artifacts/large_tool_results/result"
    assert backend.write(str(target), "private output").error is None
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    restored = backend.read(str(target), 0, 1)
    assert restored.error is None
    assert restored.file_data is not None
    assert restored.file_data["content"] == "private output"
    assert stat.S_IMODE((tmp_path / ".orcha").stat().st_mode) == 0o700
    assert (tmp_path / ".orcha/.gitignore").read_text() == "*\n"
    other = tmp_path / "second"
    other.mkdir()
    (other / ".orcha").symlink_to(tmp_path / ".orcha", target_is_directory=True)
    with pytest.raises(PermissionError):
        ensure_artifacts(other)


@pytest.mark.parametrize(
    "setting",
    [
        'allowed_roots = "bad"',
        "deny = [1]",
        'shell_env_passthrough = ["BAD=NAME"]',
        "max_read_bytes = 0",
        "max_read_bytes = true",
        'read_summary = "true"',
    ],
)
def test_security_config_rejects_invalid_values(tmp_path: Path, setting: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[tools]\n" + setting)
    with pytest.raises(SystemExit):
        load_config([], env={}, cwd=tmp_path, user_config_path=path)


def test_security_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[tools]\nallowed_roots=["../shared"]\ndeny=["private/**"]\n'
        'shell_env_passthrough=["CUSTOM_TOKEN"]\nmax_read_bytes=2048\nread_summary=true\n'
    )
    config = load_config([], env={}, cwd=tmp_path, user_config_path=path)
    assert config.tools.allowed_roots == ("../shared",)
    assert config.tools.deny == ("private/**",)
    assert config.tools.shell_env_passthrough == ("CUSTOM_TOKEN",)
    assert config.tools.max_read_bytes == 2048
    assert config.tools.read_summary is True
