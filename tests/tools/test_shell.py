import asyncio
import json
import os
from pathlib import Path

import pytest

from orcha_agent.core.tools.shell import (
    ShellSession,
    clamp_timeout,
    close_shell_tools,
    create_shell_tools,
    intercept,
    output_result,
)


@pytest.fixture
def shell(tmp_path):
    session = ShellSession(tmp_path)
    yield session
    session.close()


def test_persistent_cwd_exports_and_functions(shell, tmp_path):
    (tmp_path / "sub").mkdir()
    shell.run("cd sub; export ORCHA_TEST_VALUE='value with spaces'; greet() { printf hello; }")
    text, _ = shell.run('printf "%s\\n" "$PWD" "$ORCHA_TEST_VALUE"; greet')
    assert str(tmp_path / "sub") in text
    assert "value with spaces" in text
    assert "hello" in text


def test_cwd_env_arguments_and_background_snapshot(shell, tmp_path):
    (tmp_path / "sub").mkdir()
    text, _ = shell.run(
        'printf "%s %s" "$PWD" "$ORCHA_TEST_VALUE"', cwd="sub", env={"ORCHA_TEST_VALUE": "kept"}
    )
    assert f"{tmp_path}/sub kept" in text
    _, artifact = shell.run('printf "%s %s" "$PWD" "$ORCHA_TEST_VALUE"', background=True)
    job = shell.jobs[artifact["job_id"]]
    job.process.wait(timeout=5)
    text, _ = shell.manage("read", artifact["job_id"])
    assert f"{tmp_path}/sub\n" in text
    assert "kept" not in text
    assert "exited 0" in shell.manage("list")[0]


def test_exit_code_and_restart(shell):
    assert "Exit code: 7" in shell.run("exit 7")[0]
    assert "working" in shell.run("printf working")[0]
    assert "Exit code: 2" in shell.run('bash -c "exit 2"')[0]


@pytest.mark.parametrize(
    ("value", "expected"), [(None, 300), (0, 0), (-1, 1), (0.1, 1), (9999, 3600)]
)
def test_timeout_clamping(value, expected):
    assert clamp_timeout(value) == expected


def test_timeout_kills_group_and_restarts(shell, tmp_path):
    shell.run("export ORCHA_TEST_VALUE=preserved")
    text, artifact = shell.run('sleep 10 & printf "%s" "$!" > child; wait', timeout=1)
    assert artifact["timed_out"]
    assert "timed out after 1s" in text
    child = int((tmp_path / "child").read_text())
    # A killed child can remain a zombie until the host init reaps it.
    status = Path(f"/proc/{child}/stat")
    try:
        state = status.read_text().split()[2]
    except FileNotFoundError:
        pass  # Reaped before (or during) the read: the child is gone.
    else:
        assert state == "Z"
    assert "preserved" in shell.run('printf "%s" "$ORCHA_TEST_VALUE"')[0]


def test_background_kill_read_and_unknown_job(shell):
    _, artifact = shell.run("printf started; sleep 20", background=True, timeout=0)
    job_id = artifact["job_id"]
    assert job_id in shell.manage("list")[0]
    assert "running" in shell.manage("list")[0]
    text, artifact = shell.manage("kill", job_id)
    assert artifact["exit_code"] == -9
    assert "unknown job_id" in shell.manage("read", "missing")[0]


def test_background_timeout(shell):
    _, artifact = shell.run("sleep 10", background=True, timeout=1)
    job = shell.jobs[artifact["job_id"]]
    job.process.wait(timeout=5)
    assert job.timed_out


def test_ansi_and_truncation_recovery(shell):
    text, artifact = shell.run("printf '\\033[31mred\\033[0m\\n'; seq 1 300")
    assert "\x1b" not in text
    assert "\x1b[31m" in artifact["raw_output"]
    assert "showing lines 1-200 of 301" in text
    assert "use read(" in text
    assert Path(artifact["output_path"]).read_text().endswith("300\n")


def test_output_paging_and_giant_line(tmp_path):
    path = tmp_path / "output"
    path.write_text("one\ntwo\nthree\n")
    text, _ = output_result(path, 0, offset=1, limit=1)
    assert text.startswith("two\n")
    assert "2-2 of 3" in text
    path.write_text("x" * 100_000)
    text, artifact = output_result(path, 0)
    assert len(artifact["raw_output"]) == 20_000
    assert "100000" in text


@pytest.mark.parametrize(
    ("command", "native"),
    [
        ("cat file", "read"),
        ("head -20 file", "read"),
        ("tail file", "read"),
        ("grep foo file", "grep"),
        ('find . -name "*.py"', "glob"),
        ("ls -la", "ls"),
        ("LC_ALL=C grep foo file", "grep"),
    ],
)
def test_interceptor(command, native):
    assert f"Use the {native} tool" in intercept(command)


@pytest.mark.parametrize(
    "command",
    [
        "printf hi | grep hi",
        "cat > file",
        "find . -delete",
        "find . -exec echo {} +",
        'printf "cat file"',
        "git status",
    ],
)
def test_interceptor_leaves_composed_and_mutating_commands(command):
    assert intercept(command) is None


def test_intercept_does_not_execute(shell):
    text, _ = shell.run("ls")
    assert "not executed" in text
    assert shell.process is None


def test_env_validation_and_bad_cwd(shell):
    assert "shell identifiers" in shell.run("true", env={"BAD-NAME": "x"})[0]
    assert "does not exist" in shell.run("true", cwd="missing")[0]


def test_tool_session_isolation_and_cleanup(tmp_path):
    tools = create_shell_tools(tmp_path)
    bash, jobs = tools
    try:
        bash.invoke(
            {"command": "export ORCHA_TEST_VALUE=alpha"},
            config={"configurable": {"thread_id": "a"}},
        )
        assert "alpha" in bash.invoke(
            {"command": 'printf "%s" "$ORCHA_TEST_VALUE"'},
            config={"configurable": {"thread_id": "a"}},
        )
        assert "alpha" not in bash.invoke(
            {"command": 'printf "%s" "$ORCHA_TEST_VALUE"'},
            config={"configurable": {"thread_id": "b"}},
        )
        assert "config" not in bash.tool_call_schema.model_fields
        message = bash.invoke(
            {
                "type": "tool_call",
                "name": "bash",
                "id": "call1",
                "args": {"command": "printf artifact"},
            }
        )
        assert message.artifact["raw_output"] == "artifact"
        assert jobs.invoke({"action": "list"}) == "No background jobs."
    finally:
        close_shell_tools(tools)


@pytest.mark.asyncio
async def test_async_cancellation_cleans_up_process_group(tmp_path):
    tools = create_shell_tools(tmp_path)
    try:
        task = asyncio.create_task(
            tools[0].ainvoke({"command": 'printf "%s" "$$" > pid; sleep 20', "timeout": 0})
        )
        for _ in range(100):
            if (tmp_path / "pid").exists():
                break
            await asyncio.sleep(0.01)
        pid = int((tmp_path / "pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert "restarted" in await tools[0].ainvoke({"command": "printf restarted"})
    finally:
        close_shell_tools(tools)


def test_command_stdin_cannot_consume_control_channel(shell):
    text, _ = shell.run("read swallowed; printf 'read-status=%s' \"$?\"", timeout=1)
    assert "read-status=1" in text
    assert "timed out" not in text
    assert "healthy" in shell.run("printf healthy")[0]


@pytest.mark.asyncio
async def test_cancelling_queued_call_does_not_wait_for_active_command(tmp_path):
    tools = create_shell_tools(tmp_path)
    first = asyncio.create_task(
        tools[0].ainvoke({"command": "printf ready > ready; sleep 20", "timeout": 0})
    )
    try:
        for _ in range(100):
            if (tmp_path / "ready").exists():
                break
            await asyncio.sleep(0.01)
        queued = asyncio.create_task(tools[0].ainvoke({"command": "touch should_not_exist"}))
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, timeout=1)
        assert not (tmp_path / "should_not_exist").exists()
    finally:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        close_shell_tools(tools)


def test_close_stops_active_command_without_deadlocking(shell, tmp_path):
    import concurrent.futures
    import time

    with concurrent.futures.ThreadPoolExecutor() as executor:
        active = executor.submit(shell.run, "printf ready > ready; sleep 20", 0)
        for _ in range(100):
            if (tmp_path / "ready").exists():
                break
            time.sleep(0.01)
        closing = executor.submit(shell.close)
        closing.result(timeout=2)
        active.result(timeout=2)


def test_set_errexit_fails_and_shell_recovers(shell):
    text, artifact = shell.run("set -e; false", timeout=1)
    assert artifact["exit_code"] == 1
    assert "Exit code: 1" in text
    assert "healthy" in shell.run("printf healthy")[0]


def test_tool_invalid_inputs_have_error_status(tmp_path):
    tools = create_shell_tools(tmp_path)
    try:
        for args in [
            {"command": "true", "env": {"KEY": 123}},
            {"command": "true", "timeout": float("nan")},
            {"command": "true", "cwd": "missing"},
        ]:
            message = tools[0].invoke(
                {"type": "tool_call", "name": "bash", "id": "bad", "args": args}
            )
            assert message.status == "error"
    finally:
        close_shell_tools(tools)


def test_single_long_line_recovery_is_an_executable_byte_cursor(shell, tmp_path):
    import ast

    path = tmp_path / "long output"
    path.write_bytes(b"a" * 20_000 + b"b" * 20_000 + b"c" * 10)
    text, _ = output_result(path, 0)
    assert "last line is partial" in text
    assert "zero-based byte offset 20000" in text
    literal = text.split("use bash(command=", 1)[1].split(") (zero-based", 1)[0]
    command = ast.literal_eval(literal)
    _, artifact = shell.run(command)
    assert artifact["raw_output"] == "b" * 20_000


def test_parent_provider_keys_are_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-sensitive")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-sensitive")
    monkeypatch.setenv("XDG_AUTH_TOKEN", "xdg-sensitive")
    with_session = ShellSession(tmp_path)
    try:
        text, _ = with_session.run(
            'printf "%s:%s:%s" "$ANTHROPIC_API_KEY" "$OPENAI_API_KEY" "$XDG_AUTH_TOKEN"'
        )
        assert text.startswith("::\n")
        _, artifact = with_session.run('printf "%s" "$OPENAI_API_KEY"', background=True)
        with_session.jobs[artifact["job_id"]].process.wait(timeout=5)
        assert "openai-sensitive" not in with_session.manage("read", artifact["job_id"])[0]
    finally:
        with_session.close()
    explicit = ShellSession(tmp_path, shell_env_passthrough=["OPENAI_API_KEY"])
    try:
        assert "openai-sensitive" in explicit.run('printf "%s" "$OPENAI_API_KEY"')[0]
    finally:
        explicit.close()


def test_temporary_env_does_not_persist_even_after_restart(shell):
    shell.run("export LOCAL_VALUE=original")
    assert (
        "temporary" in shell.run('printf "%s" "$LOCAL_VALUE"', env={"LOCAL_VALUE": "temporary"})[0]
    )
    assert "original" in shell.run('printf "%s" "$LOCAL_VALUE"')[0]
    shell.run("exit 0")
    assert "original" in shell.run('printf "%s" "$LOCAL_VALUE"')[0]


def test_outputs_private_until_clipped_and_removed_on_close(tmp_path):
    session = ShellSession(tmp_path)
    control = Path(session.control.name)
    assert control.stat().st_mode & 0o777 == 0o700
    _, short = session.run("printf short")
    assert Path(short["output_path"]).parent == control
    assert not (tmp_path / ".orcha").exists()
    _, clipped = session.run("seq 1 250")
    output = Path(clipped["output_path"])
    assert output.is_relative_to(tmp_path / ".orcha")
    assert output.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / ".orcha" / ".gitignore").read_text() == "*\n"
    session.close()
    assert not control.exists()
    assert not output.exists()


def test_artifact_retention_count_and_age(shell):
    import time

    shell.artifact_root.mkdir(parents=True)
    for i in range(203):
        (shell.artifact_root / f"old-{i}").touch()
    expired = shell.artifact_root / "expired"
    expired.touch()
    old_time = time.time() - 8 * 86400
    os.utime(expired, (old_time, old_time))
    shell.run("seq 1 250")
    assert len(list(shell.artifact_root.iterdir())) == 200
    assert not expired.exists()


def test_control_files_cleaned_after_timeout_and_portable_env(shell):
    shell.run("export ROUND1_PORTABLE='space and\nnewline'")
    assert shell.env["ROUND1_PORTABLE"] == "space and\nnewline"
    shell.run("sleep 5", timeout=1)
    assert all(path.suffix == ".output" for path in Path(shell.control.name).iterdir())


@pytest.mark.parametrize(
    "command",
    [
        "tail -f file",
        "head -c 3 file",
        "grep -v pattern file",
        "ls -R",
        "cat",
        "head -",
        "find . -type d",
        "find . -iname CASE",
        "rg --files --null",
    ],
)
def test_interceptor_passes_unsupported_operations(command):
    assert intercept(command) is None


def test_rg_files_suggests_glob():
    assert "Use the glob tool" in intercept("rg --files")


def test_approval_describes_drift_and_redacts_secrets_without_starting_shell(tmp_path):
    from orcha_agent.core.tools.shell import shell_approval_description

    tools = create_shell_tools(tmp_path)
    config = {"configurable": {"thread_id": "approval"}}
    try:
        assert "Session environment delta: {}" in shell_approval_description(tools[0], config, {})
        assert not (tmp_path / ".orcha").exists()
        (tmp_path / "sub").mkdir()
        tools[0].invoke(
            {"command": "cd sub; export ROUND1_VISIBLE=yes; export ROUND1_SECRET=hidden"},
            config=config,
        )
        description = shell_approval_description(
            tools[0], config, {"cwd": "..", "env": {"OPENAI_API_KEY": "temporary-hidden"}}
        )
        assert f"Working directory (resolved): {tmp_path}\n" in description
        assert '"ROUND1_VISIBLE": "yes"' in description
        assert "ROUND1_SECRET" in description
        assert "hidden" not in description
    finally:
        close_shell_tools(tools)


def test_portable_env_capture_does_not_depend_on_external_env(shell):
    shell.run("env() { printf unavailable >&2; return 127; }; export PORTABLE_TEST=portable")
    shell.run("exit 0")
    assert "portable" in shell.run('printf "%s" "$PORTABLE_TEST"')[0]


def test_approval_omits_shell_noise_but_retains_explicit_environment(shell):
    shell.run("export REVIEW_VISIBLE=yes; export REVIEW_SECRET=hidden")
    shell.env.update({"PWD": "/changed", "OLDPWD": "/previous", "SHLVL": "9", "_": "/bin/true"})
    description = shell.approval_description({"env": {"PWD": "/requested"}})
    delta = json.loads(description.split("Session environment delta: ", 1)[1].split("\n", 1)[0])
    assert not {"PWD", "OLDPWD", "SHLVL", "_"} & delta.keys()
    assert delta["REVIEW_VISIBLE"] == "yes"
    assert delta["REVIEW_SECRET"] == "[redacted]"
    assert 'Temporary command environment: {"PWD": "/requested"}' in description


def test_artifacts_refuse_symlink_escape(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / ".orcha").symlink_to(outside, target_is_directory=True)
    tools = create_shell_tools(tmp_path)
    try:
        message = tools[0].invoke(
            {"type": "tool_call", "name": "bash", "id": "escape", "args": {"command": "seq 1 250"}}
        )
        assert message.status == "error"
        assert list(outside.iterdir()) == []
    finally:
        close_shell_tools(tools)


def test_artifact_promotion_anchors_open_before_ancestor_swap(shell, tmp_path, monkeypatch):
    outside = tmp_path.parent / (tmp_path.name + "-swapped")
    outside.mkdir()
    original_open = os.open
    swapped = False

    def swap_at_output_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            flags & os.O_WRONLY
            and str(path).endswith(".output")
            and not swapped
            and (dir_fd is not None or Path(path).parent == shell.artifact_root)
        ):
            swapped = True
            root = shell.artifact_root
            moved = root.with_name("parked-output")
            root.rename(moved)
            root.symlink_to(outside, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                root.unlink()
                moved.rename(root)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_at_output_open)
    _, artifact = shell.run("seq 1 250")
    assert swapped
    assert list(outside.iterdir()) == []
    with shell.policy.open_read(artifact["output_path"]) as result:
        assert result.read().endswith(b"250\n")


def test_artifact_promotion_rejects_nonregular_destination(shell):
    from orcha_agent.core.tools.common import ensure_artifacts

    output = shell._output()
    output.write_text("line\n" * 250)
    ensure_artifacts(shell.workspace_root, shell.policy)
    destination = shell.artifact_root / output.name
    os.mkfifo(destination)
    reader = os.open(destination, os.O_RDWR | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="regular file"):
            shell._result(output, 0)
        with pytest.raises(BlockingIOError):
            os.read(reader, 10)
    finally:
        os.close(reader)
        destination.unlink()


def test_close_cleans_private_output_when_artifact_tree_is_replaced(tmp_path, monkeypatch):
    import atexit

    session = ShellSession(tmp_path)
    control = Path(session.control.name)
    _, artifact = session.run("seq 1 250")
    filename = Path(artifact["output_path"]).name
    outside = tmp_path.parent / (tmp_path.name + "-outside-close")
    outside_output = outside / "artifacts" / "large_tool_results"
    outside_output.mkdir(parents=True)
    sentinel = outside_output / filename
    sentinel.write_text("untouched outside output")
    private = tmp_path / ".orcha"
    moved = tmp_path / "moved-artifacts"
    private.rename(moved)
    private.symlink_to(outside, target_is_directory=True)
    original_unregister = atexit.unregister
    unregistered = []

    def unregister(callback):
        unregistered.append(callback)
        original_unregister(callback)

    monkeypatch.setattr(atexit, "unregister", unregister)
    try:
        session.close()
        assert not control.exists()
        assert session.close in unregistered
        assert sentinel.read_text() == "untouched outside output"
        assert (moved / "artifacts" / "large_tool_results" / filename).is_file()
    finally:
        private.unlink()
        moved.rename(private)
        session.close()
