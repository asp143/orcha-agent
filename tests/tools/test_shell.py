import asyncio
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
    assert f"{tmp_path}/sub kept" in text
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
    assert not status.exists() or status.read_text().split()[2] == "Z"
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
