"""Session-scoped bash execution and bounded, recoverable command output."""

from __future__ import annotations

import atexit
import asyncio
import os
import math
import re
import shlex
import signal
import stat
import shutil
import sys
import json
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from orcha_agent.core.tools.common import PathPolicy, ensure_artifacts, prune_artifacts

ANSI = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")
OUTPUT_BYTES = 20_000
OUTPUT_LINES = 200
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.I)


def session_environment(passthrough: Sequence[str] = ()) -> dict[str, str]:
    explicit = set(passthrough or [])
    ordinary = {
        "PATH",
        "HOME",
        "LANG",
        "TERM",
        "COLORTERM",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TZ",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key in explicit
        or ((key in ordinary or key.startswith(("LC_", "XDG_"))) and not SECRET_NAME.search(key))
    }


def intercept(command: str) -> str | None:
    """Redirect only simple file-reading commands; leave pipelines and writes alone."""
    if any(char in command for char in "|;&><\n`$"):
        return None
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    while words and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", words[0]):
        words.pop(0)
    if not words:
        return None
    name = words[0]
    target = {
        "cat": "read",
        "head": "read",
        "tail": "read",
        "grep": "grep",
        "rg": "grep",
        "find": "glob",
        "ls": "ls",
    }.get(name)
    if target is None:
        return None
    flags = [word for word in words[1:] if word.startswith("-") and word != "-"]
    allowed = {
        "cat": {"--"},
        "head": {"-n", "--"},
        "tail": {"-n", "--"},
        "grep": {"-i", "-n", "-E", "-r", "-R", "--"},
        "rg": {"-i", "-n", "--"},
        "ls": {"-l", "-a", "-la", "-al", "--"},
        "find": {"-name", "-type"},
    }
    if name == "rg" and "--files" in flags:
        if any(flag not in {"--files", "--hidden"} for flag in flags):
            return None
        target = "glob"
    elif any(
        flag not in allowed[name] and not (name in {"head", "tail"} and re.fullmatch(r"-\d+", flag))
        for flag in flags
    ):
        return None
    if name in {"cat", "head", "tail", "grep"} and (len(words) == 1 or "-" in words[1:]):
        return None
    if name == "find" and "-type" in words:
        index = words.index("-type")
        if index + 1 >= len(words) or words[index + 1] != "f":
            return None
    return f"Use the {target} tool for this file operation. Read selectors: path:1-200, path:-25. The shell command was not executed."


def clamp_timeout(timeout: float | None) -> float:
    if timeout is not None and not math.isfinite(timeout):
        raise ValueError("timeout must be a finite number")
    return 300 if timeout is None else 0 if timeout == 0 else max(1, min(3600, timeout))


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@dataclass
class Job:
    process: subprocess.Popen[bytes]
    output: Path
    deadline: float | None
    timed_out: bool = False


class ShellSession:
    """A real bash process, serialized across calls; background jobs use snapshots."""

    def __init__(
        self,
        cwd: Path,
        artifact_root: Path | None = None,
        shell_env_passthrough: Sequence[str] = (),
        policy: PathPolicy | None = None,
    ):
        self.cwd = cwd.resolve()
        self.workspace_root = self.cwd
        self.policy = policy or PathPolicy(cwd)
        self.shell_env_passthrough = set(shell_env_passthrough)
        self.env = session_environment(shell_env_passthrough)
        self.baseline_env = self.env.copy()
        self.promoted: set[Path] = set()
        # Bash must never implicitly source startup files supplied by the host.
        self.env.pop("BASH_ENV", None)
        self.env.pop("ENV", None)
        self.artifact_root = (
            artifact_root or self.cwd / ".orcha" / "artifacts" / "large_tool_results"
        )
        self.control = tempfile.TemporaryDirectory(prefix="orcha-shell-")
        self.lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.jobs: dict[str, Job] = {}
        self.closed = False
        atexit.register(self.close)

    def close(self) -> None:
        self.closed = True
        if self.process is not None:
            _kill(self.process)
        with self.lock:
            if self.process is not None:
                _kill(self.process)
                if self.process.stdin is not None:
                    self.process.stdin.close()
                self.process = None
            for job in self.jobs.values():
                _kill(job.process)
            for output in self.promoted:
                try:
                    self.policy.unlink(output)
                except OSError:
                    # A replaced or inaccessible artifact target must stay untouched;
                    # it must not prevent cleanup of our private session outputs.
                    pass
            try:
                self.control.cleanup()
            finally:
                atexit.unregister(self.close)

    @contextmanager
    def _locked(self, cancel: threading.Event | None) -> Iterator[None]:
        while not self.lock.acquire(timeout=0.05):
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
        try:
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            yield
        finally:
            self.lock.release()

    def _start(self) -> subprocess.Popen[bytes]:
        if self.closed:
            raise RuntimeError("Shell session is closed")
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                ["bash", "--noprofile", "--norc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.cwd,
                env=self.env,
                start_new_session=True,
            )
        return self.process

    def _output(self) -> Path:
        path = Path(self.control.name) / (uuid.uuid4().hex + ".output")
        path.touch(mode=0o600)
        return path

    def _result(
        self, path: Path, code: int | None, offset: int = 0, limit: int = OUTPUT_LINES
    ) -> tuple[str, dict[str, Any]]:
        text, artifact = output_result(path, code, offset, limit)
        if artifact["clipped"]:
            ensure_artifacts(self.workspace_root, self.policy)
            self.policy.mkdir(self.artifact_root, mode=0o700)
            destination = self.policy.resolve(self.artifact_root / path.name)
            with self.policy.directory_fd(destination.parent) as parent:
                descriptor = os.open(
                    destination.name,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=parent,
                )
                with os.fdopen(descriptor, "wb") as sink, path.open("rb") as source:
                    if not stat.S_ISREG(os.fstat(sink.fileno()).st_mode):
                        raise ValueError("Artifact destination is not a regular file")
                    os.fchmod(sink.fileno(), 0o600)
                    shutil.copyfileobj(source, sink)
            self.promoted.add(destination)
            prune_artifacts(self.artifact_root)
            text = text.replace(str(path), str(destination))
            artifact["output_path"] = str(destination)
        return text, artifact

    def approval_description(self, args: dict[str, Any]) -> str:
        target = (self.cwd / str(args["cwd"])).resolve() if args.get("cwd") else self.cwd
        delta = {
            key: self.env.get(key)
            for key in self.env.keys() | self.baseline_env.keys()
            if self.env.get(key) != self.baseline_env.get(key)
        }

        def display(values: dict[str, Any]) -> str:
            return json.dumps(
                {
                    key: "[redacted]"
                    if SECRET_NAME.search(key) or key in self.shell_env_passthrough
                    else value
                    for key, value in sorted(values.items())
                }
            )

        return f"Working directory (resolved): {target}\nSession environment delta: {display(delta)}\nTemporary command environment: {display(args.get('env') or {})}"

    def run(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        background: bool = False,
        cancel: threading.Event | None = None,
    ) -> tuple[str, dict[str, Any]]:
        hint = intercept(command)
        if hint:
            return hint, {}
        if env and any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in env):
            return "Error: environment variable names must be shell identifiers.", {}
        effective = clamp_timeout(timeout)
        with self._locked(cancel):
            if self.closed:
                return "Error: shell session is closed.", {}
            target = (self.cwd / cwd).resolve() if cwd else self.cwd
            if not target.is_dir():
                return f"Error: working directory does not exist: {target}", {}
            output = self._output()
            if background:
                job_env = self.env | (env or {})
                job_env.pop("BASH_ENV", None)
                job_env.pop("ENV", None)
                with output.open("wb") as stream:
                    process = subprocess.Popen(
                        ["bash", "--noprofile", "--norc", "-c", command],
                        cwd=target,
                        env=job_env,
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                job_id = uuid.uuid4().hex[:12]
                job = Job(process, output, time.monotonic() + effective if effective else None)
                self.jobs[job_id] = job
                if effective:
                    threading.Thread(
                        target=self._watch_job, args=(job, effective), daemon=True
                    ).start()
                return (
                    f"Started background job {job_id}; use bash_jobs(action='read', job_id='{job_id}').",
                    {"job_id": job_id, "output_path": str(output)},
                )
            process = self._start()
            token = uuid.uuid4().hex
            status = Path(self.control.name) / token
            location = status.with_suffix(".cwd")
            environ = status.with_suffix(".vars")
            setup = ""
            if cwd:
                setup += f"cd -- {shlex.quote(str(target))} || exit\n"
            evaluation = f"eval {shlex.quote(command)}"
            if env:
                assignments = "\n".join(
                    f"export {key}={shlex.quote(value)}" for key, value in env.items()
                )
                evaluation = f"(\n{assignments}\n{evaluation}\n)"
            dump = "import os,json; print(json.dumps(dict(os.environ)))"
            script = (
                setup + f"{evaluation} </dev/null >{shlex.quote(str(output))} 2>&1\n"
                f"__orcha_status=$?\ncommand pwd -P >{shlex.quote(str(location))}\n"
                f"{shlex.quote(sys.executable)} -c {shlex.quote(dump)} >{shlex.quote(str(environ))}\n"
                f"printf '%s' \"$__orcha_status\" >{shlex.quote(str(status))}\n"
            )
            timed_out = False
            try:
                assert process.stdin is not None
                process.stdin.write(script.encode())
                process.stdin.flush()
                deadline = time.monotonic() + effective if effective else None
                while not status.exists() or status.stat().st_size == 0:
                    if cancel is not None and cancel.is_set():
                        raise asyncio.CancelledError
                    if process.poll() is not None:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        _kill(process)
                        break
                    time.sleep(0.01)
                if status.exists() and status.stat().st_size:
                    code = int(status.read_text())
                    self.cwd = Path(location.read_text().rstrip("\n"))
                    self.env = json.loads(environ.read_text())
                    self.env.pop("BASH_ENV", None)
                    self.env.pop("ENV", None)
                else:
                    code = process.poll()
                text, artifact = self._result(output, code)
                if timed_out:
                    text += f"\nCommand timed out after {effective:g}s; shell restarted on next call with last completed cwd/env."
                    artifact["timed_out"] = True
                return text, artifact
            except BaseException:
                _kill(process)
                raise
            finally:
                for path in (status, location, environ):
                    path.unlink(missing_ok=True)

    def _watch_job(self, job: Job, seconds: float) -> None:
        try:
            job.process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            job.timed_out = True
            _kill(job.process)

    def manage(
        self, action: str, job_id: str | None = None, offset: int = 0, limit: int = 200
    ) -> tuple[str, dict[str, Any]]:
        with self.lock:
            if action == "list":
                return "\n".join(
                    f"{key}: {'running' if job.process.poll() is None else 'timed out' if job.timed_out else f'exited {job.process.returncode}'}"
                    for key, job in self.jobs.items()
                ) or "No background jobs.", {}
            if job_id not in self.jobs:
                return f"Error: unknown job_id {job_id!r} in this session.", {}
            job = self.jobs[job_id]
            if action == "kill":
                _kill(job.process)
            text, artifact = self._result(
                job.output, job.process.poll(), max(0, offset), max(1, min(limit, 2000))
            )
            artifact["job_id"] = job_id
            return text, artifact


def output_result(
    path: Path, code: int | None, offset: int = 0, limit: int = OUTPUT_LINES
) -> tuple[str, dict[str, Any]]:
    """Scan without loading unbounded output; retain full bytes on disk."""
    selected: list[str] = []
    total = 0
    used = 0
    position = 0
    shown_end = 0
    partial_line = False
    with path.open("rb") as stream:
        continuation = False
        while chunk := stream.readline(OUTPUT_BYTES):
            if not continuation:
                total += 1
                if total > offset and len(selected) < limit and used < OUTPUT_BYTES:
                    captured = chunk[: OUTPUT_BYTES - used]
                    text_chunk = captured.decode("utf-8", errors="replace")
                    text_chunk = text_chunk.encode("utf-8")[: OUTPUT_BYTES - used].decode(
                        "utf-8", errors="ignore"
                    )
                    shown_end = position + len(captured)
                    partial_line = len(captured) < len(chunk) or not captured.endswith(b"\n")
                    selected.append(text_chunk)
                    used += len(text_chunk.encode("utf-8"))
            position += len(chunk)
            continuation = not chunk.endswith(b"\n")
    raw = "".join(selected)
    text = ANSI.sub("", raw).rstrip()
    size = path.stat().st_size
    clipped = total > offset + len(selected) or bool(selected and shown_end < size)
    if clipped:
        recover = (
            "import sys; "
            f"f=open({str(path)!r}, 'rb'); f.seek({shown_end}); "
            f"sys.stdout.buffer.write(f.read({OUTPUT_BYTES}))"
        )
        command = "python -c " + shlex.quote(recover)
        text += (
            f"\n[Output limited: showing lines {offset + 1}-{offset + len(selected)} of {total}; "
            f"{'last line is partial; ' if partial_line else ''}"
            f"showing at most {OUTPUT_BYTES} bytes of {size}. Full output: {path}; "
            f"use read(path='{path}:{offset + len(selected) + 1}+200') for later lines. "
            f"To recover the next bytes exactly, use bash(command={command!r}) "
            f"(zero-based byte offset {shown_end}, length {OUTPUT_BYTES}).]"
        )
    text += f"\nExit code: {code}" if code is not None else "\nJob is running."
    return text, {
        "clipped": clipped,
        "raw_output": raw,
        "output_path": str(path),
        "exit_code": code,
        "total_bytes": size,
        "total_lines": total,
    }


def create_shell_tools(
    cwd: Path, *, shell_env_passthrough: Sequence[str] = (), policy: PathPolicy | None = None
) -> list[BaseTool]:
    sessions: dict[str, ShellSession] = {}
    lock = threading.Lock()

    def session(config: RunnableConfig) -> ShellSession:
        configurable = config.get("configurable", {})
        key = str(configurable.get("thread_id", "default"))
        with lock:
            if key not in sessions:
                sessions[key] = ShellSession(
                    cwd, shell_env_passthrough=shell_env_passthrough, policy=policy
                )
            return sessions[key]

    def checked(result: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if result[0].startswith("Error"):
            raise ToolException(result[0])
        return result

    def bash(
        command: str,
        config: RunnableConfig,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        background: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Execute bash with persistent cwd and exported variables in this session.

        Use read/grep/glob/ls for file discovery and reading; use bash for builds,
        tests, git, and other commands. Timeout defaults to 300 seconds; nonzero
        values clamp to 1–3600; 0 disables. background returns a job_id for bash_jobs.
        env applies only to this command; explicit export commands persist when env
        is omitted. Subagents using the same thread share this shell session.
        Output is bounded; clipped output is retained for follow-up reads.
        """
        try:
            return checked(session(config).run(command, timeout, cwd, env, background))
        except (OSError, ValueError) as exc:
            raise ToolException(f"Error running bash: {exc}") from exc

    def bash_jobs(
        action: Literal["list", "read", "kill"],
        config: RunnableConfig,
        job_id: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[str, dict[str, Any]]:
        """List, read, or kill this session's background jobs. offset is zero-based lines.

        Read output incrementally with offset/limit. Killing a job terminates its
        process group. Commands launched in the background do not change shell state.
        """
        return checked(session(config).manage(action, job_id, offset, limit))

    async def abash(
        command: str,
        config: RunnableConfig,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        background: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        cancel = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                session(config).run,
                command,
                timeout,
                cwd,
                env,
                background,
                cancel,
            )
        )
        try:
            return checked(await asyncio.shield(task))
        except asyncio.CancelledError:
            cancel.set()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            raise
        except (OSError, ValueError) as exc:
            raise ToolException(f"Error running bash: {exc}") from exc

    result: list[BaseTool] = [
        StructuredTool.from_function(bash, coroutine=abash, response_format="content_and_artifact"),
        StructuredTool.from_function(bash_jobs, response_format="content_and_artifact"),
    ]

    def close() -> None:
        with lock:
            for shell in sessions.values():
                shell.close()
            sessions.clear()

    def approval(config: RunnableConfig, args: dict[str, Any]) -> str:
        key = str(config.get("configurable", {}).get("thread_id", "default"))
        with lock:
            current = sessions.get(key)
            if current is not None:
                return current.approval_description(args)
        target = (cwd / str(args["cwd"])).resolve() if args.get("cwd") else cwd.resolve()
        temporary = {
            key: "[redacted]" if SECRET_NAME.search(key) or key in shell_env_passthrough else value
            for key, value in (args.get("env") or {}).items()
        }
        return f"Working directory (resolved): {target}\nSession environment delta: {{}}\nTemporary command environment: {json.dumps(temporary)}"

    for item in result:
        _APPROVALS[id(item)] = approval
        item.handle_tool_error = True
        item.handle_validation_error = True
        # Private runtime lifecycle, never serialized into model-visible metadata.
        _CLOSERS[id(item)] = close
    return result


_CLOSERS: dict[int, Any] = {}
_APPROVALS: dict[int, Any] = {}


def shell_approval_description(tool: BaseTool, config: RunnableConfig, args: dict[str, Any]) -> str:
    callback = _APPROVALS.get(id(tool))
    return callback(config, args) if callback else ""


def close_shell_tools(tools: list[BaseTool]) -> None:
    """Release persistent processes and background groups when the app exits."""
    for item in tools:
        _APPROVALS.pop(id(item), None)
    callbacks = {_CLOSERS.pop(id(item)) for item in tools if id(item) in _CLOSERS}
    for close in callbacks:
        close()
