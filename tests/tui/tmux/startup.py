"""Real-CLI startup regression with isolated config and escaped skill symlinks."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path


def verify_startup() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[3]
    socket = "polish"
    frames = Path("/tmp/polish-frames")
    frames.mkdir(exist_ok=True)
    results: dict[str, object] = {}

    def tmux(*args: str, check: bool = True) -> str:
        return subprocess.run(
            ["tmux", "-L", socket, *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=check,
        ).stdout

    with tempfile.TemporaryDirectory(prefix="orcha-polish-startup-") as directory:
        root = Path(directory)
        home = root / "home"
        skills = home / ".claude/skills"
        skills.mkdir(parents=True)
        config = root / "config"
        config.mkdir()
        (config / "config.toml").write_text("[ui]\nnotifications = false\n")
        for index in range(6):
            target = root / "external-skills" / f"skill-{index}"
            target.mkdir(parents=True)
            (target / "SKILL.md").write_text(
                f"---\nname: skill-{index}\ndescription: isolated fixture\n---\nFixture.\n"
            )
            (skills / target.name).symlink_to(target, target_is_directory=True)
        for width, height in ((120, 40), (80, 30)):
            session = f"startup{width}-{os.getpid()}"
            command = shlex.join(
                [
                    "env",
                    "-i",
                    f"HOME={home}",
                    f"PATH={os.environ['PATH']}",
                    "TERM=xterm-256color",
                    "LANG=C.UTF-8",
                    f"XDG_CONFIG_HOME={root / 'xdg'}",
                    f"ORCHA_CONFIG_DIR={config}",
                    f"ORCHA_DB_PATH={root / f'{width}.db'}",
                    f"UV_CACHE_DIR={root / 'uv-cache'}",
                    "uv",
                    "run",
                    "--no-sync",
                    "orcha",
                    "--cwd",
                    str(home),
                ]
            )
            command = "printf 'STARTUP_HISTORY_SENTINEL\\n'\nexec " + command
            tmux(
                "-f",
                "/dev/null",
                "new-session",
                "-d",
                "-s",
                session,
                "-x",
                str(width),
                "-y",
                str(height),
                command,
            )
            try:

                def capture(*, history: bool = False, ansi: bool = False) -> str:
                    args = ["capture-pane", "-p"]
                    if ansi:
                        args.append("-e")
                    if history:
                        args.extend(["-S", "-"])
                    return tmux(*args, "-t", session)

                def ready() -> str:
                    deadline = time.monotonic() + 20
                    previous = ""
                    same = 0
                    while time.monotonic() < deadline:
                        value = capture()
                        if (
                            "Ask anything" in value
                            and "6 skills skipped" in value
                            and "orcha v" in value
                        ):
                            same = same + 1 if value == previous else 0
                            if same >= 3:
                                return value
                        previous = value
                        time.sleep(0.05)
                    raise AssertionError(
                        f"startup {width} did not settle:\n{capture(history=True)}"
                    )

                visible = ready()
                (frames / f"after-{width}.ansi").write_text(capture(ansi=True))
                (frames / f"after-{width}.txt").write_text(visible)
                assert visible.count("orcha v") == 1, visible
                assert visible.count("6 skills skipped") == 1, visible
                assert visible.index("6 skills skipped") < visible.index("orcha v"), visible
                assert "No recent sessions" in visible, visible
                assert "Tips" in visible and "Ask anything" in visible, visible
                assert "╰" in visible[visible.index("orcha v") : visible.index("Ask anything")], (
                    visible
                )
                for forbidden in (
                    "Traceback (most recent call last)",
                    "Unhandled exception",
                    "Press ENTER",
                    "escapes its discovery root",
                ):
                    assert forbidden not in capture(history=True), forbidden
                # Resize forces a fresh paint without submitting a model turn.
                tmux("resize-window", "-t", session, "-x", str(width - 1), "-y", str(height))
                ready()
                tmux("resize-window", "-t", session, "-x", str(width), "-y", str(height))
                ready()
                history = capture(history=True)
                assert history.count("STARTUP_HISTORY_SENTINEL") == 1, history
                assert history.count("orcha v") == 1, history
                assert history.count("6 skills skipped") == 1, history
                results[str(width)] = {
                    "welcome_count": 1,
                    "skill_summary_count": 1,
                    "full_welcome_visible": True,
                }
                tmux("send-keys", "-t", session, "C-d")
            finally:
                tmux("kill-session", "-t", session, check=False)
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(verify_startup(), indent=2))
