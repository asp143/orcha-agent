"""Real slash-command viewport checks; no model prompts or user config access."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def verify_panels() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[3]
    socket = f"orcha-panels-{os.getpid()}"
    frames = Path("/tmp/polish-frames/round2")
    frames.mkdir(parents=True, exist_ok=True)
    results = {}

    def tmux(*args, check=True):
        return subprocess.run(
            ["tmux", "-L", socket, *args], cwd=repo, capture_output=True, text=True, check=check
        ).stdout

    with tempfile.TemporaryDirectory(prefix="orcha-panels-") as directory:
        root = Path(directory)
        home = root / "home"
        home.mkdir()
        config = root / "config"
        config.mkdir()
        # An existing user config legitimately skips the first-run setup wizard.
        (config / "config.toml").write_text("[tui]\nnotify = false\n")
        for width, height in ((120, 40), (80, 30)):
            session = f"panels{width}"
            command = shlex.join(
                [
                    "env",
                    "-i",
                    f"HOME={home}",
                    f"PATH={os.environ['PATH']}",
                    "LANG=C.UTF-8",
                    "TERM=xterm-256color",
                    f"XDG_CONFIG_HOME={root / 'xdg'}",
                    f"ORCHA_CONFIG_DIR={config}",
                    f"ORCHA_DB_PATH={root / f'{width}.db'}",
                    sys.executable,
                    "-m",
                    "orcha_agent",
                    "--cwd",
                    str(home),
                ]
            )
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

                def capture(*extra):
                    return tmux("capture-pane", "-p", *extra, "-t", session)

                def wait_for(predicate, label):
                    deadline = time.monotonic() + 15
                    stable = 0
                    while time.monotonic() < deadline:
                        text = capture()
                        stable = stable + 1 if predicate(text) else 0
                        if stable >= 5:
                            return text
                        time.sleep(0.05)
                    raise AssertionError(f"{width}: {label}\n{capture('-S', '-')}")

                wait_for(lambda text: "Ask anything" in text and "orcha v" in text, "welcome")
                for slash, title, body in (
                    ("/status", "Status", "Segment"),
                    ("/providers", "Providers", "Auth / Keys"),
                    ("/skills", "Skills", "No skills found"),
                    ("/plugins", "Plugins", "Name"),
                    ("/mcp list", "MCP servers", "No MCP servers configured"),
                ):
                    tmux("send-keys", "-t", session, "-l", slash)
                    tmux("send-keys", "-t", session, "Enter")
                    text = wait_for(
                        lambda value: (
                            f"╭─ {title} " in value and body in value and "Ask anything" in value
                        ),
                        slash,
                    )
                    # No orphan bottom rail may appear before the visible panel.
                    before = text[: text.index(f"╭─ {title} ")]
                    assert "╰" not in before, text
                    assert "Traceback" not in text and "Press ENTER" not in text, text
                    label = slash[1:].replace(" ", "-")
                    (frames / f"{label}-{width}.txt").write_text(text)
                    (frames / f"{label}-{width}.ansi").write_text(capture("-e"))
                results[str(width)] = {"commands_visible": 5, "orphan_border": False}
                tmux("send-keys", "-t", session, "C-d")
            finally:
                tmux("kill-session", "-t", session, check=False)
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(verify_panels(), indent=2))
