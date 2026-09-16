from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


class TmuxHarness:
    def __init__(self) -> None:
        self.repo = Path(__file__).resolve().parents[3]
        self.driver = Path(__file__).with_name("driver.py")
        self.socket = f"orcha-responsive-{os.getpid()}"
        self.session = "responsive"
        descriptor, state = tempfile.mkstemp(prefix="orcha-tmux-", suffix=".state")
        os.close(descriptor)
        self.state_path = Path(state)
        self.state_path.unlink()
        self.frames: dict[str, str] = {}

    def tmux(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["tmux", "-L", self.socket, *args],
            cwd=self.repo,
            check=check,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def start(self) -> None:
        command = shlex.join([sys.executable, str(self.driver), "--state", str(self.state_path)])
        subprocess.run(
            [
                "tmux",
                "-L",
                self.socket,
                "-f",
                "/dev/null",
                "new-session",
                "-d",
                "-s",
                self.session,
                "-x",
                "100",
                "-y",
                "30",
                command,
            ],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        self.wait_until(lambda: "Ask anything" in self.capture(), "composer startup")

    def stop(self) -> None:
        self.tmux("kill-server", check=False)
        self.state_path.unlink(missing_ok=True)

    def capture(self, *, history: bool = False) -> str:
        args = ["capture-pane", "-p"]
        if history:
            args.extend(["-S", "-2000"])
        args.extend(["-t", self.session])
        return self.tmux(*args)

    def history_size(self) -> int:
        value = self.tmux(
            "display-message",
            "-p",
            "-t",
            self.session,
            "#{history_size}",
        )
        return int(value.strip())

    def states(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        return set(self.state_path.read_text(encoding="utf-8").splitlines())

    def send(self, text: str) -> None:
        self.tmux("send-keys", "-t", self.session, "-l", text)
        self.tmux("send-keys", "-t", self.session, "Enter")

    def wait_until(
        self,
        predicate: Callable[[], bool],
        description: str,
        *,
        timeout: float = 10.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        tail = self.capture(history=True)[-2000:]
        raise AssertionError(f"timed out waiting for {description}\n{tail}")

    def measure_turn(self, label: str) -> dict[str, Any]:
        self.send(f"turn-{label}")
        self.wait_until(
            lambda: f"turn-{label}-active" in self.states(),
            f"turn {label} stream",
        )
        active_history = self.history_size()
        samples: list[int] = []
        while f"turn-{label}-streamed" not in self.states():
            samples.append(self.history_size())
            time.sleep(0.05)
        if set(samples) - {active_history}:
            raise AssertionError(
                f"turn {label} leaked during repaint: {active_history=}, {samples=}"
            )
        self.wait_until(
            lambda: f"turn-{label}-done" in self.states(),
            f"turn {label} commit",
        )
        committed_history = self.history_size()
        if committed_history <= active_history:
            raise AssertionError(
                f"turn {label} did not add committed rows: {active_history=} {committed_history=}"
            )
        return {
            "active_history": active_history,
            "committed_history": committed_history,
            "commit_growth": committed_history - active_history,
            "repaint_growth": max(samples, default=active_history)
            - min(samples, default=active_history),
        }

    def measure_fanout(self) -> dict[str, Any]:
        self.send("fanout")
        self.wait_until(lambda: "fanout-active" in self.states(), "fan-out card")
        active_history = self.history_size()
        samples: list[int] = []
        card_counts: list[int] = []
        while "fanout-streamed" not in self.states():
            samples.append(self.history_size())
            card_counts.append(self.capture().count("3 agents"))
            time.sleep(0.05)
        if set(samples) - {active_history}:
            raise AssertionError(
                f"fan-out card leaked during repaint: {active_history=}, {samples=}"
            )
        if not card_counts or max(card_counts) != 1:
            raise AssertionError(f"fan-out card stacked or never rendered: {card_counts=}")
        self.wait_until(lambda: "fanout-done" in self.states(), "fan-out commit")
        full_capture = self.capture(history=True)
        card_frames = full_capture.count("3 agents")
        if card_frames != 1:
            raise AssertionError(f"expected one fan-out card frame, got {card_frames}")
        return {
            "active_history": active_history,
            "final_history": self.history_size(),
            "repaint_growth": max(samples) - min(samples),
            "card_frames": card_frames,
        }

    def verify_markers(self) -> None:
        full_capture = self.capture(history=True)
        bad = {
            marker: full_capture.count(marker)
            for label in ("A", "B")
            for index in range(1, 31)
            if (marker := f"TMUX_{label}_{index:02d}") and full_capture.count(marker) != 1
        }
        if bad:
            raise AssertionError(f"streamed markers were duplicated or missing: {bad}")

    def verify_resize(self) -> dict[str, int]:
        frames: dict[str, int] = {}
        for name, columns, rows in (("wide", 120, 35), ("narrow", 80, 24)):
            self.tmux(
                "resize-window",
                "-t",
                self.session,
                "-x",
                str(columns),
                "-y",
                str(rows),
            )
            time.sleep(0.3)
            frames[name] = self.capture().count("Ask anything")
        if frames != {"wide": 1, "narrow": 1}:
            raise AssertionError(f"resize left multiple visible frames: {frames}")
        return frames

    def verify_paste(self) -> dict[str, Any]:
        self.frames["before_paste"] = self.capture()
        payload = "paste-first\npaste-second\npaste-third\nfour\nfive\nsix"
        self.tmux("set-buffer", "--", payload)
        self.tmux("paste-buffer", "-p", "-t", self.session)
        self.wait_until(lambda: "+6 lines" in self.capture(), "collapsed paste")
        self.frames["after_paste"] = self.capture()
        if any(state.startswith("paste-submitted:") for state in self.states()):
            raise AssertionError("paste submitted without Enter")
        self.tmux("send-keys", "-t", self.session, "Enter")
        expected = "paste-submitted:" + payload.replace("\n", "|")
        self.wait_until(lambda: expected in self.states(), "whole paste submission")
        return {"lines": 6, "submitted_before_enter": False, "payload_intact": True}

    def verify_stream_resize(self) -> dict[str, Any]:
        self.send("turn-resize")
        self.wait_until(lambda: "turn-resize-active" in self.states(), "resize stream")
        self.frames["before_stream_resize"] = self.capture()
        for columns, rows in ((80, 24), (120, 40)):
            self.tmux("resize-window", "-t", self.session, "-x", str(columns), "-y", str(rows))
            time.sleep(0.15)
        self.frames["during_stream_resize"] = self.capture()
        self.wait_until(lambda: "turn-resize-done" in self.states(), "resized turn commit")
        capture = self.capture(history=True)
        counts = {
            f"TMUX_RESIZE_{index:02d}": capture.count(f"TMUX_RESIZE_{index:02d}")
            for index in range(1, 31)
        }
        if set(counts.values()) != {1}:
            raise AssertionError(f"resize duplicated or lost stream markers: {counts}")
        self.frames["after_stream_resize"] = self.capture()
        return {"markers_each": 1, "final_terminal": "120x40"}

    def verify_mouse(self) -> dict[str, bool]:
        self.send("mouse")
        self.wait_until(lambda: "mouse-active" in self.states(), "mouse stream")
        self.wait_until(lambda: "MOUSE_79" in self.capture(), "viewport tail")
        self.frames["before_wheel"] = self.capture()
        self.tmux("send-keys", "-t", self.session, "-l", "\x1b[<64;10;20M")
        self.wait_until(lambda: "MOUSE_79" not in self.capture(), "wheel scroll up")
        self.frames["after_wheel"] = self.capture()
        self.tmux("send-keys", "-t", self.session, "-l", "\x1b[<65;10;20M")
        self.wait_until(lambda: "MOUSE_79" in self.capture(), "wheel scroll down")
        self.wait_until(lambda: "mouse-done" in self.states(), "mouse turn commit")
        return {"sgr_wheel_up": True, "sgr_wheel_down": True}

    def verify_expand(self) -> dict[str, Any]:
        self.send("card")
        self.wait_until(lambda: "card-active" in self.states(), "bash card")
        self.wait_until(lambda: "demo" in self.capture(), "bash preview")
        self.frames["before_expand"] = self.capture()
        collapsed = self.capture().count("CARD_ROW")
        self.tmux("send-keys", "-t", self.session, "C-o")
        self.wait_until(lambda: self.capture().count("CARD_ROW") > collapsed, "expanded card")
        self.frames["after_expand"] = self.capture()
        expanded = self.capture().count("CARD_ROW")
        self.tmux("send-keys", "-t", self.session, "C-o")
        self.wait_until(lambda: self.capture().count("CARD_ROW") == collapsed, "collapsed card")
        self.frames["after_collapse"] = self.capture()
        self.wait_until(lambda: "card-done" in self.states(), "card completion")
        return {"collapsed_rows": collapsed, "expanded_rows": expanded}

    def run(self) -> dict[str, Any]:
        self.start()
        startup_history = self.history_size()
        if startup_history < 29:
            raise AssertionError(
                f"fixed-height startup did not reserve rows once: {startup_history=}"
            )
        paste = self.verify_paste()
        first = self.measure_turn("a")
        second = self.measure_turn("b")
        self.verify_markers()
        fanout = self.measure_fanout()
        resize_frames = self.verify_resize()
        stream_resize = self.verify_stream_resize()
        mouse = self.verify_mouse()
        expand = self.verify_expand()
        return {
            "terminal": "100x30",
            "paste": paste,
            "mouse": mouse,
            "stream_resize": stream_resize,
            "expand": expand,
            "frames": self.frames,
            "startup_reservation": startup_history,
            "turn_a": first,
            "turn_b": second,
            "fanout": fanout,
            "resize_visible_frames": resize_frames,
            "markers_each": 1,
        }


def main() -> int:
    if shutil.which("tmux") is None:
        print("SKIP: tmux is not installed")
        return 0
    harness = TmuxHarness()
    try:
        result = harness.run()
        from startup import verify_startup

        result["startup"] = verify_startup()
    finally:
        harness.stop()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
