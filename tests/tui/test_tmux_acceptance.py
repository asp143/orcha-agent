"""Keep terminal acceptance in the ordinary suite when tmux is installed."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_tmux_responsiveness_acceptance():
    script = Path(__file__).with_name("tmux") / "verify.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["turn_a"]["repaint_growth"] == 0
    assert evidence["turn_b"]["repaint_growth"] == 0
    assert evidence["paste"]["payload_intact"]
    assert evidence["stream_resize"]["markers_each"] == 1
    assert evidence["mouse"] == {"sgr_wheel_up": True, "sgr_wheel_down": True}
    print(result.stdout)
