"""Regressions for repaint recovery and synchronized-output settings."""

from types import SimpleNamespace

import pytest

from orcha_agent.tui.runtime import _PaintOutput


def paint_output(monkeypatch):
    monkeypatch.setenv("ORCHA_SYNC_OUTPUT", "1")
    calls, writes = [], []
    app = SimpleNamespace(
        output=SimpleNamespace(_buffer=[], flush=lambda: None),
        _redraw=lambda done=False: calls.append("draw"),
        invalidate=lambda: calls.append("invalidate"),
    )
    paint = _PaintOutput(app)
    paint.pump = SimpleNamespace(pending=0, error=None, write=writes.append)
    return paint, calls, writes


@pytest.mark.parametrize("pending_at_close", [0, 300_000])
@pytest.mark.parametrize("nested", [False, True])
def test_frame_close_reschedules_skipped_repaint_without_watchdog(
    monkeypatch, pending_at_close, nested
):
    paint, calls, writes = paint_output(monkeypatch)
    paint.begin_frame()
    if nested:
        paint.begin_frame()
    # Settled output fills the queue before run_in_terminal restores the UI.
    paint.pump.pending = 300_000
    paint.redraw()
    assert calls == []
    assert paint._redraw_skipped
    paint.pump.pending = pending_at_close
    if nested:
        paint.end_frame()
        assert calls == []
    paint.end_frame()
    assert calls == ["invalidate"]
    assert writes == ["\x1b[?2026h", "\x1b[?2026l"]
    # The scheduled repaint can run as soon as the writer drains, without a
    # watchdog tick. Ordinary subsequent transactions need no extra repaint.
    paint.pump.pending = 0
    paint.redraw()
    paint.begin_frame()
    paint.end_frame()
    assert calls == ["invalidate", "draw"]
    assert not paint._redraw_skipped
