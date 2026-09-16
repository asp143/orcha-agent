"""Keep optional agent dependencies out of the standalone gallery's startup."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["orcha_agent.tui.gallery", "orcha_agent.tui.overlays.hub"])
def test_gallery_and_hub_import_without_langchain(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; import sys; assert 'langchain_core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_overlay_exports_and_factories_stay_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import SimpleNamespace
import orcha_agent.tui.overlays as overlays

factories = {}
overlays.register_builtin_overlays(
    SimpleNamespace(_add_overlay=lambda owner, name, factory: factories.update({name: factory}))
)
assert not any(name.startswith('orcha_agent.tui.overlays.') for name in sys.modules)
from orcha_agent.tui.overlays import HelpOverlay
assert HelpOverlay is overlays.HelpOverlay
assert 'orcha_agent.tui.overlays.hub' not in sys.modules
assert 'orcha_agent.tui.overlays.settings' not in sys.modules
assert 'langchain_core' not in sys.modules
ctx = SimpleNamespace(registry=SimpleNamespace(commands={}), ui=SimpleNamespace())
assert isinstance(factories['help'](ctx), HelpOverlay)
try:
    overlays.missing_overlay
except AttributeError:
    pass
else:
    raise AssertionError('unknown exports must raise AttributeError')
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
