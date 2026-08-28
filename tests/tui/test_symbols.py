from __future__ import annotations

import pytest
from rich import box

from orcha_agent.tui.symbols import SYMBOL_KEYS, SYMBOL_PRESETS, resolve_symbols


def test_all_symbol_presets_cover_the_complete_surface() -> None:
    assert set(SYMBOL_PRESETS) == {"unicode", "nerd", "ascii"}
    for preset in SYMBOL_PRESETS.values():
        assert set(preset) == set(SYMBOL_KEYS)


def test_ascii_symbols_are_terminal_safe_and_supply_rich_boxes() -> None:
    symbols = resolve_symbols("ascii")

    assert all(value.isascii() for key, value in symbols.items() if key in SYMBOL_KEYS)
    assert symbols["status.success"] == "+"
    assert symbols["boxRound"] is box.ASCII
    assert symbols["boxSharp"] is box.ASCII


def test_symbol_overrides_are_validated_and_fall_back_to_preset() -> None:
    symbols = resolve_symbols(
        "unicode",
        {"status.success": "YES", "icon.model": "M"},
    )

    assert symbols["status.success"] == "YES"
    assert symbols["icon.model"] == "M"
    assert symbols["status.error"] == SYMBOL_PRESETS["unicode"]["status.error"]

    with pytest.raises(ValueError, match="unknown symbol"):
        resolve_symbols("unicode", {"missing.key": "x"})
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_symbols("unicode", {"status.success": ""})


def test_non_utf_terminal_forces_ascii_symbols() -> None:
    symbols = resolve_symbols("nerd", encoding="ascii")

    assert symbols["icon.model"] == SYMBOL_PRESETS["ascii"]["icon.model"]
