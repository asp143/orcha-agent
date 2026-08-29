from __future__ import annotations

from pathlib import Path

import pytest

from orcha_agent.tui.composer import Composer


GOLDEN_DIR = Path(__file__).with_name("golden")


def _encode_trailing_spaces(value: str) -> str:
    encoded: list[str] = []
    for line in value.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        trailing = len(body) - len(body.rstrip(" "))
        if trailing:
            body = f"{body[:-trailing]}<SP:{trailing}>"
        encoded.append(f"{body}{newline}")
    return "".join(encoded)


@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize("shape", ["box", "claude"])
def test_composer_chrome_golden(
    shape: str,
    width: int,
    update_goldens: bool,
) -> None:
    composer = Composer(
        shape=shape,
        model=lambda: "claude-sonnet-4",
        thinking=lambda: "high",
    )
    actual = _encode_trailing_spaces(
        "\n".join(
            composer.render_lines(
                ["first line", "last line"],
                width,
                scrollbar_rows={0},
            )
        )
        + "\n"
    )
    golden = GOLDEN_DIR / f"composer-{shape}.{width}.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual


def test_box_composer_truncates_an_oversized_chip_and_has_no_bottom_row() -> None:
    composer = Composer(
        shape="box",
        model=lambda: "a-model-name-that-does-not-fit",
        thinking=lambda: "high",
    )

    lines = composer.render_lines(["draft"], 20)

    assert len(lines) == 2
    assert lines[0].startswith("╭──")
    assert lines[0].endswith("──╮")
    assert len(lines[0]) == 20
    assert lines[1] == "╰─ draft" + " " * 10 + "─╯"


def test_box_composer_scrollbar_replaces_only_the_right_border() -> None:
    composer = Composer(shape="box")

    lines = composer.render_lines(["one", "two", "three"], 24, scrollbar_rows={1})

    assert lines[1].endswith("│")
    assert lines[2].endswith("█")
    assert lines[3].endswith("─╯")
