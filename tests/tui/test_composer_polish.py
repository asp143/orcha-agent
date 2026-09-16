from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.output import DummyOutput

from orcha_agent.tui.composer import Composer
from orcha_agent.tui.gallery_fixtures.composer import polish_examples
from orcha_agent.tui.overlays.paste import PasteOverlay
from orcha_agent.tui.theme import load_themes


def test_trailing_newline_does_not_invent_paste_row():
    composer = Composer()
    payload = "one\ntwo\nthree\nfour\nfive\n"
    composer.insert_paste(payload)
    assert composer.buffer.text == "[+5 lines #1]"
    assert composer.expanded_text(composer.buffer.text) == payload
    assert len(PasteOverlay(payload).rows) == 5


@pytest.mark.parametrize("key", ["/", ":"])
def test_vim_normal_command_key_enters_insert(key):
    composer = Composer()
    app = Application(editing_mode=EditingMode.VI, input=DummyInput(), output=DummyOutput())
    app.vi_state.input_mode = InputMode.NAVIGATION
    with set_app(app):
        assert composer.border_style == "class:warning"
        assert "i to insert" in composer.placeholder_fragments()[0][1]
        bindings = composer.key_bindings.get_bindings_for_keys((key,))
        binding = next(binding for binding in bindings if binding.filter())
        binding.handler(SimpleNamespace(app=app))
        assert app.vi_state.input_mode == InputMode.INSERT
        assert composer.buffer.text == "/"
        assert composer.border_style == "class:accent"


@pytest.mark.parametrize("width", [80, 120])
def test_composer_polish_gallery_golden(width, update_goldens):
    theme = load_themes()["dark"]
    examples = dict(polish_examples(theme, width))
    actual = (
        "\n".join(
            f"{name}\n" + "\n".join(f"{style}: {text!r}" for style, text in fragments)
            for name, fragments in examples.items()
        )
        + "\n"
    )
    golden = Path(__file__).with_name("golden") / f"composer-polish.{width}.txt"
    if update_goldens:
        golden.write_text(actual)
    assert golden.read_text() == actual
    idle = "".join(text for _, text in examples["Status idle"])
    running = "".join(text for _, text in examples["Status running"])
    assert idle.index("$0.04") == running.index("$0.04")
    assert idle.index("GPT-5.6 Sol") == running.index("GPT-5.6 Sol")
    assert len(idle) == len(running) == width
    if width == 80:
        assert "━" not in idle and "─" not in idle
    completion = examples["Completion"]
    assert any("bg:" in style and "history" in text for style, text in completion)
    for row in "".join(text for _, text in completion).splitlines():
        assert len(row) == width
