"""Deterministic composer examples shared with visual acceptance tests."""

COMPOSER_SHAPES = ("box", "claude", "borderless", "band", "rail")
COMPOSER_LINES = ("first line", "last line")
PASTE_EXAMPLE = "\n".join(f"line {index}" for index in range(1, 7))


def ghost_example() -> tuple[str, str]:
    """Return an argument hint through the same completer used by the composer."""
    from pathlib import Path
    from types import SimpleNamespace

    from orcha_agent.tui.complete import ComposerCompleter

    registry = SimpleNamespace(
        commands={"model": SimpleNamespace(help="Choose model")}, completers=[]
    )
    command = "/model"
    return command, ComposerCompleter(registry, Path.cwd()).argument_hint(command)


def polish_examples(theme, width: int):
    """Actual styled completion, paste, Vim and idle/running status surfaces."""
    from types import SimpleNamespace

    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.key_binding.vi_state import InputMode
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.layout.processors import TransformationInput

    from orcha_agent.core.registry import Registry
    from orcha_agent.tui.composer import Composer, PasteChipProcessor
    from orcha_agent.tui.statusline import Segment, render_statusline

    composer = Composer(theme=theme)
    composer.buffer.text = "/h"
    composer.buffer.complete_state = composer.buffer._set_completions(
        [
            Completion(name, display=name, display_meta=description)
            for name, description in (
                ("help", "Show commands and keyboard shortcuts"),
                ("history", "Browse previous conversation sessions"),
                ("hub", "Inspect active and completed agents"),
            )
        ]
    )
    composer.buffer.complete_state.complete_index = 1
    yield "Completion", composer.completion_fragments(width)
    composer.buffer.reset()
    composer.insert_paste(PASTE_EXAMPLE + "\n")
    ti = TransformationInput(
        composer.control,
        composer.buffer.document,
        0,
        lambda x: x,
        [("", composer.buffer.text)],
        width,
        3,
    )
    yield "Paste chip", PasteChipProcessor(composer).apply_transformation(ti).fragments
    app = Application(editing_mode=EditingMode.VI, input=DummyInput(), output=DummyOutput())
    composer.buffer.reset()
    with set_app(app):
        for mode in (InputMode.NAVIGATION, InputMode.INSERT):
            app.vi_state.input_mode = mode
            yield (
                f"Vim {composer.vim_mode}",
                [
                    (composer.border_style, composer._top_line(width) + "\n"),
                    *composer.placeholder_fragments(),
                ],
            )
    registry = Registry()
    segments = {
        "brand": Segment("orcha", "accent"),
        "model": Segment("GPT-5.6 Sol", "statusLineModel"),
        "cost": Segment("$0.04", "statusLineCost"),
        "mode": Segment("default", "warning"),
        "path": Segment("orcha-agent", "statusLinePath"),
        "git": Segment("fix/tui-polish-omp", "statusLineGitClean"),
        "context": Segment("25.0%/272k", "statusLineContext"),
    }
    for name in segments:
        registry._add_status_segment("gallery", name, lambda _ctx, name=name: segments[name])
    ctx = SimpleNamespace(
        registry=registry,
        cfg=SimpleNamespace(statusbar=True, composer="box"),
        console=SimpleNamespace(width=width, encoding="utf-8"),
    )
    yield "Status idle", render_statusline(ctx, theme, width=width)
    segments["brand"] = Segment("⠇ orcha 3s · thinking", "accent")
    yield "Status running", render_statusline(ctx, theme, width=width)
