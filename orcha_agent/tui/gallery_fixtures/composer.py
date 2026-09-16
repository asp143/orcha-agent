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
