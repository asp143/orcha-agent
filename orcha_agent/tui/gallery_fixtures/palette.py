"""Light accessibility colors rendered from the production theme adapter."""

from pathlib import Path

from rich.text import Text

from orcha_agent.tui.theme import apply_colorblind, load_theme_file


def light_colorblind_fixture() -> Text:
    theme = apply_colorblind(load_theme_file(Path(__file__).parents[1] / "themes" / "light.json"))
    result = Text()
    for label, foreground, background in (
        ("+ Added", "toolDiffAdded", "toolDiffAddedBg"),
        ("- Removed", "toolDiffRemoved", "toolDiffRemovedBg"),
    ):
        result.append(label, style=f"{theme.color(foreground)} on {theme.color(background)}")
        result.append("\n")
    return result
