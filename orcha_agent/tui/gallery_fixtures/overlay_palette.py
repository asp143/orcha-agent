"""Selection colour fixture derived from the live picker style adapter."""

from itertools import groupby

from rich.style import Style
from rich.text import Text

from orcha_agent.tui.overlays.select import SelectList
from orcha_agent.tui.theme import Theme


def selection_fixture(theme: Theme, width: int = 40) -> Text:
    picker = SelectList("Selection", ["Available model", "Current model", "Another model"])
    picker._move(1)
    content = picker.list_control.create_content(width, 3)
    output = Text()
    for row in range(content.line_count):
        for fragment in content.get_line(row):
            attrs = theme.pt.get_attrs_for_style_str(fragment[0])
            color = f"#{attrs.color}" if attrs.color and len(attrs.color) == 6 else None
            background = f"#{attrs.bgcolor}" if attrs.bgcolor and len(attrs.bgcolor) == 6 else None
            output.append(fragment[1], Style(color=color, bgcolor=background, bold=attrs.bold))
        output.append("\n")
    return output


def overlay_frame_fixture(theme: Theme, width: int, surface: str) -> Text:
    """Render the actual prompt-toolkit overlay, including filters and wrapped hints."""
    from types import SimpleNamespace

    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    from prompt_toolkit.output import DummyOutput

    from orcha_agent.tui.gallery_fixtures.surfaces import _Config
    from orcha_agent.tui.overlays.help import HelpOverlay
    from orcha_agent.tui.overlays.hub import HubOverlay
    from orcha_agent.tui.overlays.model import ModelOverlay
    from orcha_agent.tui.overlays.settings import SettingsOverlay
    from orcha_agent.tui.overlays.tree import TreeOverlay

    ctx = SimpleNamespace(
        cfg=_Config(),
        ui=SimpleNamespace(effective_keys={"submit": ("enter",)}),
        registry=SimpleNamespace(
            commands={"help": SimpleNamespace(help="Show commands and keybindings")},
            providers={"demo": SimpleNamespace(models=("small", "large"), available=lambda: None)},
        ),
        switch_model=lambda _: None,
        ledger=SimpleNamespace(all=lambda _: (), leaf=lambda _: None),
        session_id="gallery",
    )
    overlay = {
        "help": HelpOverlay,
        "hub": HubOverlay,
        "models": ModelOverlay,
        "tree": TreeOverlay,
        "settings": SettingsOverlay,
    }[surface](ctx)
    if isinstance(overlay, HelpOverlay):
        overlay._move(2)
    if isinstance(overlay, SettingsOverlay):
        overlay.category = overlay.categories.index("Behaviour")
        overlay._load()

    class Output(DummyOutput):
        def get_size(self) -> Size:
            return Size(rows=30 if width == 80 else 40, columns=width)

    app = Application(
        layout=Layout(overlay.container, focused_element=overlay.focus_target),
        input=DummyInput(),
        output=Output(),
    )
    with set_app(app):
        screen = Screen()
        columns, height = overlay._width(), overlay._height()
        overlay.container.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(xpos=0, ypos=0, width=columns, height=height),
            parent_style="",
            erase_bg=True,
            z_index=None,
        )
        screen.draw_all_floats()
    result = Text()
    # Resolve each repeated style once, then retain one span per adjacent run.
    # Per-cell spans otherwise generate thousands of redundant ANSI wrappers.
    styles: dict[str, Style] = {}
    for y in range(height):
        cells = (screen.data_buffer[y][x] for x in range(columns))
        for key, group in groupby(cells, key=lambda cell: cell.style):
            style = styles.get(key)
            if style is None:
                attrs = theme.pt.get_attrs_for_style_str(key)
                color = f"#{attrs.color}" if attrs.color and len(attrs.color) == 6 else None
                background = (
                    f"#{attrs.bgcolor}" if attrs.bgcolor and len(attrs.bgcolor) == 6 else None
                )
                style = styles[key] = Style(color=color, bgcolor=background, bold=attrs.bold)
            result.append("".join(cell.char for cell in group), style)
        result.append("\n")
    return result
