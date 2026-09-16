"""Selection colour fixture derived from the live picker style adapter."""

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
