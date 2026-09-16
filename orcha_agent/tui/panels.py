"""Shared, quiet chrome for read-only command summaries."""

from collections.abc import Iterable, Sequence

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def summary_panel(title: str, columns: Sequence[str], rows: Iterable[Sequence[str]]) -> Panel:
    table = Table(box=None, padding=(0, 1), collapse_padding=False, expand=True, show_edge=False)
    for index, name in enumerate(columns):
        table.add_column(
            name,
            style="cyan" if index == 0 else "dim",
            header_style="bold" if index == 0 else "dim",
        )
    for row in rows:
        table.add_row(*(Text(str(value)) for value in row))
    return Panel(
        table, title=title, title_align="left", box=box.ROUNDED, padding=(0, 1), border_style="dim"
    )


def table_panel(table: Table) -> Panel:
    title = table.title
    table.title = None
    table.box = None
    table.show_edge = False
    table.padding = (0, 1)
    table.collapse_padding = False
    return Panel(
        table, title=title, title_align="left", box=box.ROUNDED, padding=(0, 1), border_style="dim"
    )
