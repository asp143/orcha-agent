"""Startup, diagnostics, and command panel visual examples."""

from orcha_agent.tui.blocks.banner import render
from orcha_agent.tui.frame import Block
from rich.table import Table

from orcha_agent.tui.panels import summary_panel, table_panel


def runtime_examples(theme, width):
    compact = Table(title="Provider column gutters", padding=0)
    for column in ("Available", "Auth / Keys", "T/S/R/O", "Status"):
        compact.add_column(column)
    compact.add_row("yes", "GEMINI_API_KEY: no", "T S R O", "ready")
    yield "Compact provider table", table_panel(compact)
    yield (
        "Startup skill summary",
        render(
            Block(
                "warnings",
                "banner",
                data={"level": "warning", "message": "6 skills skipped · Ctrl+O for details"},
            ),
            theme,
            width,
            20,
            False,
        ),
    )
    yield (
        "Repeated warning",
        render(
            Block(
                "repeated",
                "banner",
                data={"level": "warning", "message": "Connection interrupted", "count": 3},
            ),
            theme,
            width,
            20,
            False,
        ),
    )
    yield (
        "Pinned provider error",
        render(
            Block(
                "error",
                "banner",
                data={
                    "level": "error",
                    "pinned": True,
                    "message": "Codex usage limit reached (Pro plan) · resets in 2h 14m",
                    "error_type": "UsageLimitError",
                },
            ),
            theme,
            width,
            20,
            False,
        ),
    )
    yield (
        "Status command",
        summary_panel(
            "Status", ("Segment", "Value"), [("model", "codex:gpt-5.6-sol"), ("cost", "$0.00")]
        ),
    )
    yield (
        "Skills command",
        summary_panel(
            "Skills",
            ("Command", "Description"),
            [("/skill:review", "Review a patch"), ("/skill:test", "Run focused tests")],
        ),
    )
    yield (
        "MCP command",
        summary_panel(
            "MCP servers", ("Name", "Status", "Tools", "Details"), [("local", "connected", "3", "")]
        ),
    )

    yield (
        "Providers command",
        summary_panel(
            "Providers",
            ("Prefix", "Available", "Auth / Keys", "T/S/R/O", "Status"),
            [("codex", "yes", "logged in", "T S R O", "ready")],
        ),
    )
    yield (
        "Plugins command",
        summary_panel(
            "Plugins",
            ("Name", "Version", "Source", "Status"),
            [("skills", "1.0.0", "builtin", "loaded")],
        ),
    )
