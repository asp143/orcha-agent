"""Code review transcript card renderer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from rich import box
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_symbol, theme_value

_PRIORITIES = ("P0", "P1", "P2", "P3")
_PRIORITY_ORDER = {priority: index for index, priority in enumerate(_PRIORITIES)}
_PRIORITY_TOKEN = {
    "P0": "error",
    "P1": "error",
    "P2": "warning",
    "P3": "dim",
}


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _priority(finding: Mapping[str, Any]) -> str:
    priority = str(finding.get("priority", "P3")).upper()
    return priority if priority in _PRIORITY_ORDER else "P3"


def _findings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    findings = [item for item in value if isinstance(item, Mapping)]
    return sorted(
        findings,
        key=lambda item: (
            _PRIORITY_ORDER[_priority(item)],
            str(item.get("file", "")).casefold(),
            item.get("line_start", 0)
            if type(item.get("line_start")) is int
            else 0,
            _one_line(item.get("title")).casefold(),
        ),
    )


def _location(finding: Mapping[str, Any]) -> str:
    path = str(finding.get("file") or "(unknown file)")
    start = finding.get("line_start")
    end = finding.get("line_end")
    if type(start) is not int:
        return path
    if type(end) is int and end != start:
        return f"{path}:{start}-{end}"
    return f"{path}:{start}"


def _confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not isfinite(confidence):
        return "unknown"
    if 0.0 <= confidence <= 1.0:
        return f"{confidence:.0%}"
    return f"{confidence:g}"


def _clip(line: Text, width: int) -> Text:
    clipped = line.copy()
    clipped.truncate(max(1, width), overflow="ellipsis")
    return clipped


def _finding_rows(findings: Sequence[Mapping[str, Any]], theme: Any) -> list[Text]:
    rows: list[Text] = []
    text_style = Style(color=str(theme_value(theme, "text")))
    muted_style = Style(color=str(theme_value(theme, "muted")), dim=True)

    for priority in _PRIORITIES:
        grouped = [finding for finding in findings if _priority(finding) == priority]
        if not grouped:
            continue
        token = _PRIORITY_TOKEN[priority]
        priority_style = Style(
            color=str(theme_value(theme, token)),
            bold=priority != "P3",
            dim=priority == "P3",
        )
        count = len(grouped)
        rows.append(
            Text(
                f"{priority} · {count} finding{'s' if count != 1 else ''}",
                style=priority_style,
            )
        )
        for finding in grouped:
            title = Text("  ")
            title.append(_one_line(finding.get("title")) or "Untitled finding", priority_style)
            rows.append(title)

            location = Text("    ")
            location.append(_location(finding), muted_style)
            location.append(
                f" · confidence {_confidence(finding.get('confidence'))}",
                muted_style,
            )
            rows.append(location)

            body = Text("    ")
            body.append(
                _one_line(finding.get("body")) or "No details provided.",
                muted_style if priority == "P3" else text_style,
            )
            rows.append(body)
    return rows


def _verdict_rows(block: Block, theme: Any) -> tuple[list[Text], str, Style]:
    overall = str(block.data.get("overall", "incorrect")).casefold()
    correct = overall == "correct"
    label = "Correct" if correct else "Incorrect"
    token = "success" if correct else "error"
    verdict_style = Style(color=str(theme_value(theme, token)), bold=True)
    explanation_style = Style(color=str(theme_value(theme, "text")))

    verdict = Text("Verdict · ")
    verdict.append(label, verdict_style)
    explanation = Text(_one_line(block.data.get("explanation")) or "No explanation provided.")
    explanation.stylize(explanation_style)
    return [verdict, explanation], label, verdict_style


def render(
    block: Block,
    theme: Any,
    width: int,
    budget_rows: int,
    expanded: bool,
) -> Panel | Text:
    """Render grouped findings and the review's overall verdict."""

    del expanded
    findings = _findings(block.data.get("findings"))
    finding_rows = _finding_rows(findings, theme)
    verdict_rows, verdict_label, verdict_style = _verdict_rows(block, theme)
    content_width = max(1, width - 4)

    if budget_rows <= 2 or width <= 4:
        summary = Text(f"Review · {verdict_label}: ", style=verdict_style)
        summary.append(_one_line(block.data.get("explanation")))
        return _clip(summary, max(1, width))

    capacity = max(1, budget_rows - 2)
    if finding_rows:
        rows = [*finding_rows, Text(), *verdict_rows]
    else:
        rows = verdict_rows

    if len(rows) > capacity:
        if capacity == 1:
            combined = Text(f"Verdict · {verdict_label}: ", style=verdict_style)
            combined.append(_one_line(block.data.get("explanation")))
            rows = [combined]
        else:
            available = capacity - len(verdict_rows)
            if available <= 0:
                rows = verdict_rows[:capacity]
            elif available == 1:
                rows = [Text(f"… {len(findings)} findings omitted"), *verdict_rows]
            else:
                rows = [
                    *finding_rows[: available - 1],
                    Text("… additional findings omitted", style="dim"),
                    *verdict_rows,
                ]

    content = Text("\n").join(_clip(row, content_width) for row in rows)
    return Panel(
        content,
        title=Text(f"Review · {verdict_label}", style=verdict_style),
        title_align="left",
        border_style=verdict_style,
        box=theme_symbol(theme, "boxRound", box.ROUNDED),
        padding=(0, 1),
    )


__all__ = ["render"]
