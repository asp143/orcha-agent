"""Standalone, script-free HTML transcripts from the durable session ledger."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from rich.color import Color, ColorParseError

from .ledger import CompactionEntry, Ledger, MessageEntry
from .session import SessionStore


def _color(theme: Any, token: str, fallback: str) -> str:
    colors = theme.get("colors", {}) if isinstance(theme, Mapping) else getattr(theme, "colors", {})
    value = colors.get(token, "")
    try:
        color = Color.parse(value)
        return fallback if color.is_default else color.get_truecolor().hex
    except (ColorParseError, ValueError, TypeError):
        return fallback


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, Mapping))
        )
    return json.dumps(content, ensure_ascii=False, default=str)


def _tool_output(text: str) -> str:
    lines = []
    for line in text.splitlines():
        kind = "added" if line.startswith("+") else "removed" if line.startswith("-") else "context"
        lines.append(f'<span class="{kind}">{escape(line)}</span>')
    return '<pre class="tool-output">' + "\n".join(lines) + "</pre>"


def render_session_html(store: SessionStore, session_id: str, *, theme: Any = None) -> str:
    """Render all ledger branches in chronological order, without remote assets."""
    session = store.get(session_id)
    if session is None:
        raise LookupError(f"Session not found: {session_id}")
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    # Exported transcripts never load external images, including tracking pixels.
    markdown.disable("image")
    cards: list[str] = []
    for entry in Ledger(store).all(session_id):
        metadata = f'id="{escape(entry.id, quote=True)}" data-parent="{escape(entry.parent_id or "", quote=True)}"'
        if isinstance(entry, CompactionEntry):
            cards.append(
                f"<details {metadata}><summary>Compaction</summary>{markdown.render(entry.summary)}</details>"
            )
            continue
        if not isinstance(entry, MessageEntry):
            continue
        data = entry.message.get("data", {})
        if not isinstance(data, Mapping):
            continue
        role = str(entry.message.get("type", "message"))
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        content = _text(data.get("content", ""))
        if role == "tool":
            label = escape(str(data.get("name") or "Tool result"))
            cards.append(
                f'<details class="tool" {metadata}><summary>{label}</summary>{_tool_output(content)}</details>'
            )
        else:
            cards.append(
                f'<article class="{escape(role, quote=True)}" {metadata}><h2>{escape(role.title())}</h2>{markdown.render(content)}</article>'
            )
        for call in data.get("tool_calls", []):
            if not isinstance(call, Mapping):
                continue
            name = escape(str(call.get("name", "Tool call")))
            arguments = escape(json.dumps(call.get("args", {}), ensure_ascii=False, indent=2))
            cards.append(
                f'<details class="tool"><summary>{name}</summary><pre>{arguments}</pre></details>'
            )
    variables = {
        "text": _color(theme, "text", "#e6e6e6"),
        "bg": _color(theme, "statusLineBg", "#16181d"),
        "card": _color(theme, "userMessageBg", "#242730"),
        "accent": _color(theme, "accent", "#88c0d0"),
        "border": _color(theme, "border", "#454955"),
        "added": _color(theme, "toolDiffAdded", "#a3be8c"),
        "removed": _color(theme, "toolDiffRemoved", "#bf616a"),
    }
    css = ":root{" + ";".join(f"--{key}:{value}" for key, value in variables.items()) + "}"
    css += """
body{background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif;
max-width:1000px;margin:2rem auto;padding:0 1rem}h1,h2,summary,a{color:var(--accent)}
h2{font-size:1rem;text-transform:capitalize}article,details{border:1px solid var(--border);
border-radius:8px;padding:1rem;margin:1rem 0}.user{background:var(--card)}
summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere}
code{font-family:ui-monospace,monospace}table{border-collapse:collapse}
td,th{border:1px solid var(--border);padding:.4rem}.added{color:var(--added)}
.removed{color:var(--removed)}blockquote{border-left:3px solid var(--accent);padding-left:1rem}
@media print{details{break-inside:avoid}body{max-width:none}}
"""
    title = escape(session.title or f"Session {session_id}")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; style-src &#39;unsafe-inline&#39;">'
        f"<title>{title}</title><style>{css}</style></head><body><header><h1>{title}</h1>"
        f"<p>{escape(session.cwd)} · {escape(session.created)}</p></header><main>"
        + "\n".join(cards)
        + "</main></body></html>\n"
    )


def export_session_html(
    store: SessionStore,
    session_id: str,
    path: str | Path,
    *,
    theme: Any = None,
    force: bool = False,
) -> Path:
    text = render_session_html(store, session_id, theme=theme)
    output = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_TRUNC if force else os.O_EXCL)
    descriptor = os.open(output, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(text)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return output.resolve()
