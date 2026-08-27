"""Session-management slash commands."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from orcha_agent.core.export import export_session
from orcha_agent.core.ledger import (
    AmbiguousEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    EntryNotFound,
    MessageEntry,
    ModeChangeEntry,
    ModelChangeEntry,
    OpaqueEntry,
    ResetBoundaryEntry,
)
from orcha_agent.core.plugin import PluginAPI, PluginSpec

PLUGIN = PluginSpec(name="commands_session", version="1.0.0")


def _require_no_args(ctx: Any, args: str, usage: str) -> bool:
    if args.split():
        ctx.console.error(f"Usage: {usage}")
        return False
    return True


def _model_text(model: object) -> str:
    if isinstance(model, list):
        return ", ".join(str(item) for item in model)
    return str(model)


def _semantic_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(
                    json.dumps(block, ensure_ascii=False, sort_keys=True, default=str)
                )
        return " ".join(parts)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _message_role(entry: MessageEntry) -> str:
    message_type = str(entry.message.get("type", "message")).lower()
    return {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }.get(message_type, message_type)


def _message_text(entry: MessageEntry) -> str:
    data = entry.message.get("data", {})
    if not isinstance(data, Mapping):
        return _semantic_text(data)
    return _semantic_text(data.get("content", ""))


def _marker(entry: Entry, child_count: int) -> str:
    markers: list[str] = []
    if child_count > 1:
        markers.append("⎇")
    if isinstance(entry, CompactionEntry):
        markers.append("⊟")
    if isinstance(entry, ResetBoundaryEntry):
        markers.append("⊠")
    return " ".join(markers)


def _default_label(
    entry: Entry, children: Mapping[str, list[Entry]]
) -> str:
    marker = _marker(entry, len(children.get(entry.id, ())))
    prefix = f"{marker} " if marker else ""
    if isinstance(entry, MessageEntry):
        reply_count = sum(
            isinstance(child, MessageEntry) and _message_role(child) == "assistant"
            for child in children.get(entry.id, ())
        )
        noun = "reply" if reply_count == 1 else "replies"
        return (
            f"{prefix}{entry.id} {_message_text(entry)[:60]} "
            f"({reply_count} assistant {noun})"
        )
    if isinstance(entry, CompactionEntry):
        return f"{prefix}{entry.id} {entry.summary}"
    return f"{prefix}{entry.id} reset boundary"


def _all_label(entry: Entry, children: Mapping[str, list[Entry]]) -> str:
    marker = _marker(entry, len(children.get(entry.id, ())))
    prefix = f"{marker} " if marker else ""
    if isinstance(entry, MessageEntry):
        payload = f"{_message_role(entry)}: {_message_text(entry)}"
    elif isinstance(entry, ModelChangeEntry):
        payload = f"model: {_model_text(entry.model)}"
    elif isinstance(entry, ModeChangeEntry):
        payload = f"mode: {entry.mode}"
    elif isinstance(entry, CompactionEntry):
        payload = f"compaction: {entry.summary}"
    elif isinstance(entry, ResetBoundaryEntry):
        payload = "reset boundary"
    elif isinstance(entry, CustomEntry):
        payload = f"{entry.custom_type}: {_semantic_text(entry.data)}"
    elif isinstance(entry, OpaqueEntry):
        payload = f"{entry.entry_type}: {_semantic_text(entry.payload)}"
    else:
        payload = type(entry).__name__
    return f"{prefix}{entry.id} {payload}"


async def _tree(ctx: Any, args: str) -> None:
    parts = args.split()
    if parts not in ([], ["--all"]):
        ctx.console.error("Usage: /tree [--all]")
        return

    show_all = parts == ["--all"]
    entries = ctx.ledger.all(ctx.session_id)
    by_id = {entry.id: entry for entry in entries}
    children: dict[str, list[Entry]] = {}
    for entry in entries:
        if entry.parent_id is not None:
            children.setdefault(entry.parent_id, []).append(entry)

    shown = (
        entries
        if show_all
        else [
            entry
            for entry in entries
            if (
                isinstance(entry, MessageEntry) and _message_role(entry) == "user"
            )
            or isinstance(entry, (CompactionEntry, ResetBoundaryEntry))
        ]
    )
    shown_ids = {entry.id for entry in shown}
    root = Tree(Text(f"Session {ctx.session_id}", style="bold cyan"))
    rendered_nodes: dict[str, Tree] = {}
    leaf_id = ctx.ledger.leaf(ctx.session_id)
    highlighted_id = leaf_id
    while highlighted_id is not None and highlighted_id not in shown_ids:
        highlighted = by_id.get(highlighted_id)
        highlighted_id = None if highlighted is None else highlighted.parent_id

    for entry in shown:
        parent_id = entry.parent_id
        while parent_id is not None and parent_id not in shown_ids:
            parent = by_id.get(parent_id)
            parent_id = None if parent is None else parent.parent_id
        parent_node = root if parent_id is None else rendered_nodes[parent_id]
        label = (
            _all_label(entry, children)
            if show_all
            else _default_label(entry, children)
        )
        style = "bold reverse" if entry.id == highlighted_id else None
        rendered_nodes[entry.id] = parent_node.add(Text(label, style=style))

    ctx.console.print(root)


async def _branch(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) != 1:
        ctx.console.error("Usage: /branch <id-prefix>")
        return
    prefix = parts[0]
    try:
        entry = ctx.ledger.resolve(ctx.session_id, prefix)
    except AmbiguousEntry as error:
        ctx.console.error(
            f"Ambiguous entry prefix '{prefix}': {', '.join(sorted(error.candidates))}"
        )
        return
    except EntryNotFound:
        ctx.console.error(f"Unknown entry ID: {prefix}")
        return
    await ctx.branch(entry.id)


async def _fork(ctx: Any, args: str) -> None:
    if _require_no_args(ctx, args, "/fork"):
        await ctx.fork()


async def _new(ctx: Any, args: str) -> None:
    if _require_no_args(ctx, args, "/new"):
        await ctx.new_session()


async def _clear(ctx: Any, args: str) -> None:
    if _require_no_args(ctx, args, "/clear"):
        await ctx.clear()


async def _sessions(ctx: Any, args: str) -> None:
    if not _require_no_args(ctx, args, "/sessions"):
        return
    table = Table(title="Sessions")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Model")
    table.add_column("Entries", justify="right")
    table.add_column("Working directory")
    table.add_column("Created")
    for session in ctx.session.list():
        table.add_row(
            session.thread_id,
            session.title or "",
            _model_text(session.model),
            str(ctx.ledger.count(session.thread_id)),
            session.cwd,
            session.created,
        )
    ctx.console.print(table)


async def _resume(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) != 1:
        ctx.console.error("Usage: /resume <session-id>")
        return
    await ctx.resume(parts[0])


async def _compact(ctx: Any, args: str) -> None:
    if not _require_no_args(ctx, args, "/compact"):
        return
    await ctx.compact()


async def _export(ctx: Any, args: str) -> None:
    parts = args.split()
    if len(parts) > 1:
        ctx.console.error("Usage: /export [path]")
        return
    path = Path(parts[0]) if parts else Path(f"./{ctx.session_id}.jsonl")
    output = export_session(ctx.session, ctx.session_id, path)
    ctx.console.print(f"Exported session to {output}")


def register(api: PluginAPI) -> None:
    api.add_command(
        "tree", _tree, help="Show the current session tree: /tree [--all]"
    )
    api.add_command(
        "branch", _branch, help="Branch from a ledger entry: /branch <id-prefix>"
    )
    api.add_command("fork", _fork, help="Fork the current session")
    api.add_command("new", _new, help="Start a new session")
    api.add_command("clear", _clear, help="Clear the current conversation")
    api.add_command("compact", _compact, help="Compact the current conversation")
    api.add_command("export", _export, help="Export the current session: /export [path]")
    api.add_command("sessions", _sessions, help="List saved sessions")
    api.add_command("resume", _resume, help="Resume a saved session: /resume <session-id>")
