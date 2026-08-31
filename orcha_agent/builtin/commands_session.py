"""Session-management slash commands."""

from __future__ import annotations

import errno
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from orcha_agent.core.export import export_session
from orcha_agent.core.memory_store import (
    CredentialContentError,
    MemoryConflictError,
    MemoryDocument,
    MemoryScope,
)
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
    if not parts:
        ui = getattr(ctx, "ui", None)
        if ui is not None and hasattr(ui, "show"):
            try:
                await ui.show("tree")
                return
            except RuntimeError:
                pass

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
    exact = bool(parts and parts[0] == "--exact")
    if (exact and len(parts) != 2) or (not exact and len(parts) != 1):
        ctx.console.error("Usage: /branch [--exact] <id-prefix>")
        return
    prefix = parts[-1]
    if prefix.startswith("-"):
        ctx.console.error("Usage: /branch [--exact] <id-prefix>")
        return
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

    if (
        not exact
        and isinstance(entry, MessageEntry)
        and _message_role(entry) == "user"
    ):
        owner_by_id: dict[str, str | None] = {}
        reply = None
        for candidate in ctx.ledger.all(ctx.session_id):
            role = (
                _message_role(candidate)
                if isinstance(candidate, MessageEntry)
                else None
            )
            if role == "user":
                owner = candidate.id
            else:
                owner = (
                    owner_by_id.get(candidate.parent_id)
                    if candidate.parent_id is not None
                    else None
                )
            owner_by_id[candidate.id] = owner
            if owner == entry.id and role == "assistant":
                reply = candidate
        if reply is None:
            ctx.console.error(f"No assistant reply found for user entry: {prefix}")
            return
        entry = reply
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
    ui = getattr(ctx, "ui", None)
    if ui is not None and hasattr(ui, "show"):
        try:
            await ui.show("session")
            return
        except RuntimeError:
            pass
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
    if not parts:
        ui = getattr(ctx, "ui", None)
        if ui is not None and hasattr(ui, "show"):
            try:
                await ui.show("session")
                return
            except RuntimeError:
                pass
        ctx.console.error("Usage: /resume <session-id>")
        return
    if len(parts) != 1:
        ctx.console.error("Usage: /resume <session-id>")
        return
    await ctx.resume(parts[0])


async def _compact(ctx: Any, args: str) -> None:
    if not _require_no_args(ctx, args, "/compact"):
        return
    await ctx.compact()


async def _export(ctx: Any, args: str) -> None:
    raw = args.strip()
    first_and_path = raw.split(maxsplit=1)
    force = bool(first_and_path and first_and_path[0] == "--force")
    if force:
        raw = first_and_path[1].strip() if len(first_and_path) == 2 else ""
    tokens = raw.split()
    if (tokens and tokens[0].startswith("-")) or "--force" in tokens:
        ctx.console.error("Usage: /export [--force] [path]")
        return

    path = (
        Path(raw).expanduser()
        if raw
        else Path.cwd() / f"{ctx.session_id}.jsonl"
    )
    try:
        output = export_session(
            ctx.session, ctx.session_id, path, force=force
        ).resolve()
    except FileExistsError:
        ctx.console.error(
            f"Export destination already exists: {path}. "
            "Use --force to replace it."
        )
        return
    except OSError as error:
        if error.errno == errno.ELOOP:
            ctx.console.error(f"Refusing to export through symlink: {path}")
        else:
            ctx.console.error(f"Unable to export session to {path}: {error}")
        return
    ctx.console.print(f"Exported session to {output}")


def _structured_memory(ctx: Any) -> Any | None:
    store = getattr(getattr(ctx, "session", None), "structured_memory", None)
    if store is None:
        ctx.console.error(
            "Structured memory is unavailable. Enable the opt-in Turso memory store."
        )
    return store


def _memory_workspace(ctx: Any) -> str | None:
    settings = getattr(getattr(ctx, "cfg", None), "memory_store", None)
    value = getattr(settings, "workspace", None)
    return value if isinstance(value, str) and value else None


async def _sync(ctx: Any, args: str) -> None:
    if not _require_no_args(ctx, args, "/sync"):
        return
    store = getattr(ctx, "session", None)
    if store is None or not bool(getattr(store, "supports_sync", False)):
        ctx.console.error("Sync is only available with the Turso persistence backend.")
        return
    try:
        store.sync()
    except Exception as exc:
        ctx.console.error(str(exc))
        return
    ctx.console.print("Turso sync complete (sessions and structured memories).")


async def _memory(ctx: Any, args: str) -> None:
    store = _structured_memory(ctx)
    if store is None:
        return
    parts = args.strip().split(maxsplit=1)
    action = parts[0] if parts else "list"
    remainder = parts[1] if len(parts) == 2 else ""
    workspace = _memory_workspace(ctx)

    if action == "list":
        if remainder:
            ctx.console.error("Usage: /memory list")
            return
        table = Table(title="Structured memories")
        table.add_column("ID", style="cyan")
        table.add_column("Scope")
        table.add_column("Path")
        table.add_column("Revision", justify="right")
        for document in store.all():
            if document.scope is not MemoryScope.GLOBAL and document.workspace != workspace:
                continue
            table.add_row(
                document.id,
                str(document.scope),
                str(document.path or ""),
                str(document.revision),
            )
        ctx.console.print(table)
        return

    if action == "show":
        name = remainder.strip()
        if not name:
            ctx.console.error("Usage: /memory show <id>")
            return
        matches = [
            document
            for document in store.all()
            if document.id == name
            and (
                document.scope is MemoryScope.GLOBAL
                or document.workspace == workspace
            )
        ]
        if not matches:
            ctx.console.error(f"Unknown memory: {name}")
            return
        for document in matches:
            location = str(document.scope)
            if document.path is not None:
                location += f":{document.path}"
            ctx.console.print(
                f"[{location} revision {document.revision}] {document.id}\n"
                f"{document.content}"
            )
        return

    if action == "set":
        fields = remainder.split(maxsplit=2)
        if len(fields) != 3 or fields[0] not in {"global", "workspace"}:
            ctx.console.error(
                "Usage: /memory set global|workspace <id> <content>"
            )
            return
        scope_name, name, content = fields
        if scope_name == "global":
            document = MemoryDocument.global_document(name, content)
        else:
            if workspace is None:
                ctx.console.error("A [memory_store] workspace is required.")
                return
            document = MemoryDocument.workspace_document(name, content, workspace)
        current = store.get(
            name,
            scope=document.scope,
            workspace=document.workspace,
            path=document.path,
            include_deleted=True,
        )
        expected = 0 if current is None else current.revision
        try:
            saved = store.save(document, expected_revision=expected)
        except (CredentialContentError, MemoryConflictError, ValueError) as exc:
            ctx.console.error(str(exc))
            return
        rebuild = getattr(ctx, "request_rebuild", None)
        if callable(rebuild):
            rebuild()
        ctx.console.print(f"Saved memory {saved.id} at revision {saved.revision}.")
        return

    if action == "set-path":
        fields = remainder.split(maxsplit=2)
        if len(fields) != 3:
            ctx.console.error("Usage: /memory set-path <path> <id> <content>")
            return
        scope_path, name, content = fields
        if workspace is None:
            ctx.console.error("A [memory_store] workspace is required.")
            return
        document = MemoryDocument.path_document(name, content, workspace, scope_path)
        current = store.get(
            name,
            scope=document.scope,
            workspace=workspace,
            path=scope_path,
            include_deleted=True,
        )
        expected = 0 if current is None else current.revision
        try:
            saved = store.save(document, expected_revision=expected)
        except (CredentialContentError, MemoryConflictError, ValueError) as exc:
            ctx.console.error(str(exc))
            return
        rebuild = getattr(ctx, "request_rebuild", None)
        if callable(rebuild):
            rebuild()
        ctx.console.print(f"Saved memory {saved.id} at revision {saved.revision}.")
        return

    if action == "delete":
        name = remainder.strip()
        if not name:
            ctx.console.error("Usage: /memory delete <id>")
            return
        matches = [
            document
            for document in store.all()
            if document.id == name
            and (
                document.scope is MemoryScope.GLOBAL
                or document.workspace == workspace
            )
        ]
        if not matches:
            ctx.console.error(f"Unknown memory: {name}")
            return
        if len(matches) != 1:
            ctx.console.error(
                f"Memory {name!r} exists at multiple scopes; delete is ambiguous."
            )
            return
        try:
            deleted = store.delete(matches[0])
        except MemoryConflictError as exc:
            ctx.console.error(str(exc))
            return
        rebuild = getattr(ctx, "request_rebuild", None)
        if callable(rebuild):
            rebuild()
        ctx.console.print(f"Deleted memory {deleted.id} at revision {deleted.revision}.")
        return

    ctx.console.error(
        "Usage: /memory [list|show <id>|set global|workspace <id> <content>|"
        "set-path <path> <id> <content>|delete <id>]"
    )


def register(api: PluginAPI) -> None:
    api.add_command(
        "tree", _tree, help="Show the current session tree: /tree [--all]"
    )
    api.add_command(
        "branch",
        _branch,
        help="Branch from a ledger entry: /branch [--exact] <id-prefix>",
    )
    api.add_command("fork", _fork, help="Fork the current session")
    api.add_command("new", _new, help="Start a new session")
    api.add_command("clear", _clear, help="Clear the current conversation")
    api.add_command("compact", _compact, help="Compact the current conversation")
    api.add_command(
        "export",
        _export,
        help="Export the current session: /export [--force] [path]",
    )
    api.add_command("sessions", _sessions, help="List saved sessions")
    api.add_command("resume", _resume, help="Resume a saved session: /resume <session-id>")
    api.add_command("sync", _sync, help="Synchronize the configured Turso replica")
    api.add_command(
        "memory",
        _memory,
        help="Manage structured memories: /memory [list|show|set|set-path|delete]",
    )
