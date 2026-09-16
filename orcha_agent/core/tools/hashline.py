"""Optional snapshot-tagged line edits, compatible with omp's core patch syntax."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from .common import atomic_write, resolve_path, split_text_lines
from .filesystem import FilesystemTools, _checked, _diff, _identity, parse_selector

_MASK = 0xFFFFFFFF
MAX_SNAPSHOTS = 32
SNAPSHOT_BYTE_BUDGET = 32 * 1024 * 1024


def _rotate(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (32 - bits))) & _MASK


def file_hash(text: str) -> str:
    """omp's normalized XXH32 low 16 bits; a label, never the safety check itself."""
    normalized = re.sub(r"[ \t\r]+(?=\n|$)", "", text)
    data = normalized.encode()
    p1, p2, p3, p4, p5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
    index = 0
    if len(data) >= 16:
        accumulators = [(p1 + p2) & _MASK, p2, 0, -p1 & _MASK]
        while index <= len(data) - 16:
            for lane in range(4):
                word = int.from_bytes(data[index : index + 4], "little")
                accumulators[lane] = (
                    _rotate((accumulators[lane] + word * p2) & _MASK, 13) * p1
                ) & _MASK
                index += 4
        value = (
            sum(_rotate(number, shift) for number, shift in zip(accumulators, [1, 7, 12, 18]))
            & _MASK
        )
    else:
        value = p5
    value = (value + len(data)) & _MASK
    while index <= len(data) - 4:
        value = (
            _rotate((value + int.from_bytes(data[index : index + 4], "little") * p3) & _MASK, 17)
            * p4
        ) & _MASK
        index += 4
    for byte in data[index:]:
        value = (_rotate((value + byte * p5) & _MASK, 11) * p1) & _MASK
    value ^= value >> 15
    value = value * p2 & _MASK
    value ^= value >> 13
    value = value * p3 & _MASK
    value ^= value >> 16
    return f"{value & 0xFFFF:04X}"


@dataclass
class Snapshot:
    content: str
    digest: str
    seen: set[int]


@dataclass
class Hunk:
    start: int  # zero-based, inclusive
    end: int  # exclusive; equal to start for insertions
    body: list[str]
    anchors: list[int]  # one-based snapshot anchors


@dataclass
class Section:
    path: str
    tag: str
    rows: list[str]


def _sections(patch: str) -> list[Section]:
    sections: list[Section] = []
    for row in patch.replace("\r\n", "\n").split("\n"):
        if row in ("*** Begin Patch", "*** End Patch") or not row.strip():
            continue
        header = re.fullmatch(r"\[(.+)#([0-9A-Fa-f]{4})\]", row)
        if header:
            sections.append(Section(header[1], header[2].upper(), []))
        elif sections:
            sections[-1].rows.append(row)
        else:
            raise ValueError("Start each section with a [path#TAG] header copied from read.")
    if not sections:
        raise ValueError("Patch must contain at least one [path#TAG] section.")
    return sections


def _hunks(rows: list[str], total: int) -> tuple[list[Hunk], str | None]:
    hunks: list[Hunk] = []
    operation = None
    index = 0
    while index < len(rows):
        row = rows[index]
        index += 1
        if row == "REM" or row.startswith("MV "):
            if operation is not None:
                raise ValueError("Only one MV or REM operation is allowed per file.")
            operation = row
            continue
        match = re.fullmatch(r"(PUT|CUT) ([<>]?(?:[1-9]\d*|\$)(?:\.=[1-9]\d*)?)(:?)", row)
        if not match:
            raise ValueError(
                f"Invalid operation {row!r}; use PUT N.=M:, CUT N.=M, MV path, or REM."
            )
        kind, locator, colon = match.groups()
        if kind == "PUT" and not colon:
            raise ValueError("PUT requires ':' followed by +TEXT payload rows.")
        gap = locator[0] in "<>"
        if gap:
            if kind == "CUT" or ".=" in locator:
                raise ValueError("Only PUT supports before/after gap anchors.")
            anchor = total if locator[1:] == "$" else int(locator[1:])
            if locator not in (">$", "<1") and not 1 <= anchor <= total:
                raise ValueError(f"Line {anchor} does not exist (file has {total} lines).")
            start = anchor if locator[0] == ">" else anchor - 1
            end = start
            anchors = [] if locator in (">$", "<1") else [anchor]
        else:
            if "$" in locator:
                raise ValueError("Use >$ to append at end of file.")
            endpoints = locator.split(".=")
            first, last = int(endpoints[0]), int(endpoints[-1])
            if not 1 <= first <= last <= total:
                raise ValueError(
                    f"Invalid range {locator}; endpoints are absolute lines (file has {total})."
                )
            start, end = first - 1, last
            anchors = list(range(first, last + 1))
        body: list[str] = []
        if kind == "PUT":
            while index < len(rows) and rows[index].startswith("+"):
                body.append(rows[index][1:])
                index += 1
            if not body:
                raise ValueError("PUT requires +TEXT payload rows; use CUT to delete lines.")
        hunks.append(Hunk(start, end, body, anchors))
    if operation == "REM" and hunks:
        raise ValueError("REM cannot be combined with line edits.")
    if not hunks and operation is None:
        raise ValueError("Section contains no operations.")
    return hunks, operation


def _apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """Replace only addressed byte spans; untouched line terminators stay exact."""
    if not hunks:
        return original
    rows = split_text_lines(original, keepends=True)

    def ending(row: str) -> str:
        return "\r\n" if row.endswith("\r\n") else "\n" if row.endswith("\n") else ""

    fallback = next((ending(row) for row in rows if ending(row)), "\n")
    for hunk in reversed(hunks):
        selected = rows[hunk.start : hunk.end]
        nearby = selected or rows[max(0, hunk.start - 1) : hunk.start + 1]
        eol = next((ending(row) for row in nearby if ending(row)), fallback)
        terminal = (
            ending(selected[-1])
            if selected
            else (eol if hunk.start < len(rows) or original.endswith("\n") else "")
        )
        replacement = [line + eol for line in hunk.body]
        if replacement:
            replacement[-1] = hunk.body[-1] + terminal
            if hunk.start == len(rows) and rows and not ending(rows[-1]):
                rows[-1] += eol
        rows[hunk.start : hunk.end] = replacement
    return "".join(rows)


class HashlineTools:
    def __init__(self, service: FilesystemTools):
        self.service = service
        self.snapshots: dict[tuple[str, Path, str], Snapshot] = {}

    def _remember(self, key: tuple[str, Path, str], snapshot: Snapshot) -> None:
        self.snapshots.pop(key, None)
        self.snapshots[key] = snapshot
        size = sum(len(item.content.encode()) for item in self.snapshots.values())
        # Keep the current file usable even when it alone exceeds the aggregate budget.
        while len(self.snapshots) > MAX_SNAPSHOTS or (
            size > SNAPSHOT_BYTE_BUDGET and len(self.snapshots) > 1
        ):
            oldest = next(iter(self.snapshots))
            size -= len(self.snapshots.pop(oldest).content.encode())

    def read(self, path: str, *, thread: str = "default", turn: str = "default") -> str:
        try:
            bare, _, _ = parse_selector(path, 0)
            target = resolve_path(self.service.cwd, bare)
            if not any(key[0] == thread and key[1] == target for key in self.snapshots):
                # An expired snapshot must be recoverable even in the same turn.
                self.service.repeats = {
                    key: value
                    for key, value in self.service.repeats.items()
                    if not (key[0] == thread and key[2].startswith(str(target)))
                }
        except ValueError:
            pass
        result = self.service.read(path, thread=thread, turn=turn)
        if result.startswith(("Error:", "Image file:", "Binary file:")):
            return result
        try:
            bare, _, raw = parse_selector(path, 0)
            target = resolve_path(self.service.cwd, bare)
            if not target.is_file():
                return result
            data = target.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if self.service.reads.get((thread, target)) != digest:
                return "Error: File changed during read. Read again before editing."
            content = data.decode("utf-8")
            tag = file_hash(content)
            seen: set[int] = set()
            output: list[str] = []
            for row in result.split("\n"):
                match = re.match(r"^\s*(\d+)  (.*)$", row) if not raw else None
                if match:
                    number = int(match[1])
                    actual_lines = split_text_lines(content)
                    if number <= len(actual_lines) and match[2] == actual_lines[number - 1]:
                        seen.add(number)
                    output.append(f"{match[1]}:{match[2]}")
                else:
                    output.append(row)
            previous = self.snapshots.get((thread, target, tag))
            if previous and previous.digest == digest:
                seen |= previous.seen
            self._remember((thread, target, tag), Snapshot(content, digest, seen))
            return f"[{bare}#{tag}]\n" + "\n".join(output)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"

    def _recover(self, snapshot: Snapshot, current: str, hunks: list[Hunk]) -> list[Hunk] | None:
        old, new = split_text_lines(snapshot.content), split_text_lines(current)
        mapping: dict[int, int] = {}
        for block in SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
            mapping.update(
                (block.a + offset + 1, block.b + offset + 1) for offset in range(block.size)
            )
        recovered: list[Hunk] = []
        for hunk in hunks:
            # EOF/BOF has no content anchor; external changes make its intent uncertain.
            if not hunk.anchors:
                return None
            for anchor in hunk.anchors:
                line = old[anchor - 1]
                if anchor not in mapping or old.count(line) != 1 or new.count(line) != 1:
                    return None
            anchors = [mapping[anchor] for anchor in hunk.anchors]
            if anchors != list(range(anchors[0], anchors[-1] + 1)):
                return None
            offset = anchors[0] - hunk.anchors[0]
            recovered.append(Hunk(hunk.start + offset, hunk.end + offset, hunk.body, anchors))
        return recovered

    def edit(self, patch: str, *, thread: str = "default") -> str:
        try:
            plans: list[tuple[Path, Path | None, str, str, str, bool, set[int]]] = []
            touched: set[Path] = set()
            for section in _sections(patch):
                target = resolve_path(self.service.cwd, section.path)
                if target in touched:
                    raise ValueError(
                        "Use one section per source path; duplicate/overlapping paths are not supported."
                    )
                data = target.read_bytes()
                original = data.decode("utf-8")
                digest = hashlib.sha256(data).hexdigest()
                current_tag = file_hash(original)
                snapshot = self.snapshots.get((thread, target, section.tag))
                hunks, operation = _hunks(
                    section.rows, len(split_text_lines(snapshot.content if snapshot else original))
                )
                anchors = sorted({line for hunk in hunks for line in hunk.anchors})
                recovered = False
                if snapshot is None or snapshot.digest != digest:
                    remapped = (
                        self._recover(snapshot, original, hunks)
                        if snapshot and operation is None
                        else None
                    )
                    if remapped is None:
                        lines = split_text_lines(original)
                        context = sorted(
                            {
                                n
                                for anchor in anchors
                                for n in range(max(1, anchor - 2), min(len(lines), anchor + 2) + 1)
                            }
                        )[:40]
                        reason = (
                            "file changed between read and edit"
                            if snapshot
                            else "tag is not from this session or has expired from the snapshot cache"
                        )
                        raise ValueError(
                            f"Edit rejected for {section.path}: {reason}. Current [{section.path}#{current_tag}]. "
                            f"Re-read {section.path} with read to refresh the snapshot; never invent a tag.\n"
                            + "\n".join(f"{n}:{lines[n - 1]}" for n in context)
                        )
                    hunks = remapped
                    recovered = True
                assert snapshot is not None
                unseen = set(anchors) - snapshot.seen
                if unseen:
                    numbers = ",".join(str(n) for n in sorted(unseen)[:20])
                    raise ValueError(
                        f"Snapshot never displayed lines {numbers}; read {section.path}:{numbers} before editing."
                    )
                ordered = sorted(hunks, key=lambda item: (item.start, item.end))
                for previous, following in zip(ordered, ordered[1:]):
                    if following.start < previous.end or following.start == previous.start:
                        raise ValueError(
                            "Overlapping edits are ambiguous; use one PUT for each source range."
                        )
                updated = _apply_hunks(original, ordered)
                destination = target
                if operation == "REM":
                    destination = None
                elif operation and operation.startswith("MV "):
                    destination = resolve_path(self.service.cwd, operation[3:].strip())
                    if destination.exists() or destination in touched:
                        raise ValueError(
                            f"Move destination already exists or overlaps another section: {destination}"
                        )
                touched.add(target)
                if destination is not None:
                    touched.add(destination)
                new_seen: set[int] = set()
                for block in SequenceMatcher(
                    None,
                    split_text_lines(snapshot.content),
                    split_text_lines(updated),
                    autojunk=False,
                ).get_matching_blocks():
                    new_seen.update(
                        block.b + offset + 1
                        for offset in range(block.size)
                        if block.a + offset + 1 in snapshot.seen
                    )
                for kind, _, _, start, end in SequenceMatcher(
                    None, split_text_lines(original), split_text_lines(updated), autojunk=False
                ).get_opcodes():
                    if kind in ("replace", "insert"):
                        new_seen.update(range(start + 1, end + 1))
                plans.append(
                    (target, destination, original, updated, section.path, recovered, new_seen)
                )
            # All syntax, hashes, paths and range overlap checks finish before the first write.
            # Each write is atomic; rollback restores originals if a later filesystem action fails.
            applied: list[tuple[Path, Path | None, str, int]] = []
            try:
                for target, destination, original, updated, _, _, _ in plans:
                    mode = target.stat().st_mode
                    applied.append((target, destination, original, mode))
                    if destination is not None:
                        atomic_write(destination, updated)
                        destination.chmod(mode)
                    if destination != target:
                        target.unlink()
            except OSError:
                for target, destination, original, mode in reversed(applied):
                    atomic_write(target, original)
                    target.chmod(mode)
                    if destination is not None and destination != target and destination.exists():
                        destination.unlink()
                raise
            output: list[str] = []
            for target, destination, original, updated, label, recovered, new_seen in plans:
                if recovered:
                    output.append(
                        "Recovered stale snapshot: uniquely unchanged source lines remapped to current positions."
                    )
                self.service.reads.pop((thread, target), None)
                if destination is None:
                    output.append(f"Removed {label}.\n" + _diff(label, original, ""))
                    continue
                digest = hashlib.sha256(updated.encode()).hexdigest()
                tag = file_hash(updated)
                self.service.reads[(thread, destination)] = digest
                self._remember((thread, destination, tag), Snapshot(updated, digest, new_seen))
                shown = label if destination == target else str(destination)
                output.append(f"[{shown}#{tag}]\n" + _diff(shown, original, updated))
            return "\n".join(output)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: {exc}"


def create_hashline_tools(service: FilesystemTools) -> list[BaseTool]:
    implementation = HashlineTools(service)

    @tool
    def read(path: str, runtime: ToolRuntime) -> str:
        """Read numbered text with [path#TAG] snapshot headers for hashline edit. Select ranges with file:N-M, file:N+K, file:-N or comma-separated ranges. Copy tags exactly; read every line before editing it. Repeated ranges in one turn return unchanged notices."""
        thread, turn = _identity(runtime)
        return _checked(implementation.read(path, thread=thread, turn=turn))

    @tool
    def edit(patch: str, runtime: ToolRuntime) -> str:
        """Edit snapshots using [path#TAG] sections copied from read. PUT N.=M: replaces inclusive source lines; payload rows are +TEXT. PUT <N: or PUT >N: inserts before/after N; PUT >$: appends. CUT N.=M deletes lines. MV path moves a file; REM deletes it. All anchors use original snapshot positions. Example: [a.py#ABCD]\nPUT 2:\n+replacement. Rejects unknown tags and changed/ambiguous anchors; safe unique unchanged anchors can relocate. Returns diffs and fresh tags."""
        thread, _ = _identity(runtime)
        return _checked(implementation.edit(patch, thread=thread))

    for native_tool in [read, edit]:
        native_tool.handle_tool_error = True
    return [read, edit]
