"""Conservative exact-first replacement matching, inspired by pi-edit."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

from .common import split_text_lines

_FOLD = str.maketrans(
    {
        **dict.fromkeys("“”„‟«»", '"'),
        **dict.fromkeys("‘’‚‛`´", "'"),
        **dict.fromkeys("‐‑‒–—―−", "-"),
        "\u00a0": " ",
    }
)


def normalize(value: str) -> str:
    return re.sub(r"[^\S\n]+", " ", value.translate(_FOLD)).strip()


def _normalized_lines(lines: list[str]) -> list[str]:
    indents = [len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip()) for line in lines]
    nonempty = [depth for depth, line in zip(indents, lines) if line.strip()]
    base = min(nonempty, default=0)
    unit = min((depth - base for depth in nonempty if depth > base), default=1)
    return [
        f"{(depth - base) // unit if line.strip() else 0}|{normalize(line)}"
        for depth, line in zip(indents, lines)
    ]


def _fold_offsets(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Fold typography/space runs while retaining replacement span boundaries."""
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, char in enumerate(value):
        folded = char.translate(_FOLD)
        if folded.isspace() and folded not in "\r\n":
            folded = " "
            if chars and chars[-1] == " ":
                offsets[-1] = (offsets[-1][0], index + 1)
                continue
        chars.append(folded)
        offsets.append((index, index + 1))
    return "".join(chars), offsets


def find_matches(content: str, target: str) -> list[tuple[int, int]]:
    if not target:
        raise ValueError("old_string must not be empty")
    exact = [(match.start(), match.end()) for match in re.finditer(re.escape(target), content)]
    if exact:
        return exact
    # Single-line selections need not consume the surrounding source line.
    if "\n" not in target and "\r" not in target:
        folded, mapping = _fold_offsets(content)
        needle, _ = _fold_offsets(target)
        normalized_matches = [
            (mapping[match.start()][0], mapping[match.end() - 1][1])
            for match in re.finditer(re.escape(needle), folded)
        ]
        if normalized_matches:
            return normalized_matches
    lines = split_text_lines(content, keepends=True)
    target_lines = split_text_lines(target)
    normalized = _normalized_lines(target_lines)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    candidates: list[tuple[float, int, int]] = []
    for start in range(len(lines) - len(target_lines) + 1):
        window = lines[start : start + len(target_lines)]
        actual = _normalized_lines(window)
        score = sum(
            SequenceMatcher(None, a, b, autojunk=False).ratio() for a, b in zip(normalized, actual)
        ) / len(normalized)
        end = offsets[start + len(window)]
        if not target.endswith(("\n", "\r")):
            end -= 2 if window[-1].endswith("\r\n") else int(window[-1].endswith("\n"))
        candidates.append((score, offsets[start], end))
    accepted = sorted((c for c in candidates if c[0] >= 0.95), reverse=True)
    if len(accepted) > 1 and accepted[0][0] >= 0.97 and accepted[0][0] - accepted[1][0] >= 0.08:
        accepted = accepted[:1]
    return sorted((start, end) for _, start, end in accepted)


def replace(content: str, old: str, new: str, replace_all: bool = False) -> str:
    matches = find_matches(content, old)
    if not matches:
        first = normalize(split_text_lines(old)[0]) if split_text_lines(old) else ""
        candidates = sorted(
            (
                (SequenceMatcher(None, first, normalize(line), autojunk=False).ratio(), i, line)
                for i, line in enumerate(split_text_lines(content), 1)
            ),
            reverse=True,
        )[:3]
        preview = "\n".join(
            f"line {i}: {line[:160]} (similarity {score:.2f})" for score, i, line in candidates
        )
        raise ValueError(
            "Found 0 occurrences of old_string. Read the current file and supply more exact context."
            + (f" Candidate lines:\n{preview[:2000]}" if preview else "")
        )
    if len(matches) > 1 and not replace_all:
        previews = [
            f"line {content[:start].count(chr(10)) + 1}: {split_text_lines(content[start:end])[0][:160]}"
            for start, end in matches[:20]
        ]
        raise ValueError(
            f"Found {len(matches)} occurrences of old_string; expected exactly 1. Add surrounding context or use replace_all=true.\n"
            + "\n".join(previews)
        )
    if any(end > next_start for (_, end), (next_start, _) in zip(matches, matches[1:])):
        raise ValueError(
            "Matched edit ranges overlap; add unique context and edit them separately."
        )
    for start, end in reversed(matches):
        content = content[:start] + new + content[end:]
    return content
