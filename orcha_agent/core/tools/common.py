from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import tempfile


def notice(message: str, **metadata: object) -> str:
    """Machine-readable metadata with a human-readable recovery instruction."""
    return "[notice] " + json.dumps({"message": message, **metadata}, ensure_ascii=False)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~])", "", text)


def resolve_path(cwd: Path, path: str) -> Path:
    result = Path(path).expanduser()
    return (result if result.is_absolute() else cwd / result).resolve()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def split_text_lines(content: str, *, keepends: bool = False) -> list[str]:
    """Address source lines by LF; strip CR only when it belongs to CRLF."""
    if not content:
        return []
    parts = content.split("\n")
    lines = [part + "\n" if keepends else part.removesuffix("\r") for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines
