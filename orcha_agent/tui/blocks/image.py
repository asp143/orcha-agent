"""Safe inline image transport with a useful fallback on ordinary terminals."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from typing import Any

from rich.text import Text

from orcha_agent.tui.frame import Block

from . import theme_value

_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _image(value: Any) -> tuple[bytes, str] | None:
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _image(item)
            if found is not None:
                return found
    if not isinstance(value, Mapping):
        content = getattr(value, "content", None)
        return _image(content) if content is not None else None
    for field in ("content", "result", "image", "source"):
        if field in value:
            found = _image(value[field])
            if found is not None:
                return found
    mime = str(value.get("mime_type", value.get("media_type", value.get("mime", ""))))
    data = value.get("base64", value.get("data"))
    url = value.get("image_url", value.get("url"))
    if isinstance(url, Mapping):
        url = url.get("url")
    if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
        prefix, data = url.split(";base64,", 1)
        mime = prefix[5:]
    if not isinstance(data, str) or len(data) > (_MAX_IMAGE_BYTES * 4 // 3 + 4):
        return None
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF8", "image/gif"),
    )
    actual = next((kind for signature, kind in signatures if decoded.startswith(signature)), "")
    if not actual or (mime and mime != actual) or len(decoded) > _MAX_IMAGE_BYTES:
        return None
    return decoded, actual


def image_protocol(block: Block, environ: Mapping[str, str] | None = None) -> str:
    """Return control bytes for one settled image; caller owns terminal emission."""
    found = _image(block.data)
    if found is None:
        return ""
    env = os.environ if environ is None else environ
    data, mime = found
    encoded = base64.b64encode(data).decode("ascii")
    if env.get("KITTY_WINDOW_ID") or env.get("TERM") == "xterm-kitty":
        if mime != "image/png":
            return ""
        chunks = [encoded[index : index + 4096] for index in range(0, len(encoded), 4096)]
        return "".join(
            f"\x1b_G{'a=T,f=100,q=2,' if index == 0 else ''}m={int(index < len(chunks) - 1)};{chunk}\x1b\\"
            for index, chunk in enumerate(chunks)
        )
    if env.get("TERM_PROGRAM") == "iTerm.app" or env.get("ITERM_SESSION_ID"):
        return f"\x1b]1337;File=size={len(data)};inline=1;preserveAspectRatio=1:{encoded}\x07"
    return ""


def has_image(block: Block) -> bool:
    return _image(block.data) is not None


def render(block: Block, theme: Any, width: int, budget_rows: int, expanded: bool) -> Text:
    del budget_rows, expanded
    found = _image(block.data)
    if found is None:
        label = "[Image unavailable: unsupported or invalid image data]"
    else:
        data, mime = found
        label = f"[Image · {mime.removeprefix('image/').upper()} · {len(data):,} bytes]"
    result = Text(label, style=str(theme_value(theme, "muted")))
    result.truncate(max(1, width), overflow="ellipsis")
    return result
