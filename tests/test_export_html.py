from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict

from orcha_agent.builtin.commands_session import _export
from orcha_agent.core.export_html import export_session_html, render_session_html
from orcha_agent.core.ledger import Ledger, MessageEntry
from orcha_agent.core.session import SessionStore


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attrs = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend(attrs)


@pytest.fixture
def transcript(tmp_path):
    with SessionStore(tmp_path / "sessions.db") as store:
        session = store.create(tmp_path, "fake:model", title="Review <script>alert(1)</script>")
        messages = [
            HumanMessage(
                "# Heading\n**Bold** and `code`\n<script>alert(1)</script>\n![track](https://example.test/pixel)"
            ),
            AIMessage(
                "Done", tool_calls=[{"id": "call", "name": "edit", "args": {"path": "<file>"}}]
            ),
            ToolMessage(
                "--- before\n+++ after\n-old\n+new<script>bad</script>",
                name="edit",
                tool_call_id="call",
            ),
        ]
        ledger = Ledger(store)
        for message in messages:
            ledger.append(session.thread_id, MessageEntry(message=message_to_dict(message)))
        yield store, session.thread_id


def test_standalone_markdown_cards_diffs_and_theme(transcript):
    store, session_id = transcript
    html = render_session_html(store, session_id, theme={"colors": {"accent": "#123456"}})
    parsed = Tags()
    parsed.feed(html)
    assert {"h1", "strong", "code", "details", "summary", "article"} <= set(parsed.tags)
    assert "script" not in parsed.tags
    assert "img" not in parsed.tags
    assert "--accent:#123456" in html
    assert '<span class="added">+new&lt;script&gt;' in html
    assert '<span class="removed">-old</span>' in html
    assert "&lt;file&gt;" in html
    assert not any(name in {"src", "onclick"} for name, _ in parsed.attrs)


def test_export_permissions_exclusive_force_and_symlink(transcript, tmp_path):
    store, session_id = transcript
    path = tmp_path / "session.html"
    assert export_session_html(store, session_id, path) == path
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export_session_html(store, session_id, path)
    export_session_html(store, session_id, path, force=True)
    link = tmp_path / "link.html"
    link.symlink_to(path)
    with pytest.raises(OSError):
        export_session_html(store, session_id, link, force=True)


@pytest.mark.asyncio
async def test_command_html_path_with_spaces_and_force(transcript, tmp_path):
    store, session_id = transcript
    output = []
    ctx = SimpleNamespace(
        session=store,
        session_id=session_id,
        console=SimpleNamespace(print=output.append, error=output.append),
    )
    path = tmp_path / "my session.html"
    await _export(ctx, f"--html {path}")
    assert path.read_text().startswith("<!doctype html>")
    await _export(ctx, f"--html {path}")
    assert "already exists" in output[-1]
    await _export(ctx, f"--force --html {path}")
    assert "Exported session" in output[-1]


@pytest.mark.asyncio
async def test_html_default_path(transcript, tmp_path, monkeypatch):
    store, session_id = transcript
    monkeypatch.chdir(tmp_path)
    ctx = SimpleNamespace(
        session=store, session_id=session_id, console=SimpleNamespace(print=lambda _: None)
    )
    await _export(ctx, "--html")
    assert Path(f"{session_id}.html").is_file()


@pytest.mark.parametrize(
    "tool,output,is_diff",
    [
        ("bash", "- ordinary bullet\n+ progress message", False),
        ("read", "-10 degrees\n+10 degrees", False),
        ("bash", "--- a/file\n+++ b/file\n-old\n+new", True),
        ("bash", "@@ -1 +1 @@\n-old\n+new", True),
        ("edit", "-old\n+new", True),
        ("apply_patch", "-old\n+new", True),
    ],
)
def test_tool_output_colours_only_diff_content(transcript, tool, output, is_diff):
    store, session_id = transcript
    entry = Ledger(store).append(
        session_id,
        MessageEntry(message=message_to_dict(ToolMessage(output, name=tool, tool_call_id="extra"))),
    )
    html = render_session_html(store, session_id)
    card = html.split(f'id="{entry.id}"', 1)[1].split("</details>", 1)[0]
    assert ('class="removed"' in card) is is_diff
    assert ('class="added"' in card) is is_diff
