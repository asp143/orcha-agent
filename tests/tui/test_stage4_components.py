from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document

from orcha_agent.core.events import EventBus
from orcha_agent.core.plugin import PluginAPI
from orcha_agent.core.registry import Registry
from orcha_agent.tui.complete import ComposerCompleter, PathIndex
from orcha_agent.tui.history import SQLiteHistory
from orcha_agent.tui.keys import (
    DEFAULT_BINDINGS,
    format_key_bindings,
    format_key_name,
    load_keybindings,
)
from orcha_agent.tui.queue import PromptQueue, split_submission


def test_queue_splits_arrow_and_numbered_batches_in_order() -> None:
    assert split_submission("-> first\n-> second") == ["first", "second"]
    assert split_submission("=> first\n=> second") == ["first", "second"]
    assert split_submission("1. first\n2. second\n   continued") == [
        "first",
        "second\ncontinued",
    ]
    assert split_submission("ordinary\ntext") == ["ordinary\ntext"]


def test_queue_dequeues_and_restores_last_item() -> None:
    queue = PromptQueue()
    queue.extend(["one", "two", "three"])
    assert queue.pop() == "one"
    assert queue.pop_last() == "three"
    assert queue.items == ("two",)
    assert queue.restore_text() == "two"
    assert queue.items == ()


def test_sqlite_history_skips_consecutive_duplicates_and_searches(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "history.db", cwd=tmp_path, session_id="session")
    history.append_string("first prompt")
    history.append_string("first prompt")
    history.append_string("second searchable prompt")

    assert list(history.load_history_strings()) == [
        "second searchable prompt",
        "first prompt",
    ]
    assert history.search("searchable") == ["second searchable prompt"]


def test_sqlite_history_migrates_legacy_file_only_once(tmp_path: Path) -> None:
    legacy = tmp_path / "history"
    legacy.write_text(
        "\n# old\n+first\n\n# newer\n+second\n+line\n",
        encoding="utf-8",
    )
    database = tmp_path / "history.db"
    first = SQLiteHistory(database, legacy_path=legacy)
    assert list(first.load_history_strings()) == ["second\nline", "first"]

    second = SQLiteHistory(database, legacy_path=legacy)
    assert list(second.load_history_strings()) == ["second\nline", "first"]


def test_path_index_honors_gitignore_cache_and_sensitive_exclusions(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n*.tmp\n", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    (tmp_path / "skip.tmp").write_text("", encoding="utf-8")
    (tmp_path / ".env.production").write_text("secret", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "no.py").write_text("", encoding="utf-8")
    (tmp_path / "Credentials").mkdir()
    (tmp_path / "Credentials" / "token.txt").write_text("secret", encoding="utf-8")
    index = PathIndex(tmp_path)

    first = index.paths()
    (tmp_path / "later.py").write_text("", encoding="utf-8")

    assert first == index.paths()
    assert "visible.py" in first
    assert "later.py" not in first
    assert not any(".env" in path or "Credentials" in path or "ignored" in path for path in first)
    index._cached_at = time.monotonic() - 11
    assert "later.py" in index.paths()


def test_completer_supports_commands_at_paths_tab_and_plugins(tmp_path: Path) -> None:
    (tmp_path / "space name.py").write_text("", encoding="utf-8")
    registry = Registry()

    async def command(_ctx: object, _args: str) -> None:
        return None

    registry._add_command("core", "help", command, "List help")

    def plugin(document: Document):
        if document.text_before_cursor.startswith("#"):
            return [Completion("plugin", start_position=-1, display_meta="from plugin")]
        return []

    registry._add_completer("plugin", "#", plugin, priority=10)
    completer = ComposerCompleter(registry, tmp_path)

    slash = list(completer.get_completions(Document("/he"), CompleteEvent(completion_requested=True)))
    at = list(completer.get_completions(Document('@"space'), CompleteEvent(completion_requested=True)))
    tab = list(completer.get_completions(Document("spa"), CompleteEvent(completion_requested=True)))
    custom = list(completer.get_completions(Document("#"), CompleteEvent(completion_requested=True)))

    assert slash[0].text == "help"
    assert slash[0].display_meta_text == "List help"
    assert at[0].text == '@"space name.py"'
    assert tab[0].text == '"space name.py"'
    assert custom[0].text == "plugin"

def test_bare_path_completion_indexes_only_after_explicit_tab(tmp_path: Path) -> None:
    registry = Registry()
    completer = ComposerCompleter(registry, tmp_path)
    indexed: list[bool] = []
    completer.path_index.paths = lambda: indexed.append(True) or ("alpha.py",)

    ordinary = list(
        completer.get_completions(
            Document("alp"),
            CompleteEvent(completion_requested=False),
        )
    )
    assert ordinary == []
    assert indexed == []

    explicit = list(
        completer.get_completions(
            Document("alp"),
            CompleteEvent(completion_requested=True),
        )
    )
    assert [completion.text for completion in explicit] == ["alpha.py"]
    assert indexed == [True]


def test_path_index_applies_anchored_nested_and_negated_gitignore_rules(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text(
        "/root-only.txt\n"
        "global.log\n"
        "ignored/**\n"
        "!ignored/keep/\n"
        "!ignored/keep/visible.py\n",
        encoding="utf-8",
    )
    (tmp_path / "root-only.txt").write_text("", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "root-only.txt").write_text("", encoding="utf-8")
    (tmp_path / "nested" / "global.log").write_text("", encoding="utf-8")
    (tmp_path / "nested" / ".gitignore").write_text(
        "/nested-only.txt\n*.tmp\n!keep.tmp\n",
        encoding="utf-8",
    )
    (tmp_path / "nested" / "nested-only.txt").write_text("", encoding="utf-8")
    (tmp_path / "nested" / "deeper").mkdir()
    (tmp_path / "nested" / "deeper" / "nested-only.txt").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "nested" / "drop.tmp").write_text("", encoding="utf-8")
    (tmp_path / "nested" / "keep.tmp").write_text("", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "drop.py").write_text("", encoding="utf-8")
    (tmp_path / "ignored" / "keep").mkdir()
    (tmp_path / "ignored" / "keep" / "visible.py").write_text("", encoding="utf-8")

    paths = PathIndex(tmp_path).paths()

    assert "root-only.txt" not in paths
    assert "nested/root-only.txt" in paths
    assert "nested/global.log" not in paths
    assert "nested/nested-only.txt" not in paths
    assert "nested/deeper/nested-only.txt" in paths
    assert "nested/drop.tmp" not in paths
    assert "nested/keep.tmp" in paths
    assert "ignored/drop.py" not in paths
    assert "ignored/keep/visible.py" in paths


def _api(registry: Registry, name: str = "plugin") -> PluginAPI:
    return PluginAPI(
        name=name,
        config={},
        state={},
        registry=registry,
        bus=EventBus(),
        request_rebuild=lambda: None,
    )


def test_plugin_completer_and_keybinding_conflicts_require_replace() -> None:
    registry = Registry()
    one = _api(registry, "one")
    two = _api(registry, "two")
    handler = lambda *_args: None
    one.add_completer("#", handler)
    one.add_keybinding("custom", handler, default="c-x")

    with pytest.raises(ValueError, match="replace=True"):
        two.add_completer("#", handler)
    with pytest.raises(ValueError, match="replace=True"):
        two.add_keybinding("custom", handler, default="c-y")

    two.add_completer("#", handler, replace=True)
    two.add_keybinding("custom", handler, default="c-y", replace=True)
    assert registry.completers[0].plugin == "two"
    assert registry.keybindings["custom"].plugin == "two"


def test_keybinding_overrides_lists_conflicts_and_invalid_values(tmp_path: Path) -> None:
    user = tmp_path / "keybindings.toml"
    user.write_text(
        '[bindings]\nsubmit = ["c-j", "escape enter"]\nqueue = "c-j"\nexit = "not-a-key"\n',
        encoding="utf-8",
    )
    warnings: list[str] = []

    effective = load_keybindings(user_path=user, warn=warnings.append)

    assert effective["submit"] == ("escape enter",)
    assert effective["queue"] == ("c-j",)
    assert effective["exit"] == tuple(DEFAULT_BINDINGS["exit"])
    assert any("submit" in warning and "queue" in warning for warning in warnings)
    assert any("not-a-key" in warning and "exit" in warning for warning in warnings)


def test_every_default_action_has_a_valid_effective_binding() -> None:
    effective = load_keybindings(user_path=Path("/path/that/does/not/exist"))
    assert set(effective) == set(DEFAULT_BINDINGS)
    assert all(effective[action] for action in DEFAULT_BINDINGS)


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        ("c-o", "Ctrl+O"),
        ("escape p", "Alt+P"),
        ("s-tab", "Shift+Tab"),
        ("escape enter", "Alt+Enter"),
        ("escape escape", "Esc Esc"),
    ],
)
def test_key_names_use_human_labels_for_prompt_toolkit_bindings(
    binding: str,
    expected: str,
) -> None:
    assert format_key_name(binding) == expected


def test_key_binding_lists_use_compact_hint_separator() -> None:
    assert format_key_bindings(("enter", "c-j")) == "Enter/Ctrl+J"


def test_key_conflicts_use_prompt_toolkit_canonical_sequences(tmp_path: Path) -> None:
    user = tmp_path / "keybindings.toml"
    user.write_text('[bindings]\nqueue = "c-m"\n', encoding="utf-8")
    warnings: list[str] = []

    effective = load_keybindings(user_path=user, warn=warnings.append)

    assert "enter" not in effective["submit"]
    assert effective["submit"] == ("c-j",)
    assert effective["queue"] == ("c-m",)
    assert any("submit" in warning and "queue" in warning for warning in warnings)


def test_plugin_must_explicitly_replace_core_key_action() -> None:
    registry = Registry()
    api = _api(registry)
    handler = lambda *_args: None

    with pytest.raises(ValueError, match="replace=True"):
        api.add_keybinding("submit", handler, default="c-y")

    api.add_keybinding("submit", handler, default="c-y", replace=True)
    assert registry.keybindings["submit"].handler is handler
