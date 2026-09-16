from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import DummyOutput

from orcha_agent.tui.composer import Composer, repair_paste
from orcha_agent.tui.complete import ComposerCompleter


def test_paste_repairs_both_tmux_control_encodings() -> None:
    assert (
        repair_paste("one\x1b[106;5utwo\x1b[27;5;109~three\x1b[105;5ufour")
        == "one\ntwo\nthree\tfour"
    )
    assert repair_paste("\x1b[31mred\x1b[0m\x07\x00\r\nnext") == "red\nnext"


def test_multiple_paste_chips_expand_without_recursion() -> None:
    composer = Composer()
    first = "\n".join(str(i) for i in range(5))
    composer.insert_paste(first)
    first_chip = composer.buffer.text
    composer.buffer.insert_text(" / ")
    second = first_chip + "\n" * 5
    composer.insert_paste(second)
    assert composer.expanded_text(composer.buffer.text) == first + " / " + second
    assert "[Pasted 5 lines #1]" in composer.buffer.text
    assert composer.text_rows(80) == 1


@pytest.mark.asyncio
async def test_pipe_bracketed_paste_does_not_submit_at_newlines() -> None:
    composer = Composer()
    submissions: list[str] = []
    keys = KeyBindings()

    @keys.add("enter")
    def submit(event: object) -> None:
        submissions.append(composer.expanded_text(composer.buffer.text))

    with create_pipe_input() as pipe:
        app: Application[None] = Application(
            layout=Layout(composer.container), key_bindings=keys, input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(app.run_async())
        try:
            await asyncio.sleep(0.03)
            pipe.send_text("\x1b[200~one\r\ntwo\x1b[106;5uthree\nfour\nfive")
            await asyncio.sleep(0.03)
            assert not submissions
            pipe.send_text("\x1b[201~")
            await asyncio.sleep(0.03)
            assert not submissions
            assert "[Pasted 5 lines" in composer.buffer.text
            pipe.send_text("\r")
            await asyncio.sleep(0.03)
            assert submissions == ["one\ntwo\nthree\nfour\nfive"]
        finally:
            if not task.done():
                app.exit()
            await task


@pytest.mark.asyncio
async def test_pipe_clear_recall_and_emacs_kill_ring() -> None:
    composer = Composer()
    with create_pipe_input() as pipe:
        app: Application[None] = Application(
            layout=Layout(composer.container), input=pipe, output=DummyOutput()
        )
        task = asyncio.create_task(app.run_async())
        try:
            await asyncio.sleep(0.03)
            pipe.send_text("saved draft\x18\x0b")
            await asyncio.sleep(0.03)
            assert composer.buffer.text == ""
            pipe.send_text("\x18\x12")
            await asyncio.sleep(0.03)
            assert composer.buffer.text == "saved draft"
            pipe.send_text("\x01\x0bnew\x01\x0b\x19")
            await asyncio.sleep(0.03)
            assert composer.buffer.text == "new"
            pipe.send_text("\x1by")
            await asyncio.sleep(0.03)
            assert composer.buffer.text == "saved draft"
        finally:
            if not task.done():
                app.exit()
            await task


def test_slash_argument_ghost_is_not_submitted(tmp_path: object) -> None:
    registry = SimpleNamespace(
        commands={"model": SimpleNamespace(help="Choose model")}, completers=[]
    )
    completer = ComposerCompleter(registry, str(tmp_path))
    composer = Composer(completer=completer)
    composer.buffer.text = "/model"
    composer.buffer.cursor_position = len(composer.buffer.text)
    assert composer.ghost_fragments() == [("class:composer.placeholder", " [provider/model]")]
    assert composer.buffer.text == "/model"


@pytest.mark.asyncio
async def test_runtime_paste_submit_preserves_expanded_history() -> None:
    from prompt_toolkit.history import InMemoryHistory
    from orcha_agent.tui.runtime import ApplicationRuntime

    submissions: list[str] = []
    submitted = asyncio.Event()
    history = InMemoryHistory()

    async def submit(text: str) -> None:
        submissions.append(text)
        submitted.set()

    payload = "\n".join(f"paste {index}" for index in range(6))
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(submit, history=history, input=pipe, output=DummyOutput())
        task = asyncio.create_task(runtime.run())
        try:
            pipe.send_text("\x1b[200~" + payload + "\x1b[201~")
            await asyncio.sleep(0.03)
            assert not submissions
            assert "[Pasted 6 lines" in runtime.buffer.text
            pipe.send_text("\r")
            await asyncio.wait_for(submitted.wait(), 1)
            assert submissions == [payload]
            assert list(history.get_strings())[-1] == payload
        finally:
            runtime.application.exit()
            await task


@pytest.mark.asyncio
async def test_runtime_native_vim_navigation_editing_and_cursor(tmp_path: object) -> None:
    from prompt_toolkit.cursor_shapes import CursorShape
    from prompt_toolkit.key_binding.vi_state import InputMode
    from orcha_agent.tui.runtime import ApplicationRuntime

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            tui=SimpleNamespace(vim=True), cwd=str(tmp_path), model="test", models={}, providers={}
        ),
        plugin_states={},
        persist_plugin_states=lambda: None,
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda text: asyncio.sleep(0),
            ctx=ctx,
            input=pipe,
            output=DummyOutput(),
            status=lambda: "test",
        )
        app = runtime.application
        task = asyncio.create_task(runtime.run())
        app.ttimeoutlen = 0.01
        app.timeoutlen = 0.01
        try:
            pipe.send_text("one two\x1b")
            await asyncio.sleep(0.05)
            assert app.vi_state.input_mode == InputMode.NAVIGATION
            assert app.cursor.get_cursor_shape(app) == CursorShape.BLOCK
            pipe.send_text("0w")
            await asyncio.sleep(0.03)
            assert runtime.buffer.cursor_position == 4
            pipe.send_text("b")
            await asyncio.sleep(0.03)
            assert runtime.buffer.cursor_position == 0
            pipe.send_text("$a!\x1b")
            await asyncio.sleep(0.05)
            assert runtime.buffer.text == "one two!"
            pipe.send_text("yyopasted\x1b")
            await asyncio.sleep(0.05)
            assert runtime.buffer.text == "one two!\npasted"
            pipe.send_text("ddp")
            await asyncio.sleep(0.03)
            assert runtime.buffer.text == "one two!\npasted"
            pipe.send_text("gg")
            await asyncio.sleep(0.03)
            assert runtime.buffer.document.cursor_position_row == 0
            pipe.send_text("GiX")
            await asyncio.sleep(0.03)
            assert app.vi_state.input_mode == InputMode.INSERT
            assert app.cursor.get_cursor_shape(app) == CursorShape.BEAM
            assert "X" in runtime.buffer.text
        finally:
            if not task.done():
                app.exit()
            await task


@pytest.mark.asyncio
async def test_file_index_walk_does_not_block_input_loop(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    registry = SimpleNamespace(commands={}, completers=[])
    completer = ComposerCompleter(registry, str(tmp_path))
    thread_ids: list[int] = []

    def walk() -> tuple[str, ...]:
        thread_ids.append(threading.get_ident())
        return ("alpha.py",)

    monkeypatch.setattr(completer.path_index, "_walk", walk)
    completions = [
        item async for item in completer.get_completions_async(Document("@ap"), CompleteEvent())
    ]
    assert [item.text for item in completions] == ["@alpha.py"]
    assert len(thread_ids) == 1
    assert thread_ids[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_live_composer_shape_changes_repaint_and_preserve_editing() -> None:
    from prompt_toolkit.completion import Completion

    composer = Composer()
    composer.buffer.text = "draft text"
    composer.buffer.cursor_position = 5
    composer.buffer.complete_state = composer.buffer._set_completions([Completion("draft")])
    completion_state = composer.buffer.complete_state
    container = composer.container
    with create_pipe_input() as pipe:
        app: Application[None] = Application(
            layout=Layout(composer.container), input=pipe, output=DummyOutput()
        )
        rendered = asyncio.Event()
        app.after_render += lambda _: rendered.set()
        task = asyncio.create_task(app.run_async())
        try:
            await asyncio.wait_for(rendered.wait(), 1)
            for shape, prefix in (("band", "─"), ("rail", "│ "), ("box", "╭")):
                rendered.clear()
                composer.set_shape(shape)
                app.invalidate()
                await asyncio.wait_for(rendered.wait(), 1)
                screen = app.renderer._last_screen
                assert screen is not None
                row = "".join(screen.data_buffer[0][x].char for x in range(20))
                assert row.startswith(prefix)
                assert composer.container is container
                assert composer.buffer.text == "draft text"
                assert composer.buffer.cursor_position == 5
                assert composer.buffer.complete_state is completion_state
                assert app.layout.current_buffer is composer.buffer
        finally:
            if not task.done():
                app.exit()
            await task


@pytest.mark.parametrize("edit", ["backspace", "delete", "insert", "selection"])
def test_paste_chip_edits_are_atomic(edit: str) -> None:
    from prompt_toolkit.document import Document

    composer = Composer()
    composer.buffer.insert_text("prefix ")
    composer.insert_paste("one\ntwo\nthree\nfour\nfive")
    chip = composer.buffer.text[7:]
    composer.buffer.insert_text(" suffix")
    original = composer.buffer.text
    composer.buffer.save_to_undo_stack()
    if edit == "backspace":
        composer.buffer.cursor_position = 7 + len(chip)
        composer.buffer.delete_before_cursor()
    elif edit == "delete":
        composer.buffer.cursor_position = 7
        composer.buffer.delete()
    elif edit == "insert":
        composer.buffer.cursor_position = 10
        composer.buffer.insert_text("X")
    else:
        composer.buffer.document = Document(
            composer.buffer.text[:10] + composer.buffer.text[15:], 10
        )
    assert composer.buffer.text == "prefix " + ("X" if edit == "insert" else "") + " suffix"
    assert "Pasted" not in composer.expanded_text(composer.buffer.text)
    assert composer.paste_preview() is None
    composer.buffer.undo()
    assert composer.buffer.text == original
    assert (
        composer.expanded_text(composer.buffer.text) == "prefix one\ntwo\nthree\nfour\nfive suffix"
    )


def test_paste_processor_dims_only_chip_and_preserves_positions() -> None:
    from prompt_toolkit.layout.processors import TransformationInput
    from orcha_agent.tui.composer import PasteChipProcessor

    composer = Composer()
    composer.buffer.insert_text("before ")
    composer.insert_paste("1\n2\n3\n4\n5")
    composer.buffer.insert_text(" after")
    ti = TransformationInput(
        composer.control,
        composer.buffer.document,
        0,
        lambda x: x,
        [("", composer.buffer.text)],
        80,
        3,
    )
    result = PasteChipProcessor(composer).apply_transformation(ti)
    dim = "".join(text for style, text in result.fragments if "dim" in style)
    assert dim == "[Pasted 5 lines #1]"
    assert "".join(text for _, text in result.fragments) == composer.buffer.text
    assert result.source_to_display(9) == 9


@pytest.mark.asyncio
async def test_pipe_peek_paste_key() -> None:
    composer = Composer()
    captured: list[str] = []
    composer.on_paste_peek = captured.append
    payload = "1\n2\n3\n4\n5"
    composer.insert_paste(payload)
    with create_pipe_input() as pipe:
        app = Application(layout=Layout(composer.container), input=pipe, output=DummyOutput())
        task = asyncio.create_task(app.run_async())
        try:
            await asyncio.sleep(0.02)
            pipe.send_bytes(b"\x18\x10")
            await asyncio.sleep(0.04)
            assert captured == [payload]
            assert composer.expanded_text(composer.buffer.text) == payload
        finally:
            app.exit()
            await task


@pytest.mark.asyncio
async def test_vim_navigation_escape_preserves_completion_abort_and_tree(tmp_path: object) -> None:
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.key_binding.vi_state import InputMode
    from orcha_agent.tui.runtime import ApplicationRuntime

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(
            tui=SimpleNamespace(vim=True), cwd=str(tmp_path), model="test", models={}, providers={}
        ),
        plugin_states={},
        persist_plugin_states=lambda: None,
    )
    with create_pipe_input() as pipe:
        runtime = ApplicationRuntime(
            lambda _: asyncio.sleep(0),
            ctx=ctx,
            input=pipe,
            output=DummyOutput(),
            status=lambda: "test",
        )
        app = runtime.application
        task = asyncio.create_task(runtime.run())
        app.ttimeoutlen = app.timeoutlen = 0.01
        aborted: list[bool] = []
        trees: list[bool] = []
        runtime._abort_turn = lambda: aborted.append(True)
        runtime._tree_handler = lambda _: trees.append(True)
        try:
            await asyncio.sleep(0.02)
            app.vi_state.input_mode = InputMode.NAVIGATION
            runtime.buffer._set_completions([Completion("candidate")])
            pipe.send_bytes(b"\x1b")
            await asyncio.sleep(0.04)
            assert runtime.buffer.complete_state is None
            assert not aborted and not trees
            runtime.streaming = True
            pipe.send_bytes(b"\x1b")
            await asyncio.sleep(0.04)
            assert aborted == [True]
            runtime.streaming = False
            runtime.buffer.reset()
            pipe.send_bytes(b"\x1b")
            await asyncio.sleep(0.04)
            pipe.send_bytes(b"\x1b")
            await asyncio.sleep(0.04)
            assert trees == [True]
        finally:
            app.exit()
            await task
