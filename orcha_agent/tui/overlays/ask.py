"""Tabbed multi-question ask dialog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from .base import Overlay
from .hints import key_hint


class AskOverlay(Overlay):
    """Collect ordered radio, multi-select, and custom answers."""

    def __init__(self, questions: object, **_payload: Any) -> None:
        if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
            raise TypeError("questions must be a sequence")
        self.questions = tuple(
            dict(question) for question in questions if isinstance(question, Mapping)
        )
        self.question_index = 0
        self.option_index = 0
        self._selected: dict[int, set[int]] = {}
        self._custom: dict[int, str] = {}
        self._custom_mode = False
        self.custom_buffer = Buffer(multiline=False)
        self.list_control = FormattedTextControl(self._fragments, focusable=True)
        self.custom_control = BufferControl(buffer=self.custom_buffer)
        bindings = KeyBindings()

        @bindings.add("up")
        def _up(event: Any) -> None:
            self._move_option(-1)
            event.app.invalidate()

        @bindings.add("down")
        def _down(event: Any) -> None:
            self._move_option(1)
            event.app.invalidate()

        @bindings.add("tab")
        def _next(event: Any) -> None:
            self._move_question(1)
            event.app.invalidate()

        @bindings.add("s-tab")
        def _previous(event: Any) -> None:
            self._move_question(-1)
            event.app.invalidate()

        @bindings.add(" ")
        def _space(event: Any) -> None:
            if self._custom_mode:
                self.custom_buffer.insert_text(" ")
                return
            question = self._question()
            if not self._multi(question):
                return
            if self.option_index == len(self._options(question)):
                self._start_custom(event)
                return
            selected = self._selected.setdefault(self.question_index, set())
            if self.option_index in selected:
                selected.remove(self.option_index)
            else:
                selected.add(self.option_index)
            event.app.invalidate()

        @bindings.add("enter")
        def _enter(event: Any) -> None:
            if self._custom_mode:
                value = self.custom_buffer.text.strip()
                if value:
                    self._custom[self.question_index] = value
                self._custom_mode = False
                event.app.layout.focus(self.list_control)
            else:
                question = self._question()
                if self.option_index == len(self._options(question)):
                    self._start_custom(event)
                    return
                if self._multi(question):
                    selected = self._selected.setdefault(self.question_index, set())
                    if not selected:
                        selected.add(self.option_index)
                else:
                    self._selected[self.question_index] = {self.option_index}
            if self.question_index == len(self.questions) - 1:
                self.resolve(self._result())
            event.app.invalidate()

        body = HSplit(
            [
                Window(self.list_control, wrap_lines=True),
                Window(char="─", height=1, style="class:overlay.divider"),
                Window(self.custom_control, height=1, style="class:overlay.custom"),
            ]
        )
        super().__init__(
            "Questions",
            body,
            width=0.8,
            height=0.7,
            min_height=7,
            bindings=bindings,
        )

    @property
    def focus_target(self) -> FormattedTextControl:
        return self.list_control

    def _question(self) -> dict[str, Any]:
        if not self.questions:
            return {}
        return self.questions[self.question_index]

    @staticmethod
    def _multi(question: Mapping[str, Any]) -> bool:
        return bool(
            question.get("multi")
            or question.get("multiSelect")
            or question.get("multi_select")
            or question.get("multiple")
        )

    @staticmethod
    def _options(question: Mapping[str, Any]) -> tuple[str, ...]:
        labels: list[str] = []
        raw = question.get("options", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        for option in raw:
            if isinstance(option, Mapping):
                value = option.get("label")
            else:
                value = option
            if isinstance(value, str):
                labels.append(value)
        return tuple(labels)

    def _move_option(self, delta: int) -> None:
        maximum = len(self._options(self._question()))
        self.option_index = min(maximum, max(0, self.option_index + delta))

    def _move_question(self, delta: int) -> None:
        if not self.questions:
            return
        self.question_index = (self.question_index + delta) % len(self.questions)
        self.option_index = 0
        self._custom_mode = False

    def _start_custom(self, event: Any) -> None:
        self._custom_mode = True
        self.custom_buffer.text = self._custom.get(self.question_index, "")
        self.custom_buffer.cursor_position = len(self.custom_buffer.text)
        event.app.layout.focus(self.custom_control)
        event.app.invalidate()

    def _fragments(self) -> StyleAndTextTuples:
        if not self.questions:
            return [("class:overlay.empty", "No questions\n")]
        fragments: StyleAndTextTuples = []
        tabs = "  ".join(
            f"[{index + 1} {question.get('header') or question.get('id') or 'Question'}]"
            for index, question in enumerate(self.questions)
        )
        fragments.append(("class:overlay.tabs", tabs + "\n\n"))
        question = self._question()
        fragments.append(("class:overlay.question", f"{question.get('question', '')}\n"))
        selected = self._selected.get(self.question_index, set())
        marker_on, marker_off = ("◉", "○")
        if self._multi(question):
            marker_on, marker_off = ("◉", "○")
        options = (*self._options(question), "Other (type your own)")
        for index, label in enumerate(options):
            current = index == self.option_index
            checked = index in selected or (
                index == len(options) - 1 and self.question_index in self._custom
            )
            marker = marker_on if checked else marker_off
            style = "class:overlay.selection" if current else "class:overlay.item"
            fragments.append((style, f" {marker} {label}\n"))
        if self._custom_mode:
            fragments.append(("", "\n"))
            fragments.extend(key_hint("enter", "submit answer"))
        return fragments

    def _result(self) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for index, question in enumerate(self.questions):
            options = self._options(question)
            selected = self._selected.get(index, set())
            answer: dict[str, Any] = {
                "id": str(
                    question.get("id")
                    or question.get("header")
                    or question.get("question")
                    or index
                ),
                "selectedOptions": [
                    option for offset, option in enumerate(options) if offset in selected
                ],
            }
            if index in self._custom:
                answer["customInput"] = self._custom[index]
            results.append(answer)
        return {"kind": "submit", "results": results}


__all__ = ["AskOverlay"]
