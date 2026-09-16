"""Regression coverage for terminal-report and bracketed-paste boundaries."""

from orcha_agent.tui.runtime import _TerminalReplies


def test_fragmented_paste_end_restores_terminal_reply_filtering() -> None:
    forwarded: list[str] = []
    reports: list[str] = []
    parser = _TerminalReplies(forwarded.append, reports.append)
    for chunk in ("\x1b[200~payload\x1b[20", "1~", "\x1b[?2026;1$y", "ordinary"):
        parser(chunk)
    assert "".join(forwarded) == "\x1b[200~payload\x1b[201~ordinary"
    assert reports == ["\x1b[?2026;1$y"]
    assert not parser.pasting


def test_every_paste_marker_split_retains_literal_payload() -> None:
    begin, end = "\x1b[200~", "\x1b[201~"
    payload = "code\n\x1b]11;rgb:ffff/ffff/ffff\x07"
    for start_split in range(1, len(begin)):
        for end_split in range(1, len(end)):
            forwarded: list[str] = []
            reports: list[str] = []
            parser = _TerminalReplies(forwarded.append, reports.append)
            for chunk in (
                begin[:start_split],
                begin[start_split:] + payload + end[:end_split],
                end[end_split:],
                "\x1b[I",
            ):
                parser(chunk)
            assert "".join(forwarded) == begin + payload + end
            assert reports == ["\x1b[I"]


def test_large_paste_is_forwarded_in_spans_not_per_character() -> None:
    forwarded: list[str] = []
    parser = _TerminalReplies(forwarded.append, lambda report: None)
    payload = "line of code\n" * 10_000
    parser("\x1b[200~" + payload + "\x1b[201~")
    assert "".join(forwarded) == "\x1b[200~" + payload + "\x1b[201~"
    assert len(forwarded) <= 4


def test_input_timeout_between_paste_end_fragments_preserves_marker() -> None:
    forwarded: list[str] = []
    reports: list[str] = []
    parser = _TerminalReplies(forwarded.append, reports.append)
    parser("\x1b[200~payload\x1b[20")
    parser.flush()
    parser("1~\x1b[I")
    assert "".join(forwarded) == "\x1b[200~payload\x1b[201~"
    assert reports == ["\x1b[I"]
