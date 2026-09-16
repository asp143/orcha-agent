from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orcha_agent.tui.overlays.session import SessionOverlay


@pytest.mark.parametrize(
    "query",
    [
        "parser repair",
        "Prsr rpr",
        "4bf8d91e-session-unique",
        "long-component-hidden-from-label",
        "/repo/long-component-hidden-from-label/packages/target",
    ],
)
def test_sessions_filter_precomputed_title_id_and_full_path(query: str) -> None:
    sessions = [
        SimpleNamespace(
            thread_id="4bf8d91e-session-unique",
            title=None,
            cwd="/repo/long-component-hidden-from-label/packages/target",
            created=None,
        ),
        SimpleNamespace(thread_id="other", title="Other work", cwd="/other", created=None),
    ]
    ledger = SimpleNamespace(
        count=Mock(return_value=3),
        all=Mock(
            return_value=[SimpleNamespace(message={"role": "user", "content": "Parser repair"})]
        ),
    )
    picker = SessionOverlay(
        SimpleNamespace(session=SimpleNamespace(list=lambda: sessions), ledger=ledger)
    )
    label = picker.label(sessions[0])
    assert "Parser repair" in label
    assert "long-component-hidden-from-label" not in label
    assert "4bf8d91e-session-unique" not in label

    # The filter buffer is the typing target; querying or painting must not load
    # entries or reconstruct labels after the initial session snapshot.
    assert picker.focus_target is picker.filter_control
    picker.filter.insert_text(query)
    assert picker.filtered_items == (sessions[0],)
    picker.render_text()
    picker.render_text()
    assert picker.label(sessions[0]) == label
    assert ledger.count.call_count == 2
    ledger.all.assert_called_once_with("4bf8d91e-session-unique")

    picker.filter.text = "no matching session"
    assert picker.filtered_items == ()
    picker.filter.text = ""
    assert picker.filtered_items == tuple(sessions)
