from types import SimpleNamespace

from orcha_agent.tui.overlays.model import ModelOverlay


def picker(tmp_path):
    return ModelOverlay(
        SimpleNamespace(
            cfg=SimpleNamespace(
                model="alpha:small", user_config_path=tmp_path / "config.toml", trust_cwd=False
            ),
            registry=SimpleNamespace(
                providers={
                    "alpha": SimpleNamespace(models=("small", "large"), available=lambda: None),
                    "beta": SimpleNamespace(models=("tiny",), available=lambda: None),
                }
            ),
            switch_model=lambda _: None,
        )
    )


def lines(content):
    return [
        "".join(part[1] for part in content.get_line(row)).rstrip()
        for row in range(content.line_count)
    ]


def test_live_model_groups_match_snapshot_and_selectable_count(tmp_path):
    overlay = picker(tmp_path)
    content = overlay.list_control.create_content(80, 4)
    rows = lines(content)
    assert [
        row.strip() for row in rows if row.strip() in {"alpha", "beta", "Roles", "Catalog"}
    ] == ["alpha", "beta", "Roles", "Catalog"]
    assert content.line_count == len(overlay.items) + 4
    assert rows == [row.rstrip() for row in overlay.render_text().splitlines()[:-1]]
    assert content.cursor_position.y == 1
    assert "(1/14)" in overlay.render_text()

    for selected, item in enumerate(overlay.items):
        overlay.index = selected
        content = overlay.list_control.create_content(80, 4)
        row = lines(content)[content.cursor_position.y]
        assert row.lstrip().startswith("› ") and overlay.label(item) in row
        window = SimpleNamespace(render_info=SimpleNamespace(window_height=4))
        top = overlay.list_window.get_vertical_scroll(window)
        assert top <= content.cursor_position.y < top + 4
    assert "(14/14)" in overlay.render_text()


def test_model_group_filter_and_error_rows_keep_cursor_on_item(tmp_path):
    overlay = picker(tmp_path)
    overlay.filter.text = "Browse catalog"
    content = overlay.list_control.create_content(80, 4)
    assert lines(content)[0].strip() == "Catalog"
    assert content.line_count == 2 and content.cursor_position.y == 1
    assert "(1/1)" in overlay.render_text()
    overlay._error = "Cannot switch"
    content = overlay.list_control.create_content(80, 4)
    assert content.line_count == 3 and content.cursor_position.y == 2
    assert lines(content)[0].strip() == "Cannot switch"
    overlay.filter.text = "No matching model here"
    content = overlay.list_control.create_content(80, 4)
    assert content.line_count == 2
    assert "No models registered" in lines(content)[1]
    assert content.cursor_position.y == 1
