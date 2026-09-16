from pathlib import Path

import pytest

from orcha_agent.core.config import load_config


def test_tui_table_configures_new_input_and_terminal_options(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text("""[tui]
vim = true
hyperlinks = false
mouse = false
synchronized_output = false
resize = "rebuild"
composer = "rail"
symbols = "colorblind"
colorblind = true
[tui.statusline]
preset = "powerline"
""")
    cfg = load_config([], cwd=tmp_path, env={"HOME": str(tmp_path)}, user_config_path=config)
    assert cfg.tui.vim
    assert not cfg.tui.hyperlinks
    assert cfg.tui.mouse == "off"
    assert cfg.tui.colorblind
    assert not cfg.tui.synchronized_output
    assert cfg.tui.resize == "rebuild"
    assert cfg.composer == "rail"
    assert cfg.symbols == "colorblind"
    assert cfg.statusline.preset == "powerline"


@pytest.mark.parametrize("value", ["[]", "true", '"unknown"'])
def test_invalid_resize_is_a_config_error(tmp_path: Path, value: str):
    config = tmp_path / "config.toml"
    config.write_text(f"[tui]\nresize = {value}\n")
    with pytest.raises(SystemExit):
        load_config([], cwd=tmp_path, env={"HOME": str(tmp_path)}, user_config_path=config)
