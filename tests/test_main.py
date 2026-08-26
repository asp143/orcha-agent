import sys
from pathlib import Path
from typing import Any

import pytest

from orcha_agent import __main__ as entrypoint


def test_main_loads_dotenv_from_configured_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ORCHA_TEST_SENTINEL=from-temporary-dotenv\n")
    dotenv_calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(dotenv_path: str | Path, *, override: bool) -> bool:
        dotenv_calls.append((Path(dotenv_path), override))
        return True

    async def fake_run_app(cfg: Any) -> int:
        assert cfg.cwd == tmp_path.resolve()
        return 0

    monkeypatch.setattr(entrypoint, "load_dotenv", fake_load_dotenv, raising=False)
    monkeypatch.setattr(entrypoint, "run_app", fake_run_app)
    monkeypatch.setattr(sys, "argv", ["orcha", "--cwd", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    assert dotenv_calls == [(env_file.resolve(), False)]
