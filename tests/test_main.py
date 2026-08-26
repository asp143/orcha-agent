import os
import sys
from pathlib import Path
from typing import Any

import pytest

from orcha_agent import __main__ as entrypoint


@pytest.mark.parametrize("trusted", [False, True], ids=["untrusted", "trusted"])
def test_main_loads_dotenv_only_from_trusted_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trusted: bool,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ORCHA_TEST_SENTINEL=from-temporary-dotenv\n")
    monkeypatch.delenv("ORCHA_TEST_SENTINEL", raising=False)
    dotenv_calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(dotenv_path: str | Path, *, override: bool) -> bool:
        dotenv_calls.append((Path(dotenv_path), override))
        monkeypatch.setenv("ORCHA_TEST_SENTINEL", "from-temporary-dotenv")
        return True

    async def fake_run_app(cfg: Any) -> int:
        assert cfg.cwd == tmp_path.resolve()
        return 0

    monkeypatch.setattr(entrypoint, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(entrypoint, "run_app", fake_run_app)
    argv = ["orcha", "--cwd", str(tmp_path)]
    if trusted:
        argv.append("--trust-cwd")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
    if trusted:
        assert dotenv_calls == [(env_file.resolve(), False)]
        assert os.environ["ORCHA_TEST_SENTINEL"] == "from-temporary-dotenv"
    else:
        assert dotenv_calls == []
        assert "ORCHA_TEST_SENTINEL" not in os.environ
