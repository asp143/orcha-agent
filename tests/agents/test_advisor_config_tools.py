from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orcha_agent.builtin.tools_agents import agent_tools
from orcha_agent.core.config import AdvisorConfig, AgentsConfig, Config, load_config


def _load(tmp_path: Path, user_config_path: Path | None = None) -> Config:
    return load_config(
        [],
        env={"HOME": str(tmp_path)},
        cwd=tmp_path,
        user_config_path=user_config_path or tmp_path / "missing-user.toml",
        project_config_path=tmp_path / "missing-project.toml",
    )


def _tool_map(host: Any) -> dict[str, Any]:
    return {tool.name: tool for tool in agent_tools(host)}


class _AdviceRegistry:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(agents=AgentsConfig())
        self.recorded_hosts: list[Any] = []
        self.recorded_payloads: list[dict[str, Any]] = []

    async def record_advice(self, host: Any, payload: dict[str, Any]) -> None:
        self.recorded_hosts.append(host)
        self.recorded_payloads.append(payload)
        await host.advice_outbox.put(payload)


def _child_host(
    registry: _AdviceRegistry,
    *,
    name: str,
    spawns: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        source_id=f"{name}-run",
        owner=registry,
        agent_type=SimpleNamespace(name=name, spawns=spawns),
        depth=0,
        status="idle",
        advice_outbox=asyncio.Queue(),
    )


def test_advisor_config_defaults(tmp_path: Path) -> None:
    expected = AdvisorConfig(
        enabled=False,
        model="@advisor",
        tools=("read_file", "grep", "glob"),
        immune_turns=3,
        timeout_s=30.0,
    )

    assert AdvisorConfig() == expected
    assert _load(tmp_path).advisor == expected


def test_advisor_config_loads_toml_overrides(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[advisor]
enabled = true
model = "  fake:critic  "
tools = [" read_file ", "glob"]
immune_turns = 5
timeout_s = 1.5
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert _load(tmp_path, config).advisor == AdvisorConfig(
        enabled=True,
        model="fake:critic",
        tools=("read_file", "glob"),
        immune_turns=5,
        timeout_s=1.5,
    )


@pytest.mark.parametrize(
    "setting",
    [
        'enabled = "yes"',
        'model = ""',
        'tools = "grep"',
        'tools = ["grep", ""]',
        "immune_turns = 0",
        "immune_turns = true",
        "timeout_s = 0",
        "timeout_s = false",
    ],
)
def test_advisor_config_rejects_invalid_values(
    tmp_path: Path,
    setting: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"[advisor]\n{setting}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        _load(tmp_path, config)


def test_advise_is_added_only_to_advisor_child_tools() -> None:
    registry = _AdviceRegistry()
    main = SimpleNamespace(source_id="main", agents=registry)
    normal = _child_host(registry, name="task", spawns=True)
    advisor = _child_host(registry, name="advisor", spawns=False)

    assert set(_tool_map(main)) == {"task", "hub"}
    assert set(_tool_map(normal)) == {"task", "yield", "hub"}
    assert set(_tool_map(advisor)) == {"yield", "hub", "advise"}


@pytest.mark.asyncio
async def test_advise_note_and_none_enqueue_without_settling_advisor() -> None:
    registry = _AdviceRegistry()
    advisor = _child_host(registry, name="advisor", spawns=False)
    advise = _tool_map(advisor)["advise"]

    note_response = await advise.ainvoke(
        {"note": "  Check the retry boundary.  ", "severity": "concern"}
    )
    none_response = await advise.ainvoke({"none": True})

    assert note_response == {"accepted": True, "terminal": False}
    assert none_response == {"accepted": True, "terminal": False}
    assert registry.recorded_hosts == [advisor, advisor]
    assert registry.recorded_payloads == [
        {"note": "Check the retry boundary.", "severity": "concern"},
        {"none": True},
    ]
    assert advisor.advice_outbox.get_nowait() == registry.recorded_payloads[0]
    assert advisor.advice_outbox.get_nowait() == registry.recorded_payloads[1]
    assert advisor.status == "idle"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"note": "", "severity": "nit"},
        {"note": "Conflicting assessment", "severity": "blocker", "none": True},
    ],
)
async def test_advise_rejects_empty_or_ambiguous_calls(
    payload: dict[str, Any],
) -> None:
    registry = _AdviceRegistry()
    advisor = _child_host(registry, name="advisor", spawns=False)
    advise = _tool_map(advisor)["advise"]

    with pytest.raises(
        ValueError,
        match="advise requires exactly note and severity, or none=true",
    ):
        await advise.ainvoke(payload)

    assert registry.recorded_payloads == []
    assert advisor.advice_outbox.empty()
    assert advisor.status == "idle"
