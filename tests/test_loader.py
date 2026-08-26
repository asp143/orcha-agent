from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from orcha_agent.core.config import Config, load_config
from orcha_agent.core.events import AppStart, EventBus
from orcha_agent.core.loader import load_plugins
from orcha_agent.core.plugin import ModeSpec, PluginSpec, ProviderCaps
from orcha_agent.core.registry import Registry


class EntryPoints(list[object]):
    def select(self, *, group: str) -> "EntryPoints":
        return EntryPoints(entry_point for entry_point in self if entry_point.group == group)


class EntryPoint:
    def __init__(
        self,
        module: ModuleType,
        *,
        name: str,
        group: str = "orcha_agent.plugins",
    ) -> None:
        self.group = group
        self.name = name
        self.value = module.__name__
        self.module = module.__name__
        self.attr = None
        self.extras: list[str] = []
        self.dist = SimpleNamespace(name="orcha-agent-test-plugin")
        self._module = module

    def load(self) -> ModuleType:
        return self._module


@pytest.fixture(autouse=True)
def isolate_plugin_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: EntryPoints(),
    )


def config_for(
    tmp_path: Path,
    *,
    plugin_dirs: tuple[Path, ...] = (),
    disabled: tuple[str, ...] = (),
    strict_plugins: bool = False,
) -> Config:
    cwd = tmp_path / "workspace"
    cwd.mkdir(exist_ok=True)
    return Config(
        model="anthropic:test",
        subagent_model="anthropic:test",
        summarizer_model="anthropic:test",
        mode="ask",
        backend="local_shell",
        memory=(),
        db_path=tmp_path / "sessions.db",
        cwd=cwd,
        resume=None,
        list_sessions=False,
        strict_plugins=strict_plugins,
        plugin_dirs=plugin_dirs,
        models={},
        providers={},
        plugins={"disabled": disabled},
    )


def write_mode_plugin(
    directory: Path,
    *,
    filename: str,
    name: str,
    mode: str,
    priority: int = 100,
    requires: tuple[str, ...] = (),
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        "from orcha_agent.core.plugin import ModeSpec, PluginSpec\n"
        f"PLUGIN = PluginSpec(name={name!r}, version='1.0', "
        f"requires={requires!r}, priority={priority})\n"
        "def register(api):\n"
        f"    api.add_mode({mode!r}, ModeSpec(description={name!r}, "
        "interrupt_on={}, allowed_tools=None))\n"
    )


def third_party_order(records: list[Any], names: set[str]) -> list[str]:
    return [record.name for record in records if record.name in names]


def test_discovers_plugin_from_explicit_directory(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "explicit-plugins"
    write_mode_plugin(
        plugin_dir,
        filename="directory_plugin.py",
        name="directory-plugin",
        mode="directory-mode",
    )
    registry = Registry()

    records = load_plugins(
        registry,
        EventBus(),
        config_for(tmp_path, plugin_dirs=(plugin_dir,)),
    )

    assert "directory-mode" in registry.modes
    assert "directory-plugin" in {record.name for record in records}


def test_untrusted_cwd_ignores_project_config_and_plugin(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    project_config_dir = project / ".orcha-agent"
    project_config_dir.mkdir(parents=True)
    (project_config_dir / "config.toml").write_text(
        '[core]\nmode = "project-mode"\n'
    )
    write_mode_plugin(
        project_config_dir / "plugins",
        filename="project_plugin.py",
        name="project-plugin",
        mode="project-mode-from-plugin",
    )
    cfg = load_config(
        [],
        env={"HOME": str(tmp_path)},
        cwd=project,
        user_config_path=tmp_path / "missing-user.toml",
    )
    registry = Registry()

    load_plugins(registry, EventBus(), cfg)
    captured = capsys.readouterr()
    notices = [*captured.out.splitlines(), *captured.err.splitlines()]

    assert cfg.mode == "ask"
    assert "project-mode-from-plugin" not in registry.modes
    assert len(notices) == 1
    assert "trust" in notices[0].lower()


@pytest.mark.parametrize("trust_source", ["user-config", "cli"])
def test_trusting_cwd_loads_project_config_and_plugin(
    tmp_path: Path,
    trust_source: str,
) -> None:
    project = tmp_path / "project"
    project_config_dir = project / ".orcha-agent"
    project_config_dir.mkdir(parents=True)
    (project_config_dir / "config.toml").write_text(
        '[core]\nmode = "project-mode"\n'
    )
    write_mode_plugin(
        project_config_dir / "plugins",
        filename="project_plugin.py",
        name="project-plugin",
        mode="project-mode-from-plugin",
    )
    user_config = tmp_path / "user.toml"
    argv: tuple[str, ...] = ()
    if trust_source == "user-config":
        user_config.write_text(f'[trust]\ndirs = ["{project}"]\n')
    else:
        argv = ("--trust-cwd",)
    cfg = load_config(
        argv,
        env={"HOME": str(tmp_path)},
        cwd=project,
        user_config_path=user_config,
    )
    registry = Registry()

    load_plugins(registry, EventBus(), cfg)

    assert cfg.mode == "project-mode"
    assert "project-mode-from-plugin" in registry.modes


def test_discovers_distribution_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = ModuleType("test_distribution_plugin")
    module.PLUGIN = PluginSpec(name="distribution-plugin", version="2.0", priority=25)

    def register(api: Any) -> None:
        api.add_mode(
            "distribution-mode",
            ModeSpec(description="from an entry point", interrupt_on={}, allowed_tools=None),
        )

    module.register = register
    ignored_module = ModuleType("test_unrelated_distribution_plugin")
    ignored_module.PLUGIN = PluginSpec(name="unrelated-plugin", version="1.0")

    def register_ignored(api: Any) -> None:
        api.add_mode(
            "unrelated-mode",
            ModeSpec(description="wrong entry-point group", interrupt_on={}, allowed_tools=None),
        )

    ignored_module.register = register_ignored
    entry_points = EntryPoints(
        [
            EntryPoint(module, name="distribution-plugin"),
            EntryPoint(
                ignored_module,
                name="unrelated-plugin",
                group="another_application.plugins",
            ),
        ]
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: entry_points.select(group=kwargs["group"]) if kwargs else entry_points,
    )
    registry = Registry()

    records = load_plugins(registry, EventBus(), config_for(tmp_path))

    assert "distribution-mode" in registry.modes
    assert "distribution-plugin" in {record.name for record in records}
    assert "unrelated-mode" not in registry.modes


def test_disabled_plugin_is_discovered_but_not_registered(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    write_mode_plugin(
        plugin_dir,
        filename="disabled_plugin.py",
        name="disabled-plugin",
        mode="must-not-exist",
    )
    registry = Registry()

    records = load_plugins(
        registry,
        EventBus(),
        config_for(
            tmp_path,
            plugin_dirs=(plugin_dir,),
            disabled=("disabled-plugin",),
        ),
    )

    assert "disabled-plugin" in {record.name for record in records}
    assert "must-not-exist" not in registry.modes


def test_plugins_load_in_priority_then_name_order(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    write_mode_plugin(
        plugin_dir,
        filename="zulu.py",
        name="zulu",
        mode="zulu-mode",
        priority=20,
    )
    write_mode_plugin(
        plugin_dir,
        filename="alpha.py",
        name="alpha",
        mode="alpha-mode",
        priority=20,
    )
    write_mode_plugin(
        plugin_dir,
        filename="first.py",
        name="first",
        mode="first-mode",
        priority=10,
    )

    records = load_plugins(
        Registry(),
        EventBus(),
        config_for(tmp_path, plugin_dirs=(plugin_dir,)),
    )

    assert third_party_order(records, {"alpha", "first", "zulu"}) == [
        "first",
        "alpha",
        "zulu",
    ]


def test_satisfied_requirement_loads_and_missing_requirement_skips(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    plugin_dir = tmp_path / "plugins"
    write_mode_plugin(
        plugin_dir,
        filename="base.py",
        name="base",
        mode="base-mode",
        priority=20,
    )
    write_mode_plugin(
        plugin_dir,
        filename="dependent.py",
        name="dependent",
        mode="dependent-mode",
        priority=10,
        requires=("base",),
    )
    write_mode_plugin(
        plugin_dir,
        filename="missing.py",
        name="missing-dependent",
        mode="missing-mode",
        priority=30,
        requires=("not-installed",),
    )
    registry = Registry()

    records = load_plugins(
        registry,
        EventBus(),
        config_for(tmp_path, plugin_dirs=(plugin_dir,)),
    )

    assert third_party_order(records, {"base", "dependent", "missing-dependent"}) == [
        "base",
        "dependent",
        "missing-dependent",
    ]
    assert {"base-mode", "dependent-mode"} <= set(registry.modes)
    assert "missing-mode" not in registry.modes
    assert "missing-dependent" in caplog.text
    assert "not-installed" in caplog.text


def test_failed_plugin_is_isolated_and_later_plugin_still_loads(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "isolated_bad.py").write_text(
        "from orcha_agent.core.plugin import PluginSpec\n"
        "PLUGIN = PluginSpec(name='isolated-bad', version='1.0', priority=10)\n"
        "def register(api):\n"
        "    raise RuntimeError('broken registration')\n"
    )
    write_mode_plugin(
        plugin_dir,
        filename="good.py",
        name="good",
        mode="survived-mode",
        priority=20,
    )
    registry = Registry()

    load_plugins(
        registry,
        EventBus(),
        config_for(tmp_path, plugin_dirs=(plugin_dir,)),
    )

    assert "survived-mode" in registry.modes
    assert "plugin isolated-bad failed: broken registration" in caplog.text


@pytest.mark.asyncio
async def test_failed_plugin_registration_rolls_back_registry_bus_and_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failed_module = ModuleType("test_transaction_failed_plugin")
    failed_module.PLUGIN = PluginSpec(
        name="transaction-failed",
        version="1.0",
        priority=10,
    )
    good_module = ModuleType("test_transaction_good_plugin")
    good_module.PLUGIN = PluginSpec(
        name="transaction-good",
        version="1.0",
        priority=20,
    )
    observed: list[str] = []

    def add_contributions(api: Any, label: str) -> None:
        def transaction_tool() -> str:
            return label

        async def transaction_command(ctx: Any, args: str) -> None:
            del ctx, args

        class TransactionMiddleware:
            name = "transaction-middleware"

        async def observe_start(event: AppStart) -> None:
            del event
            observed.append(label)

        api.add_command(
            "transaction-command",
            transaction_command,
            help=label,
        )
        api.add_tool(transaction_tool)
        api.add_middleware(TransactionMiddleware())
        api.add_renderer(
            "transaction-renderer",
            lambda event: f"{label}:{event!r}",
        )
        api.add_provider(
            "transaction-provider",
            lambda model, config: (label, model, config),
            capabilities=ProviderCaps(
                tool_calling=True,
                streaming=True,
                thinking=False,
                structured_output=False,
                max_context=None,
            ),
        )
        api.add_backend(
            "transaction-backend",
            lambda config: (label, config),
        )
        api.add_subagent(
            {
                "name": "transaction-subagent",
                "description": label,
                "system_prompt": f"{label} system prompt",
            }
        )
        api.add_mode(
            "transaction-mode",
            ModeSpec(description=label, interrupt_on={}, allowed_tools=None),
        )
        api.on(AppStart, observe_start)
        api.system_prompt_fragment("transaction prompt")

    def register_failed(api: Any) -> None:
        api.state["nested"]["value"] = "mutated"
        api.state["temporary"] = "must be rolled back"
        add_contributions(api, "failed")
        raise RuntimeError("registration stopped after partial contributions")

    def register_good(api: Any) -> None:
        add_contributions(api, "good")

    failed_module.register = register_failed
    good_module.register = register_good
    entry_points = EntryPoints(
        [
            EntryPoint(failed_module, name="transaction-failed"),
            EntryPoint(good_module, name="transaction-good"),
        ]
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: entry_points.select(group=kwargs["group"]) if kwargs else entry_points,
    )
    registry = Registry()
    bus = EventBus()
    state_by_plugin = {
        "transaction-failed": {
            "nested": {"value": "preserved"},
        }
    }

    records = load_plugins(
        registry,
        bus,
        config_for(tmp_path),
        state_by_plugin=state_by_plugin,
    )

    statuses = {record.name: record.status for record in records}
    assert statuses["transaction-failed"] == "failed"
    assert statuses["transaction-good"] == "loaded"
    await bus.emit(AppStart(ctx=object()))
    assert registry.tools["transaction_tool"]() == "good"
    assert registry.commands["transaction-command"].plugin == "transaction-good"
    assert registry.providers["transaction-provider"].plugin == "transaction-good"
    assert registry.backends["transaction-backend"].plugin == "transaction-good"
    assert registry.modes["transaction-mode"].description == "good"
    assert [
        entry.plugin
        for entry in registry.middleware
        if entry.name == "transaction-middleware"
    ] == ["transaction-good"]
    assert [
        entry.plugin
        for entry in registry.renderers
        if entry.name == "transaction-renderer"
    ] == ["transaction-good"]
    assert [
        entry.plugin
        for entry in registry.subagents
        if entry.name == "transaction-subagent"
    ] == ["transaction-good"]
    assert observed == ["good"]
    assert [
        fragment.plugin
        for fragment in registry.prompt_fragments
        if fragment.text == "transaction prompt"
    ] == ["transaction-good"]
    assert state_by_plugin["transaction-failed"] == {
        "nested": {"value": "preserved"},
    }


def test_strict_plugins_reraises_registration_failure(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "strict_bad.py").write_text(
        "from orcha_agent.core.plugin import PluginSpec\n"
        "PLUGIN = PluginSpec(name='strict-bad', version='1.0')\n"
        "def register(api):\n"
        "    raise RuntimeError('strict registration failure')\n"
    )

    with pytest.raises(RuntimeError, match="strict registration failure"):
        load_plugins(
            Registry(),
            EventBus(),
            config_for(
                tmp_path,
                plugin_dirs=(plugin_dir,),
                strict_plugins=True,
            ),
        )


@pytest.mark.parametrize(
    ("bad_source", "expected_exception", "message"),
    [
        ("def broken(:\n", SyntaxError, None),
        (
            "from orcha_agent.core.plugin import PluginSpec\n"
            "PLUGIN = PluginSpec(name='a-bad', version='1.0')\n",
            TypeError,
            "no callable register",
        ),
        (
            "PLUGIN = 'not a PluginSpec'\n"
            "def register(api):\n"
            "    pass\n",
            TypeError,
            "must be a PluginSpec",
        ),
        (None, RuntimeError, "entry point load failed"),
    ],
)
def test_discovery_failures_are_isolated_and_strict_mode_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_source: str | None,
    expected_exception: type[BaseException],
    message: str | None,
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    if bad_source is None:
        module = ModuleType("test_failing_entry_point")
        entry_point = EntryPoint(module, name="a-bad")

        def fail_load() -> ModuleType:
            raise RuntimeError("entry point load failed")

        monkeypatch.setattr(entry_point, "load", fail_load)
        entry_points = EntryPoints([entry_point])
        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda **kwargs: (
                entry_points.select(group=kwargs["group"]) if kwargs else entry_points
            ),
        )
        failure_source = f"entry-point:{module.__name__}"
    else:
        bad_path = plugin_dir / "a_bad.py"
        bad_path.write_text(bad_source)
        failure_source = str(bad_path.resolve())
    write_mode_plugin(
        plugin_dir,
        filename="z_good.py",
        name="z-good",
        mode="survived-discovery-failure",
    )
    registry = Registry()

    records = load_plugins(
        registry,
        EventBus(),
        config_for(tmp_path, plugin_dirs=(plugin_dir,)),
    )

    assert any(
        record.source == failure_source and record.status == "failed"
        for record in records
    )
    assert "survived-discovery-failure" in registry.modes

    with pytest.raises(expected_exception, match=message):
        load_plugins(
            Registry(),
            EventBus(),
            config_for(
                tmp_path,
                plugin_dirs=(plugin_dir,),
                strict_plugins=True,
            ),
        )
