# orcha-agent

Pluggable terminal coding agent built on LangChain deepagents. The kernel loads
built-in and third-party tools, commands, providers, modes, renderers,
middleware, backends, subagents, and event hooks through one `PluginAPI`.

## Install and run

```bash
uv sync
uv run orcha --help
uv run orcha
```

Optional model providers are extras:

```bash
uv sync --extra openai
uv sync --extra ollama
uv sync --extra google
```

Built-in provider prefixes are `anthropic:`, `openai:`, `ollama:`, `google:`,
and `langchain:`. Set the provider's documented environment variable, such as
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; orcha never stores API keys.

## Configuration

Precedence is CLI, `ORCHA_*` environment variables,
`./.orcha-agent/config.toml`, `~/.config/orcha-agent/config.toml`, then
defaults. Example:

```toml
[core]
model = "anthropic:claude-opus-5"
subagent_model = "fast"
summarizer_model = "fast"
mode = "ask"
backend = "local_shell"

[models]
fast = "anthropic:claude-haiku-4-5"

[plugins]
disabled = []
```

Modes: `ask` approves writes and execution, `edit` approves execution, `yolo`
auto-approves all tools, and `plan` exposes only read-only filesystem tools.

## Trust model

Project config (`./.orcha-agent/config.toml`), project plugins
(`./.orcha-agent/plugins/*.py`), and the project `.env` are loaded only when
the working directory is trusted. Trust a directory persistently in the user
config, or trust the current invocation explicitly:

```toml
# ~/.config/orcha-agent/config.toml
[trust]
dirs = ["/path/to/trusted/project"]
```

```bash
uv run orcha --trust-cwd
```

Project plugins execute Python with the same filesystem and shell access as
orcha, so do not trust repositories you have not reviewed.

## Plugins

Plugins are discovered from built-ins, the `orcha_agent.plugins` entry-point
group, `~/.config/orcha-agent/plugins/`, `./.orcha-agent/plugins/`, and
additional `--plugin-dir` paths. Each module exports:

```python
def register(api):
    api.add_tool(my_tool)
```

See `examples/plugins/hello.py` for a complete external plugin.

## Commands

`/help`, `/clear`, `/exit`, `/plugins`, `/providers`, `/sessions`,
`/resume <id>`, `/compact`, `/model <spec>`, and `/mode <name>`.
