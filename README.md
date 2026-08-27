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

Starting without a configured provider or API key is supported. Use `/help`,
`/providers`, `/plugins`, `/login`, or `/model` after the REPL starts.

Optional model providers are extras:

```bash
uv sync --extra openai
uv sync --extra ollama
uv sync --extra google
```

Built-in provider prefixes are `anthropic:`, `codex:`, `openai:`, `ollama:`,
`google:`, and `langchain:`. Set the provider's documented environment
variable, such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; orcha never stores
API keys.

## Codex (ChatGPT subscription) login

Codex uses a ChatGPT subscription through OAuth rather than an OpenAI API key.
Automatic mode opens a browser on a local desktop and uses device login on
headless or SSH sessions:

```bash
uv run orcha login codex
```

Select a mode explicitly when needed:

```bash
uv run orcha login codex --browser
uv run orcha login codex --device   # headless servers and SSH
uv run orcha login codex --paste    # paste a redirect URL or code
```

The same modes are available in the REPL:

```text
/login codex
/login codex browser
/login codex device
/login codex paste
/model codex:gpt-5.6-sol
```


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
banner = true

[models]
fast = "anthropic:claude-haiku-4-5"

[plugins]
disabled = []
```

Modes: `ask` approves writes and execution, `edit` approves execution, `yolo`
auto-approves all tools, and `plan` exposes only read-only filesystem tools.

Disable the startup banner with `[core] banner=false` or
`ORCHA_NO_BANNER=1`.

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

## Status bar

The prompt keeps an omp-style session footer updated with the selected model,
effort, mode, workspace, Git state, context use, cumulative tokens, and
estimated cost:

```text
󰚩 GPT-5.6 Sol · 󰪣 high · ask · deepagent · feat/pi-agent ?2 +1 · 26.3%/272k · 󰁨 12.4k↑ 3.1k↓ · 󰙺 $0.42
```

Use `/status` to print the same values vertically. Disable the footer or Nerd
Font icons in user or project configuration:

```toml
[ui]
statusbar = false
icons = false

[pricing."codex:gpt-5.6-sol"]
input = 5
output = 30
cache_read = 0.5
```


## Thinking display

Model reasoning streams before the answer in dim italic text. Main-agent reasoning
is shown by default; subagent reasoning is opt-in:

```toml
[ui]
thinking = "summary"  # default; main agent only
# thinking = "off"    # hide all reasoning
# thinking = "all"    # include subagent reasoning
```

Each reasoning block starts with `󰟶 thinking`, or `[thinking]` when
`[ui] icons=false`. `/thinking off` hides reasoning for the current session;
`/thinking on` restores main-agent summaries. The toggle is saved with the
session and restored by `/resume`.

Codex requests automatic reasoning summaries at the configured
`[providers.codex] reasoning_effort` or `medium`. OpenAI requests automatic
summaries when `[providers.openai] reasoning_effort` is set. Anthropic
requests adaptive summarized thinking while display is on.

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

`/help`, `/clear`, `/exit`, `/plugins`, `/providers`, `/login <prefix>`,
`/logout <prefix>`, `/sessions`, `/resume <id>`, `/compact`,
`/model <spec>`, `/mode <name>`, and `/thinking on|off`.
