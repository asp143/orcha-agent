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

`orcha --yolo` starts in `yolo` mode (no tool approvals); it is shorthand for
`--mode yolo`.

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

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=orcha_agent --cov-report=term-missing --cov-report=xml:coverage.xml
uv run python -m benchmarks
```

The benchmark command runs locally. CI runs lint, format checks, type checks,
and tests with coverage on Python 3.12 and 3.13, then builds and smoke-tests the
wheel in a clean virtual environment.

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

[models]
fast = "anthropic:claude-haiku-4-5"

[plugins]
disabled = []
```

Agent orchestration is configured independently from the main model:

```toml
[agents]
max_concurrency = 8
max_live_runs = 32
max_depth = 2
idle_ttl_s = 420
max_runtime_s = 0       # 0 disables the deadline
soft_request_budget = 200

[models.roles]
task = "anthropic:claude-sonnet-4-5"
scout = "fast"
reviewer = "anthropic:claude-opus-5"
advisor = "fast"

[advisor]
enabled = false
model = "@advisor"
tools = ["read_file", "grep", "glob"]
immune_turns = 3
timeout_s = 30
```

Role models fall back to the main model. The `task` role also falls back to the
legacy `core.subagent_model` when configured. Agent concurrency limits active
model turns, while `max_live_runs` caps all nonterminal workers across retained
sessions; depth limits child spawning. Idle workers park after `idle_ttl_s`, and
a positive `max_runtime_s` adds a deadline. The soft request budget asks a
worker to yield before aborting it ten requests later. Blocking task batches and
awaited hub sends use a positive `max_runtime_s` as their wait bound, or a
300-second safety bound when runtime deadlines are disabled.

When enabled, the advisor is one hidden, persistent worker per session.
`model="@advisor"` uses `[models.roles].advisor`, then the main model. It
checks completed main turns without delaying new prompts; `concern` and
`blocker` notes may trigger a follow-up no more often than once per
`immune_turns`, while `nit` is display-only. `timeout_s` bounds each check.
Optional watchdog instructions are loaded from the nearest `WATCHDOG.md` or
`~/.config/orcha-agent/WATCHDOG.md`.

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

## Terminal UI

The TUI runs inline rather than taking over the alternate screen. Settled
messages and tool output are committed to native terminal scrollback; only the
active transcript blocks, HUD, composer, and status line are redrawn. The
composer grows to eight wrapped rows and has three shapes: `box` (closed
frame), `claude` (open prompt rail), and `borderless` (text only).

Provider fallback retries currently run inside LangChain middleware without
scheduling events, so the TUI cannot show a retry countdown for them.

### UI configuration

These are all supported UI keys and their defaults:

```toml
[ui]
theme = "dark"
symbols = "nerd"
icons = true
thinking = "summary"
composer = "box"
banner = true
notify = false
statusbar = true

[ui.statusline]
preset = "default"
separator = "powerline-thin"
transparent = false
# left and right are omitted by default; lists override the preset groups.
```

`theme` is a theme name or `auto`; `symbols` is `nerd`, `unicode`, or `ascii`;
`thinking` is `summary`, `off`, or `all`; and `composer` is `box`, `claude`,
or `borderless`. `icons` is retained for compatibility: when `symbols` is
omitted, `icons=false` selects `ascii` and `icons=true` selects `nerd`.
Disable the welcome with `banner=false` or `ORCHA_NO_BANNER=1`.

The status line presets and their left/right groups are:

- `default`: `model mode path git context cost` / `subagents session`
- `ascii`: `model mode path git` / `subagents context cost`
- `minimal`: `model path` / `context`
- `compact`: `mode path git` / `context time`
- `full` and `nerd`: `model mode path git session` /
  `subagents tokens cache cost context time`

Available built-in segments are `model`, `mode`, `path`, `git`, `session`,
`subagents`, `tokens`, `cache`, `cost`, `context`, and `time`; plugins may add
more. Separators are `powerline`, `powerline-thin`, `slash`, `pipe`, `block`,
`none`, and `ascii`; the `ascii` preset also forces ASCII-safe output.
Override either group, remove status backgrounds, or both:

```toml
[ui.statusline]
preset = "compact"
separator = "pipe"
left = ["model", "mode", "path", "git"]
right = ["tokens", "context", "cost"]
transparent = true
```

`/status` prints the effective visible segments vertically when the status
line is enabled. Pricing overrides still use the top-level pricing table:

```toml
[pricing."codex:gpt-5.6-sol"]
input = 5
output = 30
cache_read = 0.5
```

### Renderer gallery

Use the non-interactive gallery to inspect every built-in block renderer in
each lifecycle state without starting a model session:

```bash
uv run orcha gallery
uv run orcha gallery --tool tool --state error --width 100 --expanded
uv run orcha gallery --plain > /tmp/orcha-gallery.txt
```

`--tool NAME` and `--state streaming|progress|success|error` filter the
matrix. `--width N` sets the simulated terminal width, `--expanded` reveals
expanded renderer details, and `--plain` disables ANSI styling for snapshots
or redirected output. Fixtures live in `orcha_agent/tui/gallery_fixtures/`.

### Themes and symbols

Built-in themes are `dark`, `light`, `ansi`, `dracula`, `nord`, and
`gruvbox`. `theme="auto"` chooses light or dark from `COLORFGBG`, defaulting
to dark when the terminal background cannot be determined. User themes are
JSON files in `~/.config/orcha-agent/themes/`. Project themes in
`./.orcha-agent/themes/` load only for a trusted working directory and take
precedence over user themes with the same filename.

A theme can define variables, any subset of color tokens, and symbol
overrides:

```json
{
  "name": "Ocean",
  "vars": {
    "blue": "#5fafff",
    "terminal": ""
  },
  "colors": {
    "accent": "$blue",
    "statusLineBg": "$terminal"
  },
  "symbols": {
    "overrides": {
      "icon.model": "M",
      "status.success": "ok"
    }
  }
}
```

Colors accept `#rrggbb`, palette indexes `0` through `255`, `$variable`
references, or `""` for the terminal default. Missing color tokens produce a
warning and inherit from `dark`; unknown tokens or invalid files are skipped
with a warning. Theme files may choose a `symbols.preset`; an explicit
`[ui] symbols` value (`nerd`, `unicode`, or `ascii`) overrides it. Setting
`icons=false` without an explicit preset forces the ASCII compatibility
preset. Theme `symbols.overrides` still apply when the terminal can encode
them; non-UTF output falls back to safe `ascii` symbols.

`/theme` opens a live-preview picker; `Esc` restores the prior theme and
`Enter` saves the selection with the session. `/theme <name>` switches
directly.

### Keybindings

Override bindings in `~/.config/orcha-agent/keybindings.toml`. A value may be
one key sequence or a list; spaces form chords such as `escape p`. The full
default action map is:

```toml
[bindings]
submit = ["enter", "c-j"]
newline = ["escape enter", "escape c-j"]
queue = "c-q"
dequeue = "escape up"
toggle_thinking = "c-t"
cycle_thinking_level = "s-tab"
expand_tools = "c-o"
model_picker = "escape p"
cycle_model = "c-p"
history_search = "c-r"
external_editor = "c-g"
clear_screen = "c-l"
interrupt = "c-c"
exit = "c-d"
tree = "escape escape"
agents = "escape a"
```

Plugins extend the action map with `PluginAPI.add_keybinding(...)` before
user overrides are applied. If two actions claim the same sequence, the last
definition wins and the TUI warns which action lost it; invalid or unknown
user entries also warn instead of replacing a working binding. `/keys` prints
the effective map, including plugin actions and conflict resolution.

### Composer, history, and queue

`Enter` submits. A trailing `\` turns that keypress into a newline, while
`Esc Enter` inserts a newline directly. In dot mode, a prompt containing only
`.` submits `keep going`. In bash mode, a prompt beginning with `!` runs the
remainder through the local shell in the working directory with a 60-second
timeout. `Ctrl+G` edits the current draft with `$VISUAL` or `$EDITOR`.

Prompt history is stored in `~/.local/share/orcha-agent/history.db` with
SQLite FTS5 search and is rebound to the active working directory and session
after `/resume`. `Ctrl+R` opens the searchable history overlay and returns the
selected prompt to the composer. Slash-command completion starts at `/`; `@`
completes project paths, while bare path completion indexes only after an
explicit `Tab`. Completion honors anchored rules, nested `.gitignore` files,
and negated descendants, does not follow directory symlinks, and excludes
`.git`, `.env*`, and `Credentials`. `Tab` accepts menu choices. Plugins may
add completion triggers.

While a turn streams, submitting or pressing `Ctrl+Q` queues prompts. A
submission made entirely of `->`/`=>` lines, or a consecutive `1.`/`2.` (or
`1)`/`2)`) numbered list, expands into a FIFO batch; otherwise multiline text
remains one prompt. `Esc Up` pulls the newest queued prompt back into the
composer, and the queue runs sequentially after the active turn.

`Esc` first closes completion. During streaming it cancels the turn and
restores the queued prompts as editable `->` lines. With the default tree
binding, double `Esc` or `Shift+Esc` opens the conversation tree from an idle,
empty composer. `Ctrl+C` clears a draft, otherwise cancels a streaming turn,
otherwise exits on a second press within one second. `Ctrl+D` exits
immediately and saves the current draft and queue. Draft, queue, thinking
level, path completion, and history scope are restored both when orcha starts
with `--resume <session-id>` and after in-app `/resume`.

### Overlays and session chrome

Pickers accept fuzzy filter text, arrows/Page Up/Page Down to move, `Enter`
to select, and `Esc` to cancel.

| Overlay | Trigger | Result |
| --- | --- | --- |
| Model | `/model` or `escape p` | Shows registered models and provider availability, then switches to the selection. |
| Session | `/sessions` or `/resume` | Shows saved-session age, directory, and entry count, then resumes the selection. |
| Tree | `/tree`, double `Esc`, or `Shift+Esc` | Shows the ledger hierarchy and branches at the selected entry. |
| Theme | `/theme` | Live-previews themes and persists the accepted selection; cancellation rolls back. |
| Approval | A tool approval interrupt | Previews shell commands, edits, or arguments and returns `approve`, `reject`, or `always` (`Y`, `N`, or `A`). |
| Ask | A plugin calls `await ctx.ui.ask(questions)` | Returns `{"kind":"submit","results":[...]}` with each answer's `id`, `selectedOptions`, and optional `customInput`. |
| History | `Ctrl+R` | Full-text searches prompt history and returns a selection to the composer. |
| Help | `/help` or `?` in an empty composer | Shows the effective command and keybinding reference; `Enter` or `Esc` closes it. |
| Agent Hub | `/agents` or `Alt+A` | Shows visible workers and jobs; inspect, message, cancel, revive, copy results, or drill into a worker transcript. |

The responsive welcome block shows the active model, mode, working directory,
recent sessions, trust/provider/plugin hints, and a rotating tip. During a
turn, a compact HUD above the composer shows up to seven todo items, running
subagents, and queued prompts. The terminal title tracks the session and adds
a spinner while working or a waiting marker for approval.

With `[ui] notify=true`, turn completion and approval requests notify only
after more than five seconds without a keypress. The TUI prefers
`notify-send` and falls back to terminal OSC 9 notifications; failures never
interrupt the session.

### Agent orchestration

Four built-in roles cover general work (`task`), read-only exploration
(`scout`), structured code review (`reviewer`), and optional post-turn guidance
(`advisor`). The main agent's `task` tool starts every item in a batch
concurrently. Nonblocking work continues in the background; completed results
are persisted and delivered through `hub`. Workers use `yield` for incremental
findings and terminal structured results, and use `hub` to list jobs, exchange
messages, wait for activity, inspect their inbox, or cancel work. The advisor
uses `advise` instead and remains alive across checks.

Open `/agents` or press `Alt+A` to inspect active, parked, and completed visible
workers. In the hub, arrows or `j`/`k` move, `/` filters, `t` toggles tree order,
`Enter` drills into a transcript, `m` messages, `x` cancels, `r` revives a
parked worker, and `y` copies its latest result.

### Thinking display

Model reasoning streams before the answer. `thinking="summary"` shows the
main agent only, `off` hides it, and `all` includes subagents. `/thinking
off|on` and `Ctrl+T` change display for the current session. The saved display
mode restores both at startup with `--resume <session-id>` and after in-app
`/resume`. `Shift+Tab` cycles the provider-gated inference level through
`off`, `low`, `medium`, `high`, and `max`.

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

Interactive pickers are used by `/help`, `/theme`, `/model`, `/sessions`,
`/resume`, and `/tree` when no explicit argument is supplied. Direct forms
remain available:

- session: `/clear`, `/new`, `/sessions`, `/resume [session-id]`,
  `/tree [--all]`, `/branch [--exact] <id-prefix>`, `/fork`, `/compact`, and
  `/export [--force] [path]`
- model and UI: `/model [provider:model[,provider:model...]]`, `/mode <name>`,
  `/thinking on|off`, `/theme [name]`, `/keys`, and `/status`
- providers and runtime: `/providers [prefix]`, `/plugins`,
  `/login <prefix> [browser|device|paste]`, `/logout <prefix>`, `/help`,
  and `/exit`
- orchestration: `/agents` and
  `/review [<base-ref>|--uncommitted|<commit>] [--fix]`

`/clear` resets the current session history, while `/new` starts a fresh
session.

`/model <spec>` (and the model picker) is remembered: the chosen model is
written to `[core] model` in `~/.config/orcha-agent/config.toml` and becomes
the default for new sessions. `--model` and `ORCHA_MODEL` still override it
for a single run.

### Parallel code review

`/review` filters the selected git diff, partitions complete hunks, and starts
1, 2, 4, or 8 reviewers for at most 100, 500, 2,000, or more changed lines.
Lockfiles, generated or minified files, binaries, and sensitive artifacts are
excluded before fan-out. Findings are validated, deduplicated by file, line,
and normalized title, sorted P0 through P3, shown in one review card, and fed
back to the main agent as guidance.

With no selector, review covers the merge base of `main` or `master` through
HEAD plus tracked and eligible untracked working-tree changes. A non-hex
`<base-ref>` replaces that base; `--uncommitted` covers index, worktree, and
eligible untracked changes; a 7-40 digit hexadecimal `<commit>` reviews only
that commit. `--fix` additionally tells the main agent to fix all P0/P1
findings immediately; it does not apply changes before the review completes.

### Export format

`/export` writes compact version-3 JSONL: a session header followed by every
ledger entry across all branches. The destination is created exclusively by
default, so an existing file is never overwritten; a leading `--force`
replaces it. The optional path is the remaining raw command text and may
contain spaces.

Unknown entry payload fields are normally flattened into the entry envelope.
If the payload contains any reserved key (`type`, `id`, `parentId`,
`timestamp`, `opaqueWrapped`, or `opaquePayload`), the export instead writes
the envelope metadata plus exactly:

```json
{"opaqueWrapped":true,"opaquePayload":{"original":"payload"}}
```

For an unknown entry type, import unwraps this exact marker-and-payload pair.
This preserves the original object losslessly even when its keys collide with
envelope metadata, without mistaking ordinary unknown fields for a wrapper.
