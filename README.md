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

The deterministic tmux responsiveness check needs no model credentials and
skips cleanly when tmux is unavailable:

```bash
uv run python tests/tui/tmux/verify.py
```

It runs an isolated 100x30 tmux server, streams two 30-line turns, renders a
three-agent fan-out card, resizes the idle TUI wider and narrower, and fails if
history grows during repaint, markers duplicate, cards stack, or visible frames
survive a resize.

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
defaults. Core behavior uses `[core]`, native tools use `[tools]`, terminal
settings use `[tui]`, and orchestration uses `[agents]`. Plugin-specific options
use `[plugins.<name>]`; the examples below can share one `config.toml`. Example:

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
tools = ["read", "grep", "glob"]
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

## Tools

Native tools are enabled by default with the local shell backend. Other backends
retain their deepagents tools. Configure them in the project or user config:

```toml
[tools]
native = true
edit_format = "replace"  # "replace" or "hashline"
read_summary = false    # opt in to outlines for eligible bare Python reads
max_read_bytes = 67108864  # 64 MB maximum source-file size
allowed_roots = []      # additional directories native file tools may access
shell_env_passthrough = []  # explicitly inherit these parent environment names
# Setting deny replaces the defaults; include every pattern you want to retain.
# deny = [".env*", "Credentials/", "*.pem", "*.key", "*.p12", "*.pfx",
#         ".ssh/id_*", "secrets.*", "credentials.json", "serviceAccountKey.json"]
```

Set `native = false` to use the original deepagents filesystem tools
(`read_file`, `write_file`, `edit_file`, `execute`, `grep`, `glob`, and `ls`).
Legacy tool names in configured scopes are mapped to native names when native
tools are enabled. The native set replaces those names rather than exposing duplicate tools;
`delete` remains available from the contained local backend. Tools retain the selected mode's
approval rules: `ask` approves edits, writes, deletion, and shell execution;
`edit` approves shell execution; `yolo` auto-approves; `plan` allows only reads.

| Tool | Arguments and behavior |
| --- | --- |
| `read` | `path`: numbered text, a directory listing, or an image note. Defaults to the first 200 lines / 20 KB; `:summary` requests an eligible Python declaration outline. |
| `edit` | `path`, `old_string`, `new_string`, optional `replace_all=false`; or transactional `edits=[{"old":"before","new":"after"}]`. Read the file first. Exact matching falls back to conservative typography, whitespace, and indentation matching; ambiguous candidates require more context. Writes atomically, preserves mode/EOL, and returns a unified diff. |
| `write` | `path`, `content`: create or overwrite a UTF-8 file, creating parent directories; returns size and diff information. |
| `bash` | `command`, optional `timeout`, `cwd`, `env`, `background`: execute builds, tests, git, and other commands in a persistent bash session. |
| `bash_jobs` | `action="list"`, `"read"`, or `"kill"`; `job_id` for read/kill, optional zero-based line `offset` and `limit=200` for reading. |
| `grep` | Regex `pattern`, optional `path`, `glob`, `case=true`, `context=0`, `limit`, `skip=0`; grouped content matches. Defaults to 20 files per page, or 200 matching lines when `path` is one file. |
| `glob` | `pattern`, optional `path`, `limit=200`, `include_hidden=false`; newest files first, grouped by directory. |
| `ls` | Optional `path="."`, `limit=200`; entries with sizes and modification times. |

Read selectors use one-based, inclusive line numbers:

```text
read(path="src/app.py:40")              # a page starting at line 40
read(path="src/app.py:40-65")           # an inclusive range
read(path="src/app.py:40+20")           # 20 lines starting at line 40
read(path="logs/build.log:-25")         # the last 25 lines
read(path="src/app.py:5-16,960-973")     # multiple ranges
read(path="src/app.py:raw")             # text without line-number gutters
read(path="src/app.py:summary")         # eligible Python declaration outline
```

Repeated reads of an unchanged range in the same user turn retain the content.
Starting with the third identical read, a reminder says re-reading will not change
the output. Oversized reads elide middle lines using a 60-line head and 25-line
tail, subject to the byte budget. Follow the returned selector to retrieve
omitted lines; `:raw` removes gutters but retains output limits.
Bare reads show the first page by default. Use `:summary`, or enable
`read_summary`, for an AST declaration outline of Python files with 500–20,000
lines and at most 2 MB. The outline identifies body ranges to read explicitly;
other languages and files outside those thresholds keep paginated reads.
Nonregular files are rejected, and images are identified by their contents.
Files above `max_read_bytes` are refused; raise that setting explicitly or use a
bounded shell command when inspecting larger files.

Native file tools restrict paths to the workspace and configured `allowed_roots`,
including checks for symlink escapes. The workspace's `.orcha/artifacts` directory
holds recoverable tool output. Relative allowed roots are resolved from the
workspace. The default deny patterns shown above block sensitive filenames and
`Credentials/` directories, including inside allowed roots. An explicit `deny`
list replaces those defaults; `deny = []` removes filename exclusions. Approval
prompts show resolved target paths. Containment and deny patterns apply only to
the file tools. `bash` is deliberately unconfined; the selected approval mode
controls shell execution, and shell commands can access resources outside the
allowed roots or matching deny patterns.

With `edit_format = "hashline"`, `read` supplies a `[path#TAG]` snapshot header
and numbered anchors. Pass `edit(patch="...")` using that exact header:

```text
[src/app.py#TAG_FROM_READ]
PUT 2.=3:
+replacement for lines two through three
PUT >$:
+append this line
```

`PUT <N:` and `PUT >N:` insert before or after a line; `CUT N` or
`CUT N.=M` remove lines. `MV new/path` moves a file and `REM` removes it.
Anchors in a section refer to the original read snapshot. Stale anchors can
be remapped only when the original text remains uniquely identifiable;
modified anchors require a fresh read. A tag observed for different file digests
is rejected for the rest of that session, including after snapshot eviction;
start a new session or use replace mode to recover from that ambiguity.

Search respects gitignore rules and excludes hidden entries by default.
`grep` uses ripgrep when available and a Python fallback otherwise, including
multiline regexes when the pattern contains a newline. `case=true` means
case-sensitive matching; `case=false` ignores case. In directory searches,
`limit` counts matching files (default and maximum 20), `skip` counts matching
files to omit, and each file shows at most 20 matching lines. When `path` names
one file, `limit` counts matching lines (default and maximum 200), and `skip`
counts matching lines to omit. Context lines do not count toward those limits.
The internal cap is 2,000 matching lines; at most the first 4 MB of each file is
scanned, with a notice when the rest is omitted. Follow the returned `skip`
cursor for another page, or search one file to inspect more matches. Invalid
ripgrep patterns return an error; fallback regex timeouts identify omitted files.
Large searches keep file descriptors bounded: files beyond the pinned rg batch
use the same timed fallback, with a notice.

Slash-free glob patterns match basenames recursively: `glob(pattern="*.py",
path="src")` includes Python files in nested directories. Patterns containing
slashes match relative paths. `glob` returns partial results after five seconds;
narrow the path/pattern or raise `limit` when a notice indicates omitted files.

```text
grep(pattern="class .*Service", path="src", glob="*.py", skip=0)
glob(pattern="**/*.py", path="src", limit=200)
bash(command="cd src; export BUILD_MODE=debug")
bash(command="printf '%s %s' \"$PWD\" \"$BUILD_MODE\"")
bash(command="uv run pytest -q", background=true)
bash_jobs(action="read", job_id="ID_FROM_BASH", offset=0, limit=200)
bash_jobs(action="kill", job_id="ID_FROM_BASH")
```

Bash cwd and exported variables survive calls with the same graph `thread_id`.
Subagents inheriting that thread share the shell; calls with distinct thread IDs
are isolated. Calls sharing a shell run serially. Per-call `env` applies only to
that command, including any exports inside that temporary subshell. To persist
a variable, use `bash(command="export BUILD_MODE=debug")` without an `env` argument. Background jobs inherit a snapshot and
do not change the foreground shell. The default timeout is 300 seconds;
nonzero values clamp to 1–3600, and `0` disables it. Timeout, cancellation,
and job termination kill the process group. After a timeout the next call
starts a shell with the last completed cwd/environment; shell functions and
other unexported state are lost. Restarting orcha starts fresh shells and jobs.
There is no PTY or interactive stdin interface; command stdin is closed.
Simple `cat`, `head`, `tail`, `grep`, `find`, and `ls` commands receive hints to
use native tools, and `rg --files` suggests `glob`. Flags with no native equivalent,
such as `tail -f` or `head -c`, pass through, as do composed and mutating commands.

New shells inherit only `PATH`, `HOME`, `LANG`, `LC_*`, `TERM`, `COLORTERM`, `USER`,
`LOGNAME`, `SHELL`, `TMPDIR`, `TZ`, and `XDG_*` from the parent. Names containing
`KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, or `AUTH` are also excluded
unless named explicitly in `shell_env_passthrough`. Provider keys therefore do
not reach the shell by default. Approval prompts include the effective working
directory, changes from the initial environment, and temporary command variables;
sensitive names and explicit passthrough values are redacted.

Bash output is limited to 200 lines / 20 KB per response. The model receives
ANSI-free text; the renderer also receives a bounded raw output artifact.
Raw command output is kept in a private temporary directory (mode `0700`). Only
clipped output is promoted to `.orcha/artifacts/large_tool_results/` in the starting
workspace, with mode `0600`, a recovery path, and counts in truncation notices.
Creating `.orcha/` also creates its own `.gitignore` containing `*`. Artifact
retention is capped at 200 files and seven days; closing the shell session removes
its temporary and promoted outputs.
For a single oversized line, follow the notice's Python byte-range command to
retrieve the omitted bytes. Background output can also be paged through
`bash_jobs`. Deepagents overflow handling remains active for oversized tool
responses.

## Optional Turso persistence and structured memory

SQLite remains the default and requires no additional dependency. Turso support is
opt in and uses a local libSQL embedded replica synchronized with a user-provided
Turso database:

```bash
uv sync --extra turso
export TURSO_DATABASE_URL="libsql://database-name-organization.turso.io"
export TURSO_AUTH_TOKEN="..."
```

Configure the replica and a stable logical workspace name in the user config:

```toml
[persistence]
backend = "turso"
replica_path = "~/.local/share/orcha-agent/turso-replica.db"
sync_on_start = true
sync_on_exit = true
# url may be set here when TURSO_DATABASE_URL is not used; it is not a secret.
# url = "libsql://database-name-organization.turso.io"

[memory_store]
backend = "hybrid"
workspace = "orcha-agent"
```

`hybrid` loads synchronized structured memories alongside configured local memory
files such as `AGENTS.md` and `CLAUDE.md`. Use `backend = "turso"` to load only
structured memory. Local repository memory files are never uploaded automatically.
The workspace is a logical cross-device identifier, not a local filesystem path.

Run `orcha sync` without starting a model or use `/sync` in the TUI. Structured
memories can be inspected and changed explicitly:

```text
/memory list
/memory show test-command
/memory set global language-preference Prefer Python examples
/memory set workspace test-command Run uv run pytest
/memory set-path orcha_agent/core persistence Preserve atomic ledger updates
/memory delete test-command
```

When structured memory is enabled, the main agent also receives `list_memories`,
`read_memory`, and `save_memory` tools. The system prompt permits writes only after
an explicit user request and rejects content that appears to contain credentials.

The Turso database contains complete session metadata, ledger entries, graph
checkpoints and pending writes, internal agent sessions, plugin state, and structured
memories. Composer prompt history in `~/.local/share/orcha-agent/history.db` remains
local. Treat the remote database as sensitive. Tokens are environment-only and are
never stored by orcha. Turso roaming currently assumes one active writer per session;
it is not a collaborative multi-writer protocol, and offline-write behavior is that
of the installed libSQL SDK rather than a guarantee made by orcha.

Selecting Turso does not upload an existing SQLite `sessions.db`; use a distinct
replica path. The adapter synchronizes before initializing a new local replica so an
existing remote schema is hydrated first. SDK, authentication, and synchronization
errors fail explicitly rather than silently falling back to SQLite.

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
composer grows to eight wrapped rows and supports five presets: `box`, `claude`,
`borderless`, `band`, and `rail`.

Provider fallback retries currently run inside LangChain middleware without
scheduling events, so the TUI cannot show a retry countdown for them.

### UI configuration

Terminal settings and their defaults:

```toml
[tui]
theme = "dark"
symbols = "nerd"
icons = true
thinking = "summary"
composer = "box"
banner = true
notify = false
statusbar = true
hyperlinks = true
vim = false
colorblind = false
mouse = "scroll"
synchronized_output = true
resize = "preserve"

[tui.statusline]
preset = "default"
separator = "powerline-thin"
transparent = false
# left and right are omitted by default; lists override the preset groups.
```

`theme` is a theme name or `auto`; `symbols` is `nerd`, `unicode`, `ascii`,
or `colorblind`; `thinking` is `summary`, `off`, or `all`; and `composer` is
`box`, `claude`, `borderless`, `band`, or `rail`. `icons` is retained for
compatibility: when `symbols` is omitted, `icons=false` selects `ascii` and `icons=true` selects `nerd`.
Disable the welcome with `banner=false` or `ORCHA_NO_BANNER=1`.

The legacy `[ui]` and `[ui.statusline]` tables remain supported for shared
appearance settings. `[tui]` overrides matching `[ui]` keys; when both specify
status-line settings, the entire `[tui.statusline]` table takes precedence.
`/settings` preserves the section supplying an existing setting. The newer
`hyperlinks`, `vim`, `colorblind`, `mouse`, `synchronized_output`, and `resize`
options belong in `[tui]`.

`vim=true` enables Vim editing in the composer. `hyperlinks=false` disables
terminal hyperlinks. Synchronized output groups terminal paint operations;
`resize="preserve"` preserves scrollback, while `resize="rebuild"` rebuilds the
visible display on resize.

The status line presets and their left/right groups are:

- `default`: `brand model mode path git context cost` / `subagents session`
- `ascii`: `brand model mode path git` / `subagents context cost`
- `minimal`: `brand model path` / `context`
- `powerline`: `brand model path git pr` / `token_rate usage context`
- `compact`: `brand mode path git` / `context time`
- `full` and `nerd`: `brand model mode path git session` /
  `subagents tokens cache cost context time`

Available built-in segments are `brand`, `model`, `mode`, `path`, `git`, `pr`,
`session`, `subagents`, `tokens`, `cache`, `cost`, `context`, `time`, `token_rate`,
`cache_hit`, `time_spent`, `hostname`, `vim`, and `usage`; plugins may add more. Separators are `powerline`, `powerline-thin`, `slash`, `pipe`, `block`,
`none`, and `ascii`; the `ascii` preset also forces ASCII-safe output.
Override either group, remove status backgrounds, or both:

```toml
[tui.statusline]
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

Built-in themes include `dark`, `light`, `ansi`, Catppuccin (latte, frappe,
macchiato, mocha), Dracula, Nord, Gruvbox, Tokyo Night, Solarized, One,
GitHub, Rosé Pine, Kanagawa and Everforest. Family variants use names such as
`gruvbox-dark`, `github-light`, `tokyo-night-day`, and `catppuccin-mocha`.
Earlier `dark-*` / `light-*` names remain accepted aliases. The original
Dracula/Nord palettes remain `dracula` / `nord`; imported variants are
`dracula-omp` / `nord-omp`. The duplicate `dark-catppuccin` palette now aliases
`catppuccin-mocha` and appears only once in the picker.

`theme="auto"` chooses light or dark from terminal background detection or
`COLORFGBG`, defaulting to dark when unavailable. User themes are JSON files in
`~/.config/orcha-agent/themes/`. Project themes in `./.orcha-agent/themes/` load
only for a trusted working directory and take precedence over user themes
with the same filename.

`mouse="scroll"` preserves native text selection by leaving button tracking off
outside overlays. The terminal wheel scrolls native scrollback; forwarded SGR
wheel reports also scroll the viewport. `mouse="full"` opts into application
wheel tracking and composer click focus; `mouse="off"` ignores viewport wheel
reports. Overlays retain their own mouse support. `symbols="colorblind"` changes
only glyphs; use `colorblind=true` to change palette colors.

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
`[tui] symbols` value (`nerd`, `unicode`, `ascii`, or `colorblind`) overrides it. Setting
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
clear_draft = "c-x c-k"
recall_draft = "c-x c-r"
peek_paste = "c-x c-p"
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

Bracketed pastes of at least five lines or 1,000 characters appear as compact
paste chips; submission expands each chip to its original text. `Ctrl+X Ctrl+P`
previews a paste, `Ctrl+X Ctrl+K` clears the draft, and `Ctrl+X Ctrl+R` recalls it.
Terminal control sequences are removed from pasted text.

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
| Settings | `/settings` | Changes terminal appearance and interaction settings. |
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

With `[tui] notify=true`, turn completion and approval requests notify when
the terminal is unfocused. Without focus reporting, five seconds of keyboard
inactivity is the fallback. The TUI prefers `notify-send`, then OSC 9 on
supported terminals, otherwise a bell; failures never interrupt the session.

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

Interactive pickers are used by `/help`, `/settings`, `/theme`, `/model`, `/sessions`,
`/resume`, and `/tree` when no explicit argument is supplied. Direct forms
remain available:

- session and persistence: `/clear`, `/new`, `/sessions`, `/resume [session-id]`,
  `/tree [--all]`, `/branch [--exact] <id-prefix>`, `/fork`, `/compact`,
  `/export [--force] [path]`, `/sync`, and `/memory ...`
- model and UI: `/model [provider:model[,provider:model...]]`, `/mode <name>`,
  `/thinking on|off`, `/theme [name]`, `/settings`, `/keys`, and `/status`
- providers and runtime: `/providers [prefix]`, `/plugins`,
  `/login <prefix> [browser|device|paste]`, `/logout <prefix>`, `/help`,
  and `/exit`
- extensions: `/skills`, `/skill <name> [args]`, `/skill:<name> [args]`, and
  `/mcp list|test|resources|prompts|add|remove|enable|disable|reload|reconnect`
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

## Skills

Put a `SKILL.md` in `.orcha-agent/skills/<name>/` or
`~/.config/orcha-agent/skills/<name>/`. Discovery searches project ancestors,
nearest first, with native skills before imported skills at each depth. Claude,
Codex, and GitHub skill directories are imported too. In an untrusted project,
user roots are searched first; conflicting project skill names are skipped with a
warning. Trusted projects retain project-first precedence with user skills as fallback.

```markdown
---
name: review
description: Review a change for correctness and missing tests
globs: ["**/*.py"]
---
Read the affected callers and report concrete findings with file references.
```

Only names and descriptions enter the prompt by default. The `skill` tool reads
bodies on demand; `skill(name="skill://review")` also accepts a `skill://<name>`
URI (the filesystem read tool does not handle that URI). For trusted skills,
`alwaysApply: true` includes the body immediately. `globs` patterns automatically
attach matching trusted skill instructions after a successful file tool call,
once per user turn, in the next model call's system context. Set
`[plugins.skills] auto_attach_globs = false` to disable this.
`hide: true` hides a skill from the model's prompt listing;
`disableModelInvocation: true` also prevents tool invocation. Both remain available
through explicit user commands. Hiding also suppresses automatic body inclusion.
Untrusted project skills never automatically include bodies (`alwaysApply` or
`globs`); explicit invocation still works. User skill directories remain trusted.
Discovery and subsequent reads reject symlinks escaping the discovery root,
and bodies are escaped inside their prompt wrappers. Use `--trust-cwd` to enable
project automatic instructions. Skill discovery has a 250 ms build gate and
continues in the background when it takes longer.

Run `/skills`, `/skill:review [args]`, or `/skill review [args]`. Skill commands
submit the body and arguments as a user turn and appear in composer completion.
Set `enabled`, `import_claude`, `import_codex`, or `import_github` to `false` under
`[plugins.skills]` to disable discovery or individual importers.

## MCP servers

MCP plugin options belong in `config.toml` under `[plugins.mcp]`; a top-level
`[mcp]` table is not supported. Server definitions live in the separate JSON file:

```toml
[plugins.mcp]
import_claude = true
import_codex = true
```

Configure `~/.config/orcha-agent/mcp.json` or trusted-project
`.orcha-agent/mcp.json`:

```json
{
  "mcpServers": {
    "local": {
      "type": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/server.py"],
      "timeout": 30,
      "enabled": true
    },
    "remote": {"type": "http", "url": "https://example.com/mcp"}
  }
}
```

Supported transports are `stdio`, streamable `http`, and legacy `sse`; `env` and
`headers` are optional mappings. Project servers require the existing project
trust setting (`--trust-cwd` for one invocation). Claude `.mcp.json` and Codex
`~/.codex/config.toml` servers are imported; disable them with `import_claude = false`
or `import_codex = false` under `[plugins.mcp]`. Server names cannot contain `__`.
Headers and URL credentials require HTTPS unless the URL host is loopback.
HTTPS URLs such as `https://user:pw@host/mcp` are accepted and use HTTP Basic
authentication over TLS; URL credentials are not categorically blocked. Codex imports recognize
`startup_timeout_sec`, `tool_timeout_sec`, `env_http_headers`, and
`bearer_token_env_var`; environment-backed headers resolve at connection time.
Malformed server entries are reported individually without disabling valid entries.

Connections start in the background. The first agent build waits at most 250 ms for them,
then can expose cached tool schemas from `~/.cache/orcha-agent/mcp/` while connection
continues. Tools use names such as `mcp__local__search`, follow exec-tier approval,
and share the ordinary output truncation limits. The status line shows connected
and enabled server counts. Failed connections retry with backoff; exit closes them.
JSON-RPC application errors are returned without reconnecting. `/mcp list` and
`/mcp test <name>` show the last connection error. Stdio diagnostics retain roughly
the last 50 lines in `~/.cache/orcha-agent/mcp/<name>.log`, with configured env values
redacted. Cache/log directories are private (0700), as are files (0600).

Try `/mcp list`, `/mcp test local`, `/mcp resources local`, or `/mcp prompts local`.
Manage configuration with `/mcp add local python /path/to/server.py`,
`/mcp add remote --url https://example.com/mcp`, `/mcp remove local`,
`/mcp enable local`, `/mcp disable local`, `/mcp reload`, and `/mcp reconnect [name]`.
Adds write to project config when trusted, otherwise user config. Removing an
imported server writes a native removal marker; enable/disable writes native
overrides. Claude `.mcp.json` and Codex TOML are never rewritten or chmodded.
Removal markers prevent lower-priority definitions from reappearing.

## Custom slash commands

Place Markdown files in `.orcha-agent/commands/*.md` or
`~/.config/orcha-agent/commands/*.md`. Recursive `.claude/commands/` and
`~/.claude/commands/` are also imported: `foo/bar.md` supplies `/foo:bar`. The short
`/bar` alias exists only for trusted, unambiguous definitions. User commands take
precedence over all untrusted project commands and aliases. With a trusted project,
native commands win over imports and project commands win within each format.
Existing built-in commands cannot be replaced. Command roots must resolve within
their project or home scope before files are enumerated; symlinked importer
directories cannot redirect discovery outside that scope.

```markdown
---
description: Explain a module
argument-hint: <path>
---
Explain $1 and list its main callers. Additional context: $@[2:]
```

Run `/explain orcha_agent/core/agent.py` after saving this as `explain.md`.
Trusted frontmatter also accepts `model` for a temporary per-turn agent, without
changing the saved model or adding model-switch ledger entries. Untrusted model
overrides are ignored with one warning per file. Templates support `$1` through `$9`, `$@`, `$ARGUMENTS`, and
one-based `$@[start:len]` slices. Arguments are appended when no placeholders exist.
Shell interpolation such as `` !`git status --short` `` runs only for trusted files
(user command directories or trusted projects), with a timeout and output cap.
Set `[plugins.file_commands] import_claude = false` to disable Claude imports.

## Context files

The context plugin walks from the current directory to the nearest Git root and
loads one file at each depth, in this priority order:
`.orcha-agent/AGENTS.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, then
`.github/copilot-instructions.md`. Global `~/.config/orcha-agent/AGENTS.md` and
`~/.claude/CLAUDE.md` precede project instructions. Broader instructions appear
before closer ones. Deeper files are listed as pointers, without loading their bodies.
Outside Git repositories, ancestry stops at home; a cwd outside home is the only
project directory searched. Builds wait at most 250 ms for discovery; late results
apply on a subsequent build, and discovery failures are logged.

References such as `@docs/conventions.md` expand relative to the importing file;
code examples remain literal. Imports are bounded, cyclic imports stop, and
untrusted project files and resolved import targets must remain inside the nearest
Git root, or inside cwd when no Git repository exists. Discovering ancestors up to
home never grants an untrusted project access to the rest of home. Absolute and `~` imports in untrusted project files stay literal; symlink
escapes are rejected. Home-scope instruction files are unaffected by the project
containment rule. Regardless of trust, instructions never inline `.ssh/`, `.aws/`,
`.gnupg/`, `.docker/`, `.kube/`, `.netrc`, `.git-credentials`, `.npmrc`, `.pypirc`,
`*.pem`, `*.key`, `*.p12`, `*.pfx`, `credentials.json`, `secrets.*`, `.env*`, or
`Credentials/`, including symlink disguises. Prompt content is framed as
`<repo-rules>` with `<file path="…">` and pointer-only `<dir-context>` entries.

```toml
[plugins.context_files]
enabled = true
import_claude = true
import_cursor = true
import_github = true
max_bytes = 65536
max_import_depth = 5
```

The ladder replaces the default root-only memory loading; explicitly configured
custom memory sources remain available. Turso's structured-memory-only mode stays
structured-only. To try it, add `AGENTS.md` at the repository root and a closer
`.orcha-agent/AGENTS.md`, launch from that directory, and ask the agent which
repository instructions apply.

### HTML session export

`/export --html [path]` saves a standalone HTML transcript using the active theme
colours. Markdown is rendered locally; tool calls and results use expandable
cards, with added/removed diff lines highlighted. Every ledger branch is included
in chronological order. No scripts, remote assets, or image requests are needed.
The default filename is `<session-id>.html`; paths may contain spaces.
As with JSONL export, existing files require `--force`, for example
`/export --html --force review.html`. The output file is private (0600).

## Declarative hooks

Add `[[hooks]]` entries to `~/.config/orcha-agent/config.toml` or trusted
`.orcha-agent/config.toml`. Project hooks load only with `--trust-cwd` or an
existing trusted directory; user and project hooks are combined. `/hooks` lists
active hooks.

```toml
[[hooks]]
event = "tool_call_before"
matcher = "write*"
command = "python scripts/check_write.py"
timeout = 10
blocking = true
```

Events are `session_start`, `session_end`, `turn_start`, `turn_end`,
`tool_call_before`, `tool_call_after`, `model_switch`, `compaction`,
`agent_spawned`, and `agent_finished`. Tool matchers are name globs; `regex:`
selects a regex over the JSON payload. Other matchers are regexes over event
text (or its JSON payload). Commands receive JSON on stdin and run in the
workspace. Alternatively, `python = "package.module:function"` invokes a sync
or async function with the payload in a bounded subprocess.

Exit 0 succeeds; exit 2 blocks a before-tool call with stderr as the explanation.
A successful before hook can return `{"block": true, "message": "reason"}` or
`{"args": {"path": "corrected-path", "content": "..."}}` to replace arguments
for file-writing tools. Hooks run in declaration order and stop on a block.
Other exit codes and timeouts produce warnings. `blocking = false` runs an
observational hook in the background; it cannot block or rewrite a call.

## First-run setup

On an interactive first launch with no user or project config and no usable
provider, orcha opens a short setup wizard inside the existing TUI. Run
`uv run orcha setup` or `/setup` to revisit it explicitly. Choose a theme,
composer style, provider sign-in or API-key environment hint, then a model.
Choose **Enter model name** for a provider without a model catalog, or configure
it later with `/model`.
The wizard never asks for or stores an API key; OAuth sign-in uses the existing
provider login flow. When the provider is usable, the selected model switches through the normal
session lifecycle. Otherwise it becomes the next-launch default while the
current session retains its model. Accepted preferences are saved in
`~/.config/orcha-agent/config.toml`, preserving other settings.

Press `Esc` at any step to skip without saving partial preferences. A completed
setup suppresses the automatic wizard on later launches. Noninteractive launches
and resumed sessions do not automatically open it.

## Magic keywords

Standalone lowercase `ultrathink` requests the highest reasoning effort for the
current turn; `orchestrate` adds a hidden system reminder to fan out independent
work with `task`. Standalone `plan` makes that turn read-only when the plan mode
is registered: only its read tools are exposed, and write/delegation calls are
blocked. These controls do not change saved model settings or the session mode.
Keywords inside fenced or inline code, XML/HTML sections, identifiers, paths,
filenames, and immediate function calls remain literal. The notices enter only
the model request and do not appear as extra conversation messages.
