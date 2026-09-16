# OMP parity integration — 2026-09-17

The four reviewed heads were merged, in order and without squashing, into
`integration/omp-parity`. Work stayed in this worktree; nothing was pushed.

## Merge history and conflict resolution

| Merge | Reviewed head | Integration commit | Conflicts and resolution |
| --- | --- | --- | --- |
| Performance | `24d4783` | `e8f487a` | None. Retained lazy startup/model resolution, fragment/revision caches, redraw coalescing, offloaded persistence, capture cursors, cached status snapshots, and bounded pickers. |
| Native tools | `32bd0eb` | `81ef3b7` | `tui/runtime.py`: kept the performance runtime and removed its fallback `/review` registration so the real built-in plugin owns that command. Native renderer keys/artifacts and tool configuration were retained. |
| Skills, MCP, commands | `8f9da4a` | `bae95f4` | `tui/runtime.py`: retained performance behavior and plugin-owned `/review`; added command-discovery waiting and guaranteed `AppExit` cleanup. `tests/test_builtins.py`: combined plugin/command expectations, including native tools, review, skills, MCP, context files, and file commands. |
| TUI parity | `c7e746b` | `54119c9` | `tui/runtime.py`, `tui/blocks/tool.py`, `tui/overlays/session.py`, and `tui/statusline.py`; details below. |

The final runtime combines redraw coalescing and revision caching with synchronized
output, watchdogs, mouse handling, and settings. Cache keys include theme changes
and individual tool expansion; live viewport captures strip hyperlink wrappers
while scrollback retains them. The tool renderer combines native argument/result
keys, hashline selectors, and raw shell artifacts with syntax highlighting, card
restyling, diffs, and terminal replay. Truncation/timeout notices remain outside
raw terminal replay so cursor control cannot erase them. Session search retains
bounded pages, cancellation, and precomputed labels while searching IDs and full
paths. Status imports and behavior combine cached snapshots with all new segments.

No golden file had a textual merge conflict. Merged renderer output was covered
by the golden and live-render checks; native read grouping and shell diagnostics
have additional integration regressions.

## Gates

Every merge passed `uv sync`, Ruff lint and format checks, Pyright, the full
ordinary and clean-environment pytest suites, `orcha gallery`, and the isolated
tmux verification harness before the next merge proceeded.

| Merge | Ordinary pytest | `env -i HOME=$HOME PATH=$PATH TERM=xterm-256color` pytest |
| --- | --- | --- |
| Performance | 1,176 passed, 2 skipped | 1,176 passed, 2 skipped |
| Native tools | 1,403 passed, 2 skipped | 1,403 passed, 2 skipped |
| Skills/MCP | 1,622 passed, 2 skipped | 1,622 passed, 2 skipped |
| Final TUI union | 1,755 passed, 2 skipped | 1,755 passed, 2 skipped |

The two skips are optional native-libsql tests; the default dependency set was
used. They are not evidence of remote Turso validation.

Integration-specific checks include:

- `tests/tui/test_omp_integration.py`: a scripted provider drives native `read`,
  `edit`, and `bash`, a skill invocation, and an in-process MCP SDK server (`MCPServer`) tool through
  the real inline runtime, checking card output and effects.
- `tests/tui/test_discovery_first_paint.py`: first paint completes while skill
  discovery and MCP startup are deliberately blocked.
- `tests/test_mcp_native_approvals.py`: MCP execution-tier approvals honor both
  legacy `execute` and native `bash` policy, explicit approvals, and tool scopes.
  The native case failed before the fix.
- Renderer/cache/picker regressions cover same-ID theme refresh, individual card
  expansion, raw-output diagnostics, native read selectors, and bounded session
  searches by hidden ID/full path.
- A pre-existing headless keybinding timing race was removed: the test waits for
  actual first paint, explicitly terminates the ambiguous Ctrl+X prefix, and
  awaits handler completion. Assertions and the action deadline were preserved;
  no retries or increased timeouts were introduced.

## Live smoke and startup

The isolated `orcha-int` tmux session used a 120×40 terminal and temporary database.
Captured welcome, `/theme`, `/settings`, `/model`, `/help`, `/skills`, `/mcp list`,
multiline paste, and `/status` frames are in `/tmp/orcha-int-frames` for this run.
The frames showed intact composer/status placement and no literal escape sequences.
`/tmp/orcha-int.err` contained zero bytes. Ctrl+D exited successfully. `/tools`
is not registered, so that optional command was not exercised.

Twenty actual `uv run orcha --help` executions measured median **73.080 ms**,
p95 **74.480 ms**, and maximum **78.787 ms**, all below the requested 150 ms.
These include uv overhead, unlike the benchmark's direct-executable help samples.

## Benchmarks

Command: `uv run python -m benchmarks`; 20 samples per case, 20 fresh startup processes. Baselines: Sprint 1 after for startup, streaming, history and sessions; Review round 1 for capture and cold ledger/fork/context; Review round 2 for fresh-store and warm ledger. Review round 1 contains no replacement streaming/startup/overlay numerical tables. Lower timing values are better. These are separate runs, not a controlled paired comparison.

Measured commit: `54119c98fb168af1a3ecd396929548368ee4ad44`; clean tree: `True`; scoped SHA-256: `55dc62a7cb77cee515afa5853a01b3898c44020a12ae602f2f6170619a55babd`.
Environment: CPython 3.12.13; `Linux-7.2.3-arch1-3-x86_64-with-glibc2.44` (same reported Python/kernel as latest baseline).

| Suite | Generated UTC |
| --- | --- |
| startup | 2026-09-16T16:41:55+00:00 |
| streaming | 2026-09-16T16:43:28+00:00 |
| ledger | 2026-09-16T16:43:32+00:00 |
| turn_capture | 2026-09-16T16:43:42+00:00 |
| history_load | 2026-09-16T16:43:55+00:00 |
| session_overlay_load | 2026-09-16T16:44:18+00:00 |

### Timings

| Case / metric | Baseline median → integration | Median Δ | Baseline p95 → integration | P95 Δ |
| --- | ---: | ---: | ---: | ---: |
| CLI help | 56.822 → 58.363 ms | +2.7% | 60.243 → 59.655 ms | -1.0% |
| Gallery plain | 155.902 → 613.425 ms | +293.5% | 167.666 → 625.619 ms | +273.1% |
| Stream 100 KiB / 1-byte chunks | 3.466 → 3.536 s/MiB | +2.0% | 3.527 → 3.577 s/MiB | +1.4% |
| Stream 100 KiB / 10-byte chunks | 0.347 → 0.353 s/MiB | +1.8% | 0.348 → 0.357 s/MiB | +2.4% |
| Stream 100 KiB / 100-byte chunks | 0.035 → 0.036 s/MiB | +1.6% | 0.036 → 0.036 s/MiB | -0.6% |
| Stream 1024 KiB / 1-byte chunks | 3.590 → 3.492 s/MiB | -2.7% | 3.753 → 3.518 s/MiB | -6.3% |
| Stream 1024 KiB / 10-byte chunks | 0.360 → 0.360 s/MiB | -0.1% | 0.364 → 0.363 s/MiB | -0.2% |
| Stream 1024 KiB / 100-byte chunks | 0.036 → 0.036 s/MiB | -0.7% | 0.037 → 0.037 s/MiB | +0.2% |
| Changed viewport 100 KiB | 86.471 → 8.992 ms | -89.6% | 192.656 → 9.076 ms | -95.3% |
| Cached viewport 100 KiB | 0.453 → 0.436 ms | -3.8% | 0.464 → 0.450 ms | -3.1% |
| Changed viewport 1024 KiB | 1310.282 → 99.735 ms | -92.4% | 1346.541 → 269.616 ms | -80.0% |
| Cached viewport 1024 KiB | 3.881 → 3.732 ms | -3.8% | 5.082 → 3.974 ms | -21.8% |
| Ledger first_load / 0 abandoned | 8.802 → 8.413 ms | -4.4% | 10.160 → 169.095 ms | +1564.3% |
| Ledger path / 0 abandoned | 2.330 → 2.342 ms | +0.5% | 2.423 → 2.396 ms | -1.1% |
| Ledger cold_path / 0 abandoned | 8.126 → 8.278 ms | +1.9% | 9.171 → 9.588 ms | +4.5% |
| Ledger fork / 0 abandoned | 7.164 → 7.327 ms | +2.3% | 8.144 → 7.844 ms | -3.7% |
| Ledger build_context / 0 abandoned | 0.378 → 0.380 ms | +0.6% | 0.399 → 0.390 ms | -2.2% |
| Ledger first_load / 10,000 abandoned | 9.633 → 8.746 ms | -9.2% | 121.931 → 10.193 ms | -91.6% |
| Ledger path / 10,000 abandoned | 2.430 → 2.394 ms | -1.5% | 2.741 → 2.469 ms | -9.9% |
| Ledger cold_path / 10,000 abandoned | 8.432 → 8.527 ms | +1.1% | 117.163 → 9.933 ms | -91.5% |
| Ledger fork / 10,000 abandoned | 7.560 → 7.351 ms | -2.8% | 9.611 → 9.359 ms | -2.6% |
| Ledger build_context / 10,000 abandoned | 0.379 → 0.382 ms | +0.9% | 0.423 → 0.401 ms | -5.1% |
| Ledger first_load / 100,000 abandoned | 9.343 → 9.412 ms | +0.7% | 120.806 → 171.825 ms | +42.2% |
| Ledger path / 100,000 abandoned | 2.555 → 2.551 ms | -0.2% | 2.625 → 3.111 ms | +18.5% |
| Ledger cold_path / 100,000 abandoned | 9.008 → 8.672 ms | -3.7% | 118.013 → 9.610 ms | -91.9% |
| Ledger fork / 100,000 abandoned | 7.877 → 7.893 ms | +0.2% | 9.601 → 9.859 ms | +2.7% |
| Ledger build_context / 100,000 abandoned | 0.383 → 0.389 ms | +1.6% | 0.396 → 0.417 ms | +5.4% |
| Capture 100 turns stable | 0.097 → 0.097 ms | -0.1% | 0.111 → 0.103 ms | -7.1% |
| Capture 100 turns unchanged 102400 bytes | 0.096 → 0.097 ms | +1.4% | 0.105 → 0.109 ms | +3.5% |
| Capture 1000 turns stable | 0.413 → 0.422 ms | +2.1% | 0.455 → 0.503 ms | +10.5% |
| Capture 1000 turns unchanged 102400 bytes | 0.408 → 0.416 ms | +1.9% | 0.428 → 0.445 ms | +4.1% |
| History data / 10,000 rows | 0.615 → 0.638 ms | +3.8% | 0.639 → 0.673 ms | +5.4% |
| History overlay / 10,000 rows | 0.816 → 0.830 ms | +1.8% | 0.863 → 0.935 ms | +8.4% |
| History data / 100,000 rows | 0.619 → 0.622 ms | +0.5% | 0.683 → 0.716 ms | +4.8% |
| History overlay / 100,000 rows | 0.810 → 0.835 ms | +3.0% | 0.845 → 0.890 ms | +5.3% |
| Full session list / 10,000 rows | 23.602 → 23.631 ms | +0.1% | 25.203 → 24.716 ms | -1.9% |
| Session overlay / 10,000 rows | 7.078 → 7.052 ms | -0.4% | 7.242 → 7.130 ms | -1.6% |
| Full session list / 100,000 rows | 395.651 → 446.417 ms | +12.8% | 399.740 → 457.015 ms | +14.3% |
| Session overlay / 100,000 rows | 7.025 → 7.022 ms | -0.0% | 7.172 → 7.352 ms | +2.5% |

### Memory and interpretation

| Startup peak RSS | Baseline median → integration | Baseline p95 → integration |
| --- | ---: | ---: |
| help | 21.475 → 22.318 MiB | 21.582 → 22.355 MiB |
| gallery_plain | 36.359 → 84.324 MiB | 36.484 → 84.719 MiB |

- CLI help remains fast. Separately measured actual `uv run orcha --help` (including uv overhead) had median 73.080 ms, p95 74.480 ms, max 78.787 ms over 20 runs; every sample was below 150 ms.
- The gallery wall-time regression came from the eager ledger import: gallery fixtures imported overlay surfaces, the overlay package imported the hub, and the hub imported `core.ledger`, loading LangChain, LangSmith, and Pydantic. Lazy overlay exports and a local ledger import now defer that dependency chain until the hub needs ledger messages; the regression was not caused by rendering.
- Changed viewport paint improves about 90–92% at the median. Rich Markdown layouts per displayed revision reports 0 versus baseline 1 at both payload sizes: instrumentation wraps base `rich.markdown.Markdown.__rich_console__` after warmup, while the new `StreamingMarkdown` caches pieces across revisions of identical text. This counts no additional base Markdown layouts for the warmed identical payload, not zero rendering or proof that changed text never needs layout.
- Fresh-store ledger tail latency varies markedly: p95 worsens to 169.095 ms for zero abandoned entries and 171.825 ms for 100,000, but improves to 10.193 ms for 10,000. Warm medians remain essentially stable. The baseline also records large tail outliers; this run did not profile the source of its outliers, so no new causal attribution is established.
- Full loading of 100,000 sessions regresses 12.8% median / 14.3% p95; the bounded session picker stays about 7 ms. Capture medians remain within approximately 2.2% of the latest baseline; 1,000-turn stable-state p95 increases 10.5%.
- History overlay median rises 1.8% at 10,000 rows and 3.0% at 100,000; p95 rises 8.4% and 5.3%, respectively. Streaming accumulation medians range from roughly -2.7% to +2.0%.
- Capture database/WAL sizes remain near the published baseline: final databases approximately 0.141/0.238 MiB for 100 turns and 0.531/0.629 MiB for 1,000 turns (stable/100 KiB state), with WAL approximately 3.352/3.454 MiB and 3.959/3.957 MiB.

Raw JSON artifacts remain ignored in `benchmarks/results/`; the benchmark command
completed successfully. The report-only commit follows the measured code commit.


## Documentation and remaining naming differences

README now presents one configuration scheme with `[core]`, `[tools]`, `[tui]`,
`[agents]`, and plugin-specific tables. It documents legacy `[ui]` precedence,
native tool-name aliases, all status presets/segments, composer/paste controls,
and consistent native skills/commands paths. TOML examples parse and headings
are unique. The manifest and lockfile contain MCP, PyYAML, and pyte together;
`uv lock --check` passed.

Two expectation differences remain explicit:

- The merged theme picker has **35 distinct themes**, rather than the task's
  nominal 36. The reviewed `c7e746b` branch also has 35 distinct theme files and
  registered IDs; no theme was lost in integration. Decide whether another
  palette is desired.
- MCP uses `[plugins.mcp]` options and `mcp.json` server definitions. There is no
  top-level `[mcp]` configuration table; README documents the implemented names.

## Execution and review

Independent renderer, runtime/persistence, extensibility/integration-test, and
documentation research or edits were delegated in parallel with explicit file
ownership. Independent static gates and read-only investigations ran concurrently.
Merge ordering, dependent conflict integration, full-suite runs, and commits stayed
serial. Final benchmarks ran after validation without concurrent test workloads.
The full combined changes received a review pass, with integration findings fixed
and covered before final validation.

## Review follow-up — 2026-09-17

The review corrections are four small commits:

- `f260563`: the outermost terminal transaction invalidates a skipped redraw when
  it closes, regardless of current queue depth. Four regression cases failed
  before the fix and pass afterward, covering nested frames and high/drained
  queues without a watchdog tick.
- `531b728`: `_PaintOutput.set_enabled()` owns the preference/capability decision.
  Constructor, terminal reports, and settings updates use it. Tests check that
  disabling synchronization mid-frame still closes the existing DECSET-2026
  transaction and that unsupported terminals remain unsynchronized.
- `2a08cf1`: overlay exports and factories load lazily; the hub imports the ledger
  only when reading a session. Fresh-process tests assert that gallery and hub
  imports do not load `langchain_core`. The attribution above now identifies the
  eager ledger dependency chain, rather than blaming gallery rendering.
- `a71e768`: the keybinding test subscribes to `after_render` before startup and
  awaits concrete thinking-rebuild, model-switch, and plugin-completion events.
  A controlled model-switch barrier demonstrates that plugin completion does
  not imply model completion. This replaces the earlier short action deadline
  and transient-renderer-state check; cleanup releases barriers and joins the
  runtime even on failure.

Final follow-up gates: `uv sync`, Ruff check/format, Pyright, gallery, tmux
verification, and the full default benchmark suite passed. Ordinary pytest:
**1,763 passed, 2 skipped in 50.24 s**; the exact clean-environment command:
**1,763 passed, 2 skipped in 46.95 s**. The skips remain optional native-libsql
checks. Tmux recorded zero repaint growth, one occurrence per marker, and passing
paste, expansion, mouse, and resize checks. Existing golden assertions passed.

### Follow-up startup measurements

The full default benchmark ran after tests, with 20 samples and no concurrent
test workloads. It measured clean code commit
`a71e76809554bfde83b47eda7a338e67afe9b4b6`, scoped SHA-256
`ab06f1b5a612c4693da75ba809ebe2230ef26640bc35d740e7ee0e3b2c106ee6`,
on CPython 3.12.13 / Linux 7.2.3. All six result files in `benchmarks/results/`
record that same clean tree. The earlier tables describe the original integration;
these startup measurements supersede its gallery regression.

| Direct-executable benchmark | Original integration median | Follow-up median | Follow-up p95 |
| --- | ---: | ---: | ---: |
| `orcha --help` | 58.363 ms | 58.889 ms | 60.788 ms |
| `orcha gallery --plain` | 613.425 ms | 226.278 ms | 230.353 ms |

Gallery median decreased **63.1%** and meets the **250 ms** target. Its peak RSS
median decreased from **84.324 MiB** to approximately **41.03 MiB**. The pre-parity
156 ms baseline remains lower; the measured follow-up does not claim to recover
that exact baseline.

An additional 20-run measurement including `uv run` overhead gave gallery median
**238.786 ms**, p95 **241.339 ms**, maximum **242.731 ms**; every sample was below
250 ms. Help median was **73.948 ms**, p95 **76.772 ms**, maximum **76.773 ms**.
Samples are retained in `/tmp/orcha-followup-startup.json` for this run.

Gallery imports, paint handling, and keybinding synchronization were implemented
and cross-reviewed in parallel with explicit file ownership. Git commits and
full-suite runs remained serial; benchmark measurements ran without concurrent
tests. No push was performed.

## Wave 2 integration — 2026-09-17

This wave starts at `4855616` on `integration/omp-parity`. The reviewed heads were
verified and integrated in order without squashing: hooks/rules/setup `1f7cf81`,
compaction/models/usage `e99ab58`, then TUI polish `2db51b7`. All work remains local;
no push was performed. Real smoke commands used temporary configuration and
session databases; no free text or pasted text was submitted to a provider.

### Merge resolutions

| Merge | Textual conflicts | Resolution |
| --- | --- | --- |
| `161836c` — hooks, rules and wizard | None | Existing follow-up redraw/import/keybinding fixes survived the automatic merge. |
| `a5313fd` — compaction, models and usage | `core/capture.py`, `core/ledger.py`, `tui/context.py`, `tui/overlays/__init__.py` | Retained durable rule reminders while adding typed compaction metadata, summary wrappers, retained-message boundaries and replacement capture. Combined context lifecycle and lazy overlay exports with model catalog/role support. |
| `715d1c6` — TUI polish | `core/agent.py`, `tui/runtime.py`, `tui/statusline.py`, `tests/test_statusline.py`, `tests/tui/test_statusline_polish.py`, `tests/tui/golden/omp-models.76.txt` | Preserved rules/hooks and the new compaction controller alongside the auto-compaction setting; kept runtime viewport/panel handling and compaction cards, retained model/usage status segments within the polished layout, and regenerated the combined model-picker golden. |

Paths in this table are relative to `orcha_agent/`, except those beginning with
`tests/`. The themed-gallery CLI flag needed by the first two merge gates was not
yet present on those feature branches. Commit `ae8f2f9` brought forward the exact
two CLI lines from the reviewed polish branch, allowing the required light-theme
gate to run before the final merge.

### Semantic integration fixes

- Automatic compaction emits declarative hook events for the new committed
  compaction markers, including shake compaction; legacy summarization events
  remain supported. Durable rule instructions survive both checkpoint capture
  and ledger reconstruction.
- Turning off automatic compaction disables the replacement controller rather
  than removing it. Removing that middleware would silently restore deepagents'
  default summarizer. Both `[compaction].enabled` and the UI preference gate
  automatic compaction; idle tasks recheck live settings after waking. Explicit
  `/compact` remains available.
- TTSR interruptions preserve provider-reported partial usage on the failed
  request's accounting row. The retry has a distinct request ID; repeated end or
  error callbacks cannot double-count the original request.
- The gallery retains lazy overlay/ledger imports. Model-role constants remain
  available without importing the provider stack during gallery rendering.
- Runtime integration assertions inspect the visible screen and scrollback
  together: polished command panels can keep subsequent committed content in the
  live viewport until the next user message.
- `/usage` now renders through the same bounded command-panel path as the other
  polished command surfaces, preserving viewport placement and status/composer
  space.
- Performance follow-up `299a19a` removes the remaining avoidable
  gallery work: lazy YAML overrides, a plugin-aware Python filename lexer fast
  path, cached/coalesced fixture styles, and avoiding a second load of all 35
  themes just to build picker labels (one dark theme still supplies the status
  fixture). Profiling attributed approximately 124 ms to overlay rendering,
  98 ms to filename lexer selection and 7 ms to YAML imports; these are profiling
  costs, not additive uninstrumented benchmark measurements.

### Gate results

`uv sync`, Ruff check and formatting, Pyright, the full pytest suite, the exact
`env -i HOME=$HOME PATH=$PATH TERM=xterm-256color uv run pytest -q` command,
default/light gallery commands, and tmux verification were required after each
merge. The benchmark suite is run only after the final merge and without
concurrent test workloads.

| Tree | Normal pytest | Clean-environment pytest | Other gates |
| --- | --- | --- | --- |
| Hooks/rules/setup | 1,913 passed, 2 skipped, 52.02 s | 1,913 passed, 2 skipped, 51.27 s | Passed |
| Compaction/models/usage | 2,034 passed, 2 skipped, 55.61 s | 2,034 passed, 2 skipped, 57.65 s | Passed |
| Final wave 2 | 2,153 passed, 2 skipped, 73.76 s | 2,153 passed, 2 skipped, 73.29 s | Passed |

The two skips are the optional native-libsql checks. The final load sweep ran
three sequential full suites alongside three sequential clean-environment
companion suites.
Each of the six runs passed 2,153 tests with two skips:

| Pair | Normal suite | Clean-environment companion |
| --- | ---: | ---: |
| 1 | 73.76 s | 73.29 s |
| 2 | 75.95 s | 74.10 s |
| 3 | 72.49 s | 72.49 s |

The first normal run used the exact gate command; normal runs 2 and 3 also disabled
`randomly` as requested. No flaky failure reproduced. An earlier six-run sweep
before the performance follow-up also passed (2,149 tests and two skips per run).

### Integration and first paint

`tests/tui/test_omp_integration.py` retains the native read/edit/bash, skill and
in-process MCP tool scenario. Its additional wave-2 scenario uses a scripted
streaming provider through `ApplicationRuntime`, `AppContext`, a real compiled
graph and a temporary `SessionStore`:

- A split-chunk TTSR match aborts the original response and retries with the rule
  reminder. A preceding bash side effect occurs exactly once; rejected suffix
  text never appears.
- A before-tool hook blocks a real native write; no destination file is created
  and the tool result is an error.
- Crossing the configured context threshold invokes the fake summarizer,
  persists a compaction entry and renders its committed card.
- `@smol` resolves to `fake:small`; main, summary and interrupted requests write
  five usage rows. The abort-triggering chunk contributes exactly 30 input and
  two output tokens, with no replay double-counting.
- Completion and rendering wait for concrete events rather than a short polling
  deadline.

The first-paint regression still holds skills and MCP discovery behind explicit
barriers while verifying that the application paints and accepts a clean exit.
No provider request is made by that startup test.

### Golden review

Only one golden conflicted textually: `omp-models.76.txt`. The model-picker
union and later gallery-fixture optimization required these updates:

| Golden | Reason |
| --- | --- |
| `tests/tui/golden/omp-models.76.txt` | Combine role/catalog columns with polished selection arrows and model rows. |
| `tests/tui/golden/polish-round2-help.80.ansi` | Coalesce identical style runs and use cached style resolution without changing help cells or styles at 80 columns. |
| `tests/tui/golden/polish-round2-help.120.ansi` | The same byte-level encoding optimization for the 120-column help fixture. |
| `tests/tui/golden/polish-round2-hub.80.ansi` | Preserve the hub selection, borders and rows while coalescing redundant ANSI style emissions. |
| `tests/tui/golden/polish-round2-hub.120.ansi` | The same unchanged hub cells/styles at 120 columns with compact ANSI output. |
| `tests/tui/golden/polish-round2-models.80.ansi` | Merge roles/browser rows with polished selection; later coalesce style output while preserving every resulting cell/style. |
| `tests/tui/golden/polish-round2-models.120.ansi` | The same merged roles/browser content and selection at 120 columns with compact style output. |
| `tests/tui/golden/polish-round2-settings.80.ansi` | Preserve Behaviour-tab values, cursor and borders while removing redundant fixture style work. |
| `tests/tui/golden/polish-round2-settings.120.ansi` | The same settings cells/styles at 120 columns after style coalescing. |
| `tests/tui/golden/polish-round2-tree.80.ansi` | Preserve tree rows, empty state and borders while coalescing identical style runs. |
| `tests/tui/golden/polish-round2-tree.120.ansi` | The same tree cells/styles at 120 columns with compact style output. |

The ten round-two ANSI fixtures were regenerated after optimization. Pyte replay
verified every terminal cell and style against the pre-optimization `715d1c6`
versions; both 80-column and 120-column surfaces received visual review in addition to
the cell/style comparison. The earlier merged model snapshots
were also eyeballed. These ANSI byte changes do not represent a changed visual
contract.

`tests/tui/golden/compaction.76.txt` arrived with the compaction branch and needed
no regeneration.

### Live terminal smoke

The isolated `int2` tmux smoke captured 60 frames: 30 at 120×40 and 30 at 80×30.
Artifacts are in `/tmp/int2-smoke/frames` for this run. Both processes used a
temporary HOME, `XDG_CONFIG_HOME`, and database, had empty stderr and exited with
Ctrl+D. The seven-line paste was inspected and never submitted.

The startup harness separately confirms six warnings collapse into one summary.
Coverage includes welcome once, `/help`, all `/settings`
tabs, `/theme` arrows and preview, `/model`, `/models browse`, `/usage`, `/hooks`,
`/rules`, `/skills`, `/mcp list`, `/status`, `/providers`, graceful provider-free
`/compact`, Alt+A, Esc Esc, and paste handling. The reviewed frames showed no
literal escapes, traceback, overlapping composer or displaced status line;
panels stayed inside the viewport.

### Wave-2 benchmarks

`uv run python -m benchmarks` passed with its default 20 repetitions and
20 fresh startup subprocesses. Measured clean commit: `299a19a539181ed6eb8e16b383cc12006223d1a9`;
scoped SHA-256: `3686cf928a67063cc0f7147ef299e78e55ab85c1447ea86a3be81a3c78ab13c3`. All six artifacts report the same
clean commit and hash. CPython 3.12.13, `Linux-7.2.3-arch1-3-x86_64-with-glibc2.44`;
generated UTC range 2026-09-16 18:53:12–18:55:34. No test jobs ran concurrently.

Compare with the exact clean wave-one follow-up JSON artifacts from `a71e768`,
preserved for this run in `/tmp/int2-wave1-benchmarks`. These support a more
precise comparison than rounding the older document tables. Startup references:

| Case | Sprint 1 median / p95 | Wave-one follow-up median / p95 | Wave 2 |
| --- | ---: | ---: | ---: |
| Direct `orcha --help` | 56.822 / 60.243 ms | 58.889 / 60.788 ms | 60.748 / 62.437 ms |
| Direct `orcha gallery --plain` | 155.902 / 167.666 ms | 226.278 / 230.353 ms | 270.670 / 273.643 ms |
| `uv run orcha --help` | — | 73.948 / 76.772 ms | 75.649 ms median; 77.305 ms maximum |
| Actual `uv run orcha gallery` | — | — | 286.683 / 289.692 ms; 294.219 ms maximum |
| Actual `uv run orcha gallery --theme light` | — | — | 286.218 / 290.555 ms; 290.893 ms maximum |

The actual-command measurements above use 20 fresh runs per command from
`/tmp/int2-cli-timing.json`. Every default and light gallery sample was below
300 ms; every help sample was below 150 ms. The plain direct-executable benchmark is a distinct command/output mode; the
actual-command values are not the same workload as the historical plain benchmark.

| Metric | Published baseline median | Wave 1 follow-up | Wave 2 | Δ vs baseline | Δ vs wave 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLI help | 56.822 ms | 58.889 ms | 60.748 ms | +6.9% | +3.2% |
| Gallery plain | 155.902 ms | 226.278 ms | 270.670 ms | +73.6% | +19.6% |
| Stream 100 KiB / 1-byte | 3.466 s/MiB | 3.465 s/MiB | 3.447 s/MiB | -0.6% | -0.5% |
| Stream 1 MiB / 1-byte | 3.590 s/MiB | 3.510 s/MiB | 3.455 s/MiB | -3.8% | -1.6% |
| Changed viewport 100 KiB | 86.471 ms | 9.144 ms | 9.267 ms | -89.3% | +1.3% |
| Cached viewport 100 KiB | 0.453 ms | 0.450 ms | 0.437 ms | -3.4% | -2.8% |
| Changed viewport 1 MiB | 1310.282 ms | 102.328 ms | 102.720 ms | -92.2% | +0.4% |
| Cached viewport 1 MiB | 3.881 ms | 3.981 ms | 3.963 ms | +2.1% | -0.5% |
| Ledger first_load / 0 abandoned | 8.802 ms | 8.394 ms | 8.360 ms | -5.0% | -0.4% |
| Ledger path / 0 abandoned | 2.330 ms | 2.347 ms | 2.287 ms | -1.8% | -2.5% |
| Ledger cold_path / 0 abandoned | 8.126 ms | 8.606 ms | 8.247 ms | +1.5% | -4.2% |
| Ledger fork / 0 abandoned | 7.164 ms | 7.868 ms | 7.319 ms | +2.2% | -7.0% |
| Ledger build_context / 0 abandoned | 0.378 ms | 0.394 ms | 0.429 ms | +13.5% | +8.8% |
| Ledger first_load / 10,000 abandoned | 9.633 ms | 9.517 ms | 8.619 ms | -10.5% | -9.4% |
| Ledger path / 10,000 abandoned | 2.430 ms | 2.695 ms | 2.384 ms | -1.9% | -11.5% |
| Ledger cold_path / 10,000 abandoned | 8.432 ms | 9.781 ms | 8.557 ms | +1.5% | -12.5% |
| Ledger fork / 10,000 abandoned | 7.560 ms | 8.408 ms | 7.501 ms | -0.8% | -10.8% |
| Ledger build_context / 10,000 abandoned | 0.379 ms | 0.388 ms | 0.429 ms | +13.2% | +10.7% |
| Ledger first_load / 100,000 abandoned | 9.343 ms | 9.509 ms | 9.063 ms | -3.0% | -4.7% |
| Ledger path / 100,000 abandoned | 2.555 ms | 2.596 ms | 2.581 ms | +1.0% | -0.6% |
| Ledger cold_path / 100,000 abandoned | 9.008 ms | 8.966 ms | 8.831 ms | -2.0% | -1.5% |
| Ledger fork / 100,000 abandoned | 7.877 ms | 7.764 ms | 7.719 ms | -2.0% | -0.6% |
| Ledger build_context / 100,000 abandoned | 0.383 ms | 0.381 ms | 0.434 ms | +13.3% | +13.9% |
| Stream 100 KiB / 10-byte | 0.347 s/MiB | 0.347 s/MiB | 0.346 s/MiB | -0.3% | -0.4% |
| Stream 100 KiB / 100-byte | 0.035 s/MiB | 0.035 s/MiB | 0.035 s/MiB | -0.4% | -0.9% |
| Stream 1 MiB / 10-byte | 0.360 s/MiB | 0.356 s/MiB | 0.357 s/MiB | -0.9% | +0.3% |
| Stream 1 MiB / 100-byte | 0.036 s/MiB | 0.036 s/MiB | 0.036 s/MiB | -1.0% | -0.2% |
| Capture 100 turns / stable | 0.097 ms | 0.099 ms | 0.101 ms | +3.8% | +1.9% |
| Capture 100 turns / unchanged 100 KiB | 0.096 ms | 0.098 ms | 0.098 ms | +1.8% | -0.2% |
| Capture 1,000 turns / stable | 0.413 ms | 0.435 ms | 0.417 ms | +1.1% | -4.0% |
| Capture 1,000 turns / unchanged 100 KiB | 0.408 ms | 0.428 ms | 0.422 ms | +3.4% | -1.3% |
| History data / 10,000 rows | 0.615 ms | 0.630 ms | 0.627 ms | +2.0% | -0.3% |
| History overlay / 10,000 rows | 0.816 ms | 0.831 ms | 0.833 ms | +2.1% | +0.3% |
| History data / 100,000 rows | 0.619 ms | 0.625 ms | 0.633 ms | +2.2% | +1.3% |
| History overlay / 100,000 rows | 0.810 ms | 0.840 ms | 0.830 ms | +2.4% | -1.3% |
| Full session list / 10,000 rows | 23.602 ms | 24.073 ms | 23.241 ms | -1.5% | -3.5% |
| Session overlay / 10,000 rows | 7.078 ms | 7.069 ms | 6.904 ms | -2.5% | -2.3% |
| Full session list / 100,000 rows | 395.651 ms | 448.951 ms | 448.612 ms | +13.4% | -0.1% |
| Session overlay / 100,000 rows | 7.025 ms | 6.965 ms | 6.921 ms | -1.5% | -0.6% |

Baselines follow the existing report: Sprint 1 after for startup/streaming/history/
sessions, Review round 1 for capture/cold ledger/fork/context, and Review round 2
for fresh-store/warm ledger. Published baseline values are rounded; wave-one
comparisons use the preserved JSON values. These are separate runs, not controlled
paired measurements.

- Plain gallery is 19.6% slower than the wave-one follow-up (+73.6% versus Sprint 1),
  with the expanded polish fixtures retained. P95 is 273.643 ms (+18.8% versus
  wave one). The separate actual default/light commands also satisfy the budget.
  Gallery peak RSS rises 41.031→43.973 MiB (+7.2%); help rises 22.238→22.484 MiB
  (+1.1%).
- Context reconstruction medians rise 8.8–13.9% versus wave one, to
  0.429–0.434 ms. At zero/100,000 abandoned entries p95 rises 15.7%/14.5%, to
  0.469/0.459 ms. The benchmark does not isolate a causal source for this change.
- Streaming accumulation medians vary −1.6% to +0.3% versus wave one. Changed
  viewport medians rise 1.3% at 100 KiB and 0.4% at 1 MiB, retaining the roughly
  89–92% improvement versus Sprint 1. Cached viewport medians improve slightly
  versus wave one; the 1 MiB p95 rises 2.4% to 4.296 ms. The existing warmed-layout
  counter remains zero additional base Rich Markdown layouts, with the same
  instrumentation caveat described in the wave-one report.
- Cold ledger tails remain variable: zero-abandoned p95 is 168.885 ms, while
  100,000-abandoned p95 changes 169.408→9.432 ms. This does not establish a general
  tail-latency improvement. Most ledger path/fork medians improve versus wave one.
- Capture medians range −4.0% to +1.9% versus wave one. The 100-turn unchanged-state
  p95 rises 18.5% to 0.124 ms; history data loading at 10,000 rows has a 19.9% p95
  increase to 0.809 ms despite a stable median. Capture databases after close are
  approximately 0.156/0.254 MiB at 100 turns and 0.547/0.645 MiB at 1,000 turns
  (stable/unchanged 100 KiB state).
- Full loading of 100,000 sessions remains about 448.612 ms: −0.1% versus wave one,
  but +13.4% versus the 395.651 ms baseline (p95 451.138 ms versus 399.740 ms,
  +12.9%). The bounded picker remains 6.921 ms, −0.6% versus wave one and −1.5%
  versus baseline. No broad unbounded-session-loading optimization was added.

Raw full benchmark JSON remains ignored in `benchmarks/results/`. The report-only
commit follows the measured clean code commit.

### Documentation, remaining choices and execution

README consolidates core, native tools, TUI/statusline, agents, compaction,
models/roles, MCP and declarative hooks into one configuration reference, with
shared path and trust guidance for rules, skills, commands and context files.
No unresolved wave-2 choice remains. Compatibility precedence is explicit: `[model_roles]`
overrides `[models.roles]`, and `[tui]` overrides `[ui]`. MCP uses `[plugins.mcp]`
and `mcp.json`. The inherited count remains 35 distinct themes; adding a 36th
would be a separate choice. XDG/`ORCHA_CONFIG_DIR` relocates config/settings and
the adjacent catalog, while some skills/context/theme discovery remains HOME-based;
full smoke isolation therefore uses temporary HOME as well.

Independent module work, conflict research, documentation, integration tests and
review were delegated with explicit file ownership. Static gates and independent
read-only checks ran concurrently. Ordered merges, dependent integration and
commits stayed serial. The requested load sweep deliberately ran concurrent
suites; final benchmark measurements ran afterward without those test jobs.
