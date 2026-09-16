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
