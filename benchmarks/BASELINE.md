# Performance baseline

This file records the Phase 0 baseline after the full benchmark suite has been run on the reference workstation. It intentionally contains no estimated or fabricated measurements.

## Reproduction

From a checkout with the project environment synchronized, run:

```console
uv run python -m benchmarks
```

The default run uses 20 timed repetitions per case and 20 fresh subprocesses per startup command. The command prints every workload table and writes schema-versioned JSON records to `benchmarks/results/`. Use `--repetitions N` and `--startup-runs N` only when deliberately changing the sampling plan. `--quick` uses one repetition and development-scale fixtures; do not use it for baseline numbers.

Record the git commit, dirty flag, working-tree SHA-256, UTC timestamp, Python/platform metadata, workload parameters, and median/p95 values directly from the generated JSON. Retain the JSON artifacts used for comparison outside version control.

The working-tree hash is scoped to `pyproject.toml`, `uv.lock`, `orcha_agent/**`, and `benchmarks/**`. Git pathspec exclusions remove nested `.env*` and `Credentials/**` paths before status, diff, or untracked content is read.

## Reference environment

- Git commit: `6d469ffe56d81507e26bbd4b9ebe4c455c1bd54e`
- Git dirty: `false`
- Scoped working-tree SHA-256: `152f8c9c2bc005d80c7dc5300b3d62348983fb9f8835cdf955a4dadcdf9b0269`
- Generated UTC: startup `2026-08-30T14:10:36+00:00`; streaming `2026-08-30T14:18:23+00:00`; ledger `2026-08-30T14:18:31+00:00`; turn capture `2026-08-30T14:18:46+00:00`; history load `2026-08-30T14:18:56+00:00`; session overlay load `2026-08-30T14:19:20+00:00`
- Python: CPython 3.12.13
- Platform: Linux 7.1.9-arch1-2, x86_64, glibc 2.44

The environment and Git metadata are consistent across all six artifacts. The UTC timestamps differ because each artifact records its own generation time. Timed and layout medians/p95s use 20 samples; startup values use 20 fresh subprocesses per command. Turn-capture WAL and final database sizes are medians of 20 samples. Fixture storage measurements use their single recorded sample.

## Startup

| Case | Wall median | Wall p95 | Peak RSS median | Peak RSS p95 |
| --- | ---: | ---: | ---: | ---: |
| `orcha --help` | 1.813893 s | 1.844982 s | 143.945 MiB | 144.379 MiB |
| `orcha gallery --plain` | 1.817839 s | 1.833039 s | 145.281 MiB | 145.723 MiB |

## Streaming

Results from `benchmarks/results/streaming.json`:

| Workload | Payload | Chunk | Median | P95 | Unit |
| --- | ---: | ---: | ---: | ---: | --- |
| Transcript/Frame CPU per MiB | 100 KiB | 1 byte | 4.133535 | 4.153654 | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 10 bytes | 0.410994 | 0.424951 | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 100 bytes | 0.041168 | 0.041659 | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 1 byte | 15.001514 | 15.218773 | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 10 bytes | 2.246917 | 2.295470 | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 100 bytes | 0.222577 | 0.232887 | seconds/MiB |
| Viewport paint | 100 KiB | N/A | 0.276840 | 0.288384 | seconds |
| Rich layouts per displayed revision | 100 KiB | N/A | 2 | 2 | layouts/revision |
| Viewport paint | 1 MiB | N/A | 2.490191 | 2.533538 | seconds |
| Rich layouts per displayed revision | 1 MiB | N/A | 2 | 2 | layouts/revision |

## Ledger

Results from `benchmarks/results/ledger.json`; all cases use 1,000 active entries:

| Operation | Abandoned entries | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| `Ledger.path` | 0 | 5.678 | 5.750 | milliseconds |
| `Ledger.fork` | 0 | 10.624 | 12.691 | milliseconds |
| `build_context` | 0 | 2.958 | 3.276 | milliseconds |
| `Ledger.path` | 10,000 | 14.208 | 15.585 | milliseconds |
| `Ledger.fork` | 10,000 | 19.646 | 23.038 | milliseconds |
| `build_context` | 10,000 | 2.933 | 3.533 | milliseconds |
| `Ledger.path` | 100,000 | 111.145 | 229.072 | milliseconds |
| `Ledger.fork` | 100,000 | 110.358 | 119.498 | milliseconds |
| `build_context` | 100,000 | 2.937 | 3.814 | milliseconds |

| Abandoned entries | Fixture database | Fixture WAL | Fixture SHM | Fixture total |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.004 MiB | 0.413 MiB | 0.031 MiB | 0.448 MiB |
| 10,000 | 0.004 MiB | 2.263 MiB | 0.031 MiB | 2.298 MiB |
| 100,000 | 19.309 MiB | 19.469 MiB | 0.062 MiB | 38.840 MiB |

## Turn capture

Results from `benchmarks/results/turn_capture.json`:

| Turns | State | Median | P95 | WAL while open | Final database size |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Stable empty state | 0.0956 ms | 0.1061 ms | 2.625 MiB | 0.121 MiB |
| 100 | Unchanged 100 KiB state | 0.2954 ms | 0.3053 ms | 4.035 MiB | 9.926 MiB |
| 1,000 | Stable empty state | 0.3435 ms | 0.3862 ms | 3.974 MiB | 0.602 MiB |
| 1,000 | Unchanged 100 KiB state | 0.5670 ms | 2.3034 ms | 4.075 MiB | 98.684 MiB |

## History and session overlay loads

Results from `benchmarks/results/history_load.json` and `benchmarks/results/session_overlay_load.json`:

| Workload | Rows | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| History data load | 10,000 | 2.510 | 2.571 | milliseconds |
| History overlay construction | 10,000 | 2.686 | 2.717 | milliseconds |
| History data load | 100,000 | 23.756 | 24.209 | milliseconds |
| History overlay construction | 100,000 | 24.525 | 24.831 | milliseconds |
| Session data load | 10,000 | 20.494 | 21.885 | milliseconds |
| Session overlay construction | 10,000 | 19.674 | 20.938 | milliseconds |
| Session data load | 100,000 | 361.306 | 362.982 | milliseconds |
| Session overlay construction | 100,000 | 361.389 | 364.764 | milliseconds |

| History rows | Fixture database | Fixture WAL |
| ---: | ---: | ---: |
| 10,000 | 0.004 MiB | 1.246 MiB |
| 100,000 | 11.582 MiB | 11.697 MiB |

| Session rows | Fixture database | Fixture WAL | Fixture SHM | Fixture total |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.004 MiB | 2.024 MiB | 0.031 MiB | 2.059 MiB |
| 100,000 | 19.758 MiB | 19.901 MiB | 0.062 MiB | 39.721 MiB |

## Sprint 1 after — 2026-09-16

Measured code commit: `fbb69db78ff4339e449d4e1a48dfc4b49c6bfff6` (clean). Scoped working-tree SHA-256: `8911e46ee6cae48bc23f3eb2eda251243bed761a126e3de0bc513b7deebd795c`. The baseline-document update follows this measured commit.

Environment: CPython 3.12.13; `Linux-7.2.3-arch1-3-x86_64-with-glibc2.44`. The optional Turso dependency (`libsql 0.1.11`) was installed for native-driver validation. The older reference used kernel 7.1.9; these comparisons are not a controlled same-kernel experiment.

Command: `uv run python -m benchmarks`, default 20 repetitions and 20 fresh startup subprocesses, run after validation with no concurrent test jobs. Raw JSON remains untracked in `benchmarks/results/`. Streaming timing now includes final text materialization and checks exact payload reconstruction.

Generated UTC: startup `2026-09-16T15:29:39+00:00`; streaming `2026-09-16T15:32:04+00:00`; ledger `2026-09-16T15:32:08+00:00`; turn_capture `2026-09-16T15:32:19+00:00`; history_load `2026-09-16T15:32:28+00:00`; session_overlay_load `2026-09-16T15:32:45+00:00`.

### Startup after

| Case | Wall median | Wall p95 | Peak RSS median | Peak RSS p95 |
| --- | ---: | ---: | ---: | ---: |
| help | 0.056822 s | 0.060243 s | 21.475 MiB | 21.582 MiB |
| gallery_plain | 0.155902 s | 0.167666 s | 36.359 MiB | 36.484 MiB |

### Streaming after

| Workload | Payload | Chunk | Median | P95 | Unit |
| --- | ---: | ---: | ---: | ---: | --- |
| Transcript/Frame CPU per MiB | 100 KiB | 1 bytes | 3.466216 | 3.526584 | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 10 bytes | 0.347289 | 0.348466 | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 100 bytes | 0.034944 | 0.036068 | seconds/MiB |
| Changed viewport paint | 100 KiB | N/A | 0.086471 | 0.192656 | seconds |
| Cached viewport paint | 100 KiB | N/A | 0.000453 | 0.000464 | seconds |
| Rich layouts per displayed revision | 100 KiB | N/A | 1.000000 | 1.000000 | layouts/revision |
| Transcript/Frame CPU per MiB | 1 MiB | 1 bytes | 3.589866 | 3.752618 | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 10 bytes | 0.360330 | 0.363682 | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 100 bytes | 0.036350 | 0.036709 | seconds/MiB |
| Changed viewport paint | 1 MiB | N/A | 1.310282 | 1.346541 | seconds |
| Cached viewport paint | 1 MiB | N/A | 0.003881 | 0.005082 | seconds |
| Rich layouts per displayed revision | 1 MiB | N/A | 1.000000 | 1.000000 | layouts/revision |

### Ledger after

All cases retain 1,000 active entries.

| Operation | Abandoned entries | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| `Ledger.path` | 0 | 12.157 | 13.219 | milliseconds |
| `Ledger.fork` | 0 | 15.958 | 124.635 | milliseconds |
| `build_context` | 0 | 4.258 | 5.013 | milliseconds |
| `Ledger.path` | 10,000 | 11.859 | 121.014 | milliseconds |
| `Ledger.fork` | 10,000 | 15.988 | 125.944 | milliseconds |
| `build_context` | 10,000 | 4.223 | 5.373 | milliseconds |
| `Ledger.path` | 100,000 | 12.077 | 122.954 | milliseconds |
| `Ledger.fork` | 100,000 | 16.348 | 127.761 | milliseconds |
| `build_context` | 100,000 | 4.200 | 4.617 | milliseconds |

### Turn capture after

| Turns | State | Median | P95 | WAL while open | Final database size |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Stable empty state | 0.0973 ms | 0.1153 ms | 3.352 MiB | 0.141 MiB |
| 100 | Unchanged 100 KiB state | 0.0955 ms | 0.1050 ms | 3.454 MiB | 0.238 MiB |
| 1,000 | Stable empty state | 0.4317 ms | 0.4700 ms | 3.963 MiB | 0.531 MiB |
| 1,000 | Unchanged 100 KiB state | 0.4321 ms | 0.5013 ms | 3.957 MiB | 0.629 MiB |

### History and session overlays after

History loads the newest 1,000 prompts; FTS still searches older prompts. Session pickers load 200 sessions per page, precompute labels, and filter older pages on a cancellable worker. The separate `SessionStore.list()` measurement still loads all sessions for callers requiring that API.

| Workload | Database rows | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| History data load | 10,000 | 0.615 | 0.639 | milliseconds |
| History overlay construction | 10,000 | 0.816 | 0.863 | milliseconds |
| History data load | 100,000 | 0.619 | 0.683 | milliseconds |
| History overlay construction | 100,000 | 0.810 | 0.845 | milliseconds |
| Full session data load | 10,000 | 23.602 | 25.203 | milliseconds |
| Session overlay construction | 10,000 | 7.078 | 7.242 | milliseconds |
| Full session data load | 100,000 | 395.651 | 399.740 | milliseconds |
| Session overlay construction | 100,000 | 7.025 | 7.172 | milliseconds |

### Additional before/after evidence

| Item | Before | After | Evidence |
| --- | --- | --- | --- |
| First-available model resolution, 3 valid fallbacks | 3 factories | 1 factory | `tests/test_models.py` |
| Build/model-switch factories including manual summarizer | 3 factories | 2 factories; manual model deferred until compact | `tests/test_tui.py`, `tests/test_lazy_summarizer.py` |
| Deserialization for 10 retained messages, load plus build | 20 calls | 10 calls | `tests/test_ledger.py`; serialized-payload mutation and consumer isolation also checked |
| Session-title reads during ordinary paint | 1 synchronous query per paint | 0 synchronous queries; one worker refresh per invalidation | `tests/test_status_snapshot.py`, including 120 repeated paints |
| Stable-state capture writes | Growing captured-ID JSON and repeated state payload | 4 changed rows per appended message, no repeated state payload | `tests/test_capture_safety.py` |
| Picker render over 10,000 items | All labels formatted | Only requested rows formatted (8 in test) | `tests/tui/test_overlay_performance.py` |
| Immediate redraw with a pending timer | A redundant trailing redraw | Pending redraw cancelled | `tests/tui/test_scheduler_performance.py` |
| Visible thinking spinner/metrics ticks | New Markdown layout every tick | No new Markdown layout for 10 ticks | `tests/tui/test_streaming_performance.py` |
| Synthetic 100 ms capture, 5 ms heartbeat | 100.174 ms maximum heartbeat gap | 5.401 ms maximum heartbeat gap | Local diagnostic; deterministic worker/cancellation regressions in `tests/tui/test_turn_host.py` |

The factory diagnostic with three `FakeListChatModel` fallbacks (1,000 local calls) measured 0.012553 ms for the old full-chain algorithm and 0.004865 ms for first-success resolution. These factory and heartbeat diagnostics are separate from the default 20-sample benchmark suite.

### Interpretation and remaining limits

- One-byte stream processing is approximately linear in payload size: 3.466 seconds/MiB at 100 KiB and 3.590 at 1 MiB, versus 4.134 and 15.002 in the old baseline.
- CTE median path time stays near 12 ms as abandoned branches grow from zero to 100,000. Payload transfer is restricted to the active ancestors. Tail latency is not uniformly improved: path/fork p95 outliers reach 128 ms in this run.
- Single deserialization reduces validator calls, but preserving mutable-payload coherence and independent returned messages adds copying. The 1,000-message context-only median increases from roughly 2.94 to 4.20 ms; unbranched path loading also costs more. This is a measured correctness/performance tradeoff, not a claimed CPU speedup.
- Cursor/state write volume is incremental. Detecting arbitrary same-ID and nested in-place mutation still requires an O(message-count) safety scan because graph messages have no immutable revision contract. At 1,000 turns, unchanged 100 KiB state improves from 0.5670 to about 0.432 ms and 98.684 to about 0.629 MiB; empty-state capture rises from 0.3435 to about 0.432 ms. Larger persistent cursor indexes also increase small-session storage and WAL overhead.
- Changed 1 MiB Markdown still requires a complete Rich layout: median 1.310 s, p95 1.347 s, above the 33 ms target. Cached repaints have p95 5.082 ms. Moving rendering off-thread or incrementally laying out Markdown remains future work; this sprint keeps existing visual output.
- Session filtering has bounded retained memory and runs off the UI loop; a rare/no-match query can still scan all pages. Initial history/session windows are intentionally bounded; earlier records remain available through search or pagination. The full session-list API remains unbounded for existing callers.

### Validation and execution

`uv sync` passed; `uv sync --extra turso` additionally enabled the native-driver tests. Final `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` passed. `uv run pytest -q`: **1,161 passed in 16.87 s, no skips**. Native libsql tests cover capture cursors, recursive paths/forks and picker queries; they use local databases, not a remote Turso account.

`uv run python tests/tui/tmux/verify.py` passed: both turns and fanout had zero repaint growth; markers occurred once each, and narrow/wide resize frames occurred once each. Existing golden files were unchanged. Existing test changes were limited to lazy-import monkeypatch targets, waiting for the asynchronous title snapshot, and the intentionally reduced eager model-factory count (plus formatting in touched files).

CLI/models, streaming, persistence, and overlays/status were implemented in parallel. Independent validation commands ran concurrently. Storage-dependent changes, integration, review fixes and Git commits were serialized. The full default benchmark run was isolated from test jobs. The implementation and measurement updates are local commits on `perf/smoothness-sprint`; no push or PR was performed.

## Review round 1 after — 2026-09-17

Measured code commit: `87436089b54bcf928379db14a55d1e391efdd984` (clean). Scoped working-tree SHA-256: `d3cce1fdb0c2998d268ee03056b77abae9c2a60a14563cffaa230add3c487d3b`. This documentation commit follows the measured code commit.

Environment: CPython 3.12.13; `Linux-7.2.3-arch1-3-x86_64-with-glibc2.44`; native libsql tests enabled. Command: `uv run python -m benchmarks`, default 20 repetitions and 20 fresh startup subprocesses, isolated from test jobs. Raw JSON remains ignored in `benchmarks/results/`.

Generated UTC: startup `2026-09-16T16:05:20+00:00`; streaming `2026-09-16T16:07:44+00:00`; ledger `2026-09-16T16:07:46+00:00`; turn_capture `2026-09-16T16:07:56+00:00`; history_load `2026-09-16T16:08:05+00:00`; session_overlay_load `2026-09-16T16:08:23+00:00`.

These results supersede the sprint 1 copying tradeoff above. Decoded messages are immutable snapshots shared by context readers. The known LangGraph missing-ID mutation boundary copies messages before handing them to the reducer. The store-owned decoded cache is bounded to 2,048 messages and 8 MiB of serialized payload, with a 64 KiB per-message admission limit; those are payload budgets, not a bound on total Python heap overhead. Full row keys detect externally edited database payloads.

### Ledger after review

All cases retain 1,000 active entries. `Ledger.path` below measures repeated reads with the decoded-message cache warm, including through newly constructed Ledger instances sharing the store. The separately added cold metric clears that cache before each timed read; it does not clear SQLite/OS caches. Garbage collection remains enabled.

| Operation | Abandoned entries | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| `Ledger.path` cold decoded cache | 0 | 8.126 | 9.171 | milliseconds |
| `Ledger.path` warm decoded cache | 0 | 2.277 | 2.329 | milliseconds |
| `Ledger.fork` | 0 | 7.164 | 8.144 | milliseconds |
| `build_context` | 0 | 0.378 | 0.399 | milliseconds |
| `Ledger.path` cold decoded cache | 10,000 | 8.432 | 117.163 | milliseconds |
| `Ledger.path` warm decoded cache | 10,000 | 2.427 | 2.849 | milliseconds |
| `Ledger.fork` | 10,000 | 7.560 | 9.611 | milliseconds |
| `build_context` | 10,000 | 0.379 | 0.423 | milliseconds |
| `Ledger.path` cold decoded cache | 100,000 | 9.008 | 118.013 | milliseconds |
| `Ledger.path` warm decoded cache | 100,000 | 2.526 | 2.593 | milliseconds |
| `Ledger.fork` | 100,000 | 7.877 | 9.601 | milliseconds |
| `build_context` | 100,000 | 0.383 | 0.396 | milliseconds |

First-load cost is at parity with pre-sprint: roughly 11 ms per 1,000 entries in the reviewer’s comparable measurement, including validation now moved into decode. Only repeated/warm reads improved (roughly 3× versus pre-sprint); the historical 5.678 ms path-only number is not a like-for-like first-load comparison across that validation boundary. The round 1 warm path median is 2.277 ms and context-only reconstruction is 0.378 ms. At 10,000 abandoned entries, warm path/fork p95 falls from sprint 1’s 121.014/125.944 ms to 2.849/9.611 ms. Fresh-store measurements are recorded below; cache-cleared measurements above must not be presented as evidence of a pre-sprint first-load speedup.

The cache-cleared unbranched median is 8.126 ms; this reuses a store/connection and is distinct from a first load with a fresh SessionStore. Cold p95 remains about 117–118 ms with abandoned branches. EXPLAIN QUERY PLAN confirms indexed `(session_id, id)` lookups for the recursive seed, parent and payload join; no abandoned-payload scan occurs. Local profiling attributes the large cold outliers to generation-2 garbage collection from repeated decoded-object allocation. The bounded cache removes that work from repeated reads; it does not solve first-load allocation cost or cache-thrashing workloads. No GC disabling or timing exclusions were used.

### Turn capture after review

| Turns | State | Median | P95 | WAL while open | Final database size |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Stable empty state | 0.0968 ms | 0.1112 ms | 3.352 MiB | 0.141 MiB |
| 100 | Unchanged 100 KiB state | 0.0959 ms | 0.1050 ms | 3.454 MiB | 0.238 MiB |
| 1,000 | Stable empty state | 0.4131 ms | 0.4551 ms | 3.957 MiB | 0.533 MiB |
| 1,000 | Unchanged 100 KiB state | 0.4082 ms | 0.4279 ms | 3.957 MiB | 0.629 MiB |

The ordinary append path skips replacement/order scans after validating an unchanged digest prefix. At 1,000 turns, unchanged 100 KiB state improves from sprint 1’s 0.4321 ms to 0.4082 ms; stable empty state improves from 0.4317 ms to 0.4131 ms. Arbitrary mutable graph messages still require the existing linear digest-safety scan.

### Review regression evidence

| Item | Before | After / evidence |
| --- | --- | --- |
| M1: one same-ID edit in 2,000 messages | 2,001 → 4,003 ledger entries | 2,001 → 2,002 entries; at most 5 changed DB rows; context matches graph (`test_capture_review.py`) |
| M1: cross-provider model switch | Stripped graph diverged from capture cursor | Targeted replacement captured, cursor reseeded, next unchanged capture writes nothing; activation failure restores cursor and ledger (`test_tui.py`) |
| M2: compaction retaining six messages | Six retained rows duplicated | Six retained rows appear once, preceding-entry boundary is set, export has no duplicates; full retention, reopen, fork and replacement cases covered (`test_capture_review.py`) |
| M3: decoded message reuse | Deepcopy at decode and deep copy at reconstruction | One validation per cold message; no further deserialization for repeated reads; explicit copy at reducer mutation boundary (`test_ledger.py`, `test_tui.py`) |
| m1: ledger tail latency | Warm 10k-abandoned path p95 121.014 ms | 2.849 ms; cold outliers remain, query plan and GC findings described above |
| m2: unchanged state on reset | State appended again | One state entry retained across reset; live files preserved (`test_capture_review.py`) |
| m3: dispatcher eviction | Scans every cached block | Visits only affected block variants; test forbids top-level traversal with 1,000 unrelated blocks (`tui/test_blocks.py`) |
| m4: capture theme cache | Object identity | Semantic theme ID; equivalent theme objects cause zero extra layouts (`tui/test_runtime.py`) |
| m5: session refresh | Raw daemon thread from async paint | Running loop uses `asyncio.to_thread`; no-loop fallback retained; awaitable completion and 120-paint coalescing tested (`test_status_snapshot.py`, `test_statusline.py`) |
| m6: recovery/documentation | Misleading force-write implication; missing restart regression | Both flag values skip unchanged state; restart/recover updates cursor with at most 7 DB changes, then no-op (`test_tui.py`) |
| m7: untagged summary | Silent snapshot reset | Explicit `unrecognized_summary` ledger marker (`test_capture_review.py`) |

### Validation and execution after review

`uv sync --extra turso`, Ruff check, Ruff format check, Pyright, the full default benchmark command, and the tmux harness passed. Final full pytest: **1,178 passed in 16.95 s, no skips**. Tmux recorded zero repaint growth for both turns and fanout, one occurrence per marker, and one narrow/wide resize frame. Golden files are unchanged.

Capture, ledger, renderer and status work/review ran concurrently. Integration and commits were serialized; independent quality gates ran concurrently, followed by the isolated full benchmark run. Existing mutation-isolation tests were intentionally changed to the requested immutable snapshot contract, with a regression at the actual consumer mutation boundary. All changes are local conventional commits on `perf/smoothness-sprint`; no push or PR was performed.

## Review round 2: fresh-store first load — 2026-09-17

`first_load_wall` opens a fresh `SessionStore` for every iteration, then times `Ledger.path`, including message decoding and validation. Store initialization and fixture construction are excluded. Each sample starts with a new SQLite connection and empty decoded-message cache; OS caches are not flushed. The source fixture remains open while each fresh reader is measured. Garbage collection remains enabled. These measurements support the first-load parity interpretation above, not a claim that cold loads beat pre-sprint. The approximately 11 ms comparison comes from the independent review; the exact medians measured in this run are shown here.

| Operation | Active entries | Abandoned entries | Median | P95 |
| --- | ---: | ---: | ---: | ---: |
| `Ledger.path` first load (fresh store) | 1,000 | 0 | 8.802 ms | 10.160 ms |
| `Ledger.path` repeated/warm | 1,000 | 0 | 2.330 ms | 2.423 ms |
| `Ledger.path` first load (fresh store) | 1,000 | 10,000 | 9.633 ms | 121.931 ms |
| `Ledger.path` repeated/warm | 1,000 | 10,000 | 2.430 ms | 2.741 ms |
| `Ledger.path` first load (fresh store) | 1,000 | 100,000 | 9.343 ms | 120.806 ms |
| `Ledger.path` repeated/warm | 1,000 | 100,000 | 2.555 ms | 2.625 ms |

Measured base commit: `9db0e1d4c18736d766e660aae731bb94de77ecb2`; dirty: `true` (only the fresh-store benchmark addition in `benchmarks/persistence.py`). Scoped working-tree SHA-256: `4f36f167c5e08ec70746b8add53a350a6ff29498f987e34d2aff7501735bdd78`. Ledger generated UTC: `2026-09-16T16:20:25+00:00`. Runtime code is unchanged from round 1. CPython 3.12.13; `Linux-7.2.3-arch1-3-x86_64-with-glibc2.44`.

Validation: `uv sync --extra turso`, Ruff check/format, Pyright, **1,178 tests passed in 16.89 s**, the full default `uv run python -m benchmarks` (20 samples per case), and the tmux harness all passed. Independent gates ran concurrently; benchmarks ran afterward in isolation. No push.
