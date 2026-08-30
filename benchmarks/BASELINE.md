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

- Git commit: TBD after full execution
- Git dirty: TBD after full execution
- Scoped working-tree SHA-256: TBD after full execution
- UTC timestamp: TBD after full execution
- Python: TBD after full execution
- Platform: TBD after full execution

## Startup

| Case | Wall median | Wall p95 | Peak RSS median | Peak RSS p95 |
| --- | ---: | ---: | ---: | ---: |
| `orcha --help` | TBD | TBD | TBD | TBD |
| `orcha gallery --plain` | TBD | TBD | TBD | TBD |

## Streaming

Fill every row from `benchmarks/results/streaming.json`.

| Workload | Payload | Chunk | Median | P95 | Unit |
| --- | ---: | ---: | ---: | ---: | --- |
| Transcript/Frame CPU per MiB | 100 KiB | 1 byte | TBD | TBD | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 10 bytes | TBD | TBD | seconds/MiB |
| Transcript/Frame CPU per MiB | 100 KiB | 100 bytes | TBD | TBD | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 1 byte | TBD | TBD | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 10 bytes | TBD | TBD | seconds/MiB |
| Transcript/Frame CPU per MiB | 1 MiB | 100 bytes | TBD | TBD | seconds/MiB |
| Viewport paint | 100 KiB | N/A | TBD | TBD | seconds |
| Rich layouts per displayed revision | 100 KiB | N/A | TBD | TBD | layouts/revision |
| Viewport paint | 1 MiB | N/A | TBD | TBD | seconds |
| Rich layouts per displayed revision | 1 MiB | N/A | TBD | TBD | layouts/revision |

## Ledger

Fill every row from `benchmarks/results/ledger.json`; all cases use 1,000 active entries.

| Operation | Abandoned entries | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| `Ledger.path` | 0 | TBD | TBD | seconds |
| `Ledger.fork` | 0 | TBD | TBD | seconds |
| `build_context` | 0 | TBD | TBD | seconds |
| `Ledger.path` | 10,000 | TBD | TBD | seconds |
| `Ledger.fork` | 10,000 | TBD | TBD | seconds |
| `build_context` | 10,000 | TBD | TBD | seconds |
| `Ledger.path` | 100,000 | TBD | TBD | seconds |
| `Ledger.fork` | 100,000 | TBD | TBD | seconds |
| `build_context` | 100,000 | TBD | TBD | seconds |

| Abandoned entries | Fixture database | Fixture WAL | Fixture SHM | Fixture total |
| ---: | ---: | ---: | ---: | ---: |
| 0 | TBD | TBD | TBD | TBD |
| 10,000 | TBD | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD | TBD |

## Turn capture

Fill every row from `benchmarks/results/turn_capture.json`.

| Turns | State | Median | P95 | WAL while open | Final database size |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Stable empty state | TBD | TBD | TBD | TBD |
| 100 | Unchanged 100 KiB state | TBD | TBD | TBD | TBD |
| 1,000 | Stable empty state | TBD | TBD | TBD | TBD |
| 1,000 | Unchanged 100 KiB state | TBD | TBD | TBD | TBD |

## History and session overlay loads

Fill every row from `benchmarks/results/history_load.json` and `benchmarks/results/session_overlay_load.json`.

| Workload | Rows | Median | P95 | Unit |
| --- | ---: | ---: | ---: | --- |
| History data load | 10,000 | TBD | TBD | seconds |
| History overlay construction | 10,000 | TBD | TBD | seconds |
| History data load | 100,000 | TBD | TBD | seconds |
| History overlay construction | 100,000 | TBD | TBD | seconds |
| Session data load | 10,000 | TBD | TBD | seconds |
| Session overlay construction | 10,000 | TBD | TBD | seconds |
| Session data load | 100,000 | TBD | TBD | seconds |
| Session overlay construction | 100,000 | TBD | TBD | seconds |

| History rows | Fixture database | Fixture WAL |
| ---: | ---: | ---: |
| 10,000 | TBD | TBD |
| 100,000 | TBD | TBD |

| Session rows | Fixture database | Fixture WAL | Fixture SHM | Fixture total |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | TBD | TBD | TBD | TBD |
| 100,000 | TBD | TBD | TBD | TBD |
