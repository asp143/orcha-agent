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
