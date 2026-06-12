# Benchmarks

These scripts emit reproducible CSV artifacts for manuscript evidence. Keep smoke modes lightweight enough for CI/local checks and reserve full sweeps for controlled machines such as `ipa`.

## Performance Suite Driver

Use the suite driver to run the maintained benchmark set with a shared output
layout and JSON command manifest:

```bash
python benchmarks/performance_suite.py --mode smoke --out-dir build/benchmarks/performance-suite
```

For full runs, execute on a controlled machine such as `ipa` and wrap the suite
with the paper repository metadata tool before promoting results:

```bash
python benchmarks/performance_suite.py --mode full --out-dir /path/to/raw-runs/performance-suite
```

The suite currently covers canonical table equivalence/cache economics,
accuracy preservation, split-parameter coverage, and adaptive timing. Use
`--case <name>` to run a subset, `--dry-run` to emit the command manifest
without executing benchmarks, `--backend <auto|pocl-cpu|cuda-gpu>` to preserve
device selection for benchmark scripts that expose a backend option, and
`--list-cases` to inspect the registered cases without importing
OpenCL-dependent benchmark modules. The manifest is written even when a case
fails, so partial controlled runs remain auditable.

## Table Equivalence And Cache Economics

```bash
python benchmarks/table_equivalence_cache.py --mode smoke --out-dir build/benchmarks/table-equivalence-cache
```

Full paper runs should be wrapped from the paper repository so hardware, OpenCL, package, commit, parameter, and result-file metadata are captured:

```bash
python /path/to/boxcode-paper/tools/run_benchmark_with_metadata.py \
  --out /path/to/raw-runs/table-equivalence-cache-metadata.json \
  --case-id table-equivalence-cache \
  --mode full \
  --repo volumential=/path/to/volumential \
  --param kernel=laplace \
  --param dimension=3 \
  --param cache_state=cold-and-warm \
  --result-file table_equivalence.csv \
  --result-file cache_economics.csv \
  --cwd /path/to/volumential \
  -- python benchmarks/table_equivalence_cache.py --mode full --out-dir /path/to/raw-runs
```

The script writes:

- `table_equivalence.csv`: max absolute/relative mismatch between direct per-level tables and a scaled canonical level-0 table.
- `cache_economics.csv`: cold build, warm load, payload bytes, cache bytes, and symmetry-reduction counts from the table manager and table diagnostics.

## Accuracy Preservation

```bash
python benchmarks/accuracy_preservation.py --mode smoke --out build/benchmarks/accuracy-preservation.csv
```

The benchmark compares canonical rescaled tables, direct per-level tables, and direct evaluation on a smooth manufactured 3D Poisson problem. Full runs should be wrapped with the paper repository metadata tool before their CSVs are promoted to manuscript data.

## Helmholtz/Yukawa Split Parameter Sweep

```bash
python benchmarks/split_parameter_sweep.py --mode smoke --out build/benchmarks/split-parameter-sweep.csv
```

The benchmark sweeps 2D scalar Helmholtz wave numbers and Yukawa screening parameters. Rows compare each split order against the highest split order in the same run for that kernel parameter, giving a split-order convergence measurement while holding the pretabulated basis-table family fixed. Each row records split cache accounting. In `_row_from_result`, the `online_remainder_s` column is a conservative upper bound: it equals `split_warm_s` when `uses_online_remainder` is true, and is `0.0` otherwise, because the current wrangler instrumentation does not isolate smooth online-remainder work from the rest of the warm FMM/List-1 evaluation.

## Adaptive Timing

```bash
python benchmarks/adaptive_timing.py --mode smoke --out build/benchmarks/adaptive-timing.csv
```

The benchmark runs 2D Laplace evaluations on deterministically adapted meshes and writes one cold-cache and one warm-cache row per case. Rows report mesh/adaptation setup, geometry construction, table build or load, FMM wall time, and the timing categories exposed by `drive_volume_fmm`. Full paper runs should be wrapped with the paper repository metadata tool before their CSVs are promoted to manuscript data.
