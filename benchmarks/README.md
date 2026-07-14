# Benchmarks

These scripts emit reproducible CSV artifacts for manuscript evidence. Keep smoke modes lightweight enough for CI/local checks and reserve full sweeps for a suitable, currently idle remote compute host. Keep infrastructure-identifying metadata private and redact it from public artifacts.

## Performance Suite Driver

Use the suite driver to run the maintained benchmark set with a shared output
layout and JSON command manifest:

```bash
python benchmarks/performance_suite.py --mode smoke --out-dir build/benchmarks/performance-suite
```

For full runs, execute on a controlled remote compute host and wrap the suite
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
fails, so partial controlled runs remain auditable. The suite passes each case
an output-local cache directory under `<out-dir>/<case>/cache`, keeping table
caches with the promoted run artifacts instead of relying on script defaults.

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

The benchmark sweeps 2D scalar Helmholtz wave numbers and Yukawa screening parameters. Each row compares the full implemented split evaluator against a direct fixed-parameter near-field table at the same parameter and application level. It separately records direct-table and RKE-channel setup/load costs, payload, repeated full applications, isolated coefficient and residual diagnostics, cold/warm strategy totals, and a linear break-even model. `--direct-levels` controls the levels provisioned by the direct setup strategy, while `--nlevels` is the application level; `--repeat-count` is the number of applications per parameter, and `break_even_repeat_count` uses the same per-parameter unit.

## Adaptive Timing

```bash
python benchmarks/adaptive_timing.py --mode smoke --out build/benchmarks/adaptive-timing.csv
```

The benchmark runs 2D Laplace evaluations on deterministically adapted meshes and writes one cold-cache and one warm-cache row per case. Rows report mesh/adaptation setup, geometry construction, table build or load, FMM wall time, and the timing categories exposed by `drive_volume_fmm`. Full paper runs should be wrapped with the paper repository metadata tool before their CSVs are promoted to manuscript data.

## DMK Effective-Density Diagnostic

```bash
python benchmarks/dmk_effective_density.py --mode smoke --slice-size 8
```

The benchmark isolates a controlled 3D Laplace DMK far-plus-residual effective
density model. The far density is the analytic Gaussian-filtered source with
multiplier `exp(-sigma_l^2 |k|^2 / 4)` and smoothing standard deviation
`sigma_l / sqrt(2)`, where `sigma_l = box_side_length / sqrt(log(1/epsilon))`.
The residual uses the all-space fourth-order Taylor/asymptotic model for
`erfc(r/sigma_l)/(4*pi*r)`, converts `u_R ~= c0 rho + c1 Delta rho + c2 Delta^2 rho`
to an equivalent density by `rho_R_eff = -Delta u_R`, and applies Volumential's
singular Laplace path to the total density. CSV/JSON/NPZ diagnostics report the
Volumential error against the analytic asymptotic split and the residual split
bias against the unsmoothed Gaussian reference. Smoke mode uses a lightweight
uniform `q=4`, `nlevels=2`, root-half-width `0.5` mesh; full mode uses the
resolved `q=4`, `nlevels=4`, root-half-width `1.0` configuration.
