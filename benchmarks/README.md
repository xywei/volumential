# Benchmarks

These scripts emit reproducible CSV artifacts for manuscript evidence. Keep smoke modes lightweight enough for CI/local checks and reserve full sweeps for controlled machines such as `ipa`.

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

The benchmark sweeps 2D scalar Helmholtz wave numbers and Yukawa screening parameters. Helmholtz rows compare against a manufactured Gaussian reference; Yukawa rows compare split near-field evaluation against a direct non-split near-field table for the same parameter. Each row records split cache accounting. The `online_remainder_s` column is a conservative upper bound equal to the split warm solve time, because the current wrangler instrumentation does not isolate smooth online-remainder work from the rest of the warm FMM/List-1 evaluation.
