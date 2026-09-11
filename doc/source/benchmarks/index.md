# Benchmarks and reproducibility

`benchmarks/` holds reproducible drivers that emit CSV and JSON evidence. They
are not micro-benchmarks and not part of the test suite: each one answers a
specific question about the library, and the answer is a file that can be
promoted into a manuscript. `benchmarks/README.md` is the per-driver reference;
this page is the contract they all share.

## Two modes

Every driver takes `--mode`:

- `--mode smoke` — small enough for CI and for a laptop-scale sanity check.
  Pull-request CI runs several of these. A smoke row is evidence that the
  driver works, never evidence about performance.
- `--mode full` — the controlled configuration. Full runs belong on a
  dedicated, otherwise idle compute host, under `tmux` and `nice`, with output
  tee'd to a log.

```bash
python benchmarks/adaptive_timing.py --mode smoke \
  --out build/benchmarks/adaptive-timing.csv
```

## The suite driver

`benchmarks/performance_suite.py` runs the maintained set with a shared output
layout and a JSON command manifest:

```bash
python benchmarks/performance_suite.py --mode smoke \
  --out-dir build/benchmarks/performance-suite
```

It currently covers canonical table equivalence and cache economics, accuracy
preservation, split-parameter coverage, and adaptive timing. `--case <name>`
runs a subset, `--list-cases` inspects the registry without importing any
OpenCL-dependent module, `--dry-run` emits the manifest without executing, and
`--backend` propagates device selection to the cases that expose it. The
manifest is written even when a case fails, so a partial controlled run stays
auditable. Each case gets an output-local cache directory under
`<out-dir>/<case>/cache`, which keeps table caches with the run artifacts
instead of leaking into a shared default.

The Paper 1 mechanism drivers — the composition, break-even, windowed-sweep,
graded-convergence and Keller-Segel drivers — have campaign-specific acceptance
gates and structured outputs, and are deliberately *not* in the suite. Run them
individually; `benchmarks/README.md` documents each one's flags and gates.

## Metadata sidecars

A CSV on its own is not evidence. Metadata reaches a promoted result by two
different routes, and they are not interchangeable.

**Per-driver sidecars.** Four drivers write a JSON sidecar themselves:
`graded_tree_convergence.py`, `gaussian_free_space.py`,
`dmk_effective_density.py` and `rke_field_demo_3d.py`. They all take `--metadata-out`, but their defaults differ, and the difference
is a provenance trap: only `graded_tree_convergence.py` derives the default
from `--out` (`<out stem>-metadata.json`, beside the CSV). The other three each
default to a fixed `build/benchmarks/<case>-metadata.json`, so a run with a
custom `--out` and no `--metadata-out` puts the sidecar somewhere else than the
CSV, and two such runs overwrite one another's metadata. **Pass
`--metadata-out` explicitly whenever you pass `--out`.**

A sidecar carries the case id, the mode, the problem definition, the full
configuration, and the driver's own verdict on the run (convergence orders,
asymptotic-regime statements, gate outcomes).

What it does **not** reliably carry is the device: only
`graded_tree_convergence.py` records the selected `--backend` in its metadata,
so for the other three the device class has to come from the wrapper below.

The remaining drivers — `table_equivalence_cache.py`,
`accuracy_preservation.py`, `split_parameter_sweep.py`, `adaptive_timing.py`
and the composition and sweep drivers — write CSV only and have no
`--metadata-out`. Passing the option to them is an error, and a run of one of
them is not self-describing.

`performance_suite.py` is neither: it takes `--manifest`, not
`--metadata-out`, and writes a JSON *command manifest* of what it ran. It
forwards `--metadata-out` to exactly one wrapped case
(`dmk-effective-density`, the only one that implements it); the others stay
CSV-only. A manifest records the commands, not the environment.

**The metadata wrapper.** Every full run, sidecar or not, is wrapped by
the paper repository's metadata tool, which captures hardware, OpenCL,
package, commit, parameter and result-file metadata. For the drivers with no
sidecar of their own, this is the *only* provenance a promoted artifact has:

```bash
python /path/to/boxcode-paper/tools/run_benchmark_with_metadata.py \
  --out /path/to/raw-runs/table-equivalence-cache-metadata.json \
  --case-id table-equivalence-cache --mode full \
  --repo volumential=/path/to/volumential \
  --param kernel=laplace --param dimension=3 \
  --result-file table_equivalence.csv --result-file cache_economics.csv \
  --cwd /path/to/volumential \
  -- python benchmarks/table_equivalence_cache.py --mode full \
     --out-dir /path/to/raw-runs
```

## What a promoted measurement must record

At minimum:

- the locked dependency commits (`uv.lock`) and the `pyfmmlib` source revision;
- the resolved OpenCL platform and device with its driver or runtime build —
  for PoCL, its LLVM version;
- the **CPU model class** and whether it has hardware FMA, plus the
  worker-thread cap in force (`OMP_NUM_THREADS`, `POCL_MAX_PTHREAD_COUNT`);
- for GPU runs, peak device memory and mean utilization;
- the compile-cache state, cold or warm (`XDG_CACHE_HOME`);
- **first-call versus warm-repeat seconds, separately**;
- the benchmark parameters, and for table work the build routing
  (`direct_build_routing`; see {doc}`../user-guide/table-build-routing`).

Two rules follow from that list and are worth stating on their own.

**Never put seconds from different device or CPU classes in one table.** The
identical PoCL build of one 3D Duffy table can differ by well over an order of
magnitude between a CPU without hardware FMA and a current one, with no code
change at all: PoCL's fp64 `sincos`, and every builtin routed through its
`fma()`, fall to software emulation when the hardware instruction is absent
([#138](https://github.com/xywei/volumential/issues/138)). A table that mixes
the two classes is measuring the pool, not the code.

**Never quote `--backend auto` for a cost claim.** `auto` prefers whatever
fp64 GPU it finds, so the same recorded command can change cost class between
hosts. Name the class — see {doc}`../getting-started/device-selection`.

**Keep infrastructure out.** Host names, user names, network details, remote
paths and load figures do not belong in the repository, in a published
artifact, or in an issue. Describe the reproducible software environment and
the experiment parameters; use neutral labels for machines. Inspect generated
CSV/JSON metadata, figures and archives for leaks, not just the source text.

## First-call versus warm

Three costs are routinely confused, and a driver that reports one total hides
all three:

1. **Table build** — the singular quadrature. Cold only; a warm run loads the
   table from SQLite in milliseconds. This is the dominant first-run cost and
   the thing the assembly strategies of
   {doc}`../design-notes/windowed-channels` exist to amortize.
2. **Code generation** — sumpy builds and compiles the expansion kernels on the
   first solve of a *process*. On a one-shot driver run this can be the
   majority of the wall clock, which is why moving such a run to a GPU buys far
   less than the per-solve speedup suggests.
3. **The solve itself** — the only number that scales with the problem.

Three different recorders produce the seconds in these CSVs, and a column
should never mix them:

- `drive_volume_fmm`'s `timing_data` mapping — the per-stage FMM times.
  `adaptive_timing.py` is the driver that asks for it.
- `NearFieldInteractionTableManager.last_get_table_timings` — cold build and
  warm load of a table, read by `adaptive_timing.py` and
  `table_equivalence_cache.py`, which is where their cold/warm rows come from.
- `time.perf_counter()` around a whole phase or solve — what the composition
  drivers report, and what the other two use for the parts outside the FMM and
  the table manager.

`volumential.phase_profile` is a different measurement and only two drivers
use it: `split_parameter_sweep.py` and `break_even_validation.py`, which
activate a `PhaseProfile` and report its `shares()`. That path synchronizes the
OpenCL queue at phase boundaries, so its numbers are not the wall or stage
times the other drivers report and the two must not be put in one column.

## Gates, and why a CSV is not a pass

Several drivers apply acceptance gates: a Yukawa split-order convergence gate,
the windowed `failed`/in-declaration-refusal/small-theta gates, far-field
resolution checks, grading checks on adaptive ladders.

Most of them report **after** the CSV has been written, so a multi-hour run
does not trade its measurements for a verdict — the composition drivers, the
split-parameter sweep and `graded_tree_convergence.py` all write first and gate
afterwards. That is a property of the individual driver, not a rule of the
suite: `rke_field_demo_3d.py` still runs
`_validate_full_order_convergence(rows)` *inside* `run_benchmark`, before
`main` reaches `write_csv`, so a convergence-gate failure there leaves no CSV,
no `.npz` and no sidecar. Check the driver before assuming a long run is
recoverable.

The consequence is the important part: **the presence of an output CSV is not
evidence of a successful run.** Check the process exit status, and check the
rows' own status columns (`windowed_status`, `far_field_status`,
`grading_status`, the mismatch columns).

Do not key tooling on a message: how a gate failure announces itself is
per-driver. `split_parameter_sweep.py` and `graded_tree_convergence.py` print
`GATE-FAILED` on stdout and exit non-zero; `FAR-FIELD-UNRESOLVED` with exit
code 2 is specific to the split sweep; `windowed_rke_sweep.py` prints an
`[error]` line and returns 1; the two adaptive composition drivers let the
validator raise, so the failure arrives as a traceback on stderr. The exit
status is the one signal all of them share.

A refusal is not a failure. A certificate that refuses, or a row that records
`far_field_status = refused_order_cap`, is a measurement of where the mechanism
stops applying — it carries its certificate and diagnostic columns and is meant
to be reported, not retried until it passes.

## Promotion

1. Run `--mode smoke` on the candidate host, and check the driver's own
   smoke-mode gates pass.
2. Run `--mode full` under the metadata wrapper, on a host whose class matches
   the claim being made, with thread caps and `--backend` set explicitly.
3. Check the exit status and the per-row status columns, not just the presence
   of output.
4. Promote the CSV together with the wrapper metadata, and with the driver's
   own sidecar where it writes one. A result without its metadata is not
   promotable.
5. If a driver's full-mode verdict states a limitation ("extend the ladder"),
   extend the ladder — do not requote a pre-asymptotic order.

## Related

- {doc}`../getting-started/device-selection` — `--backend`, `PYOPENCL_CTX`,
  thread caps.
- {doc}`../user-guide/validation_matrix` — how benchmark evidence relates to
  the test tiers.
- `benchmarks/README.md` — the per-driver reference.
