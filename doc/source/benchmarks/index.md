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
preservation, split-parameter coverage, adaptive timing, and the controlled
Gaussian split effective-density diagnostic (which emits NPZ arrays and a JSON
sidecar alongside its CSV). `--case <name>`
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

**Per-driver sidecars.** Five drivers write a JSON sidecar themselves:
`graded_tree_convergence.py`, `gaussian_free_space.py`,
`dmk_effective_density.py`, `rke_field_demo_3d.py` and
`split_parameter_sweep.py`. Every sidecar carries the mode, the problem, the
configuration of the run, and a `run_provenance` block describing the machine
that produced it (see [Run provenance](#run-provenance) below).

Do not key a reader on more than that. Case identity is spelled three ways:
`gaussian_free_space.py` and `dmk_effective_density.py` write `case_id`,
`graded_tree_convergence.py` and `split_parameter_sweep.py` write `case`, and
`rke_field_demo_3d.py` has no top-level case field at all — it carries
per-case data under `cases`.

They all take `--metadata-out`, but their defaults differ, and the difference
is a provenance trap: only `graded_tree_convergence.py` and
`split_parameter_sweep.py` derive the default from `--out`
(`<out stem>-metadata.json`, beside the CSV). The other three each default to
a fixed `build/benchmarks/<case>-metadata.json`, so a run with a custom
`--out` and no `--metadata-out` puts the sidecar somewhere else than the CSV,
and two such runs overwrite one another's metadata. **Pass `--metadata-out`
explicitly whenever you pass `--out`.**

Beyond that shared core, what a sidecar holds varies, and two axes are worth
knowing before promoting one:

- **Device.** Every sidecar now carries `run_provenance.opencl`, the
  *resolved* platform, device, driver version and device type. Three of them
  also keep the older `environment.opencl_device` block
  (`gaussian_free_space.py`, `dmk_effective_density.py`,
  `rke_field_demo_3d.py`), and `graded_tree_convergence.py` and
  `split_parameter_sweep.py` additionally record `"backend"` as a requested
  label. Read the label as a *request* and `run_provenance.opencl` as what
  answered; `command.argv` preserves an explicitly passed `--backend`.
- **Verdict.** Only `graded_tree_convergence.py` writes one — observed orders,
  the asymptotic-regime statement, the matched-error DOF advantage — and only
  on a successful run: a gate failure there writes the CSV and re-raises
  *before* the sidecar, so the failing verdict is not in a file at all.
  `rke_field_demo_3d.py` validates before it builds metadata, and the other
  two record errors and timings without a status field.

**Every other driver in `benchmarks/` has no `--metadata-out` at all** — the
two adaptive timing drivers, `table_equivalence_cache.py`,
`accuracy_preservation.py`, `windowed_rke_sweep.py`, both composition
drivers, `break_even_validation.py`, `derivative_log_preservation.py`,
`keller_segel_continuation.py`, `complex_bessel_parameterized.py` and
`complex_channel_closure.py`. Those five
above are the closed set; treat everything else as sidecar-free, and check
`--help` rather than this list if a driver is added. Passing the option to one
of them is an argparse error, and its run is not self-describing.

Two qualifications on that group, because "no sidecar" is not the same as "CSV
only":

- `windowed_rke_sweep.py` unconditionally writes
  `<out-dir>/windowed_rke_sweep_config.json` after its CSV, holding the
  resolved arguments and run information. That is mostly a *configuration*
  record, useful for reproducing the invocation — but it does carry the same
  `run_provenance` block as the sidecars, so a promoted sweep from it is
  attributable to a device.
- `adaptive_timing_3d.py` writes visualization NPZ files by default, and
  `keller_segel_continuation.py` can write field NPZ files. Those are results,
  and a promotion has to carry them with the CSV.

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

## Run provenance

`benchmarks/_provenance.py` is the one place that answers "which machine
produced this number". Every driver that writes a sidecar calls
`collect_run_provenance()` on its live OpenCL context and stores the result
under the top-level key `run_provenance`; `windowed_rke_sweep.py` stores the
same block in its config JSON. The keys are additive — nothing that was in a
sidecar before has changed name or meaning.

```json
"run_provenance": {
  "opencl": {
    "platform": "Portable Computing Language",
    "platform_version": "OpenCL 3.0 PoCL 7.0  Linux, ...",
    "device": "cpu-...",
    "device_type": "CPU",
    "driver_version": "7.0",
    "vendor": "...",
    "max_compute_units": 30
  },
  "cpu_model": "...",
  "omp_num_threads": 30,
  "pocl_max_pthread_count": 30,
  "pyvkfft_importable": false
}
```

- `opencl` is the **resolved** device, not the requested `--backend` token.
  `--backend cuda-gpu` and `cl.create_some_context(interactive=False)` are
  requests; this is what the ICD loader returned. `device_type` renders the
  `cl_device_type` bit field (`"CPU"`, `"GPU"`, `"DEFAULT|CPU"`), so a device
  that reports several bits is not flattened to one.
- `cpu_model` is the host CPU's model string. Seconds without it silently
  measure host age: the identical PoCL build of one 3D Duffy table differs by
  well over an order of magnitude between a CPU without hardware FMA and a
  current one.
- `omp_num_threads` and `pocl_max_pthread_count` are the worker-thread caps in
  force. `null` means the variable was unset, which is a different run from a
  cap of `1`; a value that is not a plain integer (`OMP_NUM_THREADS` accepts a
  per-nesting-level list) is recorded verbatim as a string.
- `pyvkfft_importable` decides a cost class, not a detail: sumpy's
  FFT-accelerated multipole-to-local uses `pyvkfft` when it is importable and
  falls back to a loopy FFT — several times the arithmetic — when it is not.

Every one of those drivers also prints a single `RESOLVED-DEVICE` line on
stdout as soon as the device is resolved, before the work starts, so a
multi-hour log says on its first lines what answered:

```text
RESOLVED-DEVICE platform='Portable Computing Language' platform_version='OpenCL 3.0 PoCL 7.0 ...' device='cpu-...' device_type=CPU driver_version='7.0'
```

None of this carries a host name, user name or path, so a sidecar can be
committed next to its CSV. `test/test_benchmark_helpers.py` pins the shape of
the block against a mocked context, so it is checked on a machine with no
OpenCL platform at all.

## What a promoted measurement must record

At minimum:

- the full locked resolution (`uv.lock`) — a commit for each Git-sourced
  dependency, a version and artifact hashes for each PyPI one — and the
  `pyfmmlib` source revision;
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

The sidecars keep (1) apart from (3) by construction, and now keep (2) apart
as well. `benchmarks/_provenance.py`'s `time_repeats()` and
`first_call_and_warm()` split a repeat series into

- `<thing>_first_call_s` — the first call of the *process* for that code
  path, including whatever code generation and kernel compilation it
  triggered;
- `<thing>_warm_s` — the median of the remaining repeats, or `null` when the
  driver made only one call. A `null` is an honest "not measured", never a
  warm number contaminated by code generation;
- `<thing>_warm_repeat_count` and `<thing>_samples_s` — the count and the
  raw series, in call order.

Where they appear:

- `gaussian_free_space.py`, `dmk_effective_density.py` and
  `graded_tree_convergence.py` take `--warm-repeats` (default 1) and record
  the split under `timing` (per ladder rung, under `fmm_timing`, for the
  graded driver). `--warm-repeats 0` restores the single timed call and
  leaves `warm_s` at `null`.
- `rke_field_demo_3d.py` already ran an untimed warm-up before its timed
  solve; that call is now timed, so the split costs nothing. Its
  `direct_solve_wall_s` / `split_solve_wall_s` remain the *warm* numbers they
  always were.
- `split_parameter_sweep.py` and `windowed_rke_sweep.py` already separate a
  warm-up from the repeatable path in their own columns
  (`_time_repeated`'s untimed first call; `classical_warmup_seconds` against
  `classical_assemble_seconds`).

Three different recorders produce the seconds in these CSVs, and a column
should never mix them:

- `drive_volume_fmm`'s `timing_data` mapping — the per-stage FMM times.
  `adaptive_timing.py` and `adaptive_timing_3d.py` ask for it.
- `NearFieldInteractionTableManager.last_get_table_timings` — cold build and
  warm load of a table, read by `adaptive_timing.py`, `adaptive_timing_3d.py`
  and `table_equivalence_cache.py`, which is where their cold/warm rows come
  from.
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
   the claim being made, with thread caps set explicitly, `PYOPENCL_CTX`
   exported, and `--backend` named for the drivers that have one — passing it
   to a driver that does not is an argparse error, not a no-op.
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
