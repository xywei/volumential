# Benchmarks and reproducibility

Volumential no longer ships benchmark drivers. The measurements that back
publications are run from the manuscript repository, which pins the Volumential
revision it measured, so a promoted number names a commit of this library
rather than somebody's working tree. The drivers' history up to their removal
stays here: `7c75ed1` is the last revision of `main` that carries them in the
tree, and they are reachable from it unchanged.

What remains on this page is the part the rest of this site links here for:
**what a measurement of this library has to record to be worth quoting**,
wherever it is run from. It is guidance for contributors, not a description of
any particular program.

## What a measurement must record

At minimum:

- **The resolved OpenCL platform, device and driver** — platform name and
  version, device name and device type, and the driver or runtime build (for
  PoCL, its LLVM version). Record what the ICD loader *returned*, never the
  device-class token that was requested: a `--backend`-style argument,
  `PYOPENCL_CTX` and `cl.create_some_context(interactive=False)` are all
  requests, and a request is not a measurement. Render `cl_device_type` as the
  bit field it is (`CPU`, `GPU`, `DEFAULT|CPU`) rather than flattening a device
  that reports several bits to one. Print the resolution once, before the work
  starts, so a multi-hour log says on its first lines what answered.
- **The Volumential revision**, and the locked resolution around it — a commit
  for each Git-sourced dependency, a version and artifact hashes for each PyPI
  one, and the `pyfmmlib` source revision.
- **The CPU model class**, and whether it has hardware FMA, plus the
  worker-thread caps in force (`OMP_NUM_THREADS`, `POCL_MAX_PTHREAD_COUNT`). An
  unset cap is a different run from a cap of `1`, so record the difference
  instead of a blank.
- **The parameters of the run** — every knob the claim depends on, and for
  table work the build routing (`direct_build_routing`; see
  {doc}`../user-guide/table-build-routing`).
- **First-call and warm-repeat seconds, separately** (below).
- **The FFT backend sumpy actually selected** for the run's queue. An
  importable `pyvkfft` is necessary but **not sufficient**: sumpy also honours
  `SUMPY_FFT_BACKEND`, refuses VkFFT on an out-of-order queue, and refuses it
  on PoCL 7 and later because that miscompiles it — so a host with `pyvkfft`
  installed routinely still runs the loopy fallback, with several times the
  arithmetic in the FFT-accelerated multipole-to-local. That is a cost class,
  not a detail. Record the selection, not the installation.
- **The compile-cache state**, cold or warm (`XDG_CACHE_HOME`), and for GPU
  runs peak device memory and mean utilization.

## First call versus warm

Three costs are routinely confused, and one reported total hides all three:

1. **Table build** — the singular quadrature. Cold only; a warm run loads the
   table from SQLite in milliseconds. This is the dominant first-run cost, and
   what the assembly strategies of {doc}`../design-notes/windowed-channels`
   exist to amortize.
2. **Code generation** — sumpy builds and compiles the expansion kernels on the
   first solve of a *process*. In a one-shot run this can be most of the wall
   clock, which is why moving such a run to a GPU buys far less than the
   per-solve speedup suggests.
3. **The solve itself** — the only number that scales with the problem.

Report the first call of the process for a given code path and the median of
the remaining repeats as two numbers, with the repeat count and the raw series
in call order beside them. Where only one call was made, say so: an absent warm
number is an honest "not measured", a warm number contaminated by code
generation is not.

Keep the recorders apart too. Per-stage FMM times from `drive_volume_fmm`'s
`timing_data`, cold build and warm load from
`NearFieldInteractionTableManager.last_get_table_timings`, and a
`time.perf_counter()` around a whole phase or solve measure different things,
and the shares from {class}`volumential.phase_profile.PhaseProfile` are
different again — that path synchronizes the OpenCL queue at phase boundaries.
Never put two of them in one column.

## Three rules

**Never put seconds from different device or CPU classes in one table.** The
identical PoCL build of one 3D Duffy table can differ by well over an order of
magnitude between a CPU without hardware FMA and a current one, with no code
change at all: PoCL's fp64 `sincos`, and every builtin routed through its
`fma()`, fall to software emulation when the hardware instruction is absent
([#138](https://github.com/xywei/volumential/issues/138)). A table that mixes
the two classes is measuring the pool, not the code.

**Never quote an `auto` device selection for a cost claim.** An `auto` rule
prefers whatever fp64 GPU it finds, so the same recorded command changes cost
class between hosts, or between two days on one host, without changing the
recorded arguments. Name the class — see
{doc}`../getting-started/device-selection`.

**Keep infrastructure out.** Host names, user names, network details, remote
paths and load figures belong in none of it: not in this repository, not in a
published artifact, not in an issue. Describe the reproducible software
environment and the experiment parameters, and use neutral labels for machines.
Inspect generated CSV and JSON metadata, figures and archives for leaks, not
just the source text.

## An output file is not a pass

The presence of an output CSV is not evidence of a successful run. Check the
process exit status, and check the status columns the run itself wrote: a gate
can fail after the rows are written, and a run that stops early can leave a
plausible-looking partial file behind. Do not key tooling on a message either —
the exit status is the one signal every run shares.

A refusal is not a failure. A certificate that refuses, or a row recording the
parameter regime where a mechanism stops applying, is a measurement of where
the mechanism stops — it carries its diagnostics and is meant to be reported,
not retried until it passes.

## Related

- {doc}`../getting-started/device-selection` — `PYOPENCL_CTX`, device classes,
  thread caps.
- {doc}`../user-guide/validation_matrix` — how measured evidence relates to the
  test tiers.
- {doc}`../user-guide/table-build-routing` — build routing, and why it belongs
  in the record.
