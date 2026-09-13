# Choosing a wrangler

{doc}`volume-fmm-workflow` describes the two far-field backends;
this page says which one to pick. Nothing here changes a default: the
drivers and the library still build `FPNDExpansionWrangler` unless you ask
for something else. It exists because the right choice depends on the
kernel, the dimension *and* the device class, and the penalty for getting it
wrong is an order of magnitude rather than a few percent.

## The short version

| kernel and dimension | CPU OpenCL device | fp64 GPU |
| --- | --- | --- |
| 3D Helmholtz, high order | **FMMLib**, with an OpenMP `pyfmmlib` | either; sumpy is already fast |
| 3D Laplace | either; measure | either |
| 2D Laplace / Helmholtz | either; measure | either |
| Yukawa, any dimension | **sumpy** (FMMLib has no Yukawa) | sumpy |
| anything using the near-field Helmholtz split | **sumpy** (see the caveat below) | sumpy |
| any kernel sumpy can differentiate but `pyfmmlib` does not implement | sumpy — it is the only option | sumpy |

`FPNDFMMLibExpansionWrangler` covers 2D and 3D Laplace and Helmholtz and
nothing else. Everywhere it does not reach, the question does not arise.

## Why 3D Helmholtz on a CPU device is the case that matters

A 3D Helmholtz evaluator solve at FMM order 23 spends essentially all of its
time in one stage. Measured on a 2013-class 40-thread CPU under PoCL, with the
OpenCL queue synchronized at every phase boundary — 3D, source order 3, 5
levels, 110,592 targets, wave number 32, FMM order 23, an assembled windowed
near-field table:

| phase | s | share |
| --- | --- | --- |
| form_multipoles | 0.12 | 0.06 % |
| coarsen_multipoles | 0.05 | 0.02 % |
| near-field table apply | 1.66 | 0.78 % |
| **multipole_to_local** | **210.61** | **99.05 %** |
| eval_multipoles + form_locals + refine_locals + eval_locals | 0.18 | 0.09 % |

sumpy's default expansion for `HelmholtzKernel(3)` at that order is the
linear-PDE-conforming volume Taylor expansion with `VolumeTaylorM2LWithFFT`:
6,627 complex translation-class entries per box over 640,584 list-2 pairs.
Inside the stage the forward FFT costs 34.5 s, the pointwise translation 92.7 s
and the inverse FFT 87.0 s. With `pyvkfft` unavailable the FFT is loopy's
fallback, roughly 7.6 times the arithmetic of a real FFT — and on PoCL 7 that
is not a choice, because it miscompiles VkFFT
([pocl/pocl#2069](https://github.com/pocl/pocl/issues/2069)).

The same geometry, table and order through `FPNDFMMLibExpansionWrangler`, whose
rotation-based M2L does roughly twenty times less total work:

| far field | `pyfmmlib` | OpenMP threads | M2L (s) | warm solve (s) |
| --- | --- | --- | --- | --- |
| sumpy/loopy on PoCL | – | – | 210.6 | 217.3 |
| FMMLib rotation M2L | 2024.1.1 wheel, no OpenMP | 8, inert | 317.5 | 312.4 |
| FMMLib rotation M2L | OpenMP build, 1 thread | 1 | 300.1 | 310.5 |
| **FMMLib rotation M2L** | **OpenMP build** | **30** | **17.3–17.9** | **23.5–24.2** |

That is **9.1x on the same hardware**, and it is entirely a threading result:
the serial FMMLib far field is *slower* than the sumpy one. The `_imany`
routines scale about 17x from 1 to 30 threads, but only once `pyfmmlib`
actually carries OpenMP, which the PyPI `2024.1.1` wheel does not. Build it as
{doc}`../getting-started/installation` describes and run that page's
`ldd`/`otool` check: `FPNDFMMLibExpansionWrangler` falls back to a serial
per-box path **without complaining**, so a mis-provisioned environment is
correct, slow, and indistinguishable from a correct one except by timing.

## On a GPU the question goes away

The same solve on a current data-centre GPU, through `--backend cuda-gpu` and
with no code change, runs the sumpy path's warm Helmholtz solve in 1.5 s
against 214 s on that CPU (139x), and a Yukawa order-12 solve in 0.40 s against
38 s (96x); the near-field table apply gains 18 to 27x. Results agree with the
committed CPU row to 2e-6 relative, and the solve's peak device memory is about
57 GB, so the problem size and the card have to be matched deliberately.

The FFT-based M2L is therefore **not** a bottleneck on a GPU, and there is no
reason to move a GPU run to FMMLib: `pyfmmlib` is host Fortran and would give
the GPU nothing to do.

## First solve versus warm solve

Every number above is a *warm* solve. The first solve of a process pays
sumpy's code generation for the expansion kernels, and at order 23 that has
been measured at 884 s — against a 1.5 s warm solve on the GPU. A one-shot
driver run is therefore mostly code generation: moving one such run to a GPU
gained 2.2x overall, not 139x.

Two consequences:

- Compare wranglers on warm solves, never on a process total.
- A one-shot run is not the workload a wrangler choice should be made for.
  Amortize the code generation over many solves, or warm the compile cache,
  before either backend's per-solve speed decides anything.

The drivers record the two separately — see
{doc}`../benchmarks/index` for `*_first_call_s` against `*_warm_s`, and for
the `run_provenance` block that says which device and which CPU a number came
from. Seconds from different device or CPU classes never belong in one table.

## Caveats before switching a 3D Helmholtz run to FMMLib

- **No near-field Helmholtz split.** The FMMLib wrangler has no
  `eval_direct_helmholtz_split_correction`, so anything using the online split
  of {doc}`helmholtz_split` stays on sumpy.
- **The backends do not agree to roundoff.** Rotation M2L and volume-Taylor
  M2L agree to about 6.2e-8 relative on the solve above — both give
  1.0760894e-4 against the manufactured solution — which is far above the
  1e-11-level agreement some committed accuracy rows assert. A gate calibrated
  against one backend has to be re-certified against the other; a switch is
  not accuracy-neutral bookkeeping.
- **M2M and L2L stop being free.** Once M2L drops from 99 % to a few percent,
  the remaining translation stages become roughly 19 % of the solve, and the
  profile to optimize next is a different one.
- **Check the OpenMP build.** Repeated because it is the single most common
  way this measurement is mis-taken: without it, FMMLib is slower than sumpy
  here, not faster.

## How to decide for your own case

1. Run the solve twice in one process and compare the *second* one. The
   drivers do this for you and record `*_first_call_s` and `*_warm_s`
   separately.
2. Name the device class explicitly — `--backend pocl-cpu` or
   `--backend cuda-gpu`, or `PYOPENCL_CTX` for the drivers that call
   `cl.create_some_context` (see {doc}`../getting-started/device-selection`).
   `auto` prefers any fp64 GPU it finds and silently changes the cost class
   between hosts.
3. On a CPU device, set the thread caps you mean (`OMP_NUM_THREADS` for the
   FMMLib far field, `POCL_MAX_PTHREAD_COUNT` for the OpenCL device) and
   record them. An FMMLib comparison at one thread measures nothing.
4. Check that the two backends agree on *your* problem before quoting either,
   at the tolerance your gates assert rather than at the tolerance that looks
   small.

## Related

- {doc}`volume-fmm-workflow` — what each wrangler is and what it plugs into.
- {doc}`../getting-started/installation` — the OpenMP `pyfmmlib` build and its
  verification.
- {doc}`../getting-started/device-selection` — `--backend`, `PYOPENCL_CTX`,
  thread caps.
- {doc}`../benchmarks/index` — what a promoted measurement must record.
- [#136](https://github.com/xywei/volumential/issues/136) — the measurements
  quoted here.
