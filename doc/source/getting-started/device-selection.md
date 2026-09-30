# Device selection

Volumential runs its near-field evaluation, table builds and (with the sumpy
backend) its expansions as OpenCL kernels. Which device those land on changes
wall-clock time by two orders of magnitude and, on a host with several
platforms, can change it silently between runs. Select the device explicitly.

## `PYOPENCL_CTX`

Anything that calls `pyopencl.create_some_context()` — `examples/laplace2d.py`,
`laplace2d_adaptive.py`, `laplace3d.py`, `poisson3d.py` and the library's own
queue-less fallbacks — honours `PYOPENCL_CTX`. It takes a
platform-substring or index, optionally with a device index:

```bash
export PYOPENCL_CTX=portable:0     # the PoCL ("Portable Computing Language") platform, device 0
export PYOPENCL_TEST=portable:0    # the same, for the pytest fixtures
```

Set the two to the same device. When `PYOPENCL_TEST` is set,
`create_some_context()` called without explicit answers takes the first device
*it* selects and ignores `PYOPENCL_CTX`, so with the two apart those callers
run on the `PYOPENCL_TEST` device.

Without it, `create_some_context()` resolves the device on its own, and which
way it goes depends on the session: at a TTY it queries interactively, and with
`sys.stdin.isatty()` false — a batch job, a `tmux` pipeline, CI — it picks a
device "in an implementation-defined manner" instead. Neither is what a
reproducible run wants: one blocks on a prompt, the other silently records a
device nobody chose. Set it even when the host has exactly one platform today.

The tests that build their own context rather than take the fixtures — the
volume FMM regressions, the full-accuracy sweeps, the windowed RKE direct
references, the manufactured batched Duffy checks, the batched FMMLib stage
tests and the shared near-field table builds — read `PYOPENCL_CTX`, not
`PYOPENCL_TEST`, and run on exactly the device it selects. Without it the fp64
ones prefer an fp64 GPU and fall back to an fp64 CPU; see
{doc}`../development/testing`.

On NixOS, also point ICD discovery at a single vendor directory, otherwise
`pyopencl` fails with `PLATFORM_NOT_FOUND_KHR` even when drivers are
installed:

```bash
export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
export OPENCL_VENDOR_PATH=/run/opengl-driver/etc/OpenCL/vendors
```

## Device classes

`PYOPENCL_CTX` names a platform and a device *index*, and an index is
host-specific: `portable:0` is the PoCL CPU on one machine and something else
entirely on the next. Code whose numbers will be quoted therefore tends to
select by *device class* instead, and to fail loudly when the class it was
asked for is absent. The conventional labels are:

| class | selects |
| --- | --- |
| `auto` | the first fp64-capable GPU on any platform, else the first fp64-capable CPU |
| `pocl-cpu` | an fp64-capable CPU device on the PoCL platform |
| `cuda-gpu` | an fp64-capable GPU device on the NVIDIA CUDA platform |

Nothing in the tree takes such an argument today — measurement code lives
outside this repository — but the vocabulary is worth keeping, because timings
recorded elsewhere are quoted in these pages.

Class selection does not replace `PYOPENCL_CTX`, it sits beside it. Anything
that reaches `cl.create_some_context(interactive=False)` has no other explicit
selection at all, and without the variable the device is chosen in an
implementation-defined way. Set it for every run.

Prefer an explicit class over `auto` for anything whose timings will be quoted.
`auto` prefers whatever fp64 GPU it finds, so the same command can change cost
class between two hosts — or between two days on one host — without changing
the recorded arguments. Record the device the run *resolved*, not the class you
asked for; see {doc}`../benchmarks/index`.

## Examples that select their own device

Every example script reads `PYOPENCL_CTX`, in one of two ways.
`laplace2d.py`, `laplace2d_adaptive.py`, `laplace3d.py` and `poisson3d.py`
leave it to `create_some_context()`, so a `PYOPENCL_TEST` that is also set wins
over it, as described above. `helmholtz2d.py`, `helmholtz3d.py` and
`branched_flow_helmholtz2d.py` pass `PYOPENCL_CTX` to `create_some_context()`
as explicit answers, so it decides the device whatever `PYOPENCL_TEST` says,
and they stop if it selects several devices or one without fp64.

Those three also differ without the variable: they do not fall through to
`create_some_context()`. Each carries a `_select_opencl_device` that enumerates
the platforms and takes the `auto` path: first fp64-capable GPU, else first
fp64-capable CPU. On a host with both a CUDA GPU and PoCL they run on the GPU
unless the variable says otherwise. `helmholtz2d.py` and `helmholtz3d.py` log
the device they resolved. Until 2026-09 those two ignored the variable.

The variable still does not decide the device of the tests that take the
`ctx_factory` fixture, which read `PYOPENCL_TEST`. Set both, and read the
device off the run rather than inferring it from the environment.

## Thread caps

Set the thread counts rather than inheriting a host default, and record the
values used alongside any promoted timing:

```bash
export OMP_NUM_THREADS=1          # FMMLib / OpenMP stages
export POCL_MAX_PTHREAD_COUNT=4   # PoCL worker threads
```

## Source and target trees

When validating evaluations *at* source nodes, set

```bash
export VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE=1
```

to fail fast if a traversal was built with separate-but-identical source and
target arrays. Pass `targets=None` to `TreeBuilder` to build a genuinely
coincident tree instead.

## Which device for which work

Table builds, channel-family assembly and the `pyfmmlib` far field are
host-side or lightly parallel work that a modern CPU serves as well as a GPU;
evaluator solves and repeated table applications are where a full-fp64 GPU
wins by one to two orders of magnitude. The first solve in a process is
dominated by sumpy code generation, so a one-shot run does not pay for a
GPU unless the compile cache (`XDG_CACHE_HOME`) is already warm. Measure the
candidate classes once, at a reduced size, before committing a campaign to
either.
