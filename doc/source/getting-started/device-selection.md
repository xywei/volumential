# Device selection

Volumential runs its near-field evaluation, table builds and (with the sumpy
backend) its expansions as OpenCL kernels. Which device those land on changes
wall-clock time by two orders of magnitude and, on a host with several
platforms, can change it silently between runs. Select the device explicitly.

## `PYOPENCL_CTX`

Anything that calls `pyopencl.create_some_context()` — `examples/laplace2d.py`,
`laplace3d.py`, `poisson3d.py`, several benchmark drivers, and the library's
own queue-less fallbacks — honours `PYOPENCL_CTX`. It takes a
platform-substring or index, optionally with a device index:

```bash
export PYOPENCL_CTX=portable:0     # the PoCL ("Portable Computing Language") platform, device 0
export PYOPENCL_TEST=portable:0    # the same, for the pytest fixtures
```

Without it, `create_some_context()` resolves the device on its own, and which
way it goes depends on the session: at a TTY it queries interactively, and with
`sys.stdin.isatty()` false — a batch job, a `tmux` pipeline, CI — it picks a
device "in an implementation-defined manner" instead. Neither is what a
reproducible run wants: one blocks on a prompt, the other silently records a
device nobody chose. Set it even when the host has exactly one platform today.

On NixOS, also point ICD discovery at a single vendor directory, otherwise
`pyopencl` fails with `PLATFORM_NOT_FOUND_KHR` even when drivers are
installed:

```bash
export OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors
export OPENCL_VENDOR_PATH=/run/opengl-driver/etc/OpenCL/vendors
```

## `--backend`

Most benchmark drivers under `benchmarks/` additionally take `--backend`, which
names a *device class* rather than a platform index and fails loudly when that
class is absent:

| `--backend` | selects |
| --- | --- |
| `auto` (default) | the first fp64-capable GPU on any platform, else the first fp64-capable CPU |
| `pocl-cpu` | an fp64-capable CPU device on the PoCL platform |
| `cuda-gpu` | an fp64-capable GPU device on the NVIDIA CUDA platform |

```bash
python benchmarks/adaptive_timing.py --mode smoke --backend pocl-cpu
python benchmarks/performance_suite.py --mode smoke --backend cuda-gpu \
  --out-dir build/benchmarks/performance-suite
```

Where a misspelled `--backend` is caught varies, and it comes down to which
selector a driver uses. `performance_suite.py` constrains the argument at the
parser. `split_parameter_sweep.py` validates against the whitelist before
touching the ICD loader, and the drivers that import its selector inherit that
— the two composition drivers, `break_even_validation.py`,
`keller_segel_continuation.py` and `rke_field_demo_3d.py`. All of those name
the typo.

`adaptive_timing.py` and `gaussian_free_space.py` carry their own selector,
which calls `cl.get_platforms()` first, and `adaptive_timing_3d.py`,
`graded_tree_convergence.py` and `dmk_effective_density.py` import theirs. On
those five, on a host with broken ICD discovery, a typo surfaces as
`PLATFORM_NOT_FOUND_KHR` rather than as an argument error — so if you see that
message from one of them, check the spelling before you check the drivers.

`--backend` does not replace `PYOPENCL_CTX`, it sits beside it. Several drivers
— `table_equivalence_cache.py`, `accuracy_preservation.py`,
`derivative_log_preservation.py`, `windowed_rke_sweep.py` — select their device
with `cl.create_some_context(interactive=False)` and expose no `--backend` at
all, and `performance_suite.py` deliberately does not propagate its own
`--backend` to the cases that have none. For those, `PYOPENCL_CTX` is the only
explicit selection there is, and without it the device is chosen in an
implementation-defined way. Set the environment variable for every run, and add
`--backend` where the driver offers it.

Prefer an explicit class over `auto` for anything whose timings will be quoted.
`auto` prefers whatever fp64 GPU it finds, so the same command can change cost
class between two hosts — or between two days on one host — without changing
the recorded arguments. Record the class you chose alongside the measurement;
see {doc}`../benchmarks/index`.

## Examples that select their own device

`PYOPENCL_CTX` does not reach every example. `helmholtz2d.py`,
`helmholtz3d.py`, `branched_flow_helmholtz2d.py` and the two
`*_split_p_convergence.py` drivers each carry a `_select_opencl_device` that
enumerates the platforms and builds a `cl.Context` directly, so the variable is
ignored. What they then choose differs:

- The first three, and `helmholtz2d_split_p_convergence.py`, take the `auto`
  path: first fp64-capable GPU, else first fp64-capable CPU. On a host with
  both a CUDA GPU and PoCL they run on the GPU.
- `helmholtz3d_split_p_convergence.py` exposes `--backend` and defaults it by
  mode — `auto` in smoke, **`pocl-cpu` at full settings**. A full run of it
  therefore lands on the CPU on that same host unless you pass
  `--backend cuda-gpu`, which is the opposite of the assumption the other four
  invite.

So the variable is not a device policy for the whole tree, and neither is the
default. Read the device off the run rather than inferring it from the
environment.

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
dominated by sumpy code generation, so a one-shot driver run does not pay for a
GPU unless the compile cache (`XDG_CACHE_HOME`) is already warm. Measure the
candidate classes once with the driver's `--mode smoke` before committing a
campaign to either.
