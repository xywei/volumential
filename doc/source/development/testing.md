# Tests and markers

The suite is deliberately tiered: a pull request should not wait on numerics
that take hours, and an expensive accuracy check should not be silently skipped
just because it is expensive. {doc}`../user-guide/validation_matrix` records
what each tier currently covers; this page is how to run them.

## Tiers

| Tier | Command | Runs in |
| --- | --- | --- |
| Smoke and regression | `uv run pytest -q` | Pull-request CI and `main` |
| Long-run | `uv run pytest --longrun` | Developer or dedicated runs |
| Full accuracy | `uv run pytest -m full_accuracy --full-accuracy` | `CI Full` (weekly and on demand), on a CPU |

There is no benchmark tier any more: the drivers that produced timing, cache
and parameter-sweep evidence are no longer in the tree, and what a measurement
of this library has to record is {doc}`../benchmarks/index`.

The commands above assume `UV_PROJECT_ENVIRONMENT` points at the conda
environment, as in
{doc}`../getting-started/installation`; without it `uv run` uses the
project's own `.venv`, which has no OpenCL runtime.

```bash
uv run pytest -q                     # default suite
uv run pytest --longrun              # include long-running checks
uv run pytest --full-accuracy        # include the high-cost accuracy markers
uv run pytest -m 'not slow'          # skip the wall-clock hogs
```

Targeted checks worth running after any environment change:

```bash
uv run pytest -q test/test_import.py
uv run pytest -q test/test_public_surface.py
uv run pytest -q test/test_duffy_tanh_sinh.py
```

## Markers and options

`test/conftest.py` owns everything the whole suite relies on, and registers its
markers in `pytest_configure` so they work regardless of which configuration
file pytest picks up.

`full_accuracy`
: High-cost derivative and direct-reference accuracy tests, and the volume FMM
  convergence, PDE-residual and split-versus-nonsplit regressions. **Skipped
  unless `--full-accuracy` is passed**, so a plain `pytest` run neither pays
  for them nor pretends to have run them.

`slow`
: Labels, but does not skip, the handful of tests that dominate the wall clock.
  `-m 'not slow'` gives a quick run.

`--longrun`
: Enables the larger table-generation and quadrature cases, through the
  `longrun` fixture.

The `ctx_factory` fixture is re-exported from `pyopencl.tools` and parametrizes
OpenCL-using tests over the available platforms, so set `PYOPENCL_TEST` (see
{doc}`../getting-started/device-selection`) to pin which one. `conftest.py`
also carries the xfail policy for OpenCL platforms known to crash, a
session-scoped `table_2d_order1` near-field table shared by every test that
needs it, and end-of-session cleanup of stray table caches.

## The device of the tests that build their own context

Some tests do not take the `ctx_factory` fixture. The volume FMM regressions
and the full-accuracy sweeps need double precision whatever platform the
fixture was pinned to, and the near-field tables that `test_table_manager.py`
and `test_nearfield_potential_table.py` share are built before any fixture is
parametrized. They get their device from `test/_opencl_test_utils.py`, which
reads `PYOPENCL_CTX`, not `PYOPENCL_TEST`:

- With `PYOPENCL_CTX` set, they run on exactly the device it selects, CPU or
  GPU. The fp64 tests skip only if that device lacks fp64. A selector that
  matches no platform is an error, not a skip.
- Without it, the fp64 tests take the first fp64 GPU, and the first fp64 CPU
  when there is none. This default never picks the `Intel(R) OpenCL`
  platform, which `conftest.py` marks as crashing on these paths; select it
  explicitly to run there anyway. The table builds keep the rule they always
  had: the first device of the first platform other than that one.

So set both variables to the same device, as
{doc}`../getting-started/device-selection` does, and the whole suite runs where
you pointed it.

A CPU runs the whole full-accuracy tier, but in a different cost class from a
GPU. On the PoCL CPU of a `CI Full` runner it took 90 minutes, 57 of them in
the two 3D split-versus-nonsplit tests of `test_volume_fmm.py`.
`VOLUMENTIAL_FULL_ACCURACY_REDUCED=1` runs those two at multipole order 16
instead of 24 and without their source-derivative pair; they then took 14
minutes and the tier 38, and `CI Full` sets it. Each figure is one run, and the
second runner had a faster CPU. The docstring of `_split_3d_full_accuracy_size`
says why neither change loosens the comparison. Leave the variable unset for
the full size, on a GPU or whenever a change touches the 3D split.

```bash
VOLUMENTIAL_FULL_ACCURACY_REDUCED=1 uv run pytest -m full_accuracy --full-accuracy
```

## Configuration

The pytest configuration is `[tool.pytest.ini_options]` in `pyproject.toml`,
and it is the only one. Do not add a `pytest.ini`: pytest reads that file
first, and whenever both exist it ignores the `pyproject.toml` table (the run
header says so, as `WARNING: ignoring pytest config in pyproject.toml!`).

## Examples as tests

Four examples run in a reduced configuration on a pull request **targeting
`main`** — `laplace2d.py`, `laplace2d_adaptive.py`, `helmholtz2d.py` and
`helmholtz3d.py`. On a stacked pull request nothing runs at all; see
{doc}`ci`.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 python examples/laplace2d.py
```

Smoke mode drops the quadrature order, level count and multipole order and uses
a separate table cache file, so it finishes in seconds. An example that stops
working in smoke mode is a broken example.

The other maintained examples — `laplace3d.py`, `poisson3d.py` and
`branched_flow_helmholtz2d.py` — run at full settings in `CI Full`, which has
no `pull_request` trigger. Nothing gates them on a pull request, so run the one
you touched yourself.

Those three smoke examples are now the whole of the `Examples (Smoke)` job. It
used to run five benchmark drivers' smoke modes beside them; the drivers left
the tree, and the job's benchmark lines left with them.
