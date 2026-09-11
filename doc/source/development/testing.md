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
| Full accuracy | `uv run pytest -m full_accuracy --full-accuracy` | GPU-capable or dedicated runners |
| Benchmarks | `python benchmarks/<name>.py --mode smoke` | Five drivers' smoke modes on a `main`-targeting pull request; full runs promoted manually |

These assume `UV_PROJECT_ENVIRONMENT` points at the conda environment, as in
{doc}`../getting-started/installation`; without it `uv run` uses the project's
own `.venv`, which has no OpenCL runtime.

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
markers in `pytest_configure` so they work regardless of which ini file pytest
picks up.

`full_accuracy`
: High-cost derivative and direct-reference accuracy tests. **Skipped unless
  `--full-accuracy` is passed**, so a plain `pytest` run neither pays for them
  nor pretends to have run them.

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

## Configuration

`pytest.ini` currently wins over `[tool.pytest.ini_options]` in
`pyproject.toml`. Both exist, and until the duplication is resolved, edit
`pytest.ini`.

## Examples as tests

Three examples run in a reduced configuration on a pull request **targeting
`main`** — `laplace2d.py`, `helmholtz2d.py` and `helmholtz3d.py`. On a stacked
pull request nothing runs at all; see {doc}`ci`.

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 python examples/laplace2d.py
```

Smoke mode drops the quadrature order, level count and multipole order and uses
a separate table cache file, so it finishes in seconds. An example that stops
working in smoke mode is a broken example.

The other maintained examples — `laplace3d.py`, `poisson3d.py`,
`branched_flow_helmholtz2d.py` and the two `*_split_p_convergence.py` drivers —
run at full settings in `CI Full`, which has no `pull_request` trigger. Nothing
gates them on a pull request, so run the one you touched yourself.

The pull-request job that runs those three examples also runs five benchmark
drivers in `--mode smoke`:
`table_equivalence_cache.py`, `accuracy_preservation.py`,
`gaussian_free_space.py`, `dmk_effective_density.py` and `adaptive_timing.py`.
The `performance_suite.py` lines beside them are `--list-cases` and
`--dry-run`, which execute no case. Every other benchmark driver's smoke mode
is ungated — run it yourself when you touch one.
