# Tests and markers

The suite is deliberately tiered: a pull request should not wait on numerics
that take hours, and an expensive accuracy check should not be silently skipped
just because it is expensive. {doc}`../user-guide/validation_matrix` records
what each tier currently covers; this page is how to run them.

## Tiers

| Tier | Command | Runs in |
| --- | --- | --- |
| Smoke and regression | `uv run --active pytest -q` | Pull-request CI and `main` |
| Long-run | `uv run --active pytest --longrun` | Developer or dedicated runs |
| Full accuracy | `uv run --active pytest -m full_accuracy --full-accuracy` | GPU-capable or dedicated runners |
| Benchmarks | `python benchmarks/<name>.py --mode smoke` | Smoke in CI; full runs promoted manually |

`--active` keeps `uv run` in the environment `uv sync --active` filled — the
conda environment with the OpenCL runtime — instead of falling back to the
project's own `.venv`.

```bash
uv run --active pytest -q                     # default suite
uv run --active pytest --longrun              # include long-running checks
uv run --active pytest --full-accuracy        # include the high-cost accuracy markers
uv run --active pytest -m 'not slow'          # skip the wall-clock hogs
```

Targeted checks worth running after any environment change:

```bash
uv run --active pytest -q test/test_import.py
uv run --active pytest -q test/test_public_surface.py
uv run --active pytest -q test/test_duffy_tanh_sinh.py
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

The maintained examples run in CI in a reduced configuration:

```bash
VOLUMENTIAL_EXAMPLE_SMOKE=1 python examples/laplace2d.py
```

Smoke mode drops the quadrature order, level count and multipole order and uses
a separate table cache file, so it finishes in seconds. An example that stops
working in smoke mode is a broken example, and the job that runs them is a gate
like any other.
