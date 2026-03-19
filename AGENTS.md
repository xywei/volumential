# Volumential Agent Guide

This repository is an older volume-potential/FMM codebase built on the
PyOpenCL + boxtree + sumpy + pytential stack. Treat it as research software with
useful core ideas, uneven modernization, and a few stale interfaces.

## Project Intent

- `volumential` computes volume potentials in 2D/3D using fast multipole ideas.
- The near-field story is table-driven: singular or nearly singular local
  interactions are precomputed and cached.
- The far-field story is wrangler-driven: boxtree traversals and expansion
  wranglers execute the FMM passes.
- The medium-term modernization goal is to make this package complement
  `pytential` rather than duplicate it. Prefer integrations that make volume and
  boundary workflows compose cleanly.

## Important Files

- `volumential/volume_fmm.py`: top-level volume FMM driver and interpolation of
  computed potentials.
- `volumential/expansion_wrangler_interface.py`: expected wrangler API.
- `volumential/expansion_wrangler_fpnd.py`: concrete wrangler implementations.
- `volumential/nearfield_potential_table.py`: near-field interaction table
  construction and lookup.
- `volumential/table_manager.py`: SQLite-backed table caching and retrieval.
- `volumential/geometry.py`: bounding boxes, tree setup, FMM geometry helpers.
- `volumential/interpolation.py`: meshmode <-> box-grid interpolation helpers.
- `volumential/function_extension.py`: pytential-facing extension logic; likely a
  key integration point for future PDE workflows.
- `volumential/singular_integral_2d.py`: Duffy/radial singular quadrature
  kernels used by the active near-field table builder.
- `test/`: best source of expected behavior for maintenance.
- `examples/`: end-to-end usage patterns for Laplace volume potentials.

## What To Read First

When starting non-trivial work, read these in roughly this order:

1. `README.md`
2. `volumential/__init__.py`
3. `volumential/volume_fmm.py`
4. `volumential/table_manager.py`
5. `volumential/nearfield_potential_table.py`
6. the most relevant test in `test/`

For pytential integration work, also read:

1. `volumential/interpolation.py`
2. `volumential/function_extension.py`
3. `test/test_interpolation.py`
4. `test/test_function_extension.py`

## Current Reality / Known Debt

- Packaging is old: `setup.py` only, no `pyproject.toml`, stale Python metadata.
- The code assumes old inducer-stack APIs in places, especially around
  `pytential`, `meshmode`, `arraycontext`, and `pyopencl`.
- Some tests and modules already document drift. In particular,
  `test/test_function_extension.py` notes that it needs updating for
  `GeometryCollection`.
- There are explicit `FIXME`s and partial implementations, especially around
  multiple source fields, 3D paths, and optional build methods.
- Interactive OpenCL context creation appears in a few places; avoid adding more
  of that during modernization.
- Historical contrib/native code has been removed; use the in-tree Python/OpenCL
  implementations as the supported path.
- `volumential/meshgen.py` depends on legacy boxtree behavior that has drifted
  out of current upstream. In particular, the old
  `boxtree.tree_interactive_build` fallback no longer exists in current
  `inducer/boxtree`.
- There is a historical fork at `xywei/boxtree`, but it appears stale and should
  not be treated as an actively maintained compatibility target.

## Preferred Modernization Direction

- Favor a small, reliable public API over preserving every legacy entry point.
- Keep the separation between near-field tables, geometry/traversal setup, and
  FMM execution explicit.
- Prefer `arraycontext`-friendly and pytential-friendly interfaces over raw,
  ad hoc array handling.
- When integrating with `pytential`, aim for workflows where boundary solves can
  feed volume evaluation without bespoke glue in user code.
- Add or revive tests before large refactors in the integration layers.

## Working Conventions

- Preserve numerical behavior unless the task is explicitly to change it.
- Be careful with cached-table semantics in `table_manager.py`; changes to table
  format, kernel definitions, or scaling rules may invalidate old SQLite caches.
- When editing public behavior, update or add focused tests in `test/`.
- Prefer extending current examples or adding small new ones over writing prose
  docs first.
- Avoid broad rewrites unless the user asks for them. Incremental stabilization
  is safer here.

## Testing Guidance

- Start with the narrowest relevant test file under `test/`.
- Good anchors:
  - `test/test_volume_fmm.py`
  - `test/test_table_manager.py`
  - `test/test_nearfield_potential_table.py`
  - `test/test_duffy_tanh_sinh.py`
  - `test/test_nearfield_interaction_completeness.py`
  - `test/test_interpolation.py`
- Expect GPU/OpenCL-dependent tests. If a test requires unavailable hardware or
  drivers, say so clearly instead of papering over it.
- If touching pytential integration, check for API drift first before assuming a
  numerical bug.

## When Unsure

- Use the tests and examples as the behavioral spec.
- Prefer the pure Python/OpenCL path over reviving old contrib code.
- If a design choice affects how `volumential` should complement `pytential`,
  choose the option that reduces duplicate abstractions and improves composable
  PDE workflows.
