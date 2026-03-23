# Volumential Agent Guide

`volumential` is a research-oriented volume-potential/FMM package built around
PyOpenCL + boxtree + sumpy + pytential. Prefer incremental stabilization over
large rewrites.

## Core Mental Model

- Near field: table-driven local interactions (`nearfield_potential_table.py`).
- Far field: traversal + wrangler FMM passes (`volume_fmm.py`,
  `expansion_wrangler_fpnd.py`).
- Caching matters: table/cache key changes can invalidate stored kernels/tables.

## Files To Know First

1. `README.md`
2. `volumential/volume_fmm.py`
3. `volumential/table_manager.py`
4. `volumential/nearfield_potential_table.py`
5. Most relevant test in `test/`

For pytential integration work, also read:

1. `volumential/interpolation.py`
2. `volumential/function_extension.py`
3. `test/test_interpolation.py`
4. `test/test_function_extension.py`

## Current Constraints

- Packaging is `pyproject.toml` + uv-first (`uv sync` / `uv run`).
- Most legacy API assumptions have been removed; remaining risk is integration
  edge cases across pyopencl/meshmode/pytential boundaries.
- `meshgen.py` uses in-tree compatibility code (`tree_interactive_build.py`)
  layered on upstream boxtree primitives (not an external legacy fork).
- OpenCL availability is environment-dependent; avoid adding more interactive
  context creation paths.

## Working Conventions

- Preserve numerical behavior unless explicitly asked to change it.
- Keep near-field, traversal setup, and FMM execution concerns separated.
- Update/add focused tests with behavior changes.
- Prefer small, reviewable edits over broad refactors.

## Testing Heuristics

- Start with the narrowest relevant tests:
  - `test/test_volume_fmm.py`
  - `test/test_table_manager.py`
  - `test/test_nearfield_potential_table.py`
  - `test/test_interpolation.py`
- If GPU/OpenCL tests cannot run, report that clearly and still run available
  CPU/unit coverage.

## Directional Preference

When design is ambiguous, choose the option that keeps `volumential`
composable with `pytential` and reduces duplicate abstractions.
