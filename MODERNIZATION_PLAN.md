# Volumential Modernization Plan

This plan reflects the current state of `volumential` as of March 2026 after a
first compatibility sweep against current upstream inducer-stack packages.

## Goal

Revive `volumential` so it complements `pytential` for PDE workflows instead of
acting as an isolated legacy codebase. The package should support maintainable,
testable volume-potential workflows that compose with modern boundary-integral
infrastructure.

## Current Situation

- Basic import can work, but only after papering over legacy assumptions in the
  environment.
- The main breakages are ecosystem drift, not one isolated bug.
- The package currently depends on a mix of removed APIs, stale tests, and old
  build/runtime assumptions.

## Confirmed Breakage Areas

### 1. Bootstrap and Packaging

- `volumential/version.py` expects generated `volumential/_git_rev.py` even in a
  source-tree test run.
- `setup.py`-only packaging is brittle for modern editable/dev workflows.
- Local default environments may be missing required compiled/runtime deps such
  as `h5py` and OpenCL stack pieces.

### 2. SciPy Quadrature API Drift

- `volumential/singular_integral_2d.py` imports
  `scipy.integrate.quadrature`, which is no longer available in current SciPy.
- Several tests in `test/test_singular_integral_2d.py` fail because the old
  quadrature calling convention no longer matches current SciPy integrators.

### 3. Test Harness Drift

- `test/conftest.py` assumes the `worker_id` fixture is always present.
- That now fails unless `pytest-xdist` is installed, even for ordinary single-
  process test runs.

### 4. Meshmode / Arraycontext Drift

- `test/test_interpolation.py` imports `flatten` from
  `meshmode.dof_array`, which no longer matches current meshmode.
- This likely indicates a larger interpolation-layer API update is needed in
  `volumential/interpolation.py`, not just a one-line test fix.

### 5. Mesh Generation / Boxtree Drift

- `volumential/meshgen.py` now uses the boxtree-based provider as the supported
  path.
- Keep this aligned with upstream `inducer/boxtree` tree-of-boxes semantics.
- Treat the old `xywei/boxtree` fork as historical reference only, not as the
  modernization target.

### 6. Pytential Integration Drift

- `volumential/function_extension.py` still uses deprecated interfaces such as
  `pytential.solve.gmres`.
- `test/test_function_extension.py` is already marked stale and says it needs to
  be updated for `GeometryCollection`.

## Strategic Direction

- Target current upstream inducer packages, not the old `xywei/boxtree` fork.
- Remove legacy compatibility assumptions instead of layering more shims than
  necessary.
- Recover a minimal reliable core first: imports, quadrature, mesh generation,
  interpolation, then deeper FMM integration.
- Keep the modernization oriented toward `pytential` interoperability.

## Phased Plan

## Phase 0: Make the Project Testable Again

- Make `volumential/version.py` tolerant of missing `_git_rev.py`.
- Modernize packaging enough to support reproducible editable installs.
- Fix `test/conftest.py` so `worker_id` is optional.
- Document a supported development environment for local and remote runs.

Success criteria:

- `pytest -q test/test_import.py`
- `pytest -q test/test_singular_integral_2d.py`
- `pytest -q test/test_table_manager.py`

collect and run in a modern env without ad hoc monkeypatching.

## Phase 1: Repair Numerical Foundation Code

- Replace the old SciPy quadrature dependency in
  `volumential/singular_integral_2d.py` with a supported integration path.
- Revalidate 2D singular quadrature behavior against existing tests.
- Check whether any table-building logic depends on quadrature subtleties that
  changed with the replacement.

Success criteria:

- `test/test_singular_integral_2d.py` passes.
- table-building tests no longer fail for quadrature reasons.

## Phase 2: Stabilize Boxtree Meshgen Path

- Keep `volumential/meshgen.py` focused on upstream boxtree support.
- Remove assumptions that require private forks.
- Treat `xywei/boxtree` as reference material only if needed to understand the
  old intent.

Success criteria:

- `test/test_volume_fmm.py` can collect.
- basic geometry/tree construction works without a private fork.

## Phase 3: Modernize Interpolation Layer

- Update `volumential/interpolation.py` and related tests for current
  `meshmode`/`arraycontext` APIs.
- Rework old DOF flattening/thawing assumptions.
- Keep the public interpolation story aligned with meshmode's current container
  conventions.

Success criteria:

- `test/test_interpolation.py` passes or has a small, explicit remaining skip set.

## Phase 4: Refresh Pytential-Facing APIs

- Update `volumential/function_extension.py` to modern pytential interfaces.
- Replace deprecated solver imports and adapt to `GeometryCollection`.
- Re-evaluate the intended user-facing workflow for boundary-to-volume coupling.

Success criteria:

- `test/test_function_extension.py` is unskipped or replaced by modernized tests.

## Phase 5: Rebuild the Volume-FMM Story

- Revisit `volumential/volume_fmm.py`, wranglers, and geometry integration after
  the bootstrap layers are fixed.
- Decide what subset of existing functionality is still core and what should be
  retired.
- Add small end-to-end tests that reflect the intended future `pytential`-
  complementary workflow.

Success criteria:

- A documented, supported path exists from boundary/mesh input to volume
  evaluation using current upstream dependencies.

## Priority Order for Immediate Work

1. `volumential/version.py`
2. `test/conftest.py`
3. `volumential/singular_integral_2d.py`
4. `volumential/meshgen.py`
5. `volumential/interpolation.py`
6. `volumential/function_extension.py`
7. `volumential/volume_fmm.py`

## Non-Goals for the First Revival Pass

- Do not try to preserve compatibility with every historical environment.
- Do not resurrect the old `xywei/boxtree` fork as a required dependency.
- Do not start with broad performance tuning before basic correctness and test
  coverage are back.
- Do not reintroduce removed contrib/native code unless the modern pure-Python
  path is clearly insufficient.
