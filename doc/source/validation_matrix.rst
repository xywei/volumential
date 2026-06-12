Validation Matrix
=================

Volumential keeps the fast pull-request test suite separate from high-cost
accuracy checks. This page records the current validation coverage so kernel,
mode, and backend gaps are visible instead of being inferred from individual
test names.

Test Tiers
----------

.. list-table:: Validation tiers
   :header-rows: 1

   * - Tier
     - How to run
     - Routine execution
     - Purpose
   * - Smoke and regression tests
     - ``pytest``
     - Pull-request CI and ``main`` pushes
     - Import checks, cache/schema regressions, table-manager behavior,
       symmetry metadata, selected FMM paths, and reduced example runs.
   * - Long-run tests
     - ``pytest --longrun``
     - Developer or dedicated experiment runs
     - Larger table-generation and quadrature cases that are too expensive for
       every pull request.
   * - Full-accuracy tests
     - ``pytest -m full_accuracy --full-accuracy``
     - Scheduled/manual ``CI Full`` workflow
     - High-cost direct-reference and split/non-split accuracy checks.
   * - Benchmarks
     - ``python benchmarks/<name>.py --mode smoke`` or full benchmark commands
     - Smoke variants in pull-request CI; full runs are promoted manually
     - Paper-facing timing, cache, and parameter-sweep evidence.

Current Capability Matrix
-------------------------

.. list-table:: Kernel and mode validation status
   :header-rows: 1

   * - Area
     - Dimensions
     - Modes
     - Numerics
     - Current coverage
     - Remaining gap
   * - Laplace free-space near-field tables
     - 2D, 3D
     - Volume-source to target nodes; scalar and derivative table entries
     - Real scalar, derivative signs, symmetry-reduced payloads
     - Smoke/regression tests cover table construction, cache schema,
       orbit reconstruction, source/target derivative signs, and direct
       per-level versus rescaled table comparisons.
     - Broader nonuniform stress and memory-scaling tests remain performance
       work rather than correctness coverage.
   * - Helmholtz split reuse
     - 2D, 3D where supported
     - Volume FMM split and non-split comparisons; directional source paths
     - Complex-valued scalar outputs and derivative paths
     - ``full_accuracy`` tests compare split outputs against non-split or
       higher-order references across parameter ladders.
     - Capability reporting is still per-test; unsupported mixed/backend
       combinations should keep raising explicit errors.
   * - Yukawa split reuse
     - 2D, 3D where supported
     - Volume FMM split and non-split comparisons; directional source paths
     - Real scalar outputs and derivative paths
     - ``full_accuracy`` tests cover split/non-split tracking and directional
       source derivative behavior.
     - Same backend/mixed-combination policy audit as Helmholtz.
   * - Cahn-Hilliard and other legacy kernels
     - As implemented by existing specialized tests
     - Focused kernel-specific checks
     - Real scalar paths
     - Existing regression tests preserve current behavior.
     - Not yet represented in a unified cross-kernel validation matrix.
   * - Matrix/vector PDE kernel families
     - Planned
     - Planned vector/tensor modes
     - Planned matrix-valued outputs
     - Tracked by kernel roadmap issues.
     - Not part of the current M1 validation closure.
   * - Periodic or hybrid boundary paths
     - Planned or experimental
     - Periodic-tail and hybrid validation workflows
     - Kernel-dependent
     - Tracked separately from free-space validation.
     - Free-space and periodic rows are not yet a single complete matrix.

CI Partitioning
---------------

Pull-request CI runs smoke and regression coverage that should stay bounded in
runtime. The scheduled/manual ``CI Full`` workflow runs the documentation and
example jobs and now includes the ``full_accuracy`` pytest marker explicitly.
This keeps expensive direct-reference accuracy checks visible in GitHub Actions
without making every pull request wait for the full matrix.

When adding a new kernel or derivative mode, update this page in the same pull
request that adds tests. A feature should not be marked as fully covered unless
there is at least one CI-friendly regression test and, for high-order numerical
claims, either a ``full_accuracy`` test or a documented benchmark/provenance
path.
