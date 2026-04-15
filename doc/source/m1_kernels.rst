M1 Kernel Coverage
==================

This page summarizes the first production kernel milestone (issue #63):
Laplace, Helmholtz, and Yukawa for free-space workloads.

Capability Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 14 14

   * - Kernel
     - Potential
     - Target gradient
     - Particle->particle
     - Volume->target
   * - Laplace
     - supported
     - supported
     - supported
     - supported
   * - Helmholtz
     - supported
     - supported
     - supported
     - supported
   * - Yukawa (Modified Helmholtz)
     - supported
     - supported
     - supported
     - supported

Execution Notes
---------------

- Free-space paths are supported for this M1 scope.
- Helmholtz supports real and complex wave numbers.
- Yukawa split mode requires real ``lam``; for mixed-complex wave numbers,
  use Helmholtz with ``k = i*lam``.

Validation in CI
----------------

The M1 kernels are covered by regression tests for:

- direct-reference accuracy checks,
- convergence or refinement behavior in representative regimes,
- PDE residual sanity checks away from singularities,
- near-field table build paths for potential and gradient kernels.

Related Follow-up
-----------------

Source-derivative and mixed derivative support matrix completion is tracked
separately in issue #71.
