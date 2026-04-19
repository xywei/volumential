M1 Kernel Coverage
==================

This page summarizes the first production kernel milestone
(`issue #63 <https://github.com/xywei/volumential/issues/63>`_):
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

Periodic Prototype
------------------

An initial Barnett-style periodization hook is available in
``drive_volume_fmm`` (sumpy backend, 2D/3D):

- Set ``periodic=True`` to enable periodic mode explicitly.
- Passing ``periodic_*`` modifier kwargs without ``periodic=True`` raises a
  ``ValueError``.
- If ``periodic=True`` and no explicit near/far controls are provided,
  defaults are ``periodic_near_shifts="nearest"`` and
  ``periodic_far_operator="auto"``.
- ``periodic_near_shifts`` adds explicit near-image lattice copies
  (``"nearest"`` for the nearest image ring).
- ``periodic_near_target_boxes`` optionally restricts near-image evaluation
  to a subset of target boxes.
- ``periodic_far_operator`` accepts a precomputed root-level far operator
  ``T_per`` that maps periodic far-field features to root local coefficients.
  Passing ``"auto"`` selects an automatic periodic far strategy.
- In 2D/3D Laplace mode, ``"auto"`` uses a strict spectral periodic runtime
  solve (reciprocal-space model) instead of fitting a root-local ``T_per``.
- For non-Laplace auto paths, ``"auto"`` builds/loads ``T_per`` offline.
- ``periodic_far_operator_basis`` optionally pins the operator basis
  (``"multipole"`` or ``"source"``). Laplace auto strict mode accepts
  ``None``/``"source"`` only.
- ``periodic_far_spectral_kmax_2d`` controls Fourier truncation in 2D auto
  spectral mode.
- ``periodic_far_spectral_kmax_3d`` controls Fourier truncation in 3D auto
  spectral mode.
- ``periodic_cell_size`` sets the periodic cell lengths (defaults to
  ``tree.root_extent`` in each dimension).
- ``periodic_far_operator_manager`` or
  ``periodic_far_operator_cache_filename`` can reuse the same SQLite cache
  backend used for near-field tables.

This interface is intended for correctness prototyping and validation against
trusted periodic references (Ewald/spectral-Ewald style) while the full
periodic workflow in `issue #67 <https://github.com/xywei/volumential/issues/67>`_
is completed.

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
separately in `issue #71 <https://github.com/xywei/volumential/issues/71>`_.
