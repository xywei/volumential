Derivative Support
==================

Volumential uses derivative wrappers from :mod:`sumpy.kernel` in two places:
near-field table construction and end-to-end FMM evaluation. This page records
the supported combinations and the expected failure policy so derivative support
does not have to be inferred from individual tests.

Wrapper Types
-------------

.. list-table:: Derivative wrapper roles
   :header-rows: 1

   * - Wrapper
     - Role
     - Runtime data
     - Notes
   * - ``AxisTargetDerivative``
     - Cartesian derivative with respect to the target coordinate.
     - Axis index.
     - Used by table-manager named kernels such as ``Laplace-Dx`` and by FMM
       target-kernel wrappers.
   * - ``AxisSourceDerivative``
     - Cartesian derivative with respect to the source coordinate.
     - Axis index.
     - Supported by the split-wrapper machinery where the backend accepts
       source-derivative kernels.
   * - ``DirectionalSourceDerivative``
     - Source derivative projected onto a user-provided direction vector.
     - Direction argument name and runtime vector.
     - Symmetry-reduced near-field tables require a fixed
       ``symmetry_source_direction`` when canonicalizing entries.

Current Support Matrix
----------------------

.. list-table:: Derivative support by path
   :header-rows: 1

   * - Path
     - Kernels
     - Target derivative
     - Source derivative
     - Mixed source/target derivative
     - Validation status
   * - Near-field table generation
     - Laplace 2D/3D
     - Supported for axis derivatives.
     - Supported for axis derivatives and fixed directional source derivatives.
     - Supported when the represented derivative kernel can be constructed and
       the symmetry action has sign/component metadata.
     - Regression tests cover orbit reconstruction, sign metadata, mixed
       full/reduced-table guards, and derivative payload reconstruction.
   * - Near-field table generation
     - Helmholtz/Yukawa split basis
     - Supported through wrapped split kernels where split metadata records the
       derivative order.
     - Supported for source wrappers accepted by the split wrapper chain;
       directional source paths require runtime direction data.
     - Supported only for wrapper combinations accepted by the split table
       builder; unsupported full/reduced or base-table combinations fail
       explicitly.
     - Full-accuracy tests cover split/non-split tracking and selected
       directional source derivative paths.
   * - Volume FMM wrangler
     - Laplace/Helmholtz/Yukawa free-space paths
     - Supported through target kernel wrappers where the backend accepts the
       derivative request.
     - Supported for axis and directional source derivative wrappers in the
       maintained split paths.
     - Mixed source/target wrapper chains are not documented as supported in
       the end-to-end FMM path unless a dedicated test covers that exact chain.
     - Regression and full-accuracy tests cover representative scalar,
       target-derivative, source-derivative, and split directional paths.
   * - Table-manager named-kernel API
     - Laplace and selected Yukawa aliases
     - Supported for documented named derivative aliases.
     - Not all source-derivative wrappers are exposed as named aliases.
     - Prefer explicit :mod:`sumpy.kernel` wrappers for mixed derivatives.
     - Table-manager tests cover parameter validation, cache-key behavior, and
       unsupported alias errors.

Failure Policy
--------------

Unsupported derivative combinations should fail before expensive table/FMM work
starts. In particular:

- missing Helmholtz/Yukawa parameters must raise ``TypeError`` or ``KeyError``
  with the missing parameter name;
- unsupported split/base-table combinations must raise ``RuntimeError`` rather
  than mixing incompatible reduced and full tables;
- nested mixed source/target derivative wrappers should be treated as
  unsupported unless the specific table/FMM path has an explicit regression
  test and scaling rule;
- directional source derivatives must provide the named runtime direction vector
  for FMM evaluation and a compatible ``symmetry_source_direction`` for
  symmetry-reduced table construction;
- public named-kernel aliases should not imply FMM support when the backend path
  does not accept the corresponding derivative wrapper.

Validation Commands
-------------------

Fast pull-request CI runs the derivative regression tests that are small enough
for routine execution. Higher-cost checks are marker-gated:

.. code-block:: bash

   pytest -m full_accuracy --full-accuracy test/test_duffy_full_accuracy.py test/test_volume_fmm.py

The scheduled/manual ``CI Full`` workflow collects these tests on GitHub-hosted
CPU runners to detect marker drift and import errors. Numerical execution of
GPU-required full-accuracy cases still requires a GPU-capable developer or
dedicated runner environment; see :doc:`validation_matrix` for the broader
validation partitioning.
