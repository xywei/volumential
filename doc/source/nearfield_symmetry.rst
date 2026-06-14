Near-Field Symmetry Reduction
=============================

Volumential reduces near-field table storage by canonicalizing table entries
under symmetry orbits over ``(source_mode, target_mode, interaction_case)``.

Orbit canonicalization
----------------------

- A full table entry is mapped to a canonical representative entry.
- The dense compatibility path uses an index map from full entry IDs to
  canonical entry IDs.
- Scalar arithmetic ORBIT reconstruction avoids sending that full-entry map to
  the GPU for full-symmetry scalar tables. The List 1 evaluator canonicalizes
  ``(source_mode, target_point, case)`` with packed per-case descriptors and
  computes the target-fastest table row directly.

Online reconstruction modes
---------------------------

Scalar full-symmetry kernels and supported axis-preserving derivative subgroups
use the arithmetic path.

- ``case_orbit_ranks`` is stored as ``uint16[n_cases]``.
- ``case_axis_perm``, ``case_axis_sign``, and ``case_axis_group`` are packed as
  ``uint8``/``int8`` arrays and passed as read-only List 1 kernel inputs.
- Axis source/target derivatives additionally pass ``axis_sign_power`` so the
  evaluator computes derivative signs analytically from the applied signed
  permutation.
- Equal-magnitude directional source derivative subgroups additionally pass the
  source direction signs and one representative sign axis so the evaluator
  applies the collinearity sign analytically.
- The evaluator decodes tensor-product source/target indices, applies the
  per-case signed permutation, resolves zero-axis flips locally, and uses a
  fixed compare-exchange sorting network for equal-absolute-value case axes.
- The table layout is ``(canonical_case_orbit, canonical_source,
  canonical_target)`` with target index fastest. This intentionally stores some
  unused scalar layout rows to remove representative hash lookups and keep GPU
  table accesses arithmetic and coalescing-friendly. Axis derivatives use the
  same target-fastest layout and pre-apply any row-level representative sign
  convention while packing the table values.

Other constrained or sign-changing kernels use the generated descriptor
fallback.

- ``transform_qpoint_map[transform, qpoint]`` and
  ``transform_case_map[transform, case]`` encode the allowed signed
  permutation group for the kernel.
- The evaluator forms all transformed ``(source_mode, target_point, case)``
  candidates with a fixed-size reduction and chooses the smallest
  representative full-entry ID. Ties prefer positive transform signs and then
  transform ID, so the rule is deterministic and warp-uniform.
- ``representative_lookup`` maps representative full-entry IDs to compact table
  rows. This lookup is an open-addressed hash sized during payload preparation;
  the generated kernel uses a fixed probe count from the descriptor.
- ``sign_lookup`` is normally empty. It is a sparse correction table for
  derivative kernels whose stabilizer contains sign-conflicting transforms but
  whose dense signed-union oracle has an existing convention that must be
  preserved for bitwise metadata equivalence.

The dense maps remain the construction-time oracle and fallback. Payload
preparation validates generated representative IDs and generated signs against
``table_entry_ids`` / ``table_entry_scales`` before the generated descriptor is
used.

GPU scheduling
--------------

The List 1 evaluator maps target boxes to a global hardware axis. The target
point loop has a launch-known ``n_q_points`` bound and guards inactive lanes with
``tid < n_box_targets`` so Loopy can legally schedule target work without
data-dependent parallel bounds.

``list1_target_batch_size`` controls optional target-lane batching. The default
is ``1`` because H200 OpenCL benchmarks for 3D Laplace target derivatives showed
one target lane per workgroup faster than a 32-lane target batch for the tested
``q=2`` and ``q=3`` cases. Setting ``list1_target_batch_size=32`` remains useful
for subgroup/warp experiments and keeps output bitwise identical in the covered
benchmarks.

For scalar 3D Laplace at ``q=3``, the dense online reconstruction maps contain
``101331`` full entries and occupy ``1215972`` bytes as ``int32`` IDs plus
``float64`` scales. Scalar arithmetic reconstruction reduces transferred
metadata to ``1532`` bytes in the CPU descriptor test. It uses a
canonical-case table layout rather than the minimal representative-only layout,
so value bytes increase relative to ``2510`` representatives while eliminating
the transform scan and representative hash lookup in the scalar online kernel.

Derivative kernels
------------------

For derivative kernels, symmetry transforms can introduce sign changes.
Axis source/target derivatives, mixed axis derivatives, axis-aligned directional
source derivatives, and equal-magnitude directional source derivatives use
signed arithmetic reconstruction rather than a sparse sign lookup. The runtime
sign is the product of the applied flips on derivative axes with odd derivative
order and, for directional source derivatives, the sign needed to keep the fixed
source direction collinear with ``d`` or ``-d``.

This applies to target- and source-derivative kernels represented with
``AxisTargetDerivative`` and ``AxisSourceDerivative`` wrappers. Directional
source derivatives are handled with sign-aware vector transforms when a fixed
source direction is provided as ``symmetry_source_direction`` on the table.
Axis-aligned directions and non-axis directions whose active components have
equal magnitude use signed arithmetic reconstruction. Other non-axis directional
subgroups keep using the generated descriptor fallback because their allowed
transforms depend on the runtime direction vector geometry.

Odd derivative entries can be invariant under a sign-changing self-transform,
for example when the derivative-axis case offset and both derivative-axis
source/target coordinates are centered. These entries are mathematically zero;
the dense signed-union oracle has an arbitrary sign convention for them. The
signed arithmetic path counts these convention-only conflicts during payload
preparation and keeps a deterministic arithmetic convention without sending a
sparse sign-correction table to the device.

For example, for a directional source derivative kernel:

.. code-block:: python

   table = NearFieldInteractionTable(
       ...,
       sumpy_kernel=DirectionalSourceDerivative(LaplaceKernel(2), "dir_vec"),
       symmetry_source_direction=np.array([1.0, 0.0]),
   )

the orbit reducer applies transforms that keep the direction collinear
(``d`` or ``-d``) and records the corresponding sign in
``canonical_scales`` for runtime reconstruction.

When using the FMM wrangler path, ``symmetry_source_direction`` is inferred from
the active directional source vector in ``source_extra_kwargs`` and applied to
table lookup metadata for that evaluation.

SQLite cache payloads
---------------------

- Schema ``2.1.0`` stores table arrays only in the ``payload`` column.
- Legacy dense blob columns are migrated away in writable mode.
- Symmetry-reduced tables persist sparse payload arrays
  (``reduced_entry_ids`` and ``reduced_data``) and do not persist NaN
  sentinels.
