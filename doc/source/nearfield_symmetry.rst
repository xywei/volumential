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

Scalar full-symmetry kernels use the arithmetic path.

- ``case_orbit_ranks`` is stored as ``uint16[n_cases]``.
- ``case_axis_perm``, ``case_axis_sign``, and ``case_axis_group`` are packed as
  ``uint8``/``int8`` arrays and passed as read-only List 1 kernel inputs.
- The evaluator decodes tensor-product source/target indices, applies the
  per-case signed permutation, resolves zero-axis flips locally, and uses a
  fixed compare-exchange sorting network for equal-absolute-value case axes.
- The table layout is ``(canonical_case_orbit, canonical_source,
  canonical_target)`` with target index fastest. This intentionally stores some
  unused scalar layout rows to remove representative hash lookups and keep GPU
  table accesses arithmetic and coalescing-friendly.

Constrained or sign-changing kernels use the generated descriptor fallback.

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

For scalar 3D Laplace at ``q=3``, the dense online reconstruction maps contain
``101331`` full entries and occupy ``1215972`` bytes as ``int32`` IDs plus
``float64`` scales. Scalar arithmetic reconstruction reduces transferred
metadata to less than ``2000`` bytes in the CPU descriptor test. It uses a
canonical-case table layout rather than the minimal representative-only layout,
so value bytes increase relative to ``2510`` representatives while eliminating
the transform scan and representative hash lookup in the scalar online kernel.

Derivative kernels
------------------

For derivative kernels, symmetry transforms can introduce sign changes.
Volumential stores a per-entry scale map (typically ``+1`` or ``-1``) so that
runtime reconstruction is sign-correct.

This applies to target-derivative kernels represented with
``AxisTargetDerivative`` wrappers. Directional source derivatives are handled
with sign-aware vector transforms when a fixed source direction is provided as
``symmetry_source_direction`` on the table.

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
