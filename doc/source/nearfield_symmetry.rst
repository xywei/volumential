Near-Field Symmetry Reduction
=============================

Volumential reduces near-field table storage by canonicalizing table entries
under symmetry orbits over ``(source_mode, target_mode, interaction_case)``.

Orbit canonicalization
----------------------

- A full table entry is mapped to a canonical representative entry.
- The dense compatibility path uses an index map from full entry IDs to
  canonical entry IDs.
- Generated ORBIT reconstruction avoids sending that full-entry map to the GPU
  for symmetry-reduced tables. The List 1 evaluator instead scans the compact
  signed-permutation descriptor, computes the representative full-entry ID in
  registers, then resolves it through a small representative hash table.

Generated online reconstruction
-------------------------------

The generated path is the intended online representation for
symmetry-reduced tables.

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
``float64`` scales. Generated reconstruction keeps the representative values at
``2510 * 8`` bytes and reduces reconstruction metadata to less than ``100000``
bytes in the CPU descriptor test.

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
