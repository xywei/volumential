Near-Field Symmetry Reduction
=============================

Volumential reduces near-field table storage by canonicalizing table entries
under symmetry orbits over ``(source_mode, target_mode, interaction_case)``.

Orbit canonicalization
----------------------

- A full table entry is mapped to a canonical representative entry.
- Runtime list1 evaluation uses an index map from full entry IDs to canonical
  entry IDs.
- This keeps table lookup O(1) while avoiding duplicated near-field entries.

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
