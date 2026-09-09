__copyright__ = "Copyright (C) 2017 - 2018 Xiaoyu Wei"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """Generated (search-derived) orbit reconstruction maps.

Fallback reconstruction for tables whose symmetry group is not one of the
closed-form arithmetic cases: the orbit transforms are discovered once and
then encoded into lookup tables.
"""

import numpy as np

from volumential.wranglers.orbit_lookup import (
    _build_open_addressed_int_lookup,
    _build_sparse_sign_lookup,
)


def _evaluate_generated_orbit_reconstruction(
    transform_qpoint_map,
    transform_case_map,
    transform_signs,
    *,
    n_cases,
    n_q_points,
):
    transform_qpoint_map = np.asarray(transform_qpoint_map, dtype=np.int64)
    transform_case_map = np.asarray(transform_case_map, dtype=np.int64)
    transform_signs = np.asarray(transform_signs, dtype=np.int8)

    if transform_qpoint_map.ndim != 2:
        raise ValueError("transform_qpoint_map must be two-dimensional")
    if transform_case_map.ndim != 2:
        raise ValueError("transform_case_map must be two-dimensional")
    if transform_qpoint_map.shape[0] != transform_case_map.shape[0]:
        raise ValueError("transform maps disagree on transform count")
    if transform_signs.shape != (transform_qpoint_map.shape[0],):
        raise ValueError("transform_signs disagrees with transform maps")

    n_transforms = int(transform_qpoint_map.shape[0])
    n_pairs = int(n_q_points) * int(n_q_points)
    n_entries = int(n_cases) * n_pairs
    transform_ids = np.arange(n_transforms, dtype=np.int64)
    sign_tie_breakers = np.where(transform_signs > 0, 0, 1).astype(np.int64)

    representative_entry_ids = np.empty(n_entries, dtype=np.int64)
    representative_scales = np.empty(n_entries, dtype=np.int8)

    for case_id in range(int(n_cases)):
        transformed_cases = transform_case_map[:, case_id]
        for source_mode_id in range(int(n_q_points)):
            transformed_sources = transform_qpoint_map[:, source_mode_id]
            row = case_id * n_pairs + source_mode_id * int(n_q_points)
            for target_point_id in range(int(n_q_points)):
                transformed_targets = transform_qpoint_map[:, target_point_id]
                candidates = (
                    transformed_cases * n_pairs
                    + transformed_sources * int(n_q_points)
                    + transformed_targets
                )
                # Prefer positive stabilizer signs, then transform id. This matches
                # the dense oracle's convention that a representative scales itself
                # by +1 even if an odd stabilizer transform also fixes the entry.
                best_key = int(
                    np.min(
                        candidates * (2 * n_transforms)
                        + sign_tie_breakers * n_transforms
                        + transform_ids
                    )
                )
                best_transform = best_key % n_transforms
                entry_id = row + target_point_id
                representative_entry_ids[entry_id] = best_key // (2 * n_transforms)
                representative_scales[entry_id] = transform_signs[best_transform]

    return representative_entry_ids, representative_scales


def _build_generated_orbit_reconstruction(table, table_entry_ids, table_entry_scales):
    if not hasattr(table, "_get_orbit_reconstruction_maps"):
        return None
    if not bool(getattr(table, "table_data_is_symmetry_reduced", False)):
        return None

    orbit_info = table._get_orbit_canonical_info()
    representative_entry_ids = np.asarray(orbit_info["entry_ids"], dtype=np.int64)
    if len(representative_entry_ids) == 0:
        return None

    reconstruction_maps = table._get_orbit_reconstruction_maps()
    transform_qpoint_map = np.asarray(
        reconstruction_maps["transform_qpoint_map"], dtype=np.int32
    )
    transform_case_map = np.asarray(
        reconstruction_maps["transform_case_map"], dtype=np.int32
    )
    transform_signs = np.asarray(reconstruction_maps["transform_signs"], dtype=np.int8)

    generated_entry_ids, generated_scales = _evaluate_generated_orbit_reconstruction(
        transform_qpoint_map,
        transform_case_map,
        transform_signs,
        n_cases=table.n_cases,
        n_q_points=table.n_q_points,
    )

    table_entry_ids = np.asarray(table_entry_ids, dtype=np.int32)
    table_entry_scales = np.asarray(table_entry_scales)
    expected_entry_ids = representative_entry_ids[table_entry_ids]
    expected_scales = np.real(table_entry_scales).astype(np.int8)

    if not np.array_equal(generated_entry_ids, expected_entry_ids):
        raise RuntimeError("generated ORBIT representative ids disagree with dense map")

    sign_correction_entry_ids = np.flatnonzero(generated_scales != expected_scales)
    sign_corrections = (
        expected_scales[sign_correction_entry_ids]
        // generated_scales[sign_correction_entry_ids]
    ).astype(np.int8)

    lookup_keys, lookup_values, max_probe_count = _build_open_addressed_int_lookup(
        representative_entry_ids
    )
    sign_lookup_keys, sign_lookup_values, sign_lookup_max_probe_count = (
        _build_sparse_sign_lookup(sign_correction_entry_ids, sign_corrections)
    )

    metadata_arrays = (
        transform_qpoint_map,
        transform_case_map,
        transform_signs,
        lookup_keys,
        lookup_values,
        sign_lookup_keys,
        sign_lookup_values,
    )

    return {
        "kind": "generated-orbit",
        "transform_qpoint_map": np.ascontiguousarray(transform_qpoint_map),
        "transform_case_map": np.ascontiguousarray(transform_case_map),
        "transform_signs": np.ascontiguousarray(transform_signs),
        "lookup_keys": lookup_keys,
        "lookup_values": lookup_values,
        "lookup_max_probe_count": int(max_probe_count),
        "sign_lookup_keys": sign_lookup_keys,
        "sign_lookup_values": sign_lookup_values,
        "sign_lookup_max_probe_count": int(sign_lookup_max_probe_count),
        "sign_correction_count": int(len(sign_correction_entry_ids)),
        "metadata_bytes": int(sum(arr.nbytes for arr in metadata_arrays)),
        "representative_entry_ids": representative_entry_ids,
    }

# vim: filetype=pyopencl:foldmethod=marker
