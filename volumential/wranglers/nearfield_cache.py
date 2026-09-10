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

__doc__ = """Device-resident near-field table payload cache.

Both wrangler backends upload the same combined table arrays to the device
and keep a small LRU of them keyed by table identity; this mixin owns that
upload-and-cache step.
"""

import numpy as np

import pyopencl as cl
import pyopencl.array

from volumential.wranglers.arithmetic_orbits import (
    _ARITHMETIC_RECONSTRUCTION_KINDS,
)
from volumential.wranglers.table_data import _prepare_table_data_and_entry_map


class NearFieldPayloadCacheMixin:
    """Uploads and caches combined near-field table data on the device.

    Expects the host class to provide ``_nearfield_device_payload_cache`` (an
    :class:`collections.OrderedDict`) and
    ``_nearfield_device_payload_cache_max``.
    """

    def _get_cached_nearfield_payload(
        self,
        cache_key,
        queue,
        table0,
        near_field_tables,
        eval_dtype,
    ):
        payload = self._nearfield_device_payload_cache.get(cache_key)
        if payload is not None:
            self._nearfield_device_payload_cache.move_to_end(cache_key)
            return payload

        distinct_numbers = set()
        for vec in table0.interaction_case_vecs:
            for cvc in vec:
                distinct_numbers.add(cvc)
        base = len(range(min(distinct_numbers), max(distinct_numbers) + 1))
        shift = -min(distinct_numbers)

        (
            table_data_combined,
            mode_nmlz_combined,
            exterior_mode_nmlz_combined,
            table_entry_ids,
            table_entry_scales,
            reconstruction_info,
        ) = _prepare_table_data_and_entry_map(near_field_tables)

        case_indices_dev = cl.array.to_device(queue, table0.case_indices)
        reconstruction_kind = reconstruction_info["kind"]
        dense_id_bytes = int(table_entry_ids.nbytes)
        dense_scale_bytes = int(table_entry_scales.size * np.dtype(eval_dtype).itemsize)

        if table_data_combined.dtype != eval_dtype:
            table_data_combined = table_data_combined.astype(eval_dtype)
        if mode_nmlz_combined.dtype != eval_dtype:
            mode_nmlz_combined = mode_nmlz_combined.astype(eval_dtype)
        if exterior_mode_nmlz_combined.dtype != eval_dtype:
            exterior_mode_nmlz_combined = exterior_mode_nmlz_combined.astype(eval_dtype)

        if reconstruction_kind == "dense":
            symmetry_maps = table0._get_online_symmetry_maps()
            mode_qpoint_map_dev = cl.array.to_device(
                queue, symmetry_maps["mode_qpoint_map"]
            )
            mode_case_map_dev = cl.array.to_device(queue, symmetry_maps["mode_case_map"])
            mode_case_scale = symmetry_maps.get("mode_case_scale")
            if mode_case_scale is None:
                mode_case_scale = np.ones(
                    (table0.n_q_points, table0.n_cases),
                    dtype=np.int8,
                )
            if table_entry_scales.dtype != eval_dtype:
                table_entry_scales = table_entry_scales.astype(eval_dtype)
            if mode_case_scale.dtype != eval_dtype:
                mode_case_scale = mode_case_scale.astype(eval_dtype)

        table_data_shapes = {
            "n_tables": len(near_field_tables),
            "quad_order": table0.quad_order,
            "n_q_points": table0.n_q_points,
            "n_cases": table0.n_cases,
            "n_table_entries": table_data_combined.shape[1],
            "reconstruction_kind": reconstruction_kind,
        }
        if reconstruction_kind in _ARITHMETIC_RECONSTRUCTION_KINDS:
            table_data_shapes.update(
                {
                    "n_arithmetic_case_orbits": int(
                        reconstruction_info["n_case_orbits"]
                    ),
                }
            )
        elif reconstruction_kind == "generated-orbit":
            table_data_shapes.update(
                {
                    "n_reconstruction_transforms": reconstruction_info[
                        "transform_signs"
                    ].shape[0],
                    "n_reconstruction_lookup_entries": reconstruction_info[
                        "lookup_keys"
                    ].shape[0],
                    "n_reconstruction_lookup_probes": int(
                        reconstruction_info["lookup_max_probe_count"]
                    ),
                    "n_reconstruction_sign_lookup_entries": reconstruction_info[
                        "sign_lookup_keys"
                    ].shape[0],
                    "n_reconstruction_sign_lookup_probes": int(
                        reconstruction_info["sign_lookup_max_probe_count"]
                    ),
                }
            )

        representative_entry_count = int(
            reconstruction_info.get(
                "representative_entry_count",
                table_data_combined.shape[1],
            )
        )
        reconstruction_diagnostics = {
            "kind": reconstruction_kind,
            "online_value_bytes": int(table_data_combined.nbytes),
            "representative_value_bytes": int(
                representative_entry_count
                * len(near_field_tables)
                * np.dtype(eval_dtype).itemsize
            ),
            "generated_reconstruction_metadata_bytes": int(
                reconstruction_info["metadata_bytes"]
            ),
            "normalizer_auxiliary_bytes": int(
                mode_nmlz_combined.nbytes + exterior_mode_nmlz_combined.nbytes
            ),
            "value_dense_block_equivalents": float(
                table_data_combined.shape[1] / table0.n_pairs
            ),
            "dense_entry_id_bytes": dense_id_bytes,
            "dense_entry_scale_bytes": dense_scale_bytes,
            "dense_metadata_bytes": int(dense_id_bytes + dense_scale_bytes),
            "generated_metadata_bytes": int(reconstruction_info["metadata_bytes"]),
            "generated_sign_correction_count": int(
                reconstruction_info.get("sign_correction_count", 0)
            ),
            "sign_convention_conflict_count": int(
                reconstruction_info.get("sign_convention_conflict_count", 0)
            ),
            "unused_arithmetic_layout_entries": int(
                reconstruction_info.get("unused_layout_entry_count", 0)
            ),
        }

        payload = {
            "base": base,
            "shift": shift,
            "case_indices_dev": case_indices_dev,
            "table_data_dev": cl.array.to_device(queue, table_data_combined),
            "mode_nmlz_dev": cl.array.to_device(queue, mode_nmlz_combined),
            "exterior_mode_nmlz_dev": cl.array.to_device(
                queue, exterior_mode_nmlz_combined
            ),
            "table_data_shapes": table_data_shapes,
            "reconstruction_diagnostics": reconstruction_diagnostics,
        }

        if reconstruction_kind in _ARITHMETIC_RECONSTRUCTION_KINDS:
            payload.update(
                {
                    "arithmetic_case_orbit_ranks_dev": cl.array.to_device(
                        queue, reconstruction_info["case_orbit_ranks"]
                    ),
                    "arithmetic_case_axis_perm_dev": cl.array.to_device(
                        queue, reconstruction_info["case_axis_perm"]
                    ),
                    "arithmetic_case_axis_sign_dev": cl.array.to_device(
                        queue, reconstruction_info["case_axis_sign"]
                    ),
                    "arithmetic_case_axis_group_dev": cl.array.to_device(
                        queue, reconstruction_info["case_axis_group"]
                    ),
                    "arithmetic_case_value_offsets_dev": cl.array.to_device(
                        queue, reconstruction_info["case_value_offsets"]
                    ),
                    "arithmetic_axis_sign_power_dev": cl.array.to_device(
                        queue, reconstruction_info["axis_sign_power"]
                    ),
                    "arithmetic_axis_direction_signs_dev": cl.array.to_device(
                        queue, reconstruction_info["axis_direction_signs"]
                    ),
                    "arithmetic_direction_sign_axis": int(
                        reconstruction_info["direction_sign_axis"]
                    ),
                }
            )
        elif reconstruction_kind == "generated-orbit":
            payload.update(
                {
                    "reconstruction_qpoint_map_dev": cl.array.to_device(
                        queue, reconstruction_info["transform_qpoint_map"]
                    ),
                    "reconstruction_case_map_dev": cl.array.to_device(
                        queue, reconstruction_info["transform_case_map"]
                    ),
                    "reconstruction_signs_dev": cl.array.to_device(
                        queue, reconstruction_info["transform_signs"]
                    ),
                    "reconstruction_lookup_keys_dev": cl.array.to_device(
                        queue, reconstruction_info["lookup_keys"]
                    ),
                    "reconstruction_lookup_values_dev": cl.array.to_device(
                        queue, reconstruction_info["lookup_values"]
                    ),
                    "reconstruction_sign_lookup_keys_dev": cl.array.to_device(
                        queue, reconstruction_info["sign_lookup_keys"]
                    ),
                    "reconstruction_sign_lookup_values_dev": cl.array.to_device(
                        queue, reconstruction_info["sign_lookup_values"]
                    ),
                }
            )
        else:
            payload.update(
                {
                    "mode_qpoint_map_dev": mode_qpoint_map_dev,
                    "mode_case_map_dev": mode_case_map_dev,
                    "mode_case_scale_dev": cl.array.to_device(queue, mode_case_scale),
                    "table_entry_ids_dev": cl.array.to_device(queue, table_entry_ids),
                    "table_entry_scales_dev": cl.array.to_device(
                        queue, table_entry_scales
                    ),
                }
            )
        self._nearfield_device_payload_cache[cache_key] = payload
        while (
            len(self._nearfield_device_payload_cache)
            > self._nearfield_device_payload_cache_max
        ):
            self._nearfield_device_payload_cache.popitem(last=False)
        return payload

# vim: filetype=pyopencl:foldmethod=marker
