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

__doc__ = """Marshalling of near-field interaction tables into device-ready arrays.

Flattens a per-level list of tables into combined value/normalizer arrays
plus the reconstruction metadata that the near-field OpenCL kernels need.
"""

import hashlib
import warnings

import numpy as np

from volumential.wranglers.arithmetic_orbits import (
    _build_fast_scalar_arithmetic_orbit_reconstruction,
    _build_scalar_arithmetic_orbit_reconstruction,
)
from volumential.wranglers.orbit_generated import (
    _build_generated_orbit_reconstruction,
)


def _table_data_fingerprint(arr, sample_bytes=4096):
    narr = np.asarray(arr)
    nbytes = int(narr.nbytes)
    if nbytes == 0:
        return (str(narr.dtype), tuple(int(s) for s in narr.shape), 0, "")

    raw = narr.view(np.uint8).reshape(-1)
    if nbytes <= 2 * sample_bytes:
        sample = bytes(raw)
    else:
        sample = bytes(raw[:sample_bytes]) + bytes(raw[-sample_bytes:])

    digest = hashlib.sha256(sample).hexdigest()
    return (str(narr.dtype), tuple(int(s) for s in narr.shape), nbytes, digest)


def _prepare_table_data_and_entry_map(table_levels):
    if not table_levels:
        raise ValueError("table_levels cannot be empty")

    table0 = table_levels[0]
    if hasattr(table0, "n_cases") and hasattr(table0, "n_pairs"):
        n_full_entries = int(table0.n_cases * table0.n_pairs)
    else:
        n_full_entries = len(table0.data)

    reduced_flags = [
        bool(getattr(table, "table_data_is_symmetry_reduced", False))
        for table in table_levels
    ]
    if any(flag != reduced_flags[0] for flag in reduced_flags[1:]):
        raise RuntimeError("mixed full/reduced near-field table storage across levels")

    if hasattr(table0, "dtype"):
        table_value_dtype = np.dtype(table0.dtype)
    elif hasattr(table0, "get_reduced_table_data"):
        _, table0_values = table0.get_reduced_table_data()
        table_value_dtype = np.asarray(table0_values).dtype
    else:
        table_value_dtype = np.asarray(table0.data).dtype

    if reduced_flags[0]:
        fast_reconstruction_info = _build_fast_scalar_arithmetic_orbit_reconstruction(
            table0
        )
        if fast_reconstruction_info is not None:
            layout_entry_ids = np.asarray(
                fast_reconstruction_info["layout_entry_ids"], dtype=np.int64
            )
            layout_entry_scales = np.asarray(
                fast_reconstruction_info["layout_entry_scales"], dtype=table_value_dtype
            )
            if len(layout_entry_ids) == 0:
                raise RuntimeError(
                    "near-field reconstruction layout contains no entries"
                )

            if not table0.has_entry_data_for_full_indices(layout_entry_ids):
                raise RuntimeError(
                    "near-field reconstruction layout references missing data"
                )
            for table in table_levels[1:]:
                if not table.has_entry_data_for_full_indices(layout_entry_ids):
                    raise RuntimeError(
                        "near-field levels disagree on reconstruction layout "
                        "availability"
                    )

            table_data_combined = np.zeros(
                (len(table_levels), len(layout_entry_ids)),
                dtype=table_value_dtype,
            )
            mode_nmlz_combined = np.zeros(
                (len(table_levels), len(table0.mode_normalizers)),
                dtype=table0.mode_normalizers.dtype,
            )
            exterior_mode_nmlz_combined = np.zeros(
                (len(table_levels), len(table0.kernel_exterior_normalizers)),
                dtype=table0.kernel_exterior_normalizers.dtype,
            )

            for lev, table in enumerate(table_levels):
                layout_values = table.get_entry_data_for_full_indices(layout_entry_ids)
                table_data_combined[lev, :] = layout_values * layout_entry_scales
                mode_nmlz_combined[lev, :] = table.mode_normalizers
                exterior_mode_nmlz_combined[lev, :] = (
                    table.kernel_exterior_normalizers
                )

            return (
                table_data_combined,
                mode_nmlz_combined,
                exterior_mode_nmlz_combined,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=table_value_dtype),
                fast_reconstruction_info,
            )

    if reduced_flags[0]:
        table_entry_scales = np.ones(n_full_entries, dtype=table_value_dtype)
        orbit_info = None
        if hasattr(table0, "_get_orbit_canonical_info"):
            orbit_info = table0._get_orbit_canonical_info()

        if orbit_info is not None and "canonical_entry_ids" in orbit_info:
            canonical_entry_ids = np.asarray(
                orbit_info["canonical_entry_ids"], dtype=np.int64
            )
            kept_entry_ids = np.asarray(orbit_info["entry_ids"], dtype=np.int64)
            if len(kept_entry_ids) == 0:
                raise RuntimeError("near-field table contains no canonical entries")

            if hasattr(table0, "has_entry_data_for_full_indices"):
                has_kept_entries = table0.has_entry_data_for_full_indices(
                    kept_entry_ids
                )
            else:
                finite_mask = np.isfinite(table0.data)
                has_kept_entries = bool(np.all(finite_mask[kept_entry_ids]))
            if not has_kept_entries:
                raise RuntimeError(
                    "near-field table is missing finite values for canonical entries"
                )

            for table in table_levels[1:]:
                if hasattr(table, "has_entry_data_for_full_indices"):
                    level_has_kept_entries = table.has_entry_data_for_full_indices(
                        kept_entry_ids
                    )
                else:
                    level_finite_mask = np.isfinite(table.data)
                    level_has_kept_entries = bool(
                        np.all(level_finite_mask[kept_entry_ids])
                    )
                if not level_has_kept_entries:
                    raise RuntimeError(
                        "near-field levels disagree on canonical entry availability"
                    )

            compact_ids = np.full(n_full_entries, -1, dtype=np.int32)
            compact_ids[kept_entry_ids] = np.arange(len(kept_entry_ids), dtype=np.int32)
            table_entry_ids = compact_ids[canonical_entry_ids]
            if "canonical_scales" in orbit_info:
                table_entry_scales = np.asarray(
                    orbit_info["canonical_scales"], dtype=table_value_dtype
                )
        else:
            if hasattr(table0, "get_reduced_table_data"):
                kept_entry_ids, _ = table0.get_reduced_table_data()
                kept_entry_ids = np.asarray(kept_entry_ids, dtype=np.int64)
                finite_mask = None
            else:
                finite_mask = np.isfinite(table0.data)
                kept_entry_ids = np.flatnonzero(finite_mask).astype(np.int64)
            for table in table_levels[1:]:
                if hasattr(table, "get_reduced_table_data"):
                    level_entry_ids, _ = table.get_reduced_table_data()
                    level_matches = np.array_equal(level_entry_ids, kept_entry_ids)
                else:
                    level_finite_mask = np.isfinite(table.data)
                    level_matches = np.array_equal(level_finite_mask, finite_mask)
                if not level_matches:
                    raise RuntimeError(
                        "near-field levels disagree on symmetry-reduced entry ids"
                    )

            if len(kept_entry_ids) == 0:
                raise RuntimeError("near-field table contains no finite entries")

            table_entry_ids = np.full(n_full_entries, -1, dtype=np.int32)
            table_entry_ids[kept_entry_ids] = np.arange(
                len(kept_entry_ids), dtype=np.int32
            )
    else:
        for table in table_levels:
            if not np.all(np.isfinite(table.data)):
                raise RuntimeError("full near-field table contains non-finite entries")
        kept_entry_ids = np.arange(n_full_entries, dtype=np.int64)
        table_entry_ids = np.full(n_full_entries, -1, dtype=np.int32)
        table_entry_ids[kept_entry_ids] = np.arange(len(kept_entry_ids), dtype=np.int32)
        table_entry_scales = np.ones(n_full_entries, dtype=table_value_dtype)

    reconstruction_info = _build_scalar_arithmetic_orbit_reconstruction(
        table0,
        table_entry_ids,
        table_entry_scales,
    )
    if reconstruction_info is None:
        reconstruction_info = _build_generated_orbit_reconstruction(
            table0,
            table_entry_ids,
            table_entry_scales,
        )
        if reconstruction_info is not None:
            warnings.warn(
                "using generated-orbit fallback for near-field table "
                "reconstruction; this compact fallback is correct but not "
                "GPU-ideal because it still uses transform scans and lookup "
                f"tables (kernel={table0.integral_knl!r})",
                RuntimeWarning,
                stacklevel=2,
            )

    if reconstruction_info is None:
        reconstruction_info = {
            "kind": "dense",
            "metadata_bytes": int(table_entry_ids.nbytes + table_entry_scales.nbytes),
        }

    layout_entry_ids = reconstruction_info.get("layout_entry_ids", kept_entry_ids)
    layout_entry_ids = np.asarray(layout_entry_ids, dtype=np.int64)
    layout_entry_scales = reconstruction_info.get("layout_entry_scales")
    if layout_entry_scales is None:
        layout_entry_scales = np.ones(len(layout_entry_ids), dtype=table_value_dtype)
    else:
        layout_entry_scales = np.asarray(layout_entry_scales, dtype=table_value_dtype)
    if layout_entry_scales.shape != layout_entry_ids.shape:
        raise RuntimeError("reconstruction layout entry scales have incompatible shape")

    used_layout_mask = layout_entry_ids >= 0
    used_layout_entry_ids = layout_entry_ids[used_layout_mask]
    used_layout_scales = layout_entry_scales[used_layout_mask]
    if len(used_layout_entry_ids) == 0:
        raise RuntimeError("near-field reconstruction layout contains no entries")
    if hasattr(table0, "has_entry_data_for_full_indices"):
        layout_entries_available = table0.has_entry_data_for_full_indices(
            used_layout_entry_ids
        )
    else:
        layout_entries_available = bool(
            np.all(np.isfinite(table0.data[used_layout_entry_ids]))
        )
    if not layout_entries_available:
        raise RuntimeError("near-field reconstruction layout references missing data")
    for table in table_levels[1:]:
        if hasattr(table, "has_entry_data_for_full_indices"):
            level_layout_entries_available = table.has_entry_data_for_full_indices(
                used_layout_entry_ids
            )
        else:
            level_layout_entries_available = bool(
                np.all(np.isfinite(table.data[used_layout_entry_ids]))
            )
        if not level_layout_entries_available:
            raise RuntimeError(
                "near-field levels disagree on reconstruction layout availability"
            )

    table_data_combined = np.zeros(
        (len(table_levels), len(layout_entry_ids)),
        dtype=table_value_dtype,
    )
    mode_nmlz_combined = np.zeros(
        (len(table_levels), len(table0.mode_normalizers)),
        dtype=table0.mode_normalizers.dtype,
    )
    exterior_mode_nmlz_combined = np.zeros(
        (len(table_levels), len(table0.kernel_exterior_normalizers)),
        dtype=table0.kernel_exterior_normalizers.dtype,
    )

    for lev, table in enumerate(table_levels):
        if hasattr(table, "get_entry_data_for_full_indices"):
            layout_values = table.get_entry_data_for_full_indices(used_layout_entry_ids)
        else:
            layout_values = table.data[used_layout_entry_ids]
        table_data_combined[lev, used_layout_mask] = layout_values * used_layout_scales
        mode_nmlz_combined[lev, :] = table.mode_normalizers
        exterior_mode_nmlz_combined[lev, :] = table.kernel_exterior_normalizers

    return (
        table_data_combined,
        mode_nmlz_combined,
        exterior_mode_nmlz_combined,
        table_entry_ids,
        table_entry_scales,
        reconstruction_info,
    )

# vim: filetype=pyopencl:foldmethod=marker
