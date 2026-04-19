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

__doc__ = """
.. autofunction:: drive_volume_fmm
.. autofunction:: interpolate_volume_potential
"""

import logging
import os
import inspect
from dataclasses import replace
from itertools import product

import numpy as np

import pyopencl as cl
import pyopencl.array  # noqa: F401
from pytools.obj_array import new_1d as obj_array_1d

try:
    from boxtree.timing import TimingRecorder
except ImportError:
    try:
        from boxtree.fmm import TimingRecorder
    except ImportError:

        class TimingRecorder:
            def __init__(self):
                self._futures = []

            def add(self, stage, future):
                self._futures.append((stage, future))

            def summarize(self):
                summary = {}
                for stage, future in self._futures:
                    if future is None:
                        continue
                    try:
                        summary[stage] = future()
                    except TypeError:
                        summary[stage] = future
                return summary


from volumential.expansion_wrangler_fpnd import (
    FPNDFMMLibExpansionWrangler,
    FPNDSumpyExpansionWrangler,
)
from volumential.expansion_wrangler_interface import ExpansionWranglerInterface


logger = logging.getLogger(__name__)
_COINCIDENT_TREE_WARNING_EMITTED = False


def _env_flag_enabled(name):
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_interpolation_target_coverage(multiplicity, queue):
    if hasattr(multiplicity, "with_queue"):
        multiplicity_dev = multiplicity.with_queue(queue)
        if int(multiplicity_dev.size) == 0:
            return
        multiplicity_host = multiplicity_dev.get(queue)
    elif hasattr(multiplicity, "get"):
        multiplicity_host = multiplicity.get(queue)
    else:
        multiplicity_host = np.asarray(multiplicity)

    if multiplicity_host.size == 0:
        return

    missing_mask = multiplicity_host <= 0
    if np.any(missing_mask):
        n_missing = int(np.count_nonzero(missing_mask))
        raise ValueError(
            "interpolation did not cover all targets "
            f"({n_missing} targets missed by interpolation lookup)"
        )


def _compute_interpolation_lookup_tol(tree, tol=1e-12):
    coord_dtype = np.dtype(tree.coord_dtype)
    root_extent = float(abs(getattr(tree, "root_extent", 1.0)))
    if not np.isfinite(root_extent) or root_extent == 0.0:
        root_extent = 1.0

    tol = max(float(tol), 64.0 * np.finfo(coord_dtype).eps * root_extent)

    nlevels = int(getattr(tree, "nlevels", 1))
    leaf_diam_cap = 0.25 * root_extent / (1 << max(nlevels - 1, 0))
    if leaf_diam_cap > 0:
        tol = min(tol, leaf_diam_cap)

    return tol


def _cast_source_field_dtype(field, dtype):
    if isinstance(field, np.ndarray):
        if field.dtype == object:
            try:
                return field.astype(dtype, copy=False)
            except (TypeError, ValueError):
                return field
        return field.astype(dtype, copy=False)

    if hasattr(field, "astype"):
        return field.astype(dtype)

    return np.asarray(field, dtype=dtype)


def _normalize_matrix_source_fields(values, expected_length, field_name):
    if expected_length is None:
        raise ValueError(
            f"{field_name} has ndim=2 but source length is unknown. "
            "Pass multiple source fields as an object array/list in "
            "(nfields,) form."
        )

    nrows, ncols = (int(values.shape[0]), int(values.shape[1]))

    if nrows == expected_length and ncols == expected_length:
        raise ValueError(
            f"{field_name} has ambiguous square shape {values.shape}; "
            "use an object array/list or an explicit (nfields, npoints) array."
        )

    if nrows == expected_length:
        if ncols == 1:
            return [values[:, 0]]

        raise ValueError(
            f"{field_name} has point-major shape {values.shape}. "
            "Use object-array/list input or transpose to (nfields, npoints)."
        )

    if ncols == expected_length:
        return [values[i] for i in range(nrows)]

    raise ValueError(
        f"{field_name} has shape {values.shape}, but one axis must match "
        f"the source count ({expected_length})."
    )


def _normalize_source_fields(
    values, dtype, *, expected_length=None, field_name="source"
):
    if isinstance(values, np.ndarray) and values.dtype == object:
        if values.ndim <= 1:
            fields = list(values.flat)
        elif values.ndim == 2:
            fields = _normalize_matrix_source_fields(
                values, expected_length, field_name
            )
        else:
            raise ValueError(f"{field_name} must be 1D or 2D, got ndim={values.ndim}")
    elif isinstance(values, (list, tuple)):
        array_values = np.asarray(values)

        if array_values.ndim == 1 and array_values.dtype != object:
            fields = [array_values]
        elif (
            array_values.ndim == 2
            and array_values.dtype != object
            and expected_length is not None
        ):
            fields = _normalize_matrix_source_fields(
                array_values, expected_length, field_name
            )
        else:
            fields = list(values)
    elif hasattr(values, "ndim") and values.ndim == 2:
        fields = _normalize_matrix_source_fields(values, expected_length, field_name)
    elif hasattr(values, "ndim") and values.ndim > 2:
        raise ValueError(f"{field_name} must be 1D or 2D, got ndim={values.ndim}")
    else:
        fields = [values]

    return obj_array_1d([_cast_source_field_dtype(field, dtype) for field in fields])


def _as_obj_array(potentials):
    if isinstance(potentials, np.ndarray) and potentials.dtype == object:
        return potentials

    return obj_array_1d([potentials])


def _is_obj_array(values):
    return isinstance(values, np.ndarray) and values.dtype == object


def _add_obj_arrays(lhs, rhs):
    lhs_is_obj = _is_obj_array(lhs)
    rhs_is_obj = _is_obj_array(rhs)

    lhs_oa = _as_obj_array(lhs)
    rhs_oa = _as_obj_array(rhs)

    if len(lhs_oa) != len(rhs_oa):
        raise ValueError("incompatible potential vector lengths")

    added = [lhs_i + rhs_i for lhs_i, rhs_i in zip(lhs_oa, rhs_oa, strict=True)]
    if len(added) == 1 and not (lhs_is_obj or rhs_is_obj):
        return added[0]

    return obj_array_1d(added)


def _coerce_obj_array_like(values, like, queue):
    like_is_obj = _is_obj_array(like)

    values_oa = _as_obj_array(values)
    like_oa = _as_obj_array(like)

    if len(values_oa) != len(like_oa):
        raise ValueError("incompatible potential vector lengths")

    coerced = []
    for value_i, like_i in zip(values_oa, like_oa, strict=True):
        if isinstance(like_i, cl.array.Array) and not isinstance(
            value_i, cl.array.Array
        ):
            if (
                isinstance(value_i, np.ndarray)
                and value_i.dtype == object
                and value_i.size
                and hasattr(value_i.flat[0], "get")
            ):
                host_value = np.asarray(
                    [entry.get(queue) for entry in value_i],
                    dtype=like_i.dtype,
                )
                coerced.append(cl.array.to_device(queue, host_value))
                continue

            coerced.append(
                cl.array.to_device(queue, np.asarray(value_i, dtype=like_i.dtype))
            )
        elif isinstance(value_i, cl.array.Array) and isinstance(like_i, np.ndarray):
            coerced.append(value_i.get(queue))
        else:
            coerced.append(value_i)

    if len(coerced) == 1 and not like_is_obj:
        return coerced[0]

    return obj_array_1d(coerced)


def _as_host_array(values, queue):
    if hasattr(values, "with_queue"):
        return np.asarray(values.with_queue(queue).get())
    if hasattr(values, "get"):
        return np.asarray(values.get(queue))
    return np.asarray(values)


def _normalize_periodic_cell_size(periodic_cell_size, dim, tree):
    if periodic_cell_size is None:
        cell_extent = float(abs(getattr(tree, "root_extent", 1.0)))
        if not np.isfinite(cell_extent) or cell_extent <= 0:
            raise ValueError("tree root_extent must be finite and positive")
        return np.full(dim, cell_extent, dtype=np.float64)

    cell_size = np.asarray(periodic_cell_size, dtype=np.float64)
    if cell_size.ndim == 0:
        cell_size = np.full(dim, float(cell_size), dtype=np.float64)
    elif cell_size.shape != (dim,):
        raise ValueError(
            "periodic_cell_size must be scalar or length-dimension vector, "
            f"got shape {cell_size.shape} for dimension {dim}"
        )

    if np.any(~np.isfinite(cell_size)) or np.any(cell_size <= 0):
        raise ValueError("periodic_cell_size entries must be finite and positive")

    return cell_size


def _normalize_periodic_near_shifts(periodic_near_shifts, dim):
    if periodic_near_shifts is None:
        return np.empty((0, dim), dtype=np.int64)

    if isinstance(periodic_near_shifts, str):
        token = periodic_near_shifts.strip().lower()
        if token in {"nearest", "barnett-nearest", "nn"}:
            periodic_near_shifts = [
                shift
                for shift in product((-1, 0, 1), repeat=dim)
                if any(component != 0 for component in shift)
            ]
        elif token in {"none", "off", "disabled"}:
            return np.empty((0, dim), dtype=np.int64)
        else:
            raise ValueError(
                "periodic_near_shifts string must be one of {'nearest', 'none'}"
            )

    shifts = np.asarray(periodic_near_shifts, dtype=np.int64)
    if shifts.ndim == 1:
        shifts = shifts.reshape(1, -1)

    if shifts.ndim != 2 or shifts.shape[1] != dim:
        raise ValueError(
            "periodic_near_shifts must have shape (nshifts, dim), "
            f"got {shifts.shape} for dim={dim}"
        )

    if shifts.size == 0:
        return np.empty((0, dim), dtype=np.int64)

    shifts = np.unique(shifts, axis=0)
    keep_mask = np.any(shifts != 0, axis=1)
    return shifts[keep_mask]


def _periodic_shift_vectors_from_indices(shift_indices, cell_size):
    if shift_indices.size == 0:
        return np.empty((0, len(cell_size)), dtype=np.float64)
    return shift_indices.astype(np.float64) * cell_size.reshape(1, -1)


def _collect_target_point_ids_for_boxes(tree, target_boxes, queue):
    if target_boxes is None:
        return None

    target_boxes_host = _as_host_array(target_boxes, queue).astype(np.int64, copy=False)
    if target_boxes_host.ndim != 1:
        raise ValueError("periodic_near_target_boxes must be a 1D array")

    if target_boxes_host.size == 0:
        return np.empty(0, dtype=np.int32)

    box_target_starts = _as_host_array(tree.box_target_starts, queue)
    box_target_counts = _as_host_array(tree.box_target_counts_nonchild, queue)

    point_ids = []
    nboxes = int(box_target_starts.size)
    for box_id in map(int, target_boxes_host):
        if box_id < 0 or box_id >= nboxes:
            raise ValueError(
                f"periodic_near_target_boxes contains out-of-range box id {box_id}"
            )

        start = int(box_target_starts[box_id])
        count = int(box_target_counts[box_id])
        if count <= 0:
            continue
        point_ids.extend(range(start, start + count))

    if not point_ids:
        return np.empty(0, dtype=np.int32)

    return np.asarray(sorted(set(point_ids)), dtype=np.int32)


def _find_root_box_id(tree, queue):
    box_levels = _as_host_array(tree.box_levels, queue).astype(np.int64, copy=False)
    root_candidates = np.flatnonzero(box_levels == 0)
    if root_candidates.size != 1:
        raise ValueError(
            f"expected exactly one root box, got {int(root_candidates.size)} candidates"
        )
    return int(root_candidates[0]), int(box_levels.size)


def _expansion_level_zero_view(wrangler, values, expansion_kind):
    if expansion_kind == "local":
        base_ibox, level_view = wrangler.local_expansions_view(values, 0)
    elif expansion_kind == "multipole":
        base_ibox, level_view = wrangler.multipole_expansions_view(values, 0)
    else:
        raise ValueError(
            f"unsupported expansion kind '{expansion_kind}', expected 'local' "
            "or 'multipole'"
        )

    return int(base_ibox), level_view


def _expansion_row_index_from_box_id(base_ibox, level_view, box_id):
    row_index = int(box_id) - int(base_ibox)
    nrows = int(level_view.shape[0]) if level_view.ndim > 1 else 1
    if row_index < 0 or row_index >= nrows:
        raise ValueError(
            f"box id {box_id} is not present in level-0 expansion view "
            f"(base={base_ibox}, nrows={nrows})"
        )
    return row_index


def _coerce_expansion_row_dtype(row_values, target_dtype):
    row_values = np.asarray(row_values)
    target_dtype = np.dtype(target_dtype)

    if np.iscomplexobj(row_values) and not np.issubdtype(
        target_dtype, np.complexfloating
    ):
        real_part = np.real(row_values)
        imag_part = np.abs(np.imag(row_values))
        scale = float(np.max(np.abs(real_part))) if row_values.size else 1.0
        imag_tol = 100.0 * np.finfo(np.float64).eps * max(1.0, scale)
        if np.any(imag_part > imag_tol):
            raise ValueError(
                "periodic far operator produced complex-valued local "
                "coefficients for a real-valued local expansion"
            )
        row_values = real_part

    return np.asarray(row_values, dtype=target_dtype)


def _extract_root_expansion_row_host(
    values,
    *,
    wrangler,
    root_box_id,
    expansion_kind,
    queue,
):
    base_ibox, level_view = _expansion_level_zero_view(
        wrangler,
        values,
        expansion_kind,
    )
    row_index = _expansion_row_index_from_box_id(base_ibox, level_view, root_box_id)

    if isinstance(level_view, cl.array.Array):
        level_host = level_view.get(queue)
    else:
        level_host = np.asarray(level_view)

    level_host = np.asarray(level_host)
    if level_host.ndim == 1:
        level_host_rows = level_host.reshape(1, -1)
    elif level_host.ndim == 2:
        level_host_rows = level_host
    else:
        raise ValueError(
            f"unsupported expansion view rank {level_host.ndim}; expected 1D or 2D"
        )

    return level_host_rows[row_index].copy()


def _inject_root_expansion_row(
    values,
    *,
    wrangler,
    root_box_id,
    expansion_kind,
    row_values,
    queue,
):
    base_ibox, level_view = _expansion_level_zero_view(
        wrangler,
        values,
        expansion_kind,
    )
    row_index = _expansion_row_index_from_box_id(base_ibox, level_view, root_box_id)

    if isinstance(level_view, cl.array.Array):
        level_host = level_view.get(queue)
    else:
        level_host = np.asarray(level_view)

    level_host = np.array(level_host, copy=True)
    original_shape = level_host.shape

    if level_host.ndim == 1:
        level_rows = level_host.reshape(1, -1)
    elif level_host.ndim == 2:
        level_rows = level_host
    else:
        raise ValueError(
            f"unsupported expansion view rank {level_host.ndim}; expected 1D or 2D"
        )

    row_values = _coerce_expansion_row_dtype(row_values, level_rows.dtype)
    ncoeff = int(level_rows.shape[1])
    if row_values.size > ncoeff:
        raise ValueError(
            f"periodic far operator produced too many {expansion_kind} "
            f"coefficients: {row_values.size} > {ncoeff}"
        )

    level_rows[row_index, : row_values.size] += row_values

    updated_level = level_rows.reshape(original_shape)
    if isinstance(level_view, cl.array.Array):
        level_view.set(updated_level)
    else:
        level_view[...] = updated_level

    return values


def _resolve_periodic_far_operator_matrix(periodic_far_operator, out_kernel):
    if isinstance(periodic_far_operator, dict):
        kname = repr(out_kernel)
        matrix = periodic_far_operator.get(kname, None)
        if matrix is None:
            try:
                matrix = periodic_far_operator.get(out_kernel, None)
            except TypeError:
                matrix = None

        if matrix is None:
            raise KeyError(
                f"missing periodic far operator matrix for output kernel {kname}"
            )
    else:
        matrix = periodic_far_operator

    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(
            f"periodic_far_operator matrices must be 2D, got ndim={matrix.ndim}"
        )

    return matrix


def _apply_periodic_far_operator_to_locals(
    *,
    local_exps,
    mpole_exps,
    source_strength,
    traversal,
    wrangler,
    periodic_far_operator,
    periodic_far_operator_basis,
    queue,
):
    if periodic_far_operator is None:
        return local_exps

    root_box_id, _ = _find_root_box_id(traversal.tree, queue)

    if periodic_far_operator_basis not in {None, "multipole", "source"}:
        raise ValueError(
            "periodic_far_operator_basis must be one of None, 'multipole', or 'source'"
        )

    source_row = None
    if source_strength is not None:
        if isinstance(source_strength, cl.array.Array):
            source_row = source_strength.get(queue)
        else:
            source_row = np.asarray(source_strength)
        source_row = np.asarray(source_row).reshape(-1)

    local_oa = _as_obj_array(local_exps)
    mpole_oa = _as_obj_array(mpole_exps)

    if len(mpole_oa) not in {1, len(local_oa)}:
        raise NotImplementedError(
            "periodic far operator currently supports either one source "
            "expansion channel or one channel per output kernel"
        )

    updated_local = []
    for out_idx, (out_kernel, local_values) in enumerate(
        zip(wrangler.tree_indep.target_kernels, local_oa, strict=True)
    ):
        src_idx = 0 if len(mpole_oa) == 1 else out_idx
        matrix = _resolve_periodic_far_operator_matrix(
            periodic_far_operator, out_kernel
        )

        if isinstance(mpole_oa[src_idx], cl.array.Array):
            mpole_row = _extract_root_expansion_row_host(
                mpole_oa[src_idx],
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="multipole",
                queue=queue,
            )
        else:
            mpole_row = _extract_root_expansion_row_host(
                np.asarray(mpole_oa[src_idx]),
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="multipole",
                queue=queue,
            )

        feature_row = None
        if periodic_far_operator_basis == "multipole":
            feature_row = mpole_row
        elif periodic_far_operator_basis == "source":
            if source_row is None:
                raise ValueError(
                    "periodic far operator basis is 'source' but source_strength "
                    "was not supplied"
                )
            feature_row = source_row
        else:
            if matrix.shape[1] == mpole_row.size:
                feature_row = mpole_row
            elif source_row is not None and matrix.shape[1] == source_row.size:
                feature_row = source_row

        if feature_row is None or matrix.shape[1] != feature_row.size:
            expected = [f"multipole={mpole_row.size}"]
            if source_row is not None:
                expected.append(f"source={source_row.size}")
            expected_text = ", ".join(expected)
            raise ValueError(
                "periodic far operator matrix has incompatible column count "
                f"({matrix.shape[1]}). Expected one of: {expected_text}"
            )

        local_corr = matrix @ feature_row

        if isinstance(local_values, cl.array.Array):
            local_values_updated = local_values.copy()
            local_values_updated = _inject_root_expansion_row(
                local_values_updated,
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="local",
                row_values=local_corr,
                queue=queue,
            )
            updated_local.append(local_values_updated)
        else:
            local_values_updated = np.asarray(local_values).copy()
            local_values_updated = _inject_root_expansion_row(
                local_values_updated,
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="local",
                row_values=local_corr,
                queue=queue,
            )
            updated_local.append(local_values_updated)

    if isinstance(local_exps, np.ndarray) and local_exps.dtype == object:
        return obj_array_1d(updated_local)
    if len(updated_local) == 1:
        return updated_local[0]
    return obj_array_1d(updated_local)


def _take_point_cloud_subset(points, point_ids, queue):
    if point_ids is None:
        return points

    point_ids = np.asarray(point_ids, dtype=np.int32)

    if isinstance(points, np.ndarray) and points.dtype == object:
        axes = list(points)
    elif isinstance(points, (list, tuple)):
        axes = list(points)
    else:
        raise TypeError("point cloud must be an object array or sequence of axes")

    if not axes:
        raise ValueError("point cloud has no coordinate axes")

    if isinstance(axes[0], cl.array.Array):
        point_ids_dev = cl.array.to_device(queue, point_ids)
        subset_axes = []
        for axis in axes:
            axis_eval = axis.with_queue(queue) if hasattr(axis, "with_queue") else axis
            subset_axes.append(axis_eval[point_ids_dev])
        return obj_array_1d(subset_axes)

    return obj_array_1d([np.take(np.asarray(axis), point_ids) for axis in axes])


def _scatter_subset_potential_to_full(subset_potential, point_ids, ntargets, queue):
    point_ids = np.asarray(point_ids, dtype=np.int32)

    if isinstance(subset_potential, cl.array.Array):
        full_host = np.zeros(int(ntargets), dtype=subset_potential.dtype)
        full_host[point_ids] = subset_potential.get(queue)
        return cl.array.to_device(queue, full_host)

    full_host = np.zeros(int(ntargets), dtype=np.asarray(subset_potential).dtype)
    full_host[point_ids] = np.asarray(subset_potential)
    return full_host


def _cast_scalar_to_axis_dtype(value, axis):
    axis_dtype = getattr(axis, "dtype", None)
    if axis_dtype is None:
        return value

    try:
        return np.dtype(axis_dtype).type(value)
    except TypeError:
        return value


def _eval_periodic_near_stencil_p2p(
    *,
    traversal,
    wrangler,
    src_weights,
    shift_vectors,
    periodic_near_target_boxes,
    queue,
):
    if shift_vectors.size == 0:
        return wrangler.output_zeros(), None

    if len(src_weights) != 1:
        raise NotImplementedError(
            "periodic near-stencil evaluation currently supports one source field"
        )

    from sumpy import P2P

    tree = traversal.tree
    actx = wrangler.tree_indep._setup_actx
    p2p = P2P(
        wrangler.tree_indep.target_kernels,
        exclude_self=False,
        value_dtypes=[wrangler.dtype],
    )

    p2p_kwargs = {}
    source_extra_kwargs = getattr(wrangler, "source_extra_kwargs", None)
    if source_extra_kwargs is not None:
        p2p_kwargs.update(source_extra_kwargs)

    kernel_extra_kwargs = getattr(wrangler, "kernel_extra_kwargs", None)
    if kernel_extra_kwargs is not None:
        p2p_kwargs.update(kernel_extra_kwargs)

    p2p_kwargs.pop("target_to_source", None)

    point_ids = _collect_target_point_ids_for_boxes(
        tree,
        periodic_near_target_boxes,
        queue,
    )

    if point_ids is not None and point_ids.size == 0:
        return wrangler.output_zeros(), None

    targets_eval = _take_point_cloud_subset(tree.targets, point_ids, queue)
    ntargets = int(getattr(tree, "ntargets", targets_eval[0].size))

    if isinstance(tree.sources, np.ndarray) and tree.sources.dtype == object:
        source_axes = list(tree.sources)
    else:
        source_axes = list(tree.sources)

    source_axes = [
        axis.with_queue(queue)
        if isinstance(axis, cl.array.Array) and hasattr(axis, "with_queue")
        else axis
        for axis in source_axes
    ]

    periodic_potentials = wrangler.output_zeros()

    for shift in shift_vectors:
        shift = np.asarray(shift, dtype=np.float64)
        shifted_sources = obj_array_1d(
            [
                axis + _cast_scalar_to_axis_dtype(shift[d], axis)
                for d, axis in enumerate(source_axes)
            ]
        )

        shifted_values = p2p(
            actx,
            targets_eval,
            shifted_sources,
            (src_weights[0],),
            **p2p_kwargs,
        )

        shifted_values = obj_array_1d(list(shifted_values))

        if point_ids is not None:
            shifted_values = obj_array_1d(
                [
                    _scatter_subset_potential_to_full(
                        pot_i,
                        point_ids,
                        ntargets,
                        queue,
                    )
                    for pot_i in shifted_values
                ]
            )

        periodic_potentials = _add_obj_arrays(periodic_potentials, shifted_values)

    return periodic_potentials, None


def _coerce_obj_array_to_template(values_oa, template):
    if isinstance(template, np.ndarray) and template.dtype == object:
        return obj_array_1d(list(values_oa))

    if len(values_oa) == 1:
        return values_oa[0]

    return obj_array_1d(list(values_oa))


def _select_periodic_check_point_ids(
    tree,
    queue,
    *,
    periodic_near_target_boxes,
    max_check_points,
    rng,
):
    if periodic_near_target_boxes is None:
        ntargets = int(getattr(tree, "ntargets", 0))
        point_ids = np.arange(ntargets, dtype=np.int32)
    else:
        point_ids = _collect_target_point_ids_for_boxes(
            tree,
            periodic_near_target_boxes,
            queue,
        )

    if point_ids is None or point_ids.size == 0:
        raise ValueError("periodic far operator build requires non-empty target set")

    if max_check_points is not None and point_ids.size > int(max_check_points):
        selected = rng.choice(
            point_ids,
            size=int(max_check_points),
            replace=False,
        )
        point_ids = np.asarray(np.sort(selected), dtype=np.int32)

    return point_ids


def _generate_periodic_far_shift_indices(dim, far_shift_radius, near_shift_indices):
    radius = int(far_shift_radius)
    if radius < 1:
        raise ValueError("far_shift_radius must be >= 1")

    near_set = set()
    near_arr = np.asarray(near_shift_indices, dtype=np.int64)
    if near_arr.size:
        near_arr = near_arr.reshape(-1, dim)
        near_set = {tuple(int(v) for v in shift) for shift in near_arr.tolist()}

    far_shifts = []
    for shift in product(range(-radius, radius + 1), repeat=dim):
        if all(component == 0 for component in shift):
            continue
        if shift in near_set:
            continue
        far_shifts.append(shift)

    if not far_shifts:
        raise ValueError("far_shift_radius is too small; no far shifts remain")

    return np.asarray(far_shifts, dtype=np.int64)


def _point_cloud_to_host_matrix(points, queue, *, point_ids=None):
    if isinstance(points, np.ndarray) and points.dtype == object:
        axes = list(points)
    elif isinstance(points, (list, tuple)):
        axes = list(points)
    else:
        raise TypeError("point cloud must be an object array or sequence of axes")

    if not axes:
        raise ValueError("point cloud has no coordinate axes")

    if point_ids is not None:
        point_ids = np.asarray(point_ids, dtype=np.int32)

    host_axes = []
    for axis in axes:
        axis_host = np.asarray(_to_host_array(axis, queue), dtype=np.float64)
        if point_ids is not None:
            axis_host = axis_host[point_ids]
        host_axes.append(axis_host)

    return np.ascontiguousarray(np.stack(host_axes, axis=1), dtype=np.float64)


def _unwrap_base_kernel(kernel):
    base_kernel = kernel
    seen = set()
    while True:
        getter = getattr(base_kernel, "get_base_kernel", None)
        if getter is None:
            return base_kernel

        kernel_id = id(base_kernel)
        if kernel_id in seen:
            return base_kernel
        seen.add(kernel_id)

        try:
            next_kernel = getter()
        except Exception:
            return base_kernel

        if next_kernel is None or next_kernel is base_kernel:
            return base_kernel

        base_kernel = next_kernel


def _laplace_target_kernel_dim(kernel):
    base_kernel = _unwrap_base_kernel(kernel)

    try:
        from sumpy.kernel import LaplaceKernel

        if isinstance(base_kernel, LaplaceKernel):
            dim = int(getattr(base_kernel, "dim", -1))
            if dim in {2, 3}:
                return dim
            return None
    except Exception:
        pass

    base_repr = repr(base_kernel)
    if "LaplaceKernel(2" in base_repr:
        return 2
    if "LaplaceKernel(3" in base_repr:
        return 3
    return None


def _build_spectral_periodic_laplace_matrix(
    *,
    source_points,
    target_points,
    cell_size,
    k_max,
):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    cell_size = np.asarray(cell_size, dtype=np.float64)

    if source_points.ndim != 2:
        raise ValueError("source_points must have shape (nsources, dim)")
    if target_points.ndim != 2:
        raise ValueError("target_points must have shape (ntargets, dim)")

    dim = int(source_points.shape[1])
    if dim not in {2, 3}:
        raise ValueError("spectral periodic Laplace matrix supports dim=2 or dim=3")
    if target_points.shape[1] != dim:
        raise ValueError("source/target dimensions do not match")
    if cell_size.shape != (dim,):
        raise ValueError(
            f"cell_size must have shape ({dim},) for spectral periodization"
        )

    k_max = int(k_max)
    if k_max < 1:
        raise ValueError("k_max must be >= 1")

    mode_range = np.arange(-k_max, k_max + 1, dtype=np.float64)
    mode_grids = np.meshgrid(*([mode_range] * dim), indexing="ij")
    nonzero_mask = np.zeros_like(mode_grids[0], dtype=bool)
    for mode_grid in mode_grids:
        nonzero_mask = np.logical_or(nonzero_mask, mode_grid != 0)

    mode_components = [mode_grid[nonzero_mask] for mode_grid in mode_grids]
    mode_vectors = np.stack(mode_components, axis=1)

    kvec = (2.0 * np.pi) * mode_vectors / cell_size.reshape(1, dim)
    k2 = np.sum(kvec * kvec, axis=1)

    src_phase = np.exp(-1j * (source_points @ kvec.T))
    tgt_phase = np.exp(1j * (target_points @ kvec.T))

    volume = float(np.prod(cell_size))
    return (tgt_phase / k2.reshape(1, -1)) @ src_phase.T / volume


def _eval_spectral_periodic_laplace_potential(
    *,
    source_points,
    target_points,
    strengths,
    cell_size,
    k_max,
):
    strengths = np.asarray(strengths)
    spectral_matrix = _build_spectral_periodic_laplace_matrix(
        source_points=source_points,
        target_points=target_points,
        cell_size=cell_size,
        k_max=k_max,
    )
    potentials = spectral_matrix @ strengths
    return np.real(potentials)


def _infer_periodic_spectral_kmax(
    *,
    tree_dim,
    periodic_far_build_info,
    periodic_far_spectral_kmax_2d,
    periodic_far_spectral_kmax_3d,
    periodic_far_shift_radius,
):
    if tree_dim == 2:
        if periodic_far_build_info is not None:
            kmax = periodic_far_build_info.get("spectral_k_max_2d")
            if kmax is not None:
                return int(kmax)
        if periodic_far_spectral_kmax_2d is not None:
            return int(periodic_far_spectral_kmax_2d)
        return max(24, 8 * int(periodic_far_shift_radius))

    if tree_dim == 3:
        if periodic_far_build_info is not None:
            kmax = periodic_far_build_info.get("spectral_k_max_3d")
            if kmax is not None:
                return int(kmax)
        if periodic_far_spectral_kmax_3d is not None:
            return int(periodic_far_spectral_kmax_3d)
        return max(10, 4 * int(periodic_far_shift_radius))

    raise ValueError(f"unsupported tree dimension {tree_dim} for spectral kmax")


def _strict_periodic_laplace_potential(
    *,
    traversal,
    wrangler,
    src_weights,
    periodic_cell_size,
    k_max,
    queue,
):
    tree = traversal.tree
    source_points_host = _point_cloud_to_host_matrix(tree.sources, queue)
    target_points_host = _point_cloud_to_host_matrix(tree.targets, queue)
    source_strength_host = np.asarray(
        _to_host_array(src_weights[0], queue),
        dtype=np.dtype(wrangler.dtype),
    )

    spectral_potential_host = _eval_spectral_periodic_laplace_potential(
        source_points=source_points_host,
        target_points=target_points_host,
        strengths=source_strength_host,
        cell_size=periodic_cell_size,
        k_max=int(k_max),
    )

    spectral_potential_host = np.ascontiguousarray(
        spectral_potential_host,
        dtype=np.dtype(wrangler.dtype),
    )

    if isinstance(src_weights[0], cl.array.Array):
        return obj_array_1d([cl.array.to_device(queue, spectral_potential_host)])

    return obj_array_1d([spectral_potential_host])


def _build_local_response_matrix(
    *,
    wrangler,
    traversal,
    queue,
    root_box_id,
    n_local_coeffs,
    check_point_ids,
):
    if len(wrangler.tree_indep.target_kernels) != 1:
        raise NotImplementedError(
            "periodic far operator build currently supports single-output kernels"
        )

    basis_dtype = np.result_type(np.dtype(wrangler.dtype), np.complex128)
    response_matrix = np.zeros(
        (int(check_point_ids.size), int(n_local_coeffs)),
        dtype=basis_dtype,
    )

    for icoef in range(int(n_local_coeffs)):
        local_exps = wrangler.local_expansion_zeros()
        local_oa = _as_obj_array(local_exps)

        local_values = local_oa[0]
        basis_row = np.zeros(int(n_local_coeffs), dtype=basis_dtype)
        basis_row[icoef] = 1

        if isinstance(local_values, cl.array.Array):
            local_values_updated = local_values.copy()
            local_values_updated = _inject_root_expansion_row(
                local_values_updated,
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="local",
                row_values=basis_row,
                queue=queue,
            )
            local_oa[0] = local_values_updated
        else:
            local_values_updated = np.asarray(local_values).copy()
            local_values_updated = _inject_root_expansion_row(
                local_values_updated,
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="local",
                row_values=basis_row,
                queue=queue,
            )
            local_oa[0] = local_values_updated

        local_exps = _coerce_obj_array_to_template(local_oa, local_exps)
        local_exps, _ = wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps,
        )
        local_result, _ = wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps,
        )

        local_eval = _as_obj_array(local_result)[0]
        if isinstance(local_eval, cl.array.Array):
            local_eval_host = local_eval.get(queue)
        else:
            local_eval_host = np.asarray(local_eval)

        response_matrix[:, icoef] = np.asarray(local_eval_host)[check_point_ids]

    return response_matrix


def _eval_shifted_p2p_on_targets(
    *,
    wrangler,
    traversal,
    src_strength,
    shift_vectors,
    target_point_ids,
    queue,
):
    if len(wrangler.tree_indep.target_kernels) != 1:
        raise NotImplementedError(
            "periodic shifted P2P evaluation currently supports one output kernel"
        )

    from sumpy import P2P

    tree = traversal.tree
    actx = wrangler.tree_indep._setup_actx
    targets_eval = _take_point_cloud_subset(tree.targets, target_point_ids, queue)

    p2p = P2P(
        wrangler.tree_indep.target_kernels,
        exclude_self=False,
        value_dtypes=[wrangler.dtype],
    )

    p2p_exclude_self = None
    target_to_source_subset_dev = None
    use_self_exclusion_for_zero_shift = bool(
        getattr(wrangler.tree_indep, "exclude_self", False)
        and getattr(tree, "sources_are_targets", False)
    )

    if use_self_exclusion_for_zero_shift:
        p2p_exclude_self = P2P(
            wrangler.tree_indep.target_kernels,
            exclude_self=True,
            value_dtypes=[wrangler.dtype],
        )
        target_to_source_subset_dev = cl.array.to_device(
            queue,
            np.asarray(target_point_ids, dtype=np.int32),
        )

    p2p_kwargs = {}
    source_extra_kwargs = getattr(wrangler, "source_extra_kwargs", None)
    if source_extra_kwargs is not None:
        p2p_kwargs.update(source_extra_kwargs)

    kernel_extra_kwargs = getattr(wrangler, "kernel_extra_kwargs", None)
    if kernel_extra_kwargs is not None:
        p2p_kwargs.update(kernel_extra_kwargs)

    p2p_kwargs.pop("target_to_source", None)

    if isinstance(tree.sources, np.ndarray) and tree.sources.dtype == object:
        source_axes = list(tree.sources)
    else:
        source_axes = list(tree.sources)

    source_axes = [
        axis.with_queue(queue)
        if isinstance(axis, cl.array.Array) and hasattr(axis, "with_queue")
        else axis
        for axis in source_axes
    ]

    total = None
    for shift in shift_vectors:
        shift = np.asarray(shift, dtype=np.float64)
        is_zero_shift = bool(np.all(shift == 0))
        shifted_sources = obj_array_1d(
            [
                axis + _cast_scalar_to_axis_dtype(shift[d], axis)
                for d, axis in enumerate(source_axes)
            ]
        )

        if is_zero_shift and p2p_exclude_self is not None:
            p2p_eval = p2p_exclude_self
            p2p_eval_kwargs = dict(p2p_kwargs)
            p2p_eval_kwargs["target_to_source"] = target_to_source_subset_dev
        else:
            p2p_eval = p2p
            p2p_eval_kwargs = p2p_kwargs

        (shifted_values,) = p2p_eval(
            actx,
            targets_eval,
            shifted_sources,
            (src_strength,),
            **p2p_eval_kwargs,
        )

        if total is None:
            total = shifted_values
        else:
            total = total + shifted_values

    if total is None:
        total = np.zeros(int(target_point_ids.size), dtype=np.dtype(wrangler.dtype))

    if isinstance(total, cl.array.Array):
        return total.get(queue)

    return np.asarray(total)


def build_periodic_far_operator(
    traversal,
    expansion_wrangler,
    *,
    periodic_cell_size=None,
    periodic_near_shifts="nearest",
    far_shift_radius=3,
    n_training_samples=None,
    rng_seed=13,
    max_check_points=None,
    spectral_k_max_2d=None,
    spectral_k_max_3d=None,
    periodic_near_target_boxes=None,
    queue=None,
    table_manager=None,
    force_recompute=False,
):
    wrangler = expansion_wrangler
    if not isinstance(wrangler, FPNDSumpyExpansionWrangler):
        raise NotImplementedError(
            "periodic far operator build currently supports only sumpy wranglers"
        )

    if queue is None:
        queue = getattr(wrangler, "queue", None)
    if queue is None:
        raise TypeError("unable to infer command queue for periodic far operator build")

    tree = traversal.tree
    dim = int(getattr(tree, "dimensions", 0))
    if dim not in (2, 3):
        raise NotImplementedError("periodic far operator build supports 2D/3D only")

    if len(wrangler.tree_indep.target_kernels) != 1:
        raise NotImplementedError(
            "periodic far operator build currently supports a single output kernel"
        )

    target_kernel = wrangler.tree_indep.target_kernels[0]

    root_box_id, _ = _find_root_box_id(tree, queue)

    periodic_cell_size = _normalize_periodic_cell_size(periodic_cell_size, dim, tree)
    near_shift_indices = _normalize_periodic_near_shifts(periodic_near_shifts, dim)
    near_shift_vectors = _periodic_shift_vectors_from_indices(
        near_shift_indices,
        periodic_cell_size,
    )

    laplace_kernel_dim = _laplace_target_kernel_dim(target_kernel)
    use_spectral_tail = laplace_kernel_dim == dim and dim in {2, 3}

    spectral_k_max = None
    if use_spectral_tail:
        if dim == 2:
            if spectral_k_max_2d is None:
                spectral_k_max_2d = max(24, 8 * int(far_shift_radius))
            spectral_k_max = int(spectral_k_max_2d)
            if spectral_k_max < 1:
                raise ValueError("spectral_k_max_2d must be >= 1")
        else:
            if spectral_k_max_3d is None:
                spectral_k_max_3d = max(10, 4 * int(far_shift_radius))
            spectral_k_max = int(spectral_k_max_3d)
            if spectral_k_max < 1:
                raise ValueError("spectral_k_max_3d must be >= 1")

        tail_model = f"spectral_{dim}d"
        far_shift_indices = np.empty((0, dim), dtype=np.int64)
        far_shift_vectors = np.empty((0, dim), dtype=np.float64)
    else:
        tail_model = "finite_stencil"
        spectral_k_max_2d = None
        spectral_k_max_3d = None
        far_shift_indices = _generate_periodic_far_shift_indices(
            dim,
            far_shift_radius,
            near_shift_indices,
        )
        far_shift_vectors = _periodic_shift_vectors_from_indices(
            far_shift_indices,
            periodic_cell_size,
        )

    mpole_template = wrangler.multipole_expansion_zeros()
    local_template = wrangler.local_expansion_zeros()
    mpole_oa = _as_obj_array(mpole_template)
    local_oa = _as_obj_array(local_template)

    mpole_values = mpole_oa[0]
    local_values = local_oa[0]

    n_multipole_coeffs = int(
        _extract_root_expansion_row_host(
            mpole_values,
            wrangler=wrangler,
            root_box_id=root_box_id,
            expansion_kind="multipole",
            queue=queue,
        ).size
    )
    n_local_coeffs = int(
        _extract_root_expansion_row_host(
            local_values,
            wrangler=wrangler,
            root_box_id=root_box_id,
            expansion_kind="local",
            queue=queue,
        ).size
    )

    nsources = int(getattr(tree, "nsources", 0))
    if nsources <= 0:
        raise ValueError("periodic far operator build requires at least one source")

    rng = np.random.default_rng(int(rng_seed))

    default_max_check_points = max(4 * n_local_coeffs, 256)
    if max_check_points is None:
        max_check_points = default_max_check_points

    check_point_ids = _select_periodic_check_point_ids(
        tree,
        queue,
        periodic_near_target_boxes=periodic_near_target_boxes,
        max_check_points=max_check_points,
        rng=rng,
    )
    n_check_points = int(check_point_ids.size)
    if n_check_points < n_local_coeffs:
        raise ValueError(
            "insufficient check points for periodic far operator build: "
            f"need at least {n_local_coeffs}, got {n_check_points}"
        )

    if n_training_samples is None:
        n_training_samples = max(2 * n_multipole_coeffs, n_multipole_coeffs + 16)
    n_training_samples = int(n_training_samples)
    if n_training_samples < 1:
        raise ValueError(
            f"n_training_samples must be a positive integer (got {n_training_samples})"
        )

    cache_config = {
        "kind": "periodic_far_operator",
        "dim": dim,
        "kernel_repr": repr(target_kernel),
        "dtype": np.dtype(wrangler.dtype).str,
        "tail_model": tail_model,
        "n_multipole_coeffs": n_multipole_coeffs,
        "n_local_coeffs": n_local_coeffs,
        "n_sources": nsources,
        "periodic_cell_size": periodic_cell_size.tolist(),
        "near_shift_indices": near_shift_indices.tolist(),
        "far_shift_radius": int(far_shift_radius),
        "spectral_k_max_2d": spectral_k_max_2d,
        "spectral_k_max_3d": spectral_k_max_3d,
        "n_training_samples": n_training_samples,
        "max_check_points": int(max_check_points),
        "rng_seed": int(rng_seed),
    }

    def _compute_periodic_operator():
        response_matrix = _build_local_response_matrix(
            wrangler=wrangler,
            traversal=traversal,
            queue=queue,
            root_box_id=root_box_id,
            n_local_coeffs=n_local_coeffs,
            check_point_ids=check_point_ids,
        )
        response_pinv = np.linalg.pinv(response_matrix, rcond=1e-12)

        response_rank = int(np.linalg.matrix_rank(response_matrix))
        if response_rank < n_local_coeffs:
            raise RuntimeError(
                "periodic far operator build failed: local response matrix rank "
                f"{response_rank} < {n_local_coeffs}"
            )

        if use_spectral_tail:
            spectral_matrix = _build_spectral_periodic_laplace_matrix(
                source_points=_point_cloud_to_host_matrix(tree.sources, queue),
                target_points=_point_cloud_to_host_matrix(
                    tree.targets,
                    queue,
                    point_ids=check_point_ids,
                ),
                cell_size=periodic_cell_size,
                k_max=spectral_k_max,
            )

            source_to_far_tail = np.asarray(
                spectral_matrix,
                dtype=np.result_type(np.dtype(wrangler.dtype), np.complex128),
            )

            direct_and_near_shift_vectors = np.vstack(
                [np.zeros((1, dim), dtype=np.float64), near_shift_vectors]
            )
            source_is_device = isinstance(mpole_values, cl.array.Array)

            for isrc in range(nsources):
                source_basis = np.zeros(nsources, dtype=np.dtype(wrangler.dtype))
                source_basis[isrc] = np.dtype(wrangler.dtype).type(1)

                if source_is_device:
                    source_basis_eval = cl.array.to_device(queue, source_basis)
                else:
                    source_basis_eval = source_basis

                direct_near_potential = _eval_shifted_p2p_on_targets(
                    wrangler=wrangler,
                    traversal=traversal,
                    src_strength=source_basis_eval,
                    shift_vectors=direct_and_near_shift_vectors,
                    target_point_ids=check_point_ids,
                    queue=queue,
                )

                source_to_far_tail[:, isrc] -= np.asarray(direct_near_potential)

            operator = response_pinv @ source_to_far_tail
            residual = response_matrix @ operator - source_to_far_tail
            residual_l2 = float(np.linalg.norm(residual))

            build_info = {
                "n_check_points": n_check_points,
                "n_training_samples": n_training_samples,
                "response_rank": response_rank,
                "sample_rank": int(nsources),
                "operator_basis": "source",
                "tail_model": tail_model,
                "spectral_k_max_2d": int(spectral_k_max_2d)
                if spectral_k_max_2d is not None
                else None,
                "spectral_k_max_3d": int(spectral_k_max_3d)
                if spectral_k_max_3d is not None
                else None,
                "residual_l2": residual_l2,
            }

            return operator, build_info

        sample_matrix_m = np.zeros(
            (n_multipole_coeffs, n_training_samples),
            dtype=np.result_type(np.dtype(wrangler.dtype), np.complex128),
        )
        sample_matrix_l = np.zeros(
            (n_local_coeffs, n_training_samples),
            dtype=np.result_type(np.dtype(wrangler.dtype), np.complex128),
        )
        sample_matrix_s = np.zeros(
            (nsources, n_training_samples),
            dtype=np.result_type(np.dtype(wrangler.dtype), np.complex128),
        )

        for isample in range(n_training_samples):
            source_sample = rng.standard_normal(nsources)
            source_sample = source_sample - np.mean(source_sample)
            source_sample = np.asarray(source_sample, dtype=np.dtype(wrangler.dtype))

            if isinstance(mpole_values, cl.array.Array):
                source_sample_dev = cl.array.to_device(queue, source_sample)
            else:
                source_sample_dev = source_sample

            source_sample_oa = obj_array_1d([source_sample_dev])
            mpole_exps, _ = wrangler.form_multipoles(
                traversal.level_start_source_box_nrs,
                traversal.source_boxes,
                source_sample_oa,
            )
            mpole_exps, _ = wrangler.coarsen_multipoles(
                traversal.level_start_source_parent_box_nrs,
                traversal.source_parent_boxes,
                mpole_exps,
            )

            mpole_eval = _as_obj_array(mpole_exps)[0]
            mpole_row = _extract_root_expansion_row_host(
                mpole_eval,
                wrangler=wrangler,
                root_box_id=root_box_id,
                expansion_kind="multipole",
                queue=queue,
            )

            far_potential = _eval_shifted_p2p_on_targets(
                wrangler=wrangler,
                traversal=traversal,
                src_strength=source_sample_dev,
                shift_vectors=far_shift_vectors,
                target_point_ids=check_point_ids,
                queue=queue,
            )

            local_coeffs = response_pinv @ far_potential

            sample_matrix_m[:, isample] = mpole_row
            sample_matrix_l[:, isample] = local_coeffs
            sample_matrix_s[:, isample] = source_sample

        sample_rank = int(np.linalg.matrix_rank(sample_matrix_m))

        operator_basis = "multipole"
        if sample_rank >= 1:
            operator = sample_matrix_l @ np.linalg.pinv(sample_matrix_m, rcond=1e-12)
            residual = operator @ sample_matrix_m - sample_matrix_l
        else:
            logger.warning(
                "periodic far operator build: root multipole samples are rank-0; "
                "falling back to source-space operator basis"
            )
            operator_basis = "source"
            operator = np.zeros(
                (n_local_coeffs, nsources),
                dtype=np.result_type(np.dtype(wrangler.dtype), np.complex128),
            )

            for isrc in range(nsources):
                source_basis = np.zeros(nsources, dtype=np.dtype(wrangler.dtype))
                source_basis[isrc] = np.dtype(wrangler.dtype).type(1)

                if isinstance(mpole_values, cl.array.Array):
                    source_basis_dev = cl.array.to_device(queue, source_basis)
                else:
                    source_basis_dev = source_basis

                far_potential = _eval_shifted_p2p_on_targets(
                    wrangler=wrangler,
                    traversal=traversal,
                    src_strength=source_basis_dev,
                    shift_vectors=far_shift_vectors,
                    target_point_ids=check_point_ids,
                    queue=queue,
                )

                operator[:, isrc] = response_pinv @ far_potential

            residual = operator @ sample_matrix_s - sample_matrix_l

        residual_l2 = float(np.linalg.norm(residual))

        build_info = {
            "n_check_points": n_check_points,
            "n_training_samples": n_training_samples,
            "response_rank": response_rank,
            "sample_rank": sample_rank,
            "operator_basis": operator_basis,
            "tail_model": "finite_stencil",
            "residual_l2": residual_l2,
        }

        return operator, build_info

    if table_manager is not None:
        operator, is_recomputed, build_info = table_manager.get_periodic_far_operator(
            cache_config,
            force_recompute=force_recompute,
            compute_callback=_compute_periodic_operator,
        )
        build_info = dict(build_info)
        build_info["is_recomputed"] = bool(is_recomputed)
        return operator, build_info

    operator, build_info = _compute_periodic_operator()
    build_info = dict(build_info)
    build_info["is_recomputed"] = True
    return operator, build_info


class _CombinedTimingFuture:
    def __init__(self, futures):
        self.futures = [future for future in futures if future is not None]

    def __call__(self):
        total = 0.0
        for future in self.futures:
            value = future()
            if isinstance(value, (int, float)):
                total += float(value)
        return total


def _debug_nan_status(label, ary):
    if not os.environ.get("VOLUMENTIAL_DEBUG_NAN"):
        return

    if isinstance(ary, np.ndarray):
        arr = ary
    else:
        arr = ary.get()

    logger.warning(
        "%s nan=%s inf=%s maxabs=%s",
        label,
        np.isnan(arr).any(),
        np.isinf(arr).any(),
        np.nanmax(np.abs(arr)) if arr.size else 0,
    )


def _debug_sample_values(label, ary, count=8):
    if not os.environ.get("VOLUMENTIAL_DEBUG_NAN"):
        return

    if isinstance(ary, np.ndarray):
        arr = ary
    else:
        arr = ary.get()

    logger.warning("%s sample=%s", label, arr[:count])


def _contains_nonfinite(values):
    values_oa = _as_obj_array(values)
    for entry in values_oa:
        arr = entry if isinstance(entry, np.ndarray) else entry.get()
        if np.isnan(arr).any() or np.isinf(arr).any():
            return True
    return False


def _to_host_array(ary, queue):
    if ary is None:
        return None
    if hasattr(ary, "with_queue"):
        return ary.with_queue(queue).get()
    if hasattr(ary, "get"):
        try:
            return ary.get(queue)
        except TypeError:
            return ary.get()
    return np.asarray(ary)


def _build_inverse_user_permutation(user_ids, nitems, *, offset=0):
    user_ids = np.asarray(user_ids)
    if user_ids.ndim != 1 or len(user_ids) != nitems:
        return None

    local_ids = user_ids - int(offset)
    if np.any(local_ids < 0) or np.any(local_ids >= nitems):
        return None

    inv = np.full(nitems, -1, dtype=np.int64)
    inv[local_ids] = np.arange(nitems, dtype=np.int64)
    if np.any(inv < 0):
        return None

    return inv


def _coords_match_same_tree_order(tree, queue, nsources, ntargets):
    if int(nsources) != int(ntargets):
        return False

    sources = getattr(tree, "sources", None)
    targets = getattr(tree, "targets", None)
    dimensions = getattr(tree, "dimensions", None)
    if sources is None or targets is None or dimensions is None:
        return False

    for iaxis in range(int(dimensions)):
        source_coords = np.asarray(_to_host_array(sources[iaxis], queue))
        target_coords = np.asarray(_to_host_array(targets[iaxis], queue))
        if source_coords.shape != target_coords.shape:
            return False

        coord_dtype = np.result_type(source_coords.dtype, target_coords.dtype)
        if np.issubdtype(coord_dtype, np.floating):
            atol = max(1e-12, 64 * float(np.finfo(coord_dtype).eps))
        else:
            atol = 0.0

        if not np.allclose(source_coords, target_coords, rtol=0.0, atol=atol):
            return False

    return True


def _looks_like_coincident_source_target_setup(tree, queue):
    if tree is None or getattr(tree, "sources_are_targets", False):
        return False

    nsources = getattr(tree, "nsources", None)
    ntargets = getattr(tree, "ntargets", None)
    if nsources is None or ntargets is None:
        return False

    if int(nsources) != int(ntargets):
        return False

    user_source_ids = _to_host_array(getattr(tree, "user_source_ids", None), queue)
    user_target_ids = _to_host_array(getattr(tree, "user_target_ids", None), queue)
    if user_source_ids is None or user_target_ids is None:
        return _coords_match_same_tree_order(tree, queue, nsources, ntargets)

    user_source_ids = np.asarray(user_source_ids)
    user_target_ids = np.asarray(user_target_ids)

    if user_source_ids.shape != user_target_ids.shape:
        return False

    if np.array_equal(user_source_ids, user_target_ids):
        return _coords_match_same_tree_order(tree, queue, nsources, ntargets)

    sources = getattr(tree, "sources", None)
    targets = getattr(tree, "targets", None)
    dimensions = getattr(tree, "dimensions", None)
    if sources is None or targets is None or dimensions is None:
        return False

    source_inv = _build_inverse_user_permutation(
        user_source_ids, int(nsources), offset=0
    )
    if source_inv is None:
        return False

    target_offset = int(np.min(user_target_ids))
    target_inv = _build_inverse_user_permutation(
        user_target_ids,
        int(ntargets),
        offset=target_offset,
    )
    if target_inv is None:
        return False

    for iaxis in range(int(dimensions)):
        source_coords = _to_host_array(sources[iaxis], queue)
        target_coords = _to_host_array(targets[iaxis], queue)

        source_sorted = np.asarray(source_coords)[source_inv]
        target_sorted = np.asarray(target_coords)[target_inv]
        if source_sorted.shape != target_sorted.shape:
            return False

        coord_dtype = np.result_type(source_sorted.dtype, target_sorted.dtype)
        if np.issubdtype(coord_dtype, np.floating):
            atol = max(1e-12, 64 * float(np.finfo(coord_dtype).eps))
        else:
            atol = 0.0

        if not np.allclose(source_sorted, target_sorted, rtol=0.0, atol=atol):
            return False

    return True


def _maybe_guard_coincident_source_target_tree(tree, queue, cache=None):
    global _COINCIDENT_TREE_WARNING_EMITTED

    cache_key = None
    if cache is not None:
        cache_key = id(tree)
        looks_like_coincident = cache.get(cache_key)
    else:
        looks_like_coincident = None

    if looks_like_coincident is None:
        looks_like_coincident = _looks_like_coincident_source_target_setup(tree, queue)
        if cache is not None:
            cache[cache_key] = looks_like_coincident

    if not looks_like_coincident:
        return False

    message = (
        "tree.sources_are_targets is False, but source/target geometry appears "
        "coincident (matching counts with equivalent user-id mappings, including "
        "offset user_target_ids, or coordinate-only fallback when user_target_ids "
        "are unavailable). This usually means the same physical points were passed "
        "as separate source/target arrays, which can add avoidable interpolation "
        "error. "
        "Build the tree with targets=None when evaluating on source nodes."
    )

    if _env_flag_enabled("VOLUMENTIAL_STRICT_SOURCE_TARGET_TREE"):
        raise ValueError(message)

    if not _COINCIDENT_TREE_WARNING_EMITTED:
        logger.warning(message)
        _COINCIDENT_TREE_WARNING_EMITTED = True

    return True


def _clone_source_side_tree_as_targets(tree, queue):
    required_attrs = (
        "user_source_ids",
        "sources",
        "box_source_starts",
        "box_source_counts_nonchild",
        "box_source_counts_cumul",
    )
    if any(getattr(tree, name, None) is None for name in required_attrs):
        return None

    user_source_ids_host = _to_host_array(tree.user_source_ids, queue)
    if user_source_ids_host is None:
        return None

    inv_user_to_tree = np.empty_like(user_source_ids_host)
    inv_user_to_tree[user_source_ids_host] = np.arange(
        len(user_source_ids_host), dtype=user_source_ids_host.dtype
    )
    sorted_target_ids = cl.array.to_device(
        queue,
        np.ascontiguousarray(inv_user_to_tree),
    )

    source_bbox_min = getattr(tree, "box_source_bounding_box_min", None)
    source_bbox_max = getattr(tree, "box_source_bounding_box_max", None)
    target_bbox_min = (
        source_bbox_min
        if source_bbox_min is not None
        else getattr(tree, "box_target_bounding_box_min", None)
    )
    target_bbox_max = (
        source_bbox_max
        if source_bbox_max is not None
        else getattr(tree, "box_target_bounding_box_max", None)
    )

    try:
        return replace(
            tree,
            sources_are_targets=True,
            targets=tree.sources,
            target_radii=tree.source_radii,
            sorted_target_ids=sorted_target_ids,
            box_target_starts=tree.box_source_starts,
            box_target_counts_nonchild=tree.box_source_counts_nonchild,
            box_target_counts_cumul=tree.box_source_counts_cumul,
            box_target_bounding_box_min=target_bbox_min,
            box_target_bounding_box_max=target_bbox_max,
        )
    except TypeError:
        return None


def _build_source_only_wrangler(traversal, wrangler, queue):
    from boxtree.array_context import (
        PyOpenCLArrayContext as BoxtreePyOpenCLArrayContext,
    )
    from boxtree.traversal import FMMTraversalBuilder

    tree = traversal.tree

    source_only = getattr(wrangler, "_source_only_fmm_context", None)
    if source_only is not None:
        cached_queue, cached_tree_id, source_traversal, source_wrangler = source_only
        if cached_queue is queue and cached_tree_id == id(tree):
            return source_traversal, source_wrangler

    boxtree_actx = BoxtreePyOpenCLArrayContext(queue)

    source_tree = _clone_source_side_tree_as_targets(tree, queue)
    if source_tree is None:
        raise ValueError(
            "source-only solve requires cloning source-side tree metadata to "
            "a sources_are_targets tree; source-only TreeBuilder reconstruction "
            "is disabled because it produces incorrect near-field results"
        )

    trav_builder = FMMTraversalBuilder(boxtree_actx)
    source_traversal, _ = trav_builder(boxtree_actx, source_tree)

    if hasattr(wrangler, "level_orders"):
        level_orders = tuple(int(order) for order in wrangler.level_orders)

        def fmm_level_to_order(kernel, kernel_args, tree, lev):
            return level_orders[min(lev, len(level_orders) - 1)]

    else:

        def fmm_level_to_order(kernel, kernel_args, tree, lev):
            return 12

    self_extra_kwargs = getattr(wrangler, "self_extra_kwargs", None)
    if self_extra_kwargs is not None:
        self_extra_kwargs = dict(self_extra_kwargs)
        if "target_to_source" in self_extra_kwargs:
            n_targets = getattr(source_tree, "ntargets", None)
            if n_targets is None:
                targets = getattr(source_tree, "targets", None)
                if targets is not None:
                    n_targets = len(targets[0])

            if n_targets is None:
                target_to_source = self_extra_kwargs["target_to_source"]
                if hasattr(target_to_source, "size"):
                    n_targets = int(target_to_source.size)
                elif hasattr(target_to_source, "__len__"):
                    n_targets = len(target_to_source)
                elif hasattr(target_to_source, "get"):
                    n_targets = len(target_to_source.get(queue))

            if n_targets is None:
                raise ValueError(
                    "source-only tree is missing target count needed to rebuild "
                    "target_to_source mapping"
                )

            self_extra_kwargs["target_to_source"] = np.arange(
                int(n_targets), dtype=np.int32
            )

    source_wrangler_kwargs = dict(
        tree_indep=wrangler.tree_indep,
        queue=queue,
        traversal=source_traversal,
        near_field_table=wrangler.near_field_table,
        dtype=wrangler.dtype,
        fmm_level_to_order=fmm_level_to_order,
        quad_order=wrangler.quad_order,
        potential_kind=getattr(wrangler, "potential_kind", 1),
        source_extra_kwargs=getattr(wrangler, "source_extra_kwargs", None),
        kernel_extra_kwargs=getattr(wrangler, "kernel_extra_kwargs", None),
        self_extra_kwargs=self_extra_kwargs,
        list1_extra_kwargs=getattr(wrangler, "list1_extra_kwargs", None),
    )

    try:
        ctor_signature = inspect.signature(type(wrangler).__init__)
    except (TypeError, ValueError):
        ctor_signature = None

    accepts_varkw = False
    ctor_param_names = set()
    if ctor_signature is not None:
        for param_name, param in ctor_signature.parameters.items():
            if param_name == "self":
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_varkw = True
            else:
                ctor_param_names.add(param_name)

    for attr_name in (
        "translation_classes_data",
        "preprocessed_mpole_dtype",
        "helmholtz_split",
        "helmholtz_split_order",
        "helmholtz_split_smooth_quad_order",
        "helmholtz_split_term_tables",
        "helmholtz_split_order1_legacy_subtraction",
    ):
        attr_value = getattr(wrangler, attr_name, None)
        if attr_value is None:
            continue
        if accepts_varkw or attr_name in ctor_param_names:
            source_wrangler_kwargs[attr_name] = attr_value

    source_wrangler = type(wrangler)(**source_wrangler_kwargs)

    wrangler._source_only_fmm_context = (
        queue,
        id(tree),
        source_traversal,
        source_wrangler,
    )
    return source_traversal, source_wrangler


# {{{ volume FMM driver


def drive_volume_fmm(
    traversal,
    expansion_wrangler,
    src_weights,
    src_func,
    direct_evaluation=False,
    timing_data=None,
    reorder_sources=True,
    reorder_potentials=True,
    **kwargs,
):
    """
    Top-level driver routine for volume potential calculation
    via fast multiple method.

    This function, and the interface it utilizes, is adapted from boxtree/fmm.py

    The fast multipole method is a two-pass algorithm:

    1. During the fist (upward) pass, the multipole expansions for all boxes
    at all levels are formed from bottom up.

    2. In the second (downward) pass, the local expansions for all boxes
    at all levels at formed from top down.

    :arg traversal: A boxtree traversal info object.
    :arg expansion_wrangler: An object implementing the expansion
                    wrangler interface.
    :arg src_weights: Source 'density/weights/charges' time quad weights..
        Passed unmodified to *expansion_wrangler*.
    :arg src_func: Source 'density/weights/charges' function.
        Passed unmodified to *expansion_wrangler*.
    :arg reorder_sources: Whether sources are in user order (if True, sources
        are reordered into tree order before conducting FMM).
    :arg reorder_potentials: Whether potentials should be in user order (if True,
        potentials are reordered into user order before return).

    Optional periodic controls (sumpy backend, 2D/3D only):

    :arg periodic: Explicit periodic-mode gate. Defaults to ``False``.
        Periodic behavior is enabled only when ``periodic=True``.
        If ``periodic=True`` and no explicit periodic near/far controls are
        provided, defaults are ``periodic_near_shifts="nearest"`` and
        ``periodic_far_operator="auto"``.
        If ``periodic=False``, passing periodic modifier kwargs raises
        :class:`ValueError`.

    :arg periodic_near_shifts: Lattice shifts for Barnett-style near-image
        correction. Accepts an ``(nshifts, dim)`` integer array in cell units,
        ``"nearest"`` for the nearest image ring, or ``"none"``.
    :arg periodic_cell_size: Scalar or length-``dim`` vector of periodic cell
        lengths. Defaults to ``tree.root_extent`` in each dimension.
    :arg periodic_near_target_boxes: Optional subset of target box ids for
        evaluating near-image corrections.
    :arg periodic_far_operator: Precomputed periodic far operator ``T_per``
        mapping periodic far-field features to root local coefficients.
        The default feature basis is root multipole coefficients; auto-build
        may fall back to source-space features when root multipole samples are
        rank-deficient.
        May be a single matrix, a dict keyed by output-kernel repr, or
        ``"auto"``.
        For 2D/3D Laplace kernels, ``"auto"`` uses strict spectral periodic
        evaluation at runtime (without fitting a root-local ``T_per``).
        For other kernels, ``"auto"`` builds/loads ``T_per`` offline.
    :arg periodic_far_operator_basis: Optional basis hint for
        ``periodic_far_operator`` columns. Supported values are
        ``"multipole"`` and ``"source"``. When omitted, the basis is inferred
        from matrix shape. In Laplace ``periodic_far_operator="auto"`` strict
        mode, only ``None`` or ``"source"`` are accepted.
    :arg periodic_far_operator_manager: Optional
        :class:`volumential.table_manager.NearFieldInteractionTableManager`
        used for loading/storing ``T_per`` in the same SQLite cache layer as
        near-field tables.
    :arg periodic_far_operator_cache_filename: Optional cache filename used when
        ``periodic_far_operator="auto"`` and no manager is passed.
    :arg periodic_far_shift_radius: Integer L-infinity radius used when building
        ``T_per`` from shifted-image samples.
    :arg periodic_far_spectral_kmax_2d: Optional Fourier truncation for 2D
        spectral periodic auto mode. In 2D Laplace auto mode, this controls the
        strict spectral runtime solve. In other auto paths, it controls the
        spectral periodic-tail builder used during ``T_per`` fitting.
    :arg periodic_far_spectral_kmax_3d: Optional Fourier truncation for 3D
        spectral periodic auto mode. In 3D Laplace auto mode, this controls the
        strict spectral runtime solve. In other auto paths, it controls the
        spectral periodic-tail builder used during ``T_per`` fitting.
    :arg periodic_far_training_samples: Number of random source samples used to
        regress ``T_per``.
    :arg periodic_far_rng_seed: RNG seed for ``T_per`` sample generation.
    :arg periodic_far_max_check_points: Maximum number of check points used in
        local-response fitting for ``T_per``.
    :arg periodic_far_force_recompute: Rebuild ``T_per`` even when a cached
        entry exists.

    Returns the potentials computed by *expansion_wrangler*.
    """
    wrangler = expansion_wrangler
    assert issubclass(type(wrangler), ExpansionWranglerInterface)

    recorder = TimingRecorder()
    logger.info("start fmm")

    dtype = wrangler.dtype
    expected_source_count = None
    if hasattr(traversal, "tree"):
        expected_source_count = getattr(traversal.tree, "nsources", None)
        if expected_source_count is None:
            expected_source_count = getattr(traversal.tree, "ntargets", None)

    src_weights = _normalize_source_fields(
        src_weights,
        dtype,
        expected_length=expected_source_count,
        field_name="src_weights",
    )
    src_func = _normalize_source_fields(
        src_func,
        dtype,
        expected_length=expected_source_count,
        field_name="src_func",
    )

    assert (ns := len(src_weights)) == len(src_func)

    list1_only = bool(kwargs.get("list1_only", False))
    auto_interpolate_targets = bool(kwargs.pop("auto_interpolate_targets", True))

    periodic_kwarg_keys = sorted(
        key for key in kwargs if key == "periodic" or key.startswith("periodic_")
    )

    periodic = kwargs.pop("periodic", False)
    if isinstance(periodic, (bool, np.bool_)):
        periodic_enabled = bool(periodic)
    else:
        raise TypeError(
            f"periodic must be a boolean flag, got {type(periodic).__name__}"
        )

    periodic_near_shifts = kwargs.pop("periodic_near_shifts", None)
    periodic_cell_size = kwargs.pop("periodic_cell_size", None)
    periodic_near_target_boxes = kwargs.pop("periodic_near_target_boxes", None)
    periodic_far_operator = kwargs.pop("periodic_far_operator", None)
    periodic_far_operator_basis = kwargs.pop("periodic_far_operator_basis", None)
    periodic_far_operator_manager = kwargs.pop("periodic_far_operator_manager", None)
    periodic_far_operator_cache_filename = kwargs.pop(
        "periodic_far_operator_cache_filename", None
    )
    periodic_far_shift_radius = kwargs.pop("periodic_far_shift_radius", 3)
    periodic_far_spectral_kmax_2d = kwargs.pop("periodic_far_spectral_kmax_2d", None)
    periodic_far_spectral_kmax_3d = kwargs.pop("periodic_far_spectral_kmax_3d", None)
    periodic_far_training_samples = kwargs.pop("periodic_far_training_samples", None)
    periodic_far_rng_seed = kwargs.pop("periodic_far_rng_seed", 13)
    periodic_far_max_check_points = kwargs.pop("periodic_far_max_check_points", None)
    periodic_far_force_recompute = bool(
        kwargs.pop("periodic_far_force_recompute", False)
    )

    periodic_modifier_keys = [key for key in periodic_kwarg_keys if key != "periodic"]
    if not periodic_enabled and periodic_modifier_keys:
        raise ValueError(
            "periodic modifier kwargs require periodic=True. "
            "Set periodic=True to enable periodic solves. "
            f"Received with periodic=False: {', '.join(periodic_modifier_keys)}"
        )

    periodic_shift_vectors = np.empty((0, 0), dtype=np.float64)
    periodic_far_build_info = None

    if "allow_list1_p2p_fallback" in kwargs:
        raise TypeError(
            "drive_volume_fmm no longer accepts allow_list1_p2p_fallback; "
            "non-finite list1 table results now raise an error"
        )

    if ns > 1 and isinstance(expansion_wrangler, FPNDFMMLibExpansionWrangler):
        raise NotImplementedError(
            "multiple source fields are not supported with FMMLib wranglers"
        )

    if (
        ns > 1
        and isinstance(expansion_wrangler, FPNDSumpyExpansionWrangler)
        and not list1_only
    ):
        raise NotImplementedError(
            "multiple source fields are currently supported only with "
            "list1_only=True for Sumpy wranglers"
        )

    queue = None
    if isinstance(src_func[0], cl.array.Array):
        queue = src_func[0].queue
    elif isinstance(src_weights[0], cl.array.Array):
        queue = src_weights[0].queue
    elif hasattr(wrangler, "queue"):
        queue = wrangler.queue

    if isinstance(expansion_wrangler, FPNDSumpyExpansionWrangler):
        assert all(isinstance(sw, cl.array.Array) for sw in src_weights)
        assert all(isinstance(sf, cl.array.Array) for sf in src_func)

    elif isinstance(expansion_wrangler, FPNDFMMLibExpansionWrangler):
        if queue is None:
            raise TypeError("unable to infer command queue for FMMLib wrangler")

        traversal = traversal.get(queue)

        if isinstance(src_weights[0], cl.array.Array):
            src_weights = obj_array_1d([sw.get(queue) for sw in src_weights])
        if isinstance(src_func[0], cl.array.Array):
            src_func = obj_array_1d([sf.get(queue) for sf in src_func])

    tree = getattr(traversal, "tree", None)

    if periodic_enabled:
        if not isinstance(expansion_wrangler, FPNDSumpyExpansionWrangler):
            raise NotImplementedError(
                "periodic support currently requires the sumpy wrangler backend"
            )

        if tree is None:
            raise ValueError("periodic support requires traversal tree metadata")

        tree_dim = int(getattr(tree, "dimensions", 0))
        if tree_dim not in (2, 3):
            raise NotImplementedError(
                "periodic support is currently implemented for 2D and 3D only"
            )

        if ns != 1:
            raise NotImplementedError(
                "periodic support currently requires a single source field"
            )

        if periodic_near_shifts is None and periodic_far_operator is None:
            periodic_near_shifts = "nearest"
            periodic_far_operator = "auto"
            logger.info(
                "periodic=True without explicit near/far controls; using "
                "defaults periodic_near_shifts='nearest', "
                "periodic_far_operator='auto'"
            )

        if periodic_far_operator is not None and periodic_near_shifts is None:
            periodic_near_shifts = "nearest"

        periodic_shift_indices = _normalize_periodic_near_shifts(
            periodic_near_shifts,
            tree_dim,
        )
        periodic_cell_size = _normalize_periodic_cell_size(
            periodic_cell_size,
            tree_dim,
            tree,
        )
        periodic_shift_vectors = _periodic_shift_vectors_from_indices(
            periodic_shift_indices,
            periodic_cell_size,
        )

        if periodic_far_operator is not None and list1_only:
            raise ValueError(
                "periodic_far_operator requires list1_only=False because far "
                "corrections are injected through local expansions"
            )

        if isinstance(periodic_far_operator, str):
            if periodic_far_operator.strip().lower() != "auto":
                raise ValueError(
                    "periodic_far_operator string must be 'auto' when provided"
                )

            laplace_dim = _laplace_target_kernel_dim(
                wrangler.tree_indep.target_kernels[0]
            )
            use_strict_periodic_laplace_runtime = bool(
                laplace_dim == tree_dim and tree_dim in {2, 3}
            )

            if use_strict_periodic_laplace_runtime:
                if periodic_far_operator_basis not in {None, "source"}:
                    raise ValueError(
                        "periodic_far_operator='auto' for 2D/3D Laplace uses "
                        "strict spectral runtime mode and only supports "
                        "periodic_far_operator_basis=None or 'source'"
                    )

                spectral_k_max = _infer_periodic_spectral_kmax(
                    tree_dim=tree_dim,
                    periodic_far_build_info=None,
                    periodic_far_spectral_kmax_2d=periodic_far_spectral_kmax_2d,
                    periodic_far_spectral_kmax_3d=periodic_far_spectral_kmax_3d,
                    periodic_far_shift_radius=periodic_far_shift_radius,
                )

                periodic_far_operator = np.zeros((1, 1), dtype=np.dtype(wrangler.dtype))
                periodic_far_build_info = {
                    "is_recomputed": False,
                    "operator_basis": "source",
                    "tail_model": f"spectral_{tree_dim}d",
                    "spectral_k_max_2d": spectral_k_max if tree_dim == 2 else None,
                    "spectral_k_max_3d": spectral_k_max if tree_dim == 3 else None,
                    "runtime_mode": "strict_spectral",
                }

                periodic_far_operator_basis = "source"

                logger.info(
                    "periodic Laplace auto mode: using strict spectral runtime "
                    "(skipping root-local periodic operator build)"
                )

            else:
                if periodic_far_operator_manager is None:
                    if periodic_far_operator_cache_filename is not None:
                        from volumential.table_manager import (
                            NearFieldInteractionTableManager,
                        )

                        with NearFieldInteractionTableManager(
                            periodic_far_operator_cache_filename,
                            root_extent=float(tree.root_extent),
                            read_only="auto",
                        ) as periodic_cache_manager:
                            periodic_far_operator, periodic_far_build_info = (
                                build_periodic_far_operator(
                                    traversal,
                                    wrangler,
                                    periodic_cell_size=periodic_cell_size,
                                    periodic_near_shifts=periodic_shift_indices,
                                    far_shift_radius=periodic_far_shift_radius,
                                    n_training_samples=periodic_far_training_samples,
                                    rng_seed=periodic_far_rng_seed,
                                    max_check_points=periodic_far_max_check_points,
                                    spectral_k_max_2d=periodic_far_spectral_kmax_2d,
                                    spectral_k_max_3d=periodic_far_spectral_kmax_3d,
                                    periodic_near_target_boxes=periodic_near_target_boxes,
                                    queue=queue,
                                    table_manager=periodic_cache_manager,
                                    force_recompute=periodic_far_force_recompute,
                                )
                            )
                    else:
                        periodic_far_operator, periodic_far_build_info = (
                            build_periodic_far_operator(
                                traversal,
                                wrangler,
                                periodic_cell_size=periodic_cell_size,
                                periodic_near_shifts=periodic_shift_indices,
                                far_shift_radius=periodic_far_shift_radius,
                                n_training_samples=periodic_far_training_samples,
                                rng_seed=periodic_far_rng_seed,
                                max_check_points=periodic_far_max_check_points,
                                spectral_k_max_2d=periodic_far_spectral_kmax_2d,
                                spectral_k_max_3d=periodic_far_spectral_kmax_3d,
                                periodic_near_target_boxes=periodic_near_target_boxes,
                                queue=queue,
                                table_manager=None,
                                force_recompute=True,
                            )
                        )
                else:
                    periodic_far_operator, periodic_far_build_info = (
                        build_periodic_far_operator(
                            traversal,
                            wrangler,
                            periodic_cell_size=periodic_cell_size,
                            periodic_near_shifts=periodic_shift_indices,
                            far_shift_radius=periodic_far_shift_radius,
                            n_training_samples=periodic_far_training_samples,
                            rng_seed=periodic_far_rng_seed,
                            max_check_points=periodic_far_max_check_points,
                            spectral_k_max_2d=periodic_far_spectral_kmax_2d,
                            spectral_k_max_3d=periodic_far_spectral_kmax_3d,
                            periodic_near_target_boxes=periodic_near_target_boxes,
                            queue=queue,
                            table_manager=periodic_far_operator_manager,
                            force_recompute=periodic_far_force_recompute,
                        )
                    )

            if periodic_far_operator_basis is None:
                periodic_far_operator_basis = periodic_far_build_info.get(
                    "operator_basis"
                )

            logger.info(
                "periodic far operator build: recomputed=%s, n_check_points=%s, "
                "n_training_samples=%s, residual_l2=%s",
                periodic_far_build_info.get("is_recomputed", True),
                periodic_far_build_info.get("n_check_points", "n/a"),
                periodic_far_build_info.get("n_training_samples", "n/a"),
                periodic_far_build_info.get("residual_l2", "n/a"),
            )

    coincident_tree_guard_cache = getattr(
        wrangler, "_coincident_tree_guard_cache", None
    )
    if coincident_tree_guard_cache is None:
        coincident_tree_guard_cache = {}
        wrangler._coincident_tree_guard_cache = coincident_tree_guard_cache

    if tree is not None and not getattr(tree, "sources_are_targets", False):
        _maybe_guard_coincident_source_target_tree(
            tree,
            queue,
            cache=coincident_tree_guard_cache,
        )

    if (
        auto_interpolate_targets
        and isinstance(expansion_wrangler, FPNDSumpyExpansionWrangler)
        and not direct_evaluation
        and tree is not None
        and not getattr(tree, "sources_are_targets", False)
    ):
        if periodic_enabled:
            raise NotImplementedError(
                "periodic mode currently requires coincident source/target trees; "
                "set auto_interpolate_targets=False and use sources_are_targets=True"
            )

        logger.info(
            "non-coincident source/target tree detected; "
            "solving on source modes and interpolating to requested targets"
        )

        source_traversal, source_wrangler = _build_source_only_wrangler(
            traversal, wrangler, queue
        )

        source_result = drive_volume_fmm(
            source_traversal,
            source_wrangler,
            src_weights,
            src_func,
            direct_evaluation=direct_evaluation,
            timing_data=timing_data,
            reorder_sources=reorder_sources,
            reorder_potentials=False,
            auto_interpolate_targets=False,
            **kwargs,
        )

        source_result_oa = _as_obj_array(source_result)
        interpolated = []
        for source_result_i in source_result_oa:
            interpolated_i = interpolate_volume_potential(
                tree.targets,
                source_traversal,
                source_wrangler,
                source_result_i,
                potential_in_tree_order=True,
                use_mode_to_source_ids=True,
            )
            interpolated.append(interpolated_i)

        if len(interpolated) == 1:
            result = obj_array_1d([interpolated[0]])
        else:
            result = obj_array_1d(interpolated)

        if reorder_potentials:
            logger.debug("reorder interpolated potentials")
            result = wrangler.reorder_potentials(result)

        logger.debug("finalize interpolated potentials")
        return wrangler.finalize_potentials(result)

    if reorder_sources:
        logger.debug("reorder source weights")
        for idx_s in range(ns):
            src_weights[idx_s] = wrangler.reorder_sources(src_weights[idx_s])
            src_func[idx_s] = wrangler.reorder_sources(src_func[idx_s])

    strict_periodic_laplace = False
    tree_dim = int(getattr(tree, "dimensions", 0)) if tree is not None else 0
    laplace_dim = None
    if periodic_enabled and periodic_far_operator is not None and tree is not None:
        laplace_dim = _laplace_target_kernel_dim(wrangler.tree_indep.target_kernels[0])
        tail_model = None
        runtime_mode = None
        if periodic_far_build_info is not None:
            tail_model = periodic_far_build_info.get("tail_model")
            runtime_mode = periodic_far_build_info.get("runtime_mode")

        strict_periodic_laplace = bool(
            ns == 1
            and laplace_dim == tree_dim
            and tree_dim in {2, 3}
            and (
                runtime_mode == "strict_spectral"
                or (
                    periodic_far_operator_basis == "source"
                    and isinstance(tail_model, str)
                    and tail_model.startswith("spectral_")
                )
            )
        )

    if strict_periodic_laplace:
        k_max = _infer_periodic_spectral_kmax(
            tree_dim=tree_dim,
            periodic_far_build_info=periodic_far_build_info,
            periodic_far_spectral_kmax_2d=periodic_far_spectral_kmax_2d,
            periodic_far_spectral_kmax_3d=periodic_far_spectral_kmax_3d,
            periodic_far_shift_radius=periodic_far_shift_radius,
        )

        result = _strict_periodic_laplace_potential(
            traversal=traversal,
            wrangler=wrangler,
            src_weights=src_weights,
            periodic_cell_size=periodic_cell_size,
            k_max=k_max,
            queue=queue,
        )

        if reorder_potentials:
            logger.debug("reorder strict periodic spectral potentials")
            result = wrangler.reorder_potentials(result)

        logger.debug("finalize strict periodic spectral potentials")
        result = wrangler.finalize_potentials(result)
        logger.info("periodic Laplace solve completed via strict spectral evaluation")
        return result

    # {{{ Construct local multipoles

    logger.debug("construct local multipoles")
    mpole_exps, timing_future = wrangler.form_multipoles(
        traversal.level_start_source_box_nrs, traversal.source_boxes, src_weights
    )
    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ Propagate multipoles upward

    logger.debug("propagate multipoles upward")
    mpole_exps, timing_future = wrangler.coarsen_multipoles(
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        mpole_exps,
    )
    recorder.add("coarsen_multipoles", timing_future)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ (Optional) direct evaluation of everything and return
    if direct_evaluation:
        if ns > 1:
            raise NotImplementedError(
                "direct_evaluation=True with multiple source fields is not supported"
            )

        print("Warning: NOT USING FMM (forcing global p2p)")
        if len(src_weights) != len(src_func):
            print(
                "Using P2P with different src/tgt discretizations can be "
                "unstable when targets are close to the sources while not "
                "be exactly the same"
            )

        from sumpy import P2P

        p2p = P2P(
            wrangler.tree_indep.target_kernels,
            wrangler.tree_indep.exclude_self,
            value_dtypes=[wrangler.dtype],
        )

        p2p_extra_kwargs = {}
        source_extra_kwargs = getattr(wrangler, "source_extra_kwargs", None)
        if source_extra_kwargs is not None:
            p2p_extra_kwargs.update(source_extra_kwargs)

        kernel_extra_kwargs = getattr(wrangler, "kernel_extra_kwargs", None)
        if kernel_extra_kwargs is not None:
            p2p_extra_kwargs.update(kernel_extra_kwargs)

        p2p_queue = wrangler.tree_indep._setup_actx.queue

        if queue is None:
            raise TypeError("unable to infer command queue for direct evaluation")

        (sw,) = src_weights
        target_to_source = cl.array.to_device(
            p2p_queue,
            np.arange(traversal.tree.ntargets, dtype=np.int32),
        )
        p2p_kwargs = dict(p2p_extra_kwargs)
        if wrangler.tree_indep.exclude_self:
            p2p_kwargs["target_to_source"] = target_to_source

        (ref_pot,) = p2p(
            wrangler.tree_indep._setup_actx,
            traversal.tree.targets,
            traversal.tree.sources,
            (sw,),
            **p2p_kwargs,
        )

        if isinstance(wrangler, FPNDSumpyExpansionWrangler):
            potentials = obj_array_1d([ref_pot])
        else:
            potentials = ref_pot

        if periodic_enabled and periodic_shift_vectors.size:
            periodic_near_potentials, periodic_near_timing = (
                _eval_periodic_near_stencil_p2p(
                    traversal=traversal,
                    wrangler=wrangler,
                    src_weights=src_weights,
                    shift_vectors=periodic_shift_vectors,
                    periodic_near_target_boxes=periodic_near_target_boxes,
                    queue=queue,
                )
            )
            if periodic_near_timing is not None:
                recorder.add("eval_periodic_near", periodic_near_timing)
            potentials = _add_obj_arrays(potentials, periodic_near_potentials)

        if periodic_enabled and periodic_far_operator is not None:
            periodic_local_exps = wrangler.local_expansion_zeros()
            periodic_local_exps = _apply_periodic_far_operator_to_locals(
                local_exps=periodic_local_exps,
                mpole_exps=mpole_exps,
                source_strength=src_weights[0],
                traversal=traversal,
                wrangler=wrangler,
                periodic_far_operator=periodic_far_operator,
                periodic_far_operator_basis=periodic_far_operator_basis,
                queue=queue,
            )
            periodic_local_exps, periodic_refine_timing = wrangler.refine_locals(
                traversal.level_start_target_or_target_parent_box_nrs,
                traversal.target_or_target_parent_boxes,
                periodic_local_exps,
            )
            if periodic_refine_timing is not None:
                recorder.add("refine_periodic_locals", periodic_refine_timing)

            periodic_local_result, periodic_eval_timing = wrangler.eval_locals(
                traversal.level_start_target_box_nrs,
                traversal.target_boxes,
                periodic_local_exps,
            )
            if periodic_eval_timing is not None:
                recorder.add("eval_periodic_locals", periodic_eval_timing)
            potentials = _add_obj_arrays(potentials, periodic_local_result)

        _debug_nan_status("global_p2p", potentials)

        assert traversal.from_sep_close_smaller_starts is None
        assert traversal.from_sep_close_bigger_starts is None

        result = potentials
        if reorder_potentials:
            logger.debug("reorder potentials")
            result = wrangler.reorder_potentials(result)

        logger.debug("finalize potentials")
        result = wrangler.finalize_potentials(result)

        logger.info("direct p2p complete")

        return result

    # }}} End Stage

    # {{{ Direct evaluation from neighbor source boxes ("list 1")

    logger.debug("direct evaluation from neighbor source boxes ('list 1')")
    # look up in the prebuilt table
    # this step also constructs the output array
    direct_timing_futures = []
    potentials = None
    for idx_s, field in enumerate(src_func):
        field_potentials, timing_future = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            field,
        )

        if _contains_nonfinite(field_potentials):
            raise RuntimeError(
                "table-based list1 evaluation produced non-finite values for "
                f"field {idx_s}; check near-field table data integrity"
            )

        direct_timing_futures.append(timing_future)

        if (
            hasattr(wrangler, "helmholtz_split")
            and getattr(wrangler, "helmholtz_split")
            and hasattr(wrangler, "eval_direct_helmholtz_split_correction")
        ):
            correction, correction_timing_future = (
                wrangler.eval_direct_helmholtz_split_correction(
                    traversal.target_boxes,
                    traversal.neighbor_source_boxes_starts,
                    traversal.neighbor_source_boxes_lists,
                    src_weights[idx_s],
                    src_func=field,
                )
            )
            if correction_timing_future is not None:
                direct_timing_futures.append(correction_timing_future)
            correction = _coerce_obj_array_like(correction, field_potentials, queue)
            if _contains_nonfinite(correction):
                raise RuntimeError(
                    "Helmholtz split list1 correction produced non-finite values "
                    f"for field {idx_s}"
                )
            field_potentials = _add_obj_arrays(field_potentials, correction)
            if _contains_nonfinite(field_potentials):
                raise RuntimeError(
                    "combined list1 potential produced non-finite values after "
                    f"Helmholtz split correction for field {idx_s}"
                )

        if potentials is None:
            potentials = field_potentials
        else:
            potentials = _add_obj_arrays(potentials, field_potentials)

    assert potentials is not None
    timing_future = _CombinedTimingFuture(direct_timing_futures)
    recorder.add("eval_direct", timing_future)
    _debug_nan_status("fmm_l1_potentials", _as_obj_array(potentials)[0])

    if periodic_enabled and periodic_shift_vectors.size:
        periodic_near_potentials, periodic_near_timing = (
            _eval_periodic_near_stencil_p2p(
                traversal=traversal,
                wrangler=wrangler,
                src_weights=src_weights,
                shift_vectors=periodic_shift_vectors,
                periodic_near_target_boxes=periodic_near_target_boxes,
                queue=queue,
            )
        )
        if periodic_near_timing is not None:
            recorder.add("eval_periodic_near", periodic_near_timing)
        potentials = _add_obj_arrays(potentials, periodic_near_potentials)
        _debug_nan_status("fmm_periodic_near_potentials", _as_obj_array(potentials)[0])

    # Return list 1 only, for debugging
    # 'list1_only' takes precedence over 'exclude_list1'
    if list1_only:
        result = potentials
        if reorder_potentials:
            logger.debug("reorder potentials")
            result = wrangler.reorder_potentials(potentials)

        logger.debug("finalize potentials")
        result = wrangler.finalize_potentials(result)

        logger.info("fmm complete with list 1 only")
        logger.info("fmm complete")
        logger.warning("only list 1 results are returned")

        return result

    # Do not include list 1
    if "exclude_list1" in kwargs and kwargs["exclude_list1"]:
        logger.info("Using zeros for list 1")
        logger.warning("list 1 interactions are not included")
        potentials = wrangler.output_zeros()

    # these potentials are called alpha in [1]

    # }}}

    # {{{ Translate separated siblings' ("list 2") mpoles to local

    logger.debug("translate separated siblings' ('list 2') mpoles to local")
    local_exps, timing_future = wrangler.multipole_to_local(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        mpole_exps,
    )

    if periodic_enabled and periodic_far_operator is not None:
        local_exps = _apply_periodic_far_operator_to_locals(
            local_exps=local_exps,
            mpole_exps=mpole_exps,
            source_strength=src_weights[0],
            traversal=traversal,
            wrangler=wrangler,
            periodic_far_operator=periodic_far_operator,
            periodic_far_operator_basis=periodic_far_operator_basis,
            queue=queue,
        )

    _debug_nan_status("local_exps_after_m2l", local_exps[0])
    recorder.add("multipole_to_local", timing_future)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ Evaluate sep. smaller mpoles ("list 3") at particles

    logger.debug("evaluate sep. smaller mpoles at particles ('list 3 far')")

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
        traversal.target_boxes_sep_smaller_by_source_level,
        traversal.from_sep_smaller_by_level,
        mpole_exps,
    )
    _debug_nan_status("mpole_result", mpole_result[0])
    recorder.add("eval_multipoles", timing_future)

    potentials = _add_obj_arrays(potentials, mpole_result)
    _debug_nan_status("potentials_after_mpole_eval", _as_obj_array(potentials)[0])

    # these potentials are called beta in [1]

    # volume fmm does not work with list 3 close currently
    # but list 3 should be empty with our use cases
    assert traversal.from_sep_close_smaller_starts is None
    # if traversal.from_sep_close_smaller_starts is not None:
    #    logger.debug("evaluate separated close smaller interactions directly "
    #                 "('list 3 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_boxes, traversal.from_sep_close_smaller_starts,
    #        traversal.from_sep_close_smaller_lists, src_weights)

    # }}}

    # {{{ Form locals for separated bigger source boxes ("list 4")

    logger.debug("form locals for separated bigger source boxes ('list 4 far')")

    local_result, timing_future = wrangler.form_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts,
        traversal.from_sep_bigger_lists,
        src_weights,
    )
    _debug_nan_status("local_result", local_result[0])

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    # volume fmm does not work with list 4 currently
    assert traversal.from_sep_close_bigger_starts is None
    # if traversal.from_sep_close_bigger_starts is not None:
    #    logger.debug("evaluate separated close bigger interactions directly "
    #                 "('list 4 close')")

    #    potentials = potentials + wrangler.eval_direct(
    #        traversal.target_or_target_parent_boxes,
    #        traversal.from_sep_close_bigger_starts,
    #        traversal.from_sep_close_bigger_lists, src_weights)

    # }}}

    # {{{ Propagate local_exps downward

    logger.debug("propagate local_exps downward")
    # import numpy.linalg as la

    local_exps, timing_future = wrangler.refine_locals(
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        local_exps,
    )
    _debug_nan_status("local_exps_after_refine", local_exps[0])

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ Evaluate locals

    logger.debug("evaluate locals")

    local_result, timing_future = wrangler.eval_locals(
        traversal.level_start_target_box_nrs, traversal.target_boxes, local_exps
    )
    _debug_nan_status("local_eval", local_result[0])

    recorder.add("eval_locals", timing_future)

    potentials = _add_obj_arrays(potentials, local_result)
    _debug_nan_status("potentials_after_local_eval", _as_obj_array(potentials)[0])

    # }}}

    # {{{ Reorder potentials
    result = potentials
    if reorder_potentials:
        logger.debug("reorder potentials")
        result = wrangler.reorder_potentials(potentials)

    logger.debug("finalize potentials")
    result = wrangler.finalize_potentials(result)

    logger.info("fmm complete")
    # }}} End Reorder potentials

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result


# }}} End volume FMM driver


# {{{ free form interpolation of potentials


def compute_barycentric_lagrange_params(q_order):

    # 1d quad points and weights
    q_points_1d, q_weights_1d = np.polynomial.legendre.leggauss(q_order)
    q_points_1d = (q_points_1d + 1) * 0.5
    q_weights_1d *= 0.5

    if q_order == 1:
        return (
            np.asarray(q_points_1d, dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )

    # interpolation weights for barycentric Lagrange interpolation
    from scipy.interpolate import BarycentricInterpolator as Interpolator

    interp = Interpolator(xi=q_points_1d, yi=None)
    interp_weights = interp.wi
    interp_points = interp.xi

    return (interp_points, interp_weights)


def _infer_tree_local_interp_points_1d(tree, traversal, q_order, queue):
    """Infer per-box 1D interpolation nodes from tree geometry.

    This is used for source-only trees built from source samples only, where the
    tree root bounding box may differ from the original source-grid root box.
    In that case, the local coordinates of source samples in a leaf box are not
    the canonical Legendre-Gauss points.
    """
    if not hasattr(traversal, "target_boxes"):
        return None

    target_boxes = traversal.target_boxes
    if hasattr(target_boxes, "with_queue"):
        target_boxes_host = target_boxes.with_queue(queue).get()
    else:
        target_boxes_host = target_boxes.get(queue)

    if len(target_boxes_host) == 0:
        return None

    box_id = int(target_boxes_host[0])

    box_target_starts = tree.box_target_starts
    box_target_counts_cumul = tree.box_target_counts_cumul
    box_levels = tree.box_levels
    box_centers = tree.box_centers

    if hasattr(box_target_starts, "with_queue"):
        box_target_starts_host = box_target_starts.with_queue(queue).get()
    else:
        box_target_starts_host = box_target_starts.get(queue)

    if hasattr(box_target_counts_cumul, "with_queue"):
        box_target_counts_host = box_target_counts_cumul.with_queue(queue).get()
    else:
        box_target_counts_host = box_target_counts_cumul.get(queue)

    if hasattr(box_levels, "with_queue"):
        box_levels_host = box_levels.with_queue(queue).get()
    else:
        box_levels_host = box_levels.get(queue)

    if hasattr(box_centers, "with_queue"):
        box_centers_host = box_centers.with_queue(queue).get()
    else:
        box_centers_host = box_centers.get(queue)

    start = int(box_target_starts_host[box_id])
    count = int(box_target_counts_host[box_id])
    if count <= 0:
        return None

    expected_modes = q_order**tree.dimensions
    if count < expected_modes:
        return None

    level = int(box_levels_host[box_id])
    extent = float(tree.root_extent) * (1.0 / (2**level))
    center_x = float(box_centers_host[0, box_id])

    xcoords = tree.targets[0]
    if hasattr(xcoords, "with_queue"):
        xcoords_host = xcoords.with_queue(queue).get()
    else:
        xcoords_host = xcoords.get(queue)

    xcoords_box = xcoords_host[start : start + count]
    xcoords_tpl = (xcoords_box - center_x) / extent + 0.5

    # Robustly identify the q_order unique interpolation nodes.
    rounded = np.round(xcoords_tpl, decimals=14)
    interp_points = np.sort(np.unique(rounded))
    if len(interp_points) != q_order:
        return None

    return interp_points


def _build_box_mode_to_source_ids(
    tree, traversal, q_order, interp_points_1d, queue, match_tol=None
):
    """Map per-box lexicographic mode ids to source indices in tree order."""
    if not hasattr(traversal, "target_boxes"):
        return None

    target_boxes = traversal.target_boxes
    if hasattr(target_boxes, "with_queue"):
        target_boxes_host = target_boxes.with_queue(queue).get()
    else:
        target_boxes_host = target_boxes.get(queue)

    if len(target_boxes_host) == 0:
        return None

    box_target_starts = tree.box_target_starts
    box_target_counts_cumul = tree.box_target_counts_cumul
    box_levels = tree.box_levels
    box_centers = tree.box_centers

    if hasattr(box_target_starts, "with_queue"):
        box_target_starts_host = box_target_starts.with_queue(queue).get()
    else:
        box_target_starts_host = box_target_starts.get(queue)

    if hasattr(box_target_counts_cumul, "with_queue"):
        box_target_counts_host = box_target_counts_cumul.with_queue(queue).get()
    else:
        box_target_counts_host = box_target_counts_cumul.get(queue)

    if hasattr(box_levels, "with_queue"):
        box_levels_host = box_levels.with_queue(queue).get()
    else:
        box_levels_host = box_levels.get(queue)

    if hasattr(box_centers, "with_queue"):
        box_centers_host = box_centers.with_queue(queue).get()
    else:
        box_centers_host = box_centers.get(queue)

    src_coords_host = []
    for iaxis in range(tree.dimensions):
        coords = tree.sources[iaxis]
        if hasattr(coords, "with_queue"):
            src_coords_host.append(coords.with_queue(queue).get())
        else:
            src_coords_host.append(coords.get(queue))

    expected_modes = q_order**tree.dimensions
    mode_to_source = np.arange(len(src_coords_host[0]), dtype=np.int32)

    if match_tol is None:
        common_dtype = np.result_type(
            interp_points_1d.dtype, *[c.dtype for c in src_coords_host]
        )
        if np.issubdtype(common_dtype, np.floating):
            tol = max(1e-12, 32 * float(np.finfo(common_dtype).eps))
        else:
            tol = 1e-12
    else:
        tol = float(match_tol)

    for box_id in map(int, target_boxes_host):
        start = int(box_target_starts_host[box_id])
        count = int(box_target_counts_host[box_id])
        if count != expected_modes:
            continue

        level = int(box_levels_host[box_id])
        extent = float(tree.root_extent) * (1.0 / (2**level))
        centers = [
            float(box_centers_host[iaxis, box_id]) for iaxis in range(tree.dimensions)
        ]

        axis_mode_ids = []
        valid = True
        max_dist = None
        for iaxis in range(tree.dimensions):
            tplt = (
                src_coords_host[iaxis][start : start + count] - centers[iaxis]
            ) / extent + 0.5
            dist = np.abs(tplt[:, np.newaxis] - interp_points_1d[np.newaxis, :])
            ids = np.argmin(dist, axis=1)
            max_dist = float(np.max(np.min(dist, axis=1)))
            if max_dist > tol:
                valid = False
                break
            axis_mode_ids.append(ids)

        if not valid:
            raise ValueError(
                "could not build mode-to-source ids for box "
                f"{box_id}: unmatched_interp_node "
                f"(max_dist={max_dist:.3e}, tol={tol:.3e})"
            )

        mode_ids = axis_mode_ids[0].astype(np.int32)
        for iaxis in range(1, tree.dimensions):
            mode_ids = mode_ids * q_order + axis_mode_ids[iaxis]

        if len(np.unique(mode_ids)) != expected_modes:
            raise ValueError(
                "could not build mode-to-source ids for box "
                f"{box_id}: duplicate_mode_ids"
            )

        mode_to_source[start + mode_ids] = np.arange(
            start, start + count, dtype=np.int32
        )

    return cl.array.to_device(queue, mode_to_source)


def interpolate_volume_potential(
    target_points,
    traversal,
    wrangler,
    potential,
    potential_in_tree_order=False,
    target_radii=None,
    **kwargs,
):
    """
    Interpolate the volume potential, only works for tensor-product quadrature
    formulae. target_points and potential should be an cl array.

    :arg wrangler: Used only for general info (nothing sumpy kernel specific).
        May also be None if the needed information is passed by kwargs.
    :arg potential_in_tree_order: Whether the potential is in tree order (as
        opposed to in user order).
    :arg leaves_near_ball_starts/leaves_near_ball_lists: Optional target-major
        lookup lists. If omitted, lookup lists are built from target points.
    """
    if wrangler is not None:
        dim = next(iter(wrangler.near_field_table.values()))[0].dim
        tree = wrangler.tree
        queue = getattr(potential, "queue", None)
        if queue is None:
            queue = wrangler.queue
        q_order = wrangler.quad_order
        dtype = wrangler.dtype
    else:
        dim = kwargs["dim"]
        tree = kwargs["tree"]
        queue = kwargs["queue"]
        q_order = kwargs["q_order"]
        dtype = kwargs["dtype"]

    coord_dtype = tree.coord_dtype
    n_points = len(target_points[0])

    from volumential.expansion_wrangler_fpnd import FPNDFMMLibExpansionWrangler

    if isinstance(wrangler, FPNDFMMLibExpansionWrangler):
        tree = tree.to_device(queue)
        traversal = traversal.to_device(queue)

    assert dim == len(target_points)
    evt = None

    deprecated_kwargs = [
        name
        for name in (
            "lbl_lookup",
            "balls_near_box_starts",
            "balls_near_box_lists",
            "use_numpy_interpolation",
        )
        if name in kwargs
    ]
    if deprecated_kwargs:
        raise TypeError(
            "interpolate_volume_potential no longer accepts legacy "
            f"interpolation arguments: {', '.join(sorted(deprecated_kwargs))}. "
            "Use target-major leaves_near_ball_starts/lists or let the "
            "function build lookup data automatically."
        )

    leaves_near_ball_starts = kwargs.get("leaves_near_ball_starts")
    leaves_near_ball_lists = kwargs.get("leaves_near_ball_lists")

    if (leaves_near_ball_starts is None) != (leaves_near_ball_lists is None):
        raise ValueError(
            "leaves_near_ball_starts and leaves_near_ball_lists must be "
            "provided together"
        )

    if leaves_near_ball_starts is None:
        from boxtree.area_query import AreaQueryBuilder
        from boxtree.array_context import (
            PyOpenCLArrayContext as BoxtreePyOpenCLArrayContext,
        )

        boxtree_actx = BoxtreePyOpenCLArrayContext(queue)
        area_query_builder = AreaQueryBuilder(boxtree_actx)

        if target_radii is None:
            lookup_tol = _compute_interpolation_lookup_tol(tree)
            target_radii = cl.array.to_device(
                queue,
                np.full(n_points, lookup_tol, dtype=np.dtype(coord_dtype)),
            )

        area_query, evt = area_query_builder(
            boxtree_actx,
            tree,
            target_points,
            target_radii,
        )
        leaves_near_ball_starts = area_query.leaves_near_ball_starts
        leaves_near_ball_lists = area_query.leaves_near_ball_lists

    assert leaves_near_ball_starts is not None
    assert leaves_near_ball_lists is not None

    pout = cl.array.zeros(queue, n_points, dtype=dtype)
    multiplicity = cl.array.zeros(queue, n_points, dtype=dtype)

    if evt is not None:
        pout.add_event(evt)

    # map all boxes to a template [0,1]^2 so that the interpolation
    # weights and modes can be precomputed
    interp_points_1d = _infer_tree_local_interp_points_1d(
        tree, traversal, q_order, queue
    )
    if interp_points_1d is None:
        (blpoints, blweights) = compute_barycentric_lagrange_params(q_order)
    elif len(interp_points_1d) == 1:
        blpoints = np.asarray(interp_points_1d, dtype=np.float64)
        blweights = np.ones(1, dtype=np.float64)
    else:
        from scipy.interpolate import BarycentricInterpolator as Interpolator

        interp = Interpolator(xi=interp_points_1d, yi=None)
        blpoints = interp.xi
        blweights = interp.wi
    blpoints = cl.array.to_device(queue, blpoints)
    blweights = cl.array.to_device(queue, blweights)
    use_mode_to_source_ids = bool(kwargs.pop("use_mode_to_source_ids", False))
    if use_mode_to_source_ids:
        mode_to_source_match_tol = kwargs.pop("mode_to_source_match_tol", None)
        mode_to_source_ids = _build_box_mode_to_source_ids(
            tree,
            traversal,
            q_order,
            np.asarray(blpoints.get(queue)),
            queue,
            match_tol=mode_to_source_match_tol,
        )
    else:
        mode_to_source_ids = None

    if mode_to_source_ids is None:
        mode_to_source_ids = cl.array.arange(
            queue, len(tree.sources[0]), dtype=np.int32
        )

    # {{{ loopy kernel for interpolation

    if dim == 1:
        code_target_coords_assignment = """target_coords_x[target_point_id]"""
    if dim == 2:
        code_target_coords_assignment = """(
        target_coords_x[target_point_id]
            if iaxis == 0
            else target_coords_y[target_point_id])"""
    elif dim == 3:
        code_target_coords_assignment = """(
        target_coords_x[target_point_id]
            if iaxis == 0
            else (
                target_coords_y[target_point_id]
                    if iaxis == 1
                    else target_coords_z[target_point_id]))"""
    else:
        raise NotImplementedError

    if dim == 1:
        code_mode_index_assignment = """mid"""
    elif dim == 2:
        code_mode_index_assignment = """(
        mid / Q_ORDER
            if iaxis == 0
            else mid % Q_ORDER)""".replace("Q_ORDER", "q_order")
    elif dim == 3:
        code_mode_index_assignment = """(
        mid / (Q_ORDER * Q_ORDER)
            if iaxis == 0
            else (
                mid % (Q_ORDER * Q_ORDER) / Q_ORDER
                    if iaxis == 1
                    else mid % (Q_ORDER * Q_ORDER) % Q_ORDER))""".replace(
            "Q_ORDER", "q_order"
        )
    else:
        raise NotImplementedError

    import loopy

    lpknl = loopy.make_kernel(
        [
            "{ [ target_point_id, iaxis ] : 0 <= target_point_id < n_points "
            "and 0 <= iaxis < dim }",
            "{ [ near_box_nr, mid, mjd, mkd ] : "
            "nearby_leaves_begin <= near_box_nr < nearby_leaves_end "
            "and 0 <= mid < n_box_modes_expected and 0 <= mjd < q_order "
            "and 0 <= mkd < q_order }",
        ],
        """
            for target_point_id
                <> nearby_leaves_begin = leaves_near_ball_starts[target_point_id]
                <> nearby_leaves_end   = leaves_near_ball_starts[target_point_id + 1]

                    <> p_target = 0 {id=p_target_init}
                    <> multiplicity_target = 0 {id=mult_target_init}

                    for near_box_nr
                        <> target_box_id  = leaves_near_ball_lists[near_box_nr]
                        <> box_level      = box_levels[target_box_id]
                        <> box_mode_beg   = box_target_starts[target_box_id]
                        <> n_box_modes    = box_target_counts_cumul[target_box_id]

                        if n_box_modes == n_box_modes_expected
                            <> box_extent = root_extent * (1.0 / (2**box_level))

                            for iaxis
                                <> box_center[iaxis] = box_centers[iaxis, target_box_id] {dup=iaxis}
                                <> real_coord[iaxis] = TARGET_COORDS_ASSIGNMENT {dup=iaxis}
                            end

                            # Map target point to template box
                            for iaxis
                                <> tplt_coord[iaxis] = (real_coord[iaxis] - box_center[iaxis]
                                    ) / box_extent + 0.5 {dup=iaxis}
                            end

                            # Precompute denominators
                            for iaxis
                                <> denom[iaxis] = 0.0 {id=reinit_denom,dup=iaxis}
                            end

                            for iaxis, mjd
                                 <> diff[iaxis, mjd] = ( \
                                                  1 \
                                                      if tplt_coord[iaxis] == barycentric_lagrange_points[mjd] \
                                                      else tplt_coord[iaxis] - barycentric_lagrange_points[mjd]) \
                                                  {id=diff, dep=reinit_denom, dup=iaxis:mjd}
                                 denom[iaxis] = denom[iaxis] + \
                                         barycentric_lagrange_weights[mjd] / diff[iaxis, mjd] \
                                         {id=denom, dep=diff, dup=iaxis:mjd}
                            end

                            for mid
                                # Find the coeff of each mode
                                <> mode_id      = box_mode_beg + mid
                                <> source_id    = mode_to_source_ids[mode_id]
                                <> mode_id_user = user_mode_ids[source_id]
                                <> mode_coeff   = potential[mode_id_user]

                                # Mode id in each direction
                                for iaxis
                                    idx[iaxis] = MODE_INDEX_ASSIGNMENT {id=mode_indices,dup=iaxis}
                                end

                                # Interpolate mode value in each direction
                                for iaxis
                                    <> numerator[iaxis] = (barycentric_lagrange_weights[idx[iaxis]]
                                                        / diff[iaxis, idx[iaxis]]) {id=numerator,dep=diff:mode_indices,dup=iaxis}
                                    <> mode_val[iaxis] = numerator[iaxis] / denom[iaxis] {id=mode_val,dep=numerator:denom,dup=iaxis}
                                end

                                # Fix when target point coincide with a quad point
                                for mkd, iaxis
                                    mode_val[iaxis] = (
                                            (1 if mkd == idx[iaxis] else 0)
                                                if tplt_coord[iaxis] == barycentric_lagrange_points[mkd]
                                                else mode_val[iaxis]) {id=fix_mode_val, dep=mode_val:mode_indices, dup=iaxis}
                                end

                                <> prod_mode_val = product(iaxis,
                                    mode_val[iaxis]) {id=pmod,dep=fix_mode_val,dup=iaxis}

                            end

                            p_target = p_target + sum(mid,
                                mode_coeff * prod_mode_val
                                ) {id=p_target_acc,dep=pmod}
                            multiplicity_target = multiplicity_target + 1 {id=mult_target_acc,dep=p_target_acc}
                        end
                    end

                    p_out[target_point_id] = p_target {id=p_out,dep=p_target_init:p_target_acc}
                    multiplicity[target_point_id] = multiplicity_target {id=mult_out,dep=mult_target_init:mult_target_acc}
                end

            """.replace(  # noqa: E501
            "TARGET_COORDS_ASSIGNMENT", code_target_coords_assignment
        )
        .replace("MODE_INDEX_ASSIGNMENT", code_mode_index_assignment)
        .replace("Q_ORDER", "q_order"),
        [
            loopy.TemporaryVariable("idx", np.int32, "dim,"),
            loopy.GlobalArg("box_centers", None, "dim, aligned_nboxes"),
            loopy.GlobalArg("leaves_near_ball_starts", None, None),
            loopy.GlobalArg("leaves_near_ball_lists", None, None),
            loopy.GlobalArg("multiplicity", None, None),
            loopy.GlobalArg("p_out", None, None),
            loopy.ValueArg("aligned_nboxes", np.int32),
            loopy.ValueArg("dim", np.int32),
            loopy.ValueArg("q_order", np.int32),
            loopy.ValueArg("n_box_modes_expected", np.int32),
            loopy.ValueArg("n_points", np.int32),
            "...",
        ],
        lang_version=(2018, 2),
    )
    # }}} End loopy kernel for interpolation

    # loopy does not directly support object arrays
    if dim == 1:
        target_coords_knl_kwargs = {"target_coords_x": target_points[0]}
    elif dim == 2:
        target_coords_knl_kwargs = {
            "target_coords_x": target_points[0],
            "target_coords_y": target_points[1],
        }
    elif dim == 3:
        target_coords_knl_kwargs = {
            "target_coords_x": target_points[0],
            "target_coords_y": target_points[1],
            "target_coords_z": target_points[2],
        }
    else:
        raise NotImplementedError

    if potential_in_tree_order:
        user_mode_ids = cl.array.arange(
            queue, len(tree.user_source_ids), dtype=tree.user_source_ids.dtype
        )
    else:
        # fetching from user_source_ids converts potential to tree order
        user_mode_ids = tree.user_source_ids

    lpknl = loopy.set_options(lpknl, return_dict=True)
    lpknl = loopy.fix_parameters(lpknl, dim=int(dim), q_order=int(q_order))
    lpknl = loopy.tag_inames(lpknl, {"target_point_id": "g.0"})
    lpknl = loopy.remove_unused_inames(lpknl)
    lpknl_exec = lpknl.executor(queue.context)

    kernel_kwargs = {
        "box_centers": tree.box_centers,
        "box_levels": tree.box_levels,
        "barycentric_lagrange_weights": blweights,
        "barycentric_lagrange_points": blpoints,
        "box_target_starts": tree.box_target_starts,
        "box_target_counts_cumul": tree.box_target_counts_cumul,
        "potential": potential,
        "user_mode_ids": user_mode_ids,
        "mode_to_source_ids": mode_to_source_ids,
        "leaves_near_ball_starts": leaves_near_ball_starts,
        "leaves_near_ball_lists": leaves_near_ball_lists,
        "n_box_modes_expected": int(q_order**dim),
        "n_points": n_points,
        "root_extent": tree.root_extent,
        **target_coords_knl_kwargs,
    }

    evt, res_dict = lpknl_exec(
        queue,
        p_out=pout,
        multiplicity=multiplicity,
        **kernel_kwargs,
    )

    assert pout is res_dict["p_out"]
    assert multiplicity is res_dict["multiplicity"]
    pout.add_event(evt)
    multiplicity.add_event(evt)

    _ensure_interpolation_target_coverage(multiplicity, queue)

    return pout / multiplicity


# }}} End free form interpolation of potentials

# vim: filetype=pyopencl:fdm=marker
