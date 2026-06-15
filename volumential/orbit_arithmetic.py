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

import itertools
import math

import numpy as np


def _encode_mode_axes(axes, q_order):
    result = 0
    for axis_value in axes:
        result = result * int(q_order) + int(axis_value)
    return int(result)


def _decode_mode_axes(mode_id, q_order, dim):
    axes = [0] * int(dim)
    residual = int(mode_id)
    for iaxis in range(int(dim) - 1, -1, -1):
        axes[iaxis] = residual % int(q_order)
        residual //= int(q_order)
    return axes


def _case_arithmetic_axis_descriptors(
    case_vec,
    *,
    axis_groups=None,
    direction_signs=None,
):
    case_vec = [int(val) for val in case_vec]
    dim = len(case_vec)
    if axis_groups is None:
        axis_group_specs = (tuple(range(dim)),)
    else:
        axis_group_specs = axis_groups
    axis_group_specs = tuple(
        tuple(int(axis) for axis in group) for group in axis_group_specs
    )
    direction_signs = np.zeros(dim, dtype=np.int8) if direction_signs is None else (
        np.asarray(direction_signs, dtype=np.int8)
    )

    axis_perm = np.empty(dim, dtype=np.uint8)
    axis_signs = np.empty(dim, dtype=np.int8)
    axis_group_ids = np.empty(dim, dtype=np.uint8)
    next_runtime_group = 0

    def sign_to_negative(value):
        value = int(value)
        if value > 0:
            return -1
        if value < 0:
            return 1
        return 0

    active_group_specs = []
    inactive_group_specs = []
    for group in axis_group_specs:
        out_axes = tuple(sorted(group))
        active_direction_axes = [
            axis for axis in out_axes if direction_signs[axis] != 0
        ]
        if active_direction_axes:
            if len(active_direction_axes) != len(out_axes):
                raise ValueError("directional arithmetic groups cannot mix active axes")
            active_group_specs.append(out_axes)
        else:
            inactive_group_specs.append(out_axes)

    if active_group_specs:
        from itertools import permutations, product

        sorted_active_axes = tuple(
            sorted(axis for group in active_group_specs for axis in group)
        )
        permutation_options = [
            tuple(permutations(group)) for group in active_group_specs
        ]
        best_key = None
        best_axis_perm = None
        best_axis_sign = None

        for line_sign in (-1, 1):
            for permuted_groups in product(*permutation_options):
                candidate_axis_perm = {}
                candidate_axis_sign = {}
                for out_axes, permuted_axes in zip(
                    active_group_specs,
                    permuted_groups,
                    strict=True,
                ):
                    for out_axis, in_axis in zip(out_axes, permuted_axes, strict=True):
                        sign = (
                            int(line_sign)
                            * int(direction_signs[out_axis])
                            * int(direction_signs[in_axis])
                        )
                        candidate_axis_perm[out_axis] = int(in_axis)
                        candidate_axis_sign[out_axis] = int(sign)

                key = (
                    tuple(
                        int(case_vec[candidate_axis_perm[axis]])
                        * int(candidate_axis_sign[axis])
                        for axis in sorted_active_axes
                    ),
                    tuple(candidate_axis_perm[axis] for axis in sorted_active_axes),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_axis_perm = candidate_axis_perm
                    best_axis_sign = candidate_axis_sign

        for out_axis in sorted_active_axes:
            axis_perm[out_axis] = best_axis_perm[out_axis]
            axis_signs[out_axis] = best_axis_sign[out_axis]
            # Avoid over-canonicalizing directional stabilizers: source/target
            # flips in active directional groups are tied by one line sign.
            axis_group_ids[out_axis] = next_runtime_group
            next_runtime_group += 1

    for out_axes in inactive_group_specs:
        free_order = sorted(
            out_axes,
            key=lambda iaxis: (-abs(case_vec[iaxis]), iaxis),
        )

        last_abs = None
        current_runtime_group = next_runtime_group - 1
        for out_axis, in_axis in zip(out_axes, free_order, strict=True):
            value = int(case_vec[in_axis])
            abs_value = abs(value)
            if last_abs != abs_value:
                current_runtime_group += 1
                last_abs = abs_value
            axis_perm[out_axis] = in_axis
            axis_signs[out_axis] = sign_to_negative(value)
            axis_group_ids[out_axis] = current_runtime_group

        next_runtime_group = current_runtime_group + 1

    return axis_perm, axis_signs, axis_group_ids


def _canonical_case_from_axis_descriptors(case_vec, axis_perm, axis_sign):
    canonical = []
    for in_axis, sign in zip(axis_perm, axis_sign, strict=True):
        value = int(case_vec[int(in_axis)])
        sign = int(sign)
        canonical.append(0 if sign == 0 else value * sign)

    return canonical


def _kernel_axis_preserving_arithmetic_symmetry(table, reconstruction_maps):
    dim = int(table.dim)
    fixed_axes = set()
    axis_sign_power = np.zeros(dim, dtype=np.uint8)
    direction_signs = np.zeros(dim, dtype=np.int8)
    direction_sign_axis = -1
    direction_abs_group = np.full(dim, -1, dtype=np.int16)
    direction_sign_power = np.uint8(0)
    direction_vector = None
    has_directional_source = False

    kernel = table.integral_knl
    while kernel is not None:
        cls_name = kernel.__class__.__name__
        if cls_name in {"AxisTargetDerivative", "AxisSourceDerivative"}:
            axis = int(kernel.axis)
            if axis < 0 or axis >= dim:
                return None
            fixed_axes.add(axis)
            axis_sign_power[axis] ^= np.uint8(1)
            kernel = kernel.inner_kernel
            continue

        if cls_name == "DirectionalSourceDerivative":
            source_direction = getattr(kernel, "_volumential_source_direction", None)
            if source_direction is None:
                return None
            source_direction = np.asarray(source_direction, dtype=np.float64).ravel()
            if source_direction.shape != (dim,):
                return None
            direction_sign_power ^= np.uint8(1)
            if has_directional_source:
                if not np.allclose(source_direction, direction_vector):
                    return None
                kernel = kernel.inner_kernel
                continue

            axis_candidates = np.flatnonzero(
                np.abs(source_direction) > 100 * np.finfo(np.float64).eps
            )
            if len(axis_candidates) == 0:
                return None

            active_axes = tuple(sorted(int(axis) for axis in axis_candidates))
            active_axis_list = list(active_axes)
            direction_signs[active_axis_list] = np.sign(
                source_direction[active_axis_list]
            ).astype(np.int8)
            active_abs_groups = []
            for axis in active_axes:
                abs_value = abs(float(source_direction[axis]))
                for group_id, (group_abs, group_axes) in enumerate(active_abs_groups):
                    if np.isclose(abs_value, group_abs):
                        group_axes.append(axis)
                        direction_abs_group[axis] = group_id
                        break
                else:
                    direction_abs_group[axis] = len(active_abs_groups)
                    active_abs_groups.append((abs_value, [axis]))
            direction_sign_axis = int(active_axes[0])
            direction_vector = source_direction.copy()
            has_directional_source = True
            kernel = kernel.inner_kernel
            continue

        if hasattr(kernel, "inner_kernel"):
            kernel = kernel.inner_kernel
            continue

        break

    from itertools import permutations, product

    if not has_directional_source:
        free_axes = tuple(axis for axis in range(dim) if axis not in fixed_axes)
        axis_groups = tuple((axis,) for axis in sorted(fixed_axes))
        if free_axes:
            axis_groups += (free_axes,)
    else:
        axis_groups = []
        for group_id in sorted(
            int(group_id)
            for group_id in np.unique(direction_abs_group)
            if group_id >= 0
        ):
            group_axes = tuple(
                axis
                for axis in range(dim)
                if int(direction_abs_group[axis]) == group_id
            )
            fixed_group_axes = tuple(axis for axis in group_axes if axis in fixed_axes)
            free_group_axes = tuple(
                axis for axis in group_axes if axis not in fixed_axes
            )
            axis_groups.extend((axis,) for axis in fixed_group_axes)
            if free_group_axes:
                axis_groups.append(free_group_axes)

        inactive_fixed_axes = tuple(
            axis
            for axis in sorted(fixed_axes)
            if int(direction_abs_group[axis]) < 0
        )
        inactive_free_axes = tuple(
            axis
            for axis in range(dim)
            if int(direction_abs_group[axis]) < 0 and axis not in fixed_axes
        )
        axis_groups.extend((axis,) for axis in inactive_fixed_axes)
        if inactive_free_axes:
            axis_groups.append(inactive_free_axes)
        axis_groups = tuple(axis_groups)

    expected = set()
    for sign_tuple in product([-1, 1], repeat=dim):
        for perm_tuple in permutations(range(dim)):
            line_sign = 1
            if not has_directional_source:
                if any(int(perm_tuple[axis]) != axis for axis in fixed_axes):
                    continue
            else:
                line_sign = None
                valid = True
                if any(int(perm_tuple[axis]) != axis for axis in fixed_axes):
                    continue
                for out_axis in range(dim):
                    in_axis = int(perm_tuple[out_axis])
                    out_dir = int(direction_signs[out_axis])
                    in_dir = int(direction_signs[in_axis])
                    if out_dir == 0 or in_dir == 0:
                        if out_dir != in_dir:
                            valid = False
                            break
                        continue
                    if int(direction_abs_group[out_axis]) != int(
                        direction_abs_group[in_axis]
                    ):
                        valid = False
                        break

                    candidate = int(sign_tuple[in_axis]) * out_dir * in_dir
                    if line_sign is None:
                        line_sign = candidate
                    elif candidate != line_sign:
                        valid = False
                        break

                if not valid or line_sign is None:
                    continue

            transform_sign = 1
            for axis in range(dim):
                if int(axis_sign_power[axis]):
                    transform_sign *= int(sign_tuple[axis])
            if int(direction_sign_power):
                transform_sign *= int(line_sign)

            expected.add(
                (
                    tuple(int(val) for val in sign_tuple),
                    tuple(int(val) for val in perm_tuple),
                    int(transform_sign),
                )
            )

    signatures = reconstruction_maps.get("transform_signatures")
    if signatures is None:
        return None

    actual = {
        (
            tuple(int(val) for val in sign_tuple),
            tuple(int(val) for val in perm_tuple),
            int(transform_sign),
        )
        for (sign_tuple, perm_tuple), transform_sign in zip(
            signatures,
            np.asarray(reconstruction_maps["transform_signs"], dtype=np.int8),
            strict=True,
        )
    }

    if actual != expected:
        return None

    if len(actual) != len(signatures):
        return None

    return {
        "axis_groups": axis_groups,
        "axis_sign_power": axis_sign_power,
        "axis_direction_signs": direction_signs,
        "direction_sign_axis": int(direction_sign_axis if direction_sign_power else -1),
    }


def _evaluate_arithmetic_orbit_entry(
    case_id,
    source_mode_id,
    target_point_id,
    *,
    case_orbit_ranks,
    case_axis_perm,
    case_axis_sign,
    case_axis_group,
    axis_sign_power=None,
    axis_direction_signs=None,
    direction_sign_axis=-1,
    q_order,
    dim,
):
    if axis_sign_power is None:
        axis_sign_power = np.zeros(int(dim), dtype=np.uint8)
    if axis_direction_signs is None:
        axis_direction_signs = np.zeros(int(dim), dtype=np.int8)

    source_axes_raw = _decode_mode_axes(source_mode_id, q_order, dim)
    target_axes_raw = _decode_mode_axes(target_point_id, q_order, dim)

    source_axes = []
    target_axes = []
    groups = []
    entry_sign = 1
    for out_axis in range(int(dim)):
        in_axis = int(case_axis_perm[case_id, out_axis])
        axis_sign = int(case_axis_sign[case_id, out_axis])
        source_axis = source_axes_raw[in_axis]
        target_axis = target_axes_raw[in_axis]
        applied_axis_sign = 1

        if axis_sign < 0:
            source_axis = int(q_order) - 1 - source_axis
            target_axis = int(q_order) - 1 - target_axis
            applied_axis_sign = -1
        elif axis_sign == 0:
            flipped_source = int(q_order) - 1 - source_axis
            flipped_target = int(q_order) - 1 - target_axis
            if (flipped_source, flipped_target) < (source_axis, target_axis):
                source_axis = flipped_source
                target_axis = flipped_target
                applied_axis_sign = -1

        if int(axis_sign_power[in_axis]):
            entry_sign *= applied_axis_sign
        if int(direction_sign_axis) == out_axis:
            entry_sign *= (
                applied_axis_sign
                * int(axis_direction_signs[in_axis])
                * int(axis_direction_signs[out_axis])
            )

        source_axes.append(source_axis)
        target_axes.append(target_axis)
        groups.append(int(case_axis_group[case_id, out_axis]))

    def compare_swap(iaxis, jaxis):
        if groups[iaxis] != groups[jaxis]:
            return
        if (source_axes[jaxis], target_axes[jaxis]) < (
            source_axes[iaxis],
            target_axes[iaxis],
        ):
            source_axes[iaxis], source_axes[jaxis] = (
                source_axes[jaxis],
                source_axes[iaxis],
            )
            target_axes[iaxis], target_axes[jaxis] = (
                target_axes[jaxis],
                target_axes[iaxis],
            )

    if int(dim) == 2:
        compare_swap(0, 1)
    elif int(dim) == 3:
        compare_swap(0, 1)
        compare_swap(1, 2)
        compare_swap(0, 1)
        compare_swap(0, 2)
    else:
        raise NotImplementedError("scalar arithmetic ORBIT supports dim 2 or 3")

    canonical_source = _encode_mode_axes(source_axes, q_order)
    canonical_target = _encode_mode_axes(target_axes, q_order)
    n_q_points = int(q_order) ** int(dim)
    entry_id = (
        int(case_orbit_ranks[case_id]) * n_q_points * n_q_points
        + canonical_source * n_q_points
        + canonical_target
    )
    return int(entry_id), int(entry_sign)


def _evaluate_scalar_arithmetic_entry(
    case_id,
    source_mode_id,
    target_point_id,
    *,
    case_orbit_ranks,
    case_axis_perm,
    case_axis_sign,
    case_axis_group,
    q_order,
    dim,
):
    entry_id, _ = _evaluate_arithmetic_orbit_entry(
        case_id,
        source_mode_id,
        target_point_id,
        case_orbit_ranks=case_orbit_ranks,
        case_axis_perm=case_axis_perm,
        case_axis_sign=case_axis_sign,
        case_axis_group=case_axis_group,
        q_order=q_order,
        dim=dim,
    )
    return entry_id


def _rank_multiset(values, alphabet_size):
    values = [int(value) for value in values]
    alphabet_size = int(alphabet_size)
    rank = 0
    min_allowed = 0
    nvalues = len(values)

    for i, value in enumerate(values):
        remaining = nvalues - i - 1
        for candidate in range(min_allowed, value):
            rank += math.comb(alphabet_size - candidate + remaining - 1, remaining)
        min_allowed = value

    return int(rank)


def _compact_arithmetic_case_value_count(case_axis_sign, case_axis_group, q_order):
    case_axis_sign = np.asarray(case_axis_sign, dtype=np.int8)
    case_axis_group = np.asarray(case_axis_group, dtype=np.int64)
    pair_count = int(q_order) * int(q_order)
    folded_pair_count = (pair_count + 1) // 2
    value_count = 1

    for group_id in range(int(np.max(case_axis_group)) + 1):
        axes = np.flatnonzero(case_axis_group == group_id)
        if len(axes) == 0:
            continue
        alphabet_size = (
            folded_pair_count
            if np.any(case_axis_sign[axes] == 0)
            else pair_count
        )
        value_count *= math.comb(int(alphabet_size) + len(axes) - 1, len(axes))

    return int(value_count)


def _evaluate_compact_arithmetic_orbit_entry(
    case_id,
    source_mode_id,
    target_point_id,
    *,
    case_orbit_ranks,
    case_value_offsets,
    case_axis_perm,
    case_axis_sign,
    case_axis_group,
    axis_sign_power=None,
    axis_direction_signs=None,
    direction_sign_axis=-1,
    q_order,
    dim,
):
    if axis_sign_power is None:
        axis_sign_power = np.zeros(int(dim), dtype=np.uint8)
    if axis_direction_signs is None:
        axis_direction_signs = np.zeros(int(dim), dtype=np.int8)

    source_axes_raw = _decode_mode_axes(source_mode_id, q_order, dim)
    target_axes_raw = _decode_mode_axes(target_point_id, q_order, dim)

    source_axes = []
    target_axes = []
    groups = []
    axis_signs = []
    entry_sign = 1
    for out_axis in range(int(dim)):
        in_axis = int(case_axis_perm[case_id, out_axis])
        axis_sign = int(case_axis_sign[case_id, out_axis])
        source_axis = source_axes_raw[in_axis]
        target_axis = target_axes_raw[in_axis]
        applied_axis_sign = 1

        if axis_sign < 0:
            source_axis = int(q_order) - 1 - source_axis
            target_axis = int(q_order) - 1 - target_axis
            applied_axis_sign = -1
        elif axis_sign == 0:
            flipped_source = int(q_order) - 1 - source_axis
            flipped_target = int(q_order) - 1 - target_axis
            if (flipped_source, flipped_target) < (source_axis, target_axis):
                source_axis = flipped_source
                target_axis = flipped_target
                applied_axis_sign = -1

        if int(axis_sign_power[in_axis]):
            entry_sign *= applied_axis_sign
        if int(direction_sign_axis) == out_axis:
            entry_sign *= (
                applied_axis_sign
                * int(axis_direction_signs[in_axis])
                * int(axis_direction_signs[out_axis])
            )

        source_axes.append(source_axis)
        target_axes.append(target_axis)
        groups.append(int(case_axis_group[case_id, out_axis]))
        axis_signs.append(axis_sign)

    pair_codes = [
        int(source_axis) * int(q_order) + int(target_axis)
        for source_axis, target_axis in zip(source_axes, target_axes, strict=True)
    ]
    pair_count = int(q_order) * int(q_order)
    folded_pair_count = (pair_count + 1) // 2

    group_ranks = []
    group_value_counts = []
    for group_id in range(max(groups) + 1):
        axes = [axis for axis, group in enumerate(groups) if group == group_id]
        if not axes:
            group_ranks.append(0)
            group_value_counts.append(1)
            continue

        alphabet_size = (
            folded_pair_count
            if any(axis_signs[axis] == 0 for axis in axes)
            else pair_count
        )
        group_values = sorted(pair_codes[axis] for axis in axes)
        group_ranks.append(_rank_multiset(group_values, alphabet_size))
        group_value_counts.append(
            math.comb(int(alphabet_size) + len(axes) - 1, len(axes))
        )

    compact_pair_rank = 0
    for group_rank, group_value_count in zip(
        group_ranks,
        group_value_counts,
        strict=True,
    ):
        compact_pair_rank = compact_pair_rank * int(group_value_count) + int(group_rank)

    case_rank = int(case_orbit_ranks[case_id])
    return int(case_value_offsets[case_rank]) + compact_pair_rank, int(entry_sign)


def _entry_has_odd_reconstruction_stabilizer(
    case_id,
    source_mode_id,
    target_point_id,
    reconstruction_maps,
):
    transform_case_map = np.asarray(reconstruction_maps["transform_case_map"])
    transform_qpoint_map = np.asarray(reconstruction_maps["transform_qpoint_map"])
    transform_signs = np.asarray(reconstruction_maps["transform_signs"])

    for transform_id, transform_sign in enumerate(transform_signs):
        if int(transform_sign) >= 0:
            continue
        if int(transform_case_map[transform_id, case_id]) != int(case_id):
            continue
        if int(transform_qpoint_map[transform_id, source_mode_id]) != int(
            source_mode_id
        ):
            continue
        if int(transform_qpoint_map[transform_id, target_point_id]) != int(
            target_point_id
        ):
            continue
        return True

    return False


def build_arithmetic_case_metadata(table, arithmetic_symmetry=None):
    if arithmetic_symmetry is None:
        axis_groups = None
        axis_sign_power = np.zeros(int(table.dim), dtype=np.uint8)
        axis_direction_signs = np.zeros(int(table.dim), dtype=np.int8)
        direction_sign_axis = -1
    else:
        axis_groups = arithmetic_symmetry["axis_groups"]
        axis_sign_power = np.asarray(
            arithmetic_symmetry["axis_sign_power"], dtype=np.uint8
        )
        axis_direction_signs = np.asarray(
            arithmetic_symmetry["axis_direction_signs"], dtype=np.int8
        )
        direction_sign_axis = int(arithmetic_symmetry["direction_sign_axis"])

    case_vecs = np.asarray(table.interaction_case_vecs, dtype=np.int32)
    canonical_case_ids = np.empty(table.n_cases, dtype=np.int32)
    case_axis_perm = np.empty((table.n_cases, table.dim), dtype=np.uint8)
    case_axis_sign = np.empty((table.n_cases, table.dim), dtype=np.int8)
    case_axis_group = np.empty((table.n_cases, table.dim), dtype=np.uint8)

    for case_id, case_vec in enumerate(case_vecs):
        axis_perm, axis_sign, axis_group = _case_arithmetic_axis_descriptors(
            case_vec,
            axis_groups=axis_groups,
            direction_signs=axis_direction_signs,
        )
        canonical_vec = _canonical_case_from_axis_descriptors(
            case_vec,
            axis_perm,
            axis_sign,
        )
        canonical_case_ids[case_id] = int(
            table.case_indices[table.case_encode(canonical_vec)]
        )
        case_axis_perm[case_id, :] = axis_perm
        case_axis_sign[case_id, :] = axis_sign
        case_axis_group[case_id, :] = axis_group

    unique_canonical_case_ids = np.unique(canonical_case_ids)
    if len(unique_canonical_case_ids) > np.iinfo(np.uint16).max:
        return None

    canonical_case_rank_by_id = {
        int(case_id): rank for rank, case_id in enumerate(unique_canonical_case_ids)
    }
    case_orbit_ranks = np.empty(table.n_cases, dtype=np.uint16)
    for case_id in range(table.n_cases):
        case_orbit_ranks[case_id] = canonical_case_rank_by_id[
            int(canonical_case_ids[case_id])
        ]

    case_value_counts = np.empty(len(unique_canonical_case_ids), dtype=np.int64)
    for case_rank, canonical_case_id in enumerate(unique_canonical_case_ids):
        case_value_counts[case_rank] = _compact_arithmetic_case_value_count(
            case_axis_sign[canonical_case_id],
            case_axis_group[canonical_case_id],
            table.quad_order,
        )

    case_value_offsets_i64 = np.zeros(
        len(unique_canonical_case_ids) + 1, dtype=np.int64
    )
    case_value_offsets_i64[1:] = np.cumsum(case_value_counts, dtype=np.int64)
    if case_value_offsets_i64[-1] > np.iinfo(np.int32).max:
        return None

    return {
        "canonical_case_ids": unique_canonical_case_ids.astype(np.int32),
        "case_orbit_ranks": case_orbit_ranks,
        "case_axis_perm": case_axis_perm,
        "case_axis_sign": case_axis_sign,
        "case_axis_group": case_axis_group,
        "case_value_offsets": case_value_offsets_i64.astype(np.int32),
        "case_value_offsets_i64": case_value_offsets_i64,
        "axis_sign_power": axis_sign_power,
        "axis_direction_signs": axis_direction_signs,
        "direction_sign_axis": int(direction_sign_axis),
        "n_compact_entries": int(case_value_offsets_i64[-1]),
    }


def enumerate_scalar_arithmetic_representatives(table, metadata):
    pair_count = int(table.quad_order) * int(table.quad_order)
    folded_pair_count = (pair_count + 1) // 2
    canonical_case_ids = np.asarray(metadata["canonical_case_ids"], dtype=np.int32)
    case_axis_sign = np.asarray(metadata["case_axis_sign"], dtype=np.int8)
    case_axis_group = np.asarray(metadata["case_axis_group"], dtype=np.uint8)
    case_value_offsets = np.asarray(metadata["case_value_offsets_i64"], dtype=np.int64)

    entry_ids = np.empty(int(metadata["n_compact_entries"]), dtype=np.int64)
    for case_rank, case_id_raw in enumerate(canonical_case_ids.tolist()):
        case_id = int(case_id_raw)
        group_options = []
        case_groups = case_axis_group[case_id]
        case_signs = case_axis_sign[case_id]
        for group_id in range(int(np.max(case_groups)) + 1):
            axes = np.flatnonzero(case_groups == group_id).astype(int).tolist()
            alphabet_size = (
                folded_pair_count if np.any(case_signs[axes] == 0) else pair_count
            )
            options = itertools.combinations_with_replacement(
                range(alphabet_size), len(axes)
            )
            group_options.append((axes, tuple(options)))

        out = int(case_value_offsets[case_rank])
        for combo_by_group in itertools.product(*(opts for _, opts in group_options)):
            source_axes = [0] * int(table.dim)
            target_axes = [0] * int(table.dim)
            for (axes, _), values in zip(group_options, combo_by_group, strict=True):
                for axis, pair_code in zip(axes, values, strict=True):
                    source_axes[axis] = int(pair_code) // int(table.quad_order)
                    target_axes[axis] = int(pair_code) % int(table.quad_order)

            source_mode_id = _encode_mode_axes(source_axes, table.quad_order)
            target_point_id = _encode_mode_axes(target_axes, table.quad_order)
            entry_ids[out] = (
                int(case_id) * int(table.n_pairs)
                + source_mode_id * int(table.n_q_points)
                + target_point_id
            )
            out += 1

    return entry_ids
