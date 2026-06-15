__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryShellPDETargetReduction:
    """Discrete target-node reduction for homogeneous PDE kernels.

    The free target nodes are the tensor-product boundary shell. Interior target
    nodes are recovered from the shell by enforcing the collocation Laplace
    equation on the target grid.
    """

    quad_order: int
    dim: int
    boundary_ids: np.ndarray
    interior_ids: np.ndarray
    full_to_boundary: np.ndarray
    full_to_interior: np.ndarray
    recovery_matrix: np.ndarray

    @property
    def n_q_points(self):
        return self.quad_order**self.dim

    @property
    def n_boundary_points(self):
        return int(self.boundary_ids.size)

    @property
    def n_interior_points(self):
        return int(self.interior_ids.size)


def _mode_axes(mode_id, q_order, dim):
    axes = [0] * dim
    rem = int(mode_id)
    for iaxis in range(dim - 1, -1, -1):
        axes[iaxis] = rem % q_order
        rem //= q_order
    return tuple(axes)


def _mode_id(axes, q_order):
    result = 0
    for axis in axes:
        result = result * q_order + int(axis)
    return result


def _lagrange_second_derivative_matrix(nodes):
    nodes = np.asarray(nodes, dtype=np.float64)
    q_order = int(nodes.size)
    vand = np.vander(nodes, N=q_order, increasing=True)
    d2 = np.zeros((q_order, q_order), dtype=np.float64)

    for ibasis in range(q_order):
        rhs = np.zeros(q_order, dtype=np.float64)
        rhs[ibasis] = 1.0
        coeffs = np.linalg.solve(vand, rhs)
        for inode, xval in enumerate(nodes):
            total = 0.0
            for degree in range(2, q_order):
                total += (
                    degree
                    * (degree - 1)
                    * coeffs[degree]
                    * xval ** (degree - 2)
                )
            d2[inode, ibasis] = total

    return d2


def _axis_nodes_from_q_points(q_points, q_order, dim):
    q_points = np.asarray(q_points, dtype=np.float64)
    if q_points.shape != (q_order**dim, dim):
        raise ValueError(
            "q_points has shape "
            f"{q_points.shape}, expected {(q_order**dim, dim)}"
        )

    axis_nodes = []
    for iaxis in range(dim):
        nodes = np.unique(q_points[:, iaxis])
        if nodes.size != q_order:
            raise ValueError(
                f"axis {iaxis} has {nodes.size} unique nodes, expected {q_order}"
            )
        axis_nodes.append(np.asarray(nodes, dtype=np.float64))
    return axis_nodes


def build_laplace_boundary_shell_reduction(q_points, q_order, dim):
    """Build a boundary-shell target reduction for a tensor-product grid."""

    q_order = int(q_order)
    dim = int(dim)
    if dim not in (2, 3):
        raise NotImplementedError("PDE boundary-shell targets support dim 2 or 3")

    n_q_points = q_order**dim
    boundary_ids = []
    interior_ids = []
    for mode_id in range(n_q_points):
        axes = _mode_axes(mode_id, q_order, dim)
        if any(axis == 0 or axis == q_order - 1 for axis in axes):
            boundary_ids.append(mode_id)
        else:
            interior_ids.append(mode_id)

    boundary_ids = np.asarray(boundary_ids, dtype=np.int32)
    interior_ids = np.asarray(interior_ids, dtype=np.int32)

    full_to_boundary = np.full(n_q_points, -1, dtype=np.int32)
    full_to_boundary[boundary_ids] = np.arange(boundary_ids.size, dtype=np.int32)
    full_to_interior = np.full(n_q_points, -1, dtype=np.int32)
    full_to_interior[interior_ids] = np.arange(interior_ids.size, dtype=np.int32)

    if interior_ids.size == 0:
        recovery_matrix = np.zeros((0, boundary_ids.size), dtype=np.float64)
        return BoundaryShellPDETargetReduction(
            quad_order=q_order,
            dim=dim,
            boundary_ids=boundary_ids,
            interior_ids=interior_ids,
            full_to_boundary=full_to_boundary,
            full_to_interior=full_to_interior,
            recovery_matrix=recovery_matrix,
        )

    axis_nodes = _axis_nodes_from_q_points(q_points, q_order, dim)
    axis_d2 = [_lagrange_second_derivative_matrix(nodes) for nodes in axis_nodes]

    interior_rows = np.zeros((interior_ids.size, n_q_points), dtype=np.float64)
    for irow, full_id in enumerate(interior_ids):
        row_axes = list(_mode_axes(full_id, q_order, dim))
        for iaxis in range(dim):
            for col_axis in range(q_order):
                col_axes = list(row_axes)
                col_axes[iaxis] = col_axis
                col_id = _mode_id(col_axes, q_order)
                interior_rows[irow, col_id] += axis_d2[iaxis][
                    row_axes[iaxis], col_axis
                ]

    interior_matrix = interior_rows[:, interior_ids]
    boundary_matrix = interior_rows[:, boundary_ids]
    recovery_matrix = -np.linalg.solve(interior_matrix, boundary_matrix)

    return BoundaryShellPDETargetReduction(
        quad_order=q_order,
        dim=dim,
        boundary_ids=boundary_ids,
        interior_ids=interior_ids,
        full_to_boundary=full_to_boundary,
        full_to_interior=full_to_interior,
        recovery_matrix=recovery_matrix,
    )


def pack_boundary_shell_table(full_table_data, reduction, n_cases):
    """Pack full table data by retaining only boundary-shell target nodes."""

    full_table_data = np.asarray(full_table_data)
    n_q_points = reduction.n_q_points
    view = full_table_data.reshape(n_cases, n_q_points, n_q_points)
    return np.ascontiguousarray(view[:, :, reduction.boundary_ids].reshape(-1))


def self_case_correction_table(full_table_data, reduction, self_case_id, n_cases):
    """Return exact same-box correction after harmonic shell recovery.

    This correction lets the online recovery handle the inhomogeneous same-box
    term while the table itself stores only boundary-shell target nodes.
    """

    full_table_data = np.asarray(full_table_data)
    n_q_points = reduction.n_q_points
    if reduction.n_interior_points == 0:
        return np.zeros((0, n_q_points), dtype=full_table_data.dtype)

    view = full_table_data.reshape(n_cases, n_q_points, n_q_points)
    same_case = view[int(self_case_id)]
    boundary_values = same_case[:, reduction.boundary_ids].T
    interior_values = same_case[:, reduction.interior_ids].T
    recovered_self = reduction.recovery_matrix @ boundary_values
    return np.ascontiguousarray(interior_values - recovered_self)


def reconstruct_boundary_shell_table(
    boundary_table_data,
    self_correction_table_data,
    reduction,
    self_case_id,
    n_cases,
):
    """Reconstruct full target-node table data from shell-compressed data."""

    boundary_table_data = np.asarray(boundary_table_data)
    self_correction_table_data = np.asarray(self_correction_table_data)
    n_q_points = reduction.n_q_points
    dtype = np.result_type(boundary_table_data, self_correction_table_data)
    result = np.zeros((n_cases, n_q_points, n_q_points), dtype=dtype)
    shell_view = boundary_table_data.reshape(
        n_cases, n_q_points, reduction.n_boundary_points
    )

    for case_id in range(n_cases):
        case_result = result[case_id]
        case_result[:, reduction.boundary_ids] = shell_view[case_id]
        if reduction.n_interior_points:
            boundary_values = shell_view[case_id].T
            interior_values = reduction.recovery_matrix @ boundary_values
            if case_id == int(self_case_id):
                interior_values = interior_values + self_correction_table_data
            case_result[:, reduction.interior_ids] = interior_values.T

    return np.ascontiguousarray(result.reshape(-1))


def boundary_shell_reconstruction_diagnostics(
    full_table_data,
    reduction,
    self_case_id,
    n_cases,
):
    """Measure the loss from using shell recovery for homogeneous cases."""

    boundary_data = pack_boundary_shell_table(full_table_data, reduction, n_cases)
    self_correction = self_case_correction_table(
        full_table_data, reduction, self_case_id, n_cases
    )
    reconstructed = reconstruct_boundary_shell_table(
        boundary_data,
        self_correction,
        reduction,
        self_case_id,
        n_cases,
    )
    full_table_data = np.asarray(full_table_data)
    diff = reconstructed - full_table_data
    full_norm = float(np.linalg.norm(full_table_data.reshape(-1)))
    diff_norm = float(np.linalg.norm(diff.reshape(-1)))
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    max_ref = float(np.max(np.abs(full_table_data))) if full_table_data.size else 0.0

    return {
        "pde_reconstruction_max_abs_error": max_abs,
        "pde_reconstruction_rel_l2_error": (
            0.0 if full_norm == 0.0 else diff_norm / full_norm
        ),
        "pde_reconstruction_rel_linf_error": (
            0.0 if max_ref == 0.0 else max_abs / max_ref
        ),
    }
