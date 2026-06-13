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

__doc__ = """Stable Lagrange basis evaluation helpers."""

import numpy as np


def _validate_integer_order(order, name, minimum):
    if isinstance(order, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {order!r}")

    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {order!r}") from exc

    if order_int != order:
        raise ValueError(f"{name} must be an integer, got {order!r}")
    if order_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {order!r}")

    return order_int


def barycentric_lagrange_weights(nodes):
    """Return first-kind barycentric weights for distinct interpolation nodes."""

    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("nodes must be one-dimensional")

    diffs = nodes[:, np.newaxis] - nodes[np.newaxis, :]
    np.fill_diagonal(diffs, 1.0)
    if np.any(diffs == 0):
        raise ValueError("nodes must be distinct")

    return 1.0 / np.prod(diffs, axis=1)


def evaluate_lagrange_basis_1d(nodes, index, x, weights=None):
    """Evaluate one Lagrange basis function with the barycentric formula."""

    index = _validate_integer_order(index, "index", minimum=0)
    nodes = np.asarray(nodes, dtype=np.float64)
    if index < 0 or index >= nodes.size:
        raise ValueError(f"basis index {index} outside node set of size {nodes.size}")
    if weights is None:
        weights = barycentric_lagrange_weights(nodes)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != nodes.shape:
            raise ValueError("weights must have the same shape as nodes")

    x_arr = np.asarray(x, dtype=np.float64)
    diffs = x_arr[..., np.newaxis] - nodes
    flat_diffs = diffs.reshape(-1, nodes.size)
    exact_node = flat_diffs == 0.0
    exact_row = np.any(exact_node, axis=1)
    flat_result = np.empty(flat_diffs.shape[0], dtype=np.float64)

    if np.any(exact_row):
        flat_result[exact_row] = exact_node[exact_row, index]
    if np.any(~exact_row):
        terms = weights / flat_diffs[~exact_row]
        flat_result[~exact_row] = terms[:, index] / np.sum(terms, axis=1)

    result = flat_result.reshape(x_arr.shape)
    if np.isscalar(x):
        return float(result)
    return result


def evaluate_tensor_lagrange_mode(nodes_by_axis, mode_axes, coords, weights_by_axis=None):
    """Evaluate a tensor-product Lagrange mode using barycentric form."""

    if len(nodes_by_axis) != len(mode_axes) or len(coords) != len(mode_axes):
        raise ValueError("nodes, mode axes, and coordinates must have matching dims")
    if weights_by_axis is None:
        weights_by_axis = [None] * len(mode_axes)
    elif len(weights_by_axis) != len(mode_axes):
        raise ValueError("weights and mode axes must have matching dimensions")

    result = None
    for nodes, index, coord, weights in zip(
        nodes_by_axis,
        mode_axes,
        coords,
        weights_by_axis,
    ):
        axis_values = evaluate_lagrange_basis_1d(nodes, index, coord, weights=weights)
        if result is None:
            result = axis_values
        else:
            result = result * axis_values

    return result


def lagrange_basis_shifted_coefficients(nodes, index, center, max_order=None):
    """Return coefficients of one basis in powers of ``x-center``."""

    index = _validate_integer_order(index, "index", minimum=0)
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("nodes must be one-dimensional")
    if index < 0 or index >= nodes.size:
        raise ValueError(f"basis index {index} outside node set of size {nodes.size}")
    if max_order is not None:
        max_order = _validate_integer_order(max_order, "max_order", minimum=0)

    roots = np.delete(nodes, index)
    coeffs = np.polynomial.polynomial.polyfromroots(roots - center)
    coeffs = coeffs / np.prod(nodes[index] - roots)
    if max_order is not None:
        coeffs = coeffs[: max_order + 1]
    return coeffs


def tensor_lagrange_mode_shifted_coefficients(
    nodes_by_axis,
    mode_axes,
    center,
    max_total_order=None,
):
    """Return target-centered coefficients for a tensor Lagrange mode."""

    if len(nodes_by_axis) != len(mode_axes) or len(center) != len(mode_axes):
        raise ValueError("nodes, mode axes, and center must have matching dims")
    if max_total_order is not None:
        max_total_order = _validate_integer_order(
            max_total_order,
            "max_total_order",
            minimum=0,
        )

    axis_coeffs = [
        lagrange_basis_shifted_coefficients(
            nodes,
            index,
            axis_center,
            max_order=max_total_order,
        )
        for nodes, index, axis_center in zip(nodes_by_axis, mode_axes, center)
    ]

    coeffs = axis_coeffs[0]
    for axis_coeff in axis_coeffs[1:]:
        coeffs = np.multiply.outer(coeffs, axis_coeff)

    if max_total_order is not None:
        for idx in np.ndindex(coeffs.shape):
            if sum(idx) > max_total_order:
                coeffs[idx] = 0.0

    return coeffs


def legendre_gauss_lagrange_data(order, a=0.0, b=1.0):
    """Return Legendre-Gauss nodes and barycentric weights on ``[a, b]``."""

    order = _validate_integer_order(order, "order", minimum=1)
    nodes, _ = np.polynomial.legendre.leggauss(order)
    center = 0.5 * (a + b)
    radius = 0.5 * (b - a)
    nodes = center + radius * nodes
    return nodes, barycentric_lagrange_weights(nodes)


def mode_index_to_axes(mode_index, q_order, dim):
    """Return tensor-product axis indices for a flat mode index."""

    mode_index = _validate_integer_order(mode_index, "mode_index", minimum=0)
    q_order = _validate_integer_order(q_order, "q_order", minimum=1)
    dim = _validate_integer_order(dim, "dim", minimum=1)
    if mode_index >= q_order**dim:
        raise ValueError(f"mode_index must be in [0, {q_order**dim}), got {mode_index}")

    axes = np.empty(dim, dtype=np.int32)
    residual = mode_index
    for axis in range(dim - 1, -1, -1):
        axes[axis] = residual % q_order
        residual //= q_order
    return axes


# vim: foldmethod=marker:filetype=python
