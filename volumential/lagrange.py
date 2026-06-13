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

    signs = np.prod(np.sign(diffs), axis=1)
    log_abs_weights = -np.sum(np.log(np.abs(diffs)), axis=1)
    return signs * np.exp(log_abs_weights - np.max(log_abs_weights))


def evaluate_lagrange_basis_1d(nodes, index, x, weights=None):
    """Evaluate one Lagrange basis function with the barycentric formula."""

    index = _validate_integer_order(index, "index", minimum=0)
    nodes = np.asarray(nodes, dtype=np.float64)
    if index >= nodes.size:
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


# vim: foldmethod=marker:filetype=python
