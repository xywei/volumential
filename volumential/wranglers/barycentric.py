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

__doc__ = """Barycentric interpolation matrices and Gauss-Legendre nodes.

These build the one-dimensional operators that the Helmholtz-split
smooth-correction quadrature uses to resample box values onto a finer
tensor-product rule.
"""

import numpy as np

from volumential.lagrange import barycentric_lagrange_weights


def _gauss_legendre_nodes_and_weights(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return (0.5 * (nodes + 1.0), 0.5 * weights)


def _barycentric_interp_matrix(source_nodes, target_nodes):
    source_nodes = np.asarray(source_nodes, dtype=np.float64)
    target_nodes = np.asarray(target_nodes, dtype=np.float64)

    if source_nodes.size == 1:
        return np.ones((target_nodes.size, 1), dtype=np.float64)

    weights = barycentric_lagrange_weights(source_nodes)
    interp_mat = np.empty((target_nodes.size, source_nodes.size), dtype=np.float64)

    for i, x_tgt in enumerate(target_nodes):
        diff = x_tgt - source_nodes
        hit = np.where(diff == 0.0)[0]
        if hit.size:
            interp_mat[i, :] = 0.0
            interp_mat[i, int(hit[0])] = 1.0
            continue

        terms = weights / diff
        interp_mat[i, :] = terms / np.sum(terms)

    return interp_mat

# vim: filetype=pyopencl:foldmethod=marker
