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

import numpy as np

from volumential import lagrange


def test_high_order_weights_remain_finite_after_float32_cast():
    q_order = 512
    nodes = 0.5 * (np.polynomial.legendre.leggauss(q_order)[0] + 1.0)

    weights = lagrange.barycentric_lagrange_weights(nodes)
    weights32 = weights.astype(np.float32)

    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(weights32))
    np.testing.assert_allclose(
        np.max(np.abs(weights)), 1.0, rtol=0.0, atol=0.0
    )

    terms = weights32 / (np.float32(0.5) - nodes.astype(np.float32))
    basis_values = terms / np.sum(terms)

    assert np.all(np.isfinite(basis_values))
    np.testing.assert_allclose(
        np.sum(basis_values), 1.0, rtol=1.0e-6, atol=1.0e-6
    )


# vim: foldmethod=marker:filetype=python
