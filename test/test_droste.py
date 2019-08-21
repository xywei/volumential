__copyright__ = "Copyright (C) 2019 Xiaoyu Wei"

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

import pytest
import loopy as lp
import numpy as np
import pyopencl as cl
from volumential.droste import DrosteReduced
from numpy.polynomial.chebyshev import chebval

# setup ctx_getter fixture
from pyopencl.tools import (  # NOQA
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)


def drive_test_cheb_poly(queue, deg, nnodes):

    droster = DrosteReduced()
    code = droster.codegen_basis_eval(0)

    nodes = np.linspace(-1, 1, nnodes)
    nodes_dev = cl.array.to_device(queue, nodes)

    knl = lp.make_kernel(
            """{ [i, f0, p0]: 0<=i<n and 0<=f0<d and 0<=p0<d }
            """,
            ["""
            for i
                <> template_mapped_point_tmp[0] = nodes[i] {id=mpoint}

                for f0
                    EVAL_CHEB_POINT
                    results[f0, i] = basis_eval0
                end
            end
            """.replace('EVAL_CHEB_POINT', code)],
            lang_version=(2018, 2)
            )

    evt, res = knl(queue, nodes=nodes_dev, d=deg)
    evt.wait()

    cheb_vals_lp = res[0].get()

    cheb_coefs = np.zeros(deg)
    for m in range(deg):
        cheb_coefs.fill(0)
        cheb_coefs[m] = 1
        cheb_vals_np = chebval(nodes, cheb_coefs)
        assert np.allclose(cheb_vals_lp[m], cheb_vals_np)


def test_cheb_poly(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)
    nnodes = 100
    drive_test_cheb_poly(queue, 5, nnodes)
    drive_test_cheb_poly(queue, 15, nnodes)
    drive_test_cheb_poly(queue, 25, nnodes)
    drive_test_cheb_poly(queue, 35, nnodes)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])


# vim: filetype=pyopencl:foldmethod=marker
