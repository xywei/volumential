
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


def test_cheb_poly(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    nnodes = 100
    drive_test_cheb_poly(queue, 5, nnodes)
    drive_test_cheb_poly(queue, 15, nnodes)
    drive_test_cheb_poly(queue, 25, nnodes)
    drive_test_cheb_poly(queue, 35, nnodes)


def drive_test_cheb_table(
        queue, integral_kernel, q_order, source_box_r,
        n_brick_q_points=120, kernel_symmetry_tags=None):

    from pypvfmm import cheb_utils
    dtype = np.float64
    cheb_degree = q_order
    assert integral_kernel.dim == 3  # PvFMM only supports 3D

    droster = DrosteReduced(
            integral_kernel, cheb_degree,
            [(0, 0, 0), ], n_brick_q_points, kernel_symmetry_tags)

    q_points = droster.get_target_points()
    assert len(q_points) == droster.ntgt_points ** droster.dim
    t = np.array([pt[-1] for pt in q_points[: droster.ntgt_points]])

    # table via droste
    cheb_table = droster.build_cheb_table(
            queue,
            source_box_extent=source_box_r,
            alpha=0, nlevels=1,
            n_brick_quad_points=n_brick_q_points,
            adaptive_level=False, adaptive_quadrature=False)

    print(cheb_table)

    # handle AxisTargetDerivative to extract the component of interest
    # (pvfmm returns the full gradient)
    from sumpy.kernel import AxisTargetDerivative
    if isinstance(integral_kernel, AxisTargetDerivative):
        uu_offset = integral_kernel.axis * (cheb_degree * 3)**3
    else:
        uu_offset = 0

    from itertools import product
    for t0, t1, t2 in product(range(cheb_degree), repeat=3):
        target = np.array([t[t0], t[t1], t[t2]], dtype=dtype) * source_box_r

        # table via pypvfmm
        uu = cheb_utils.integ(cheb_degree * 3, target,
                source_box_r, n_brick_q_points, integral_kernel)

        err = 0.  # l_inf error for the target (t0, t1, t2)
        for f0, f1, f2 in product(range(cheb_degree), repeat=3):
            resid = abs(
                    (uu[
                        uu_offset + f2 * (cheb_degree * 3)**2
                        + f1 * (cheb_degree * 3) + f0
                        ])
                    - (cheb_table[f2, f1, f0, t2, t1, t0, 0])
                    ) / (source_box_r ** 3)
            err = max(err, resid)
            if resid > 1e-12:
                print("Error found at (%d, %d, %d): %f" % (f0, f1, f2, resid))
        assert err < 1e-12


def drive_test_cheb_tables_laplace3d(requires_pypvfmm, ctx_factory, q_order):
    """Test Chebyshev table vs PvFMM's computation for
    box-self interactions.
    """
    from sumpy.kernel import LaplaceKernel

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    kernel = LaplaceKernel(3)
    source_box_r = 3.14

    drive_test_cheb_table(
            queue, kernel, q_order, source_box_r,
            n_brick_q_points=120, kernel_symmetry_tags=None)


def drive_test_cheb_tables_grad_laplace3d(
        requires_pypvfmm, ctx_factory, q_order, axis):
    from sumpy.kernel import LaplaceKernel, AxisTargetDerivative
    from volumential.list1_symmetry import Flip, Swap

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    kernel = AxisTargetDerivative(axis, LaplaceKernel(3))
    source_box_r = 1.69

    from itertools import product
    symmetry_tags = [Flip(iaxis) for iaxis in range(3) if iaxis != axis] + \
            [Swap(iaxis, jaxis) for iaxis, jaxis in product(range(3), repeat=2)
                    if (iaxis < jaxis and iaxis != axis and jaxis != axis)]
    print(symmetry_tags)

    drive_test_cheb_table(
            queue, kernel, q_order, source_box_r,
            n_brick_q_points=120, kernel_symmetry_tags=symmetry_tags)


@pytest.mark.parametrize("q_order", [1, 2])
def test_cheb_tables_laplace3d_quick(requires_pypvfmm, ctx_factory, q_order):
    drive_test_cheb_tables_laplace3d(requires_pypvfmm, ctx_factory, q_order)


@pytest.mark.parametrize("q_order", [1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cheb_tables_grad_laplace3d_quick(
        requires_pypvfmm, ctx_factory, q_order, axis):
    drive_test_cheb_tables_grad_laplace3d(
            requires_pypvfmm, ctx_factory, q_order, axis)


@pytest.mark.parametrize("q_order", [3, 4, 5, 6])
def test_cheb_tables_laplace3d_slow(longrun, requires_pypvfmm, ctx_factory, q_order):
    drive_test_cheb_tables_laplace3d(requires_pypvfmm, ctx_factory, q_order)


@pytest.mark.parametrize("q_order", [3, 4, 5, 6])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cheb_tables_grad_laplace3d_slow(
        longrun, requires_pypvfmm, ctx_factory, q_order, axis):
    drive_test_cheb_tables_grad_laplace3d(
            requires_pypvfmm, ctx_factory, q_order, axis)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])


# vim: filetype=pyopencl:foldmethod=marker
