
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

import numpy as np
import numpy.linalg as la
import pytest

import pyopencl as cl
import pyopencl.clmath  # noqa
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory)
from pytential.target import PointsTarget

from volumential.function_extension import compute_harmonic_extension


@pytest.mark.skip(reason="this test needs to be updated to use GeometryCollection")
def test_harmonic_extension_exterior_3d(ctx_factory):

    dim = 3  # noqa: F841
    mesh_order = 8
    fmm_order = 3
    bdry_quad_order = mesh_order
    bdry_ovsmp_quad_order = 4 * bdry_quad_order
    qbx_order = mesh_order

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_icosphere
    from meshmode.mesh.refinement import refine_uniformly

    radius = 2.5
    base_mesh = generate_icosphere(radius, order=mesh_order)
    mesh = refine_uniformly(base_mesh, 1)

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            ).with_refinement()
    density_discr = qbx.density_discr

    ntgts = 200
    rho = np.random.rand(ntgts) * 3 + radius + 0.05
    theta = np.random.rand(ntgts) * 2 * np.pi
    phi = (np.random.rand(ntgts) - 0.5) * np.pi

    nodes = density_discr.nodes().with_queue(queue)
    source = np.array([0, 1, 2])

    def test_func(x):
        return 1.0/la.norm(x.get()-source[:, None], axis=0)

    f = cl.array.to_device(queue, test_func(nodes))

    targets = cl.array.to_device(queue,
            np.array([
                rho * np.cos(phi) * np.cos(theta),
                rho * np.cos(phi) * np.sin(theta),
                rho * np.sin(phi)])
            )

    exact_f = test_func(targets)

    ext_f, _ = compute_harmonic_extension(
            queue, PointsTarget(targets), qbx, density_discr,
            f, loc_sign=1)

    assert np.linalg.norm(exact_f - ext_f.get()) < 1e-3 * np.linalg.norm(exact_f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])


# vim: filetype=pyopencl:foldmethod=marker
