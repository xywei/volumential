__copyright__ = "Copyright (C) 2020 Xiaoyu Wei"

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
import pyopencl as cl

from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory)
from volumential.interpolation import (ElementsToSourcesLookupBuilder,
        interpolate_from_meshmode)
from volumential.geometry import BoundingBoxFactory, BoxFMMGeometryFactory


# {{{ test data

def random_polynomial_func(dim, degree, seed=None):
    """Makes a random polynomial function.
    """
    if seed is not None:
        np.random.seed(seed)
    coefs = np.random.rand(*((degree + 1, ) * dim))

    def poly_func(pts):
        if dim == 1 and len(pts.shape) == 1:
            pts = pts.reshape([1, -1])
        assert pts.shape[0] == dim
        npts = pts.shape[1]
        res = np.zeros(npts)
        for deg in np.ndindex(coefs.shape):
            for iaxis in range(dim):
                res += coefs[deg] * pts[iaxis, :]**deg[iaxis]
        return res

    return poly_func


def eval_func_on_discr_nodes(queue, discr, func):
    nodes = discr.nodes().get(queue)
    fvals = func(nodes)
    return cl.array.to_device(queue, fvals)

# }}} End test data


def drive_test_from_meshmode_interpolation_2d(
        cl_ctx, queue,
        degree, nel_1d,
        n_levels, q_order,
        a=-0.5, b=0.5, seed=0):
    """
    meshmode mesh control: nel_1d, degree
    volumential mesh control: n_levels, q_order
    """
    dim = 2

    mesh = generate_regular_rect_mesh(
            a=(a, a),
            b=(b, b),
            n=(nel_1d, nel_1d))

    group_factory = PolynomialWarpAndBlendGroupFactory(order=degree)
    discr = Discretization(cl_ctx, mesh, group_factory)

    bbox_fac = BoundingBoxFactory(dim=2)
    boxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=n_levels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=mesh, mesh_padding_factor=0.)
    boxgeo = boxfmm_fac(queue)
    lookup_fac = ElementsToSourcesLookupBuilder(
            cl_ctx, tree=boxgeo.tree, discr=discr)
    lookup, evt = lookup_fac(queue)

    func = random_polynomial_func(dim, degree, seed)

    dof_vec = eval_func_on_discr_nodes(queue, discr, func)
    res = interpolate_from_meshmode(queue, dof_vec, lookup).get(queue)

    tree = boxgeo.tree.get(queue)
    ref = func(np.vstack(tree.sources))

    assert np.allclose(ref, res)


def test_from_meshmode_interpolation_2d_1(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_from_meshmode_interpolation_2d(cl_ctx, queue, 1, 8, 5, 3)


if __name__ == '__main__':
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_from_meshmode_interpolation_2d(cl_ctx, queue, 1, 8, 5, 3)
