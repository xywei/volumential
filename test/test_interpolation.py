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
import pytest
import numpy as np
import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import flatten, thaw
from meshmode.mesh.generation import generate_regular_rect_mesh
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory)
from volumential.interpolation import (
        ElementsToSourcesLookupBuilder, LeavesToNodesLookupBuilder,
        interpolate_from_meshmode, interpolate_to_meshmode)
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
            mono = np.ones(npts)
            for iaxis in range(dim):
                mono += pts[iaxis, :]**deg[iaxis]
            res += coefs[deg] * mono
        return res

    return poly_func


def eval_func_on_discr_nodes(queue, discr, func):
    arr_ctx = PyOpenCLArrayContext(queue)
    nodes = np.vstack([
        c.get() for c in flatten(thaw(arr_ctx, discr.nodes()))])
    fvals = func(nodes)
    return cl.array.to_device(queue, fvals)

# }}} End test data


def drive_test_from_meshmode_exact_interpolation_2d(
        cl_ctx, queue,
        degree, nel_1d, n_levels, q_order,
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

    arr_ctx = PyOpenCLArrayContext(queue)
    group_factory = PolynomialWarpAndBlendGroupFactory(order=degree)
    discr = Discretization(arr_ctx, mesh, group_factory)

    bbox_fac = BoundingBoxFactory(dim=2)
    boxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=n_levels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=mesh, mesh_padding_factor=0.)
    boxgeo = boxfmm_fac(queue)
    lookup_fac = ElementsToSourcesLookupBuilder(
            cl_ctx, tree=boxgeo.tree, discr=discr)
    lookup, evt = lookup_fac(queue)

    # algebraically exact interpolation
    func = random_polynomial_func(dim, degree, seed)

    dof_vec = eval_func_on_discr_nodes(queue, discr, func)
    res = interpolate_from_meshmode(queue, dof_vec, lookup).get(queue)

    tree = boxgeo.tree.get(queue)
    ref = func(np.vstack(tree.sources))

    return np.allclose(ref, res)


def drive_test_from_meshmode_interpolation_2d(
        cl_ctx, queue,
        degree, nel_1d, n_levels, q_order,
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

    arr_ctx = PyOpenCLArrayContext(queue)
    group_factory = PolynomialWarpAndBlendGroupFactory(order=degree)
    discr = Discretization(arr_ctx, mesh, group_factory)

    bbox_fac = BoundingBoxFactory(dim=2)
    boxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=n_levels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=mesh, mesh_padding_factor=0.)
    boxgeo = boxfmm_fac(queue)
    lookup_fac = ElementsToSourcesLookupBuilder(
            cl_ctx, tree=boxgeo.tree, discr=discr)
    lookup, evt = lookup_fac(queue)

    def func(pts):
        x, y = pts
        return np.sin(x + y) * x

    dof_vec = eval_func_on_discr_nodes(queue, discr, func)
    res = interpolate_from_meshmode(queue, dof_vec, lookup).get(queue)

    tree = boxgeo.tree.get(queue)
    ref = func(np.vstack(tree.sources))

    resid = np.linalg.norm(ref - res, ord=np.inf)
    return resid


def drive_test_to_meshmode_exact_interpolation_2d(
        cl_ctx, queue,
        degree, nel_1d, n_levels, q_order,
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

    arr_ctx = PyOpenCLArrayContext(queue)
    group_factory = PolynomialWarpAndBlendGroupFactory(order=degree)
    discr = Discretization(arr_ctx, mesh, group_factory)

    bbox_fac = BoundingBoxFactory(dim=2)
    boxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=n_levels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=mesh, mesh_padding_factor=0.)
    boxgeo = boxfmm_fac(queue)
    lookup_fac = LeavesToNodesLookupBuilder(
            cl_ctx, trav=boxgeo.trav, discr=discr)
    lookup, evt = lookup_fac(queue)

    # algebraically exact interpolation
    func = random_polynomial_func(dim, degree, seed)

    tree = boxgeo.tree.get(queue)
    potential = func(np.vstack(tree.sources))
    res = interpolate_to_meshmode(queue, potential, lookup).get(queue)

    ref = eval_func_on_discr_nodes(queue, discr, func).get(queue)
    return np.allclose(ref, res)


def drive_test_to_meshmode_interpolation_2d(
        cl_ctx, queue,
        degree, nel_1d, n_levels, q_order,
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

    arr_ctx = PyOpenCLArrayContext(queue)
    group_factory = PolynomialWarpAndBlendGroupFactory(order=degree)
    discr = Discretization(arr_ctx, mesh, group_factory)

    bbox_fac = BoundingBoxFactory(dim=2)
    boxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=n_levels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=mesh, mesh_padding_factor=0.)
    boxgeo = boxfmm_fac(queue)
    lookup_fac = LeavesToNodesLookupBuilder(
            cl_ctx, trav=boxgeo.trav, discr=discr)
    lookup, evt = lookup_fac(queue)

    def func(pts):
        x, y = pts
        return np.sin(x + y) * x

    tree = boxgeo.tree.get(queue)
    potential = func(np.vstack(tree.sources))
    res = interpolate_to_meshmode(queue, potential, lookup).get(queue)

    ref = eval_func_on_discr_nodes(queue, discr, func).get(queue)
    resid = np.linalg.norm(ref - res, ord=np.inf)
    return resid


@pytest.mark.parametrize("params", [
    [1, 8, 3, 1], [1, 8, 5, 2], [1, 8, 7, 3],
    [2, 16, 3, 1], [2, 32, 5, 2], [2, 64, 7, 3],
    ])
def test_from_meshmode_interpolation_2d_exact(ctx_factory, params):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    assert drive_test_from_meshmode_exact_interpolation_2d(
            cl_ctx, queue, *params)


@pytest.mark.parametrize("params", [
    [1, 32, 5, 1], [2, 64, 5, 3], [3, 64, 5, 4]
    ])
def test_from_meshmode_interpolation_2d_nonexact(ctx_factory, params):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    assert drive_test_from_meshmode_interpolation_2d(
            cl_ctx, queue, *params) < 1e-3


@pytest.mark.parametrize("params", [
    [1, 8, 3, 2], [1, 8, 5, 2], [1, 8, 7, 3],
    [2, 16, 3, 3], [3, 32, 5, 4], [4, 64, 7, 5],
    ])
def test_to_meshmode_interpolation_2d_exact(ctx_factory, params):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    assert drive_test_to_meshmode_exact_interpolation_2d(
            cl_ctx, queue, *params)


@pytest.mark.parametrize("params", [
    [1, 32, 7, 2], [2, 64, 5, 3], [3, 64, 6, 4]
    ])
def test_to_meshmode_interpolation_2d_nonexact(ctx_factory, params):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    assert drive_test_to_meshmode_interpolation_2d(
            cl_ctx, queue, *params) < 1e-3


if __name__ == '__main__':
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    drive_test_from_meshmode_interpolation_2d(cl_ctx, queue, 1, 8, 5, 3)