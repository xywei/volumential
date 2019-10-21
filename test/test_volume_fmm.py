from __future__ import division, absolute_import, print_function

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

# import volumential.volume_fmm as vfmm

import numpy as np
import pyopencl as cl

# import pyopencl.array
# import boxtree as bt
# import sumpy as sp
# import volumential as vm

from functools import partial

import logging

logger = logging.getLogger(__name__)

import volumential.meshgen as mg

import pytest

# {{{ make sure context getter works


def test_cl_ctx_getter(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(
        ctx,
        """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g,
        __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """,
    ).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    assert np.linalg.norm(res_np - (a_np + b_np)) < 1e-12


# }}}

# {{{ laplace volume potential


@pytest.fixture
def laplace_problem(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dim = 2
    dtype = np.float64

    q_order = 2  # quadrature order
    n_levels = 3  # 2^(n_levels-1) subintervals in 1D

    # adaptive_mesh = True
    n_refinement_loops = 100
    refined_n_cells = 1000
    rratio_top = 0.2
    rratio_bot = 0.5

    # bounding box
    a = -1
    b = 1

    m_order = 15  # multipole order

    alpha = 160 / np.sqrt(2)

    def source_field(x):
        assert len(x) == dim
        assert dim == 2
        norm2 = x[0] ** 2 + x[1] ** 2
        lap_u = (4 * alpha ** 2 * norm2 - 4 * alpha) * np.exp(-alpha * norm2)
        return -lap_u

    def exact_solu(x, y):
        norm2 = x ** 2 + y ** 2
        return np.exp(-alpha * norm2)

    # {{{ generate quad points

    mesh = mg.MeshGen2D(q_order, n_levels)
    iloop = 0
    while mesh.n_active_cells() < refined_n_cells:
        iloop += 1
        crtr = np.array(
            [
                np.abs(source_field(c) * m)
                for (c, m) in zip(mesh.get_cell_centers(), mesh.get_cell_measures())
            ]
        )
        mesh.update_mesh(crtr, rratio_top, rratio_bot)
        if iloop > n_refinement_loops:
            print("Max number of refinement loops reached.")
            break

    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()
    # q_radii = None

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points_org = q_points
    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import make_obj_array

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)]
    )

    q_weights = cl.array.to_device(queue, q_weights)
    # q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(
        queue, np.array([source_field(qp) for qp in q_points_org])
    )

    # particle_weigt = source_val * q_weight

    # }}} End discretize the source field

    # {{{ build tree and traversals

    from boxtree.tools import AXIS_NAMES

    axis_names = AXIS_NAMES[:dim]

    from pytools import single_valued

    coord_dtype = single_valued(coord.dtype for coord in q_points)
    from boxtree.bounding_box import make_bounding_box_dtype

    bbox_type, _ = make_bounding_box_dtype(ctx.devices[0], dim, coord_dtype)

    bbox = np.empty(1, bbox_type)
    for ax in axis_names:
        bbox["min_" + ax] = a
        bbox["max_" + ax] = b

    # tune max_particles_in_box to reconstruct the mesh
    from boxtree import TreeBuilder

    tb = TreeBuilder(ctx)
    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=q_order ** 2 * 4 - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    # }}} End build tree and traversals

    # {{{ build near field potential table

    from volumential.table_manager import NearFieldInteractionTableManager

    tm = NearFieldInteractionTableManager("../nft.hdf5")
    nftable, _ = tm.get_table(dim, "Laplace", q_order)

    # }}} End build near field potential table

    # {{{ sumpy expansion for laplace kernel

    from sumpy.kernel import LaplaceKernel

    # from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    # from sumpy.expansion.local import VolumeTaylorLocalExpansion

    from sumpy.expansion.multipole import (
        LaplaceConformingVolumeTaylorMultipoleExpansion,
    )
    from sumpy.expansion.local import LaplaceConformingVolumeTaylorLocalExpansion

    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion
    # local_expn_class = VolumeTaylorLocalExpansion
    # mpole_expn_class = VolumeTaylorMultipoleExpansion

    exclude_self = True
    from volumential.expansion_wrangler_fpnd import (
            FPNDExpansionWranglerCodeContainer,
            FPNDExpansionWrangler)

    wcc = FPNDExpansionWranglerCodeContainer(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        out_kernels,
        exclude_self=exclude_self,
    )

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue,
        tree=tree,
        near_field_table=nftable,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
    )

    # }}} End sumpy expansion for laplace kernel

    return trav, wrangler, source_vals, q_weights


# }}} End laplace volume potential


@pytest.mark.skipif(
    mg.provider != "meshgen_dealii", reason="Adaptive mesh module is not available"
)
def test_volume_fmm_laplace(laplace_problem):

    trav, wrangler, source_vals, q_weights = laplace_problem

    from volumential.volume_fmm import drive_volume_fmm

    direct_pot, = drive_volume_fmm(
        trav, wrangler, source_vals * q_weights, source_vals, direct_evaluation=True
    )

    fmm_pot, = drive_volume_fmm(
        trav, wrangler, source_vals * q_weights, source_vals, direct_evaluation=False
    )

    assert max(abs(direct_pot - fmm_pot)) < 1e-6


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])


# vim: filetype=pyopencl:foldmethod=marker
