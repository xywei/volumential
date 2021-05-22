from __future__ import absolute_import, division, print_function

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

import subprocess
import numpy as np
import pyopencl as cl
import pyopencl.array  # NOQA
from functools import partial


def drive_test_completeness(ctx, queue, dim, q_order):

    n_levels = 2  # 2^(n_levels-1) subintervals in 1D, must be at least 2

    # bounding box
    a = -1
    b = 1

    dtype = np.float64

    def source_field(x):
        assert len(x) == dim
        return 1

    # {{{ generate quad points

    import volumential.meshgen as mg

    q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
        degree=q_order, level=n_levels, dim=dim
    )

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points_org = q_points
    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import make_obj_array

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)]
    )

    q_weights = cl.array.to_device(queue, q_weights)
    if q_radii is not None:
        q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(
        queue, np.array([source_field(qp) for qp in q_points_org])
    )

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
    # TODO: use points from FieldPlotter are used as target points for better
    # visuals
    from boxtree import TreeBuilder

    tb = TreeBuilder(ctx)
    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=q_order ** dim * (2 ** dim) - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    # }}} End build tree and traversals

    from volumential.table_manager import NearFieldInteractionTableManager

    subprocess.check_call(['rm', '-f', 'nft-test-completeness.hdf5'])
    with NearFieldInteractionTableManager("nft-test-completeness.hdf5",
                                          progress_bar=False) as tm:

        nft, _ = tm.get_table(
            dim, "Constant", q_order, queue=queue, n_levels=1, alpha=0,
            compute_method="DrosteSum", n_brick_quad_points=50,
            adaptive_level=False, use_symmetry=True)

    # {{{ expansion wrangler

    from sumpy.kernel import LaplaceKernel
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import VolumeTaylorLocalExpansion

    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    local_expn_class = partial(VolumeTaylorLocalExpansion, use_rscale=None)
    mpole_expn_class = partial(VolumeTaylorMultipoleExpansion, use_rscale=None)

    from volumential.expansion_wrangler_fpnd import (
            FPNDExpansionWranglerCodeContainer,
            FPNDExpansionWrangler)

    wcc = FPNDExpansionWranglerCodeContainer(
        ctx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        out_kernels,
        exclude_self=True,
    )

    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue,
        tree=tree,
        near_field_table=nft,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: 1,
        quad_order=q_order,
    )

    # }}} End sumpy expansion for laplace kernel
    pot = wrangler.eval_direct(
        trav.target_boxes, trav.neighbor_source_boxes_starts,
        trav.neighbor_source_boxes_lists, mode_coefs=source_vals)
    pot = pot[0]
    for p in pot[0]:
        assert(abs(p - 2**dim) < 1e-8)


def test_completeness_1(ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    drive_test_completeness(ctx, queue, 2, 1)
    drive_test_completeness(ctx, queue, 3, 1)


def test_completeness(longrun, ctx_factory):

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    for q in range(2, 4):
        drive_test_completeness(ctx, queue, 2, q)
        drive_test_completeness(ctx, queue, 3, q)
