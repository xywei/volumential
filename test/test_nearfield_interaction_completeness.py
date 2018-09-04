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

import numpy as np
import pyopencl as cl
import pyopencl.array # NOQA

from functools import partial


def drive_test_completeness(q_order):

    dim = 2

    n_levels = 2  # 2^(n_levels-1) subintervals in 1D, must be at least 2

    # bounding box
    a = -1
    b = 1

    dtype = np.float64

    def source_field(x):
        assert (len(x) == dim)
        return 1

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # {{{ generate quad points

    import volumential.meshgen as mg
    q_points, q_weights, q_radii = mg.make_uniform_cubic_grid(
        degree=q_order, level=n_levels)

    assert (len(q_points) == len(q_weights))
    assert (q_points.shape[1] == dim)

    q_points_org = q_points
    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import make_obj_array
    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

    q_weights = cl.array.to_device(queue, q_weights)
    q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(queue,
                                     np.array(
                                         [source_field(qp) for qp in q_points_org]))

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
        max_particles_in_box=q_order**2 * 4 - 1,
        kind="adaptive-level-restricted")

    from boxtree.traversal import FMMTraversalBuilder
    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    # }}} End build tree and traversals

    from volumential.table_manager import NearFieldInteractionTableManager
    tm = NearFieldInteractionTableManager("nft.hdf5")
    nft, _ = tm.get_table(dim, "Constant", q_order)

    # {{{ expansion wrangler

    from sumpy.kernel import LaplaceKernel
    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import VolumeTaylorLocalExpansion

    knl = LaplaceKernel(dim)
    out_kernels = [knl]
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion

    from volumential.expansion_wrangler_interface \
            import ExpansionWranglerCodeContainer
    wcc = ExpansionWranglerCodeContainer(ctx,
                                         partial(mpole_expn_class, knl),
                                         partial(local_expn_class, knl),
                                         out_kernels,
                                         exclude_self=True)

    from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler
    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue,
        tree=tree,
        near_field_table=nft,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: 1,
        quad_order=q_order)

    # }}} End sumpy expansion for laplace kernel
    pot = wrangler.eval_direct(trav.target_boxes,
                               trav.neighbor_source_boxes_starts,
                               trav.neighbor_source_boxes_lists,
                               mode_coefs=source_vals)
    pot = pot[0]
    for p in pot[0]:
        assert(abs(p - 4) < 1e-12)


def test_completeness():
    for q in range(1, 4):
        drive_test_completeness(q)
