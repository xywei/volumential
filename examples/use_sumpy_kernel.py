""" This example evaluates the volume potential over
    [-1,1]^2 with a Sumpy EpxressionKernel.
"""

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

import logging
import numpy as np
import pyopencl as cl
from volumential.tools import ScalarFieldExpressionEvaluation as Eval
from volumential.table_manager import NearFieldInteractionTableManager
from sumpy.expansion import DefaultExpansionFactory

import loopy as lp

from pymbolic import var
from pymbolic.primitives import make_sym_vector
from sumpy.kernel import ExpressionKernel, KernelArgument
from sumpy.symbolic import pymbolic_real_norm_2

from functools import partial

logger = logging.getLogger(__name__)

if 1:
    # verbose
    logging.basicConfig(level=logging.INFO)
else:
    # clean
    logging.basicConfig(level=logging.CRITICAL)


class GaussianKernel(ExpressionKernel):

    init_arg_names = ("dim", "standart_deviation_name")

    def __init__(self, dim, standard_deviation_name="sigma"):
        sigma = var(standard_deviation_name)
        r = pymbolic_real_norm_2(make_sym_vector("d", dim))
        expr = var("exp")(-1/2 * (r / sigma)**2)
        scaling = 1 / (np.sqrt((2 * np.pi)**dim) * sigma)

        super(GaussianKernel, self).__init__(
            dim,
            expression=expr,
            global_scaling_const=scaling,
            is_complex_valued=False)

        self.standard_deviation_name = standard_deviation_name

    def __getinitargs__(self):
        return (self.dim, self.standard_deviation_name)

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, (self.dim, self.standard_deviation_name))

    def __repr__(self):
        return "GaussKnl%dD(%s)" % (self.dim, self.standard_deviation_name)

    def get_args(self):
        sigma_dtype = np.float64
        return [KernelArgument(
            loopy_arg=lp.ValueArg(self.standard_deviation_name, sigma_dtype),)]

    mapper_method = "map_expression_kernel"


def main():

    print("*************************")
    print("* Setting up...")
    print("*************************")

    dim = 2
    table_filename = "nft_gauss.hdf5"

    print("Using table cache:", table_filename)

    q_order = 10   # quadrature order
    n_levels = 7  # 2^(n_levels-1) subintervals in 1D
    dtype = np.float64

    m_order = 10  # multipole order
    force_direct_evaluation = False

    print("Multipole order =", m_order)

    x = var("x")
    y = var("y")
    source_expr = var("sin")(x) * y
    logger.info("Source expr: " + str(source_expr))

    # bounding box
    a = -0.5
    b = 0.5
    root_table_source_extent = (b - a)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    source_eval = Eval(dim, source_expr, [x, y])

    # {{{ generate quad points

    import volumential.meshgen as mg

    mesh = mg.MeshGen2D(q_order, n_levels, a, b, queue=queue)
    mesh.print_info()
    q_points = mesh.get_q_points()
    q_weights = mesh.get_q_weights()

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import make_obj_array

    q_points = make_obj_array(
        [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

    q_weights = cl.array.to_device(queue, q_weights)

    # }}}

    source_vals = cl.array.to_device(
        queue, source_eval(queue, np.array([coords.get() for coords in q_points]))
    )

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
        max_particles_in_box=q_order ** 2 * 4 - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    # }}} End build tree and traversals

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent,
        queue=queue)

    # condition such that the table file suits the domain
    assert (abs(
        int((b - a) / root_table_source_extent) * root_table_source_extent
        - (b - a)) < 1e-15)

    # uniform tree
    min_lev = tree.nlevels
    max_lev = tree.nlevels

    integral_kernel = GaussianKernel(dim)
    extra_kernel_kwargs = {"sigma": np.float64(0.05)}
    out_kernels = [integral_kernel]

    nftable = []
    for lev in range(min_lev, max_lev + 1):
        print("Getting table at level", lev)
        tb, _ = tm.get_table(
            dim=dim, q_order=q_order,
            sumpy_knl=integral_kernel, kernel_type=str(integral_kernel),
            source_box_level=lev,
            compute_method="DrosteSum", queue=queue,
            n_brick_quad_poitns=25,
            adaptive_level=True, adaptive_quadrature=True, use_symmetry=False,
            alpha=0., nlevels=1, force_recompute=False,
            extra_kernel_kwargs=extra_kernel_kwargs
        )
        nftable.append(tb)

    print("Using table list of length", len(nftable))

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(integral_kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(integral_kernel)

    exclude_self = True

    from volumential.expansion_wrangler_fpnd import (
            FPNDExpansionWranglerCodeContainer,
            FPNDExpansionWrangler
            )

    wcc = FPNDExpansionWranglerCodeContainer(
        ctx,
        partial(mpole_expn_class, integral_kernel),
        partial(local_expn_class, integral_kernel),
        out_kernels,
        exclude_self=exclude_self,
    )

    if exclude_self:
        target_to_source = np.arange(tree.ntargets, dtype=np.int32)
        self_extra_kwargs = {"target_to_source": target_to_source}
    else:
        self_extra_kwargs = {}

    wrangler = FPNDExpansionWrangler(
        code_container=wcc, queue=queue, tree=tree,
        near_field_table=nftable, dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
        kernel_extra_kwargs=extra_kernel_kwargs,
    )

    print("*************************")
    print("* Performing FMM ...")
    print("*************************")

    from volumential.volume_fmm import drive_volume_fmm

    import time
    queue.finish()

    t0 = time.time()

    pot, = drive_volume_fmm(
        trav,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=force_direct_evaluation,
    )
    queue.finish()

    t1 = time.time()

    print("Finished in %.2f seconds." % (t1 - t0))
    print("(%e points per second)" % (
        len(q_weights) / (t1 - t0)
        ))


if __name__ == '__main__':
    main()


# vim: filetype=pyopencl:foldmethod=marker
