""" This example evaluates the volume potential over
    [-1,1]^2 with the Helmholtz kernel.
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

import pymbolic as pmbl
from functools import partial


logger = logging.getLogger(__name__)

if 1:
    # verbose
    logging.basicConfig(level=logging.INFO)
else:
    # clean
    logging.basicConfig(level=logging.CRITICAL)


def main():

    print("*************************")
    print("* Setting up...")
    print("*************************")

    dim = 2

    download_table = False
    table_filename = "nft_helmholtz2d.hdf5"

    print("Using table cache:", table_filename)

    q_order = 5   # quadrature order
    n_levels = 7  # 2^(n_levels-1) subintervals in 1D

    dtype = np.complex128
    m_order = 20  # multipole order
    force_direct_evaluation = False

    print("Multipole order =", m_order)

    k = 3
    alpha = 160

    from sumpy.kernel import HelmholtzKernel
    integral_kernel = HelmholtzKernel(dim)
    out_kernels = [integral_kernel]

    # FIXME: specify dtype from the kernel interface
    extra_kernel_kwargs = {"k": np.float64(k)}

    x = pmbl.var("x")
    y = pmbl.var("y")
    expp = pmbl.var("exp")

    norm2 = x ** 2 + y ** 2
    solu_expr = expp(-alpha * norm2)
    source_expr = -(
        (4 * alpha ** 2 * norm2 - 4 * alpha) * expp(-alpha * norm2)
        + k**2 * solu_expr
        )

    logger.info("Source expr: " + str(source_expr))
    logger.info("Solu expr: " + str(solu_expr))

    # bounding box
    a = -1
    b = 1
    root_table_source_extent = 2

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    source_eval = Eval(dim, source_expr, [x, y])

    # {{{ generate quad points

    import volumential.meshgen as mg

    # Show meshgen info
    mg.greet()

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
    # q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(
        queue, source_eval(queue, np.array([coords.get() for coords in q_points]))
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

    bbox2 = np.array([[a, b], [a, b]])
    tree2, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox2,
        max_particles_in_box=q_order ** 2 * 4 - 1,
        kind="adaptive-level-restricted",
    )

    from boxtree.traversal import FMMTraversalBuilder

    tg = FMMTraversalBuilder(ctx)
    trav, _ = tg(queue, tree)

    # }}} End build tree and traversals

    # {{{ build near field potential table

    from volumential.table_manager import NearFieldInteractionTableManager
    import os

    if download_table and (not os.path.isfile(table_filename)):
        import json
        with open("table_urls.json", 'r') as fp:
            urls = json.load(fp)

        print("Downloading table from %s" % urls['Helmholtz2D'])
        import subprocess
        subprocess.call(["wget", "-q", urls['Helmholtz2D'], table_filename])

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent,
        queue=queue, dtype=dtype)

    assert (abs(
      int((b - a) / root_table_source_extent) * root_table_source_extent
      - (b - a)) < 1e-15)
    nftable = []

    # FIXME: need at leat two levels even for uniform trees?
    for lev in range(tree.nlevels - 1, tree.nlevels + 1):
        print("Getting table at level", lev)
        tb, _ = tm.get_table(
            dim, kernel_type="Helmholtz", sumpy_knl=integral_kernel,
            q_order=q_order, source_box_level=lev,
            compute_method="DrosteSum", queue=queue,
            n_brick_quad_points=50, adaptive_level=True,
            adaptive_quadrature=True, use_symmetry=True,
            alpha=0.1, nlevels=15,
            extra_kernel_kwargs=extra_kernel_kwargs
        )
        nftable.append(tb)

    print("Using table list of length", len(nftable))

    # }}} End build near field potential table

    # {{{ sumpy expansion for laplace kernel

    from sumpy.expansion import DefaultExpansionFactory
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

    # FIXME: unify naming convention: kernel_extra_kwargs vs extra_kernel_kwargs
    wrangler = FPNDExpansionWrangler(
        code_container=wcc,
        queue=queue, tree=tree,
        near_field_table=nftable,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
        kernel_extra_kwargs=extra_kernel_kwargs
    )

    # }}} End sumpy expansion for laplace kernel

    print("*************************")
    print("* Performing FMM ...")
    print("*************************")

    # {{{ conduct fmm computation

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

    # }}} End conduct fmm computation

    print("*************************")
    print("* Postprocessing ...")
    print("*************************")

    # {{{ postprocess and plot

    # print(pot)

    solu_eval = Eval(dim, solu_expr, [x, y])

    x = q_points[0].get()
    y = q_points[1].get()
    ze = solu_eval(queue, np.array([x, y]))
    zs = pot.get()

    print_error = True
    if print_error:
        err = np.max(np.abs(ze - zs))
        print("Error =", err)

    # Interpolated surface
    if 0:
        h = 0.005
        out_x = np.arange(a, b + h, h)
        out_y = np.arange(a, b + h, h)
        oxx, oyy = np.meshgrid(out_x, out_y)
        out_targets = make_obj_array(
            [
                cl.array.to_device(queue, oxx.flatten()),
                cl.array.to_device(queue, oyy.flatten()),
            ]
        )

        from volumential.volume_fmm import interpolate_volume_potential

        # src = source_field([q.get() for q in q_points])
        # src = cl.array.to_device(queue, src)
        interp_pot = interpolate_volume_potential(out_targets, trav, wrangler, pot)
        opot = interp_pot.get()

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        plt3d = plt.figure()
        ax = Axes3D(plt3d)  # noqa
        surf = ax.plot_surface(oxx, oyy, opot.reshape(oxx.shape))  # noqa
        # ax.scatter(x, y, src.get())
        # ax.set_zlim(-0.25, 0.25)

        plt.draw()
        plt.show()

    # Boxtree
    if 0:
        import matplotlib.pyplot as plt

        if dim == 2:
            # plt.plot(q_points[0].get(), q_points[1].get(), ".")
            pass

        from boxtree.visualization import TreePlotter

        plotter = TreePlotter(tree.get(queue=queue))
        plotter.draw_tree(fill=False, edgecolor="black")
        # plotter.draw_box_numbers()
        plotter.set_bounding_box()
        plt.gca().set_aspect("equal")

        plt.draw()
        # plt.show()
        plt.savefig("tree.png")

    # Direct p2p
    if 0:
        print("Performing P2P")
        pot_direct, = drive_volume_fmm(
            trav, wrangler, source_vals * q_weights,
            source_vals, direct_evaluation=True)
        zds = pot_direct.get()
        zs = pot.get()

        print("P2P-FMM diff =", np.max(np.abs(zs - zds)))

        print("P2P Error =", np.max(np.abs(ze - zds)))

        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        x = q_points[0].get()
        y = q_points[1].get()
        plt.scatter(x, y, c=np.log(abs(zs-zds)) / np.log(10), cmap=cm.jet)
        plt.colorbar()

        plt.xlabel("Multipole order = " + str(m_order))

        plt.draw()
        plt.show()
        """

    # Scatter plot
    if 0:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        x = q_points[0].get()
        y = q_points[1].get()
        ze = solu_eval(queue, np.array([x, y]))
        zs = pot.get()

        plt3d = plt.figure()
        ax = Axes3D(plt3d)
        ax.scatter(x, y, zs, s=1)
        # ax.scatter(x, y, source_field([q.get() for q in q_points]), s=1)
        # import matplotlib.cm as cm

        # ax.scatter(x, y, zs, c=np.log(abs(zs-zds)), cmap=cm.jet)
        # plt.gca().set_aspect("equal")

        # ax.set_xlim3d([-1, 1])
        # ax.set_ylim3d([-1, 1])
        # ax.set_zlim3d([np.min(z), np.max(z)])
        # ax.set_zlim3d([-0.002, 0.00])

        plt.draw()
        plt.show()
        # plt.savefig("exact.png")

    # }}} End postprocess and plot


if __name__ == '__main__':
    main()


# vim: filetype=pyopencl:foldmethod=marker
