""" This example evaluates the volume potential over
    [-1,1]^3 with the Laplace kernel.
"""
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

import logging

import numpy as np
import pyopencl as cl

import pymbolic as pmbl
import pymbolic.functions
from volumential.tools import ScalarFieldExpressionEvaluation as Eval

from functools import partial

verbose = True
logger = logging.getLogger(__name__)
if verbose:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.CRITICAL)


def main():

    print("*************************")
    print("* Setting up...")
    print("*************************")

    dim = 3
    # download precomputation results for the 3D Laplace kernel
    download_table = True
    table_filename = "nft_laplace3d.hdf5"

    logger.info("Using table cache: " + table_filename)

    q_order = 7  # quadrature order
    n_levels = 5
    use_multilevel_table = False

    adaptive_mesh = False
    n_refinement_loops = 100
    refined_n_cells = 5e5
    rratio_top = 0.2
    rratio_bot = 0.5

    dtype = np.float64

    m_order = 10  # multipole order
    force_direct_evaluation = False

    logger.info("Multipole order = " + str(m_order))
    logger.info("Quad order = " + str(q_order))
    logger.info("N_levels = " + str(n_levels))

    # a solution that is nearly zero at the boundary
    # exp(-40) = 4.25e-18
    alpha = 80
    x = pmbl.var("x")
    y = pmbl.var("y")
    z = pmbl.var("z")
    expp = pmbl.var("exp")

    norm2 = x ** 2 + y ** 2 + z ** 2
    source_expr = -(4 * alpha ** 2 * norm2 - 6 * alpha) * expp(-alpha * norm2)
    solu_expr = expp(-alpha * norm2)

    logger.info("Source expr: " + str(source_expr))
    logger.info("Solu expr: " + str(solu_expr))

    # bounding box
    a = -0.5
    b = 0.5
    root_table_source_extent = 2

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # logger.info("Summary of params: " + get_param_summary())
    source_eval = Eval(dim, source_expr, [x, y, z])

    # {{{ generate quad points

    import volumential.meshgen as mg

    # Show meshgen info
    mg.greet()

    mesh = mg.MeshGen3D(q_order, n_levels, a, b, queue=queue)
    if not adaptive_mesh:
        mesh.print_info()
        q_points = mesh.get_q_points()
        q_weights = mesh.get_q_weights()
    else:
        iloop = -1
        while mesh.n_active_cells() < refined_n_cells:
            iloop += 1
            cell_centers = mesh.get_cell_centers()
            cell_measures = mesh.get_cell_measures()
            density_vals = source_eval(
                queue,
                np.array([[center[d] for center in cell_centers]
                    for d in range(dim)]))
            crtr = np.abs(cell_measures * density_vals)
            mesh.update_mesh(crtr, rratio_top, rratio_bot)
            if iloop > n_refinement_loops:
                print("Max number of refinement loops reached.")
                break

        mesh.print_info()
        q_points = mesh.get_q_points()
        q_weights = mesh.get_q_weights()

    if 1:
        try:
            mesh.generate_gmsh("box_grid.msh")
        except Exception as e:
            print(e)
            pass

        legacy_msh_file = True
        if legacy_msh_file:
            import os

            os.system("gmsh box_grid.msh convert_grid -")

    assert len(q_points) == len(q_weights)
    assert q_points.shape[1] == dim

    q_points = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import make_obj_array

    q_points = make_obj_array(
            [cl.array.to_device(queue, q_points[i]) for i in range(dim)])

    q_weights = cl.array.to_device(queue, q_weights)

    # }}}

    # {{{ discretize the source field

    logger.info("discretizing source field")
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
    print("building tree")
    from boxtree import TreeBuilder

    tb = TreeBuilder(ctx)
    tree, _ = tb(
        queue,
        particles=q_points,
        targets=q_points,
        bbox=bbox,
        max_particles_in_box=q_order ** 3 * 8 - 1,
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

        print("Downloading table from %s" % urls['Laplace3D'])
        import subprocess
        subprocess.call(["wget", "-q", urls['Laplace3D'], table_filename])

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent,
        queue=queue)

    if use_multilevel_table:
        logger.info("Using multilevel tables")
        assert (
            abs(
                int((b - a)
                    / root_table_source_extent) * root_table_source_extent
                - (b - a))
            < 1e-15)
        nftable = []
        for lev in range(0, tree.nlevels + 1):
            print("Getting table at level", lev)
            tb, _ = tm.get_table(
                dim,
                "Laplace",
                q_order,
                source_box_level=lev,
                compute_method="DrosteSum",
                queue=queue,
                n_brick_quad_points=120,
                adaptive_level=False,
                use_symmetry=True,
                alpha=0,
                n_levels=1,
            )
            nftable.append(tb)

        print("Using table list of length", len(nftable))

    else:
        logger.info("Using single level table")
        force_recompute = False
        # 15 levels are sufficient (the inner most brick is 1e-15**3 in volume)
        nftable, _ = tm.get_table(
            dim,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            compute_method="DrosteSum",
            queue=queue,
            n_brick_quad_points=120,
            adaptive_level=False,
            use_symmetry=True,
            alpha=0,
            n_levels=1,
        )

    # }}} End build near field potential table

    # {{{ sumpy expansion for laplace kernel

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    knl = LaplaceKernel(dim)
    out_kernels = [knl]

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

    exclude_self = True
    from volumential.expansion_wrangler_fpnd import (
            FPNDExpansionWrangler,
            FPNDExpansionWranglerCodeContainer)

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
        list1_only=False)

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

    solu_eval = Eval(dim, solu_expr, [x, y, z])
    # x = q_points[0].get()
    # y = q_points[1].get()
    # z = q_points[2].get()
    test_x = np.array([0.0])
    test_y = np.array([0.0])
    test_z = np.array([0.0])
    test_nodes = make_obj_array(
        # get() first for CL compatibility issues
        [
            cl.array.to_device(queue, test_x),
            cl.array.to_device(queue, test_y),
            cl.array.to_device(queue, test_z),
        ]
    )

    from volumential.volume_fmm import interpolate_volume_potential

    ze = solu_eval(queue, np.array([test_x, test_y, test_z]))
    zs = interpolate_volume_potential(test_nodes, trav, wrangler, pot).get()

    print_error = True
    if print_error:
        err = np.max(np.abs(ze - zs))
        print("Error =", err)

    # Boxtree
    if 0:
        import matplotlib.pyplot as plt

        if dim == 2:
            plt.plot(q_points[0].get(), q_points[1].get(), ".")

        from boxtree.visualization import TreePlotter

        plotter = TreePlotter(tree.get(queue=queue))
        plotter.draw_tree(fill=False, edgecolor="black")
        # plotter.draw_box_numbers()
        plotter.set_bounding_box()
        plt.gca().set_aspect("equal")

        plt.draw()
        plt.show()
        # plt.savefig("tree.png")

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

    # Write vtk
    if 0:
        from meshmode.mesh.io import read_gmsh

        modemesh = read_gmsh("box_grid.msh", force_ambient_dim=None)
        from meshmode.discretization.poly_element import (
            LegendreGaussLobattoTensorProductGroupFactory,
        )
        from meshmode.discretization import Discretization

        box_discr = Discretization(
            ctx, modemesh, LegendreGaussLobattoTensorProductGroupFactory(q_order)
        )

        box_nodes_x = box_discr.nodes()[0].with_queue(queue).get()
        box_nodes_y = box_discr.nodes()[1].with_queue(queue).get()
        box_nodes_z = box_discr.nodes()[2].with_queue(queue).get()
        box_nodes = make_obj_array(
            # get() first for CL compatibility issues
            [
                cl.array.to_device(queue, box_nodes_x),
                cl.array.to_device(queue, box_nodes_y),
                cl.array.to_device(queue, box_nodes_z),
            ]
        )

        visual_order = 1
        from meshmode.discretization.visualization import make_visualizer

        vis = make_visualizer(queue, box_discr, visual_order)

        from volumential.volume_fmm import interpolate_volume_potential

        volume_potential = interpolate_volume_potential(
                box_nodes, trav, wrangler, pot)

        # qx = q_points[0].get()
        # qy = q_points[1].get()
        # qz = q_points[2].get()
        exact_solution = cl.array.to_device(
            queue, solu_eval(queue,
                np.array([box_nodes_x, box_nodes_y, box_nodes_z])))

        # clean up the mess
        def clean_file(filename):
            import os

            try:
                os.remove(filename)
            except OSError:
                pass

        vtu_filename = "laplace3d.vtu"
        clean_file(vtu_filename)
        vis.write_vtk_file(
            vtu_filename,
            [
                ("VolPot", volume_potential),
                # ("SrcDensity", source_density),
                ("ExactSol", exact_solution),
                ("Error", volume_potential - exact_solution),
            ],
        )
        print("Written file " + vtu_filename)

    # }}} End postprocess and plot


if __name__ == '__main__':
    main()


# vim: filetype=python.pyopencl:foldmethod=marker
