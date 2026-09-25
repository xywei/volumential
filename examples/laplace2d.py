"""This example evaluates the volume potential over
[-0.5, 0.5]^2 with the Laplace kernel.

The source is ``f = -Laplacian(u)`` for the Gaussian ``u = exp(-alpha |x|^2)``,
so ``u`` is the whole-space solution that the computed potential is compared
against. The integral itself runs over the box only; outside the box the
Gaussian factor is at most ``exp(-40)`` for ``alpha = 160``, so the source mass
the box leaves out is at rounding level.

Set ``VOLUMENTIAL_GALLERY_OUTPUT_DIR`` to write the documentation gallery figures
(SVG, requires matplotlib) from the computed data after the solve.
"""

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
import os
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from functools import partial

import numpy as np

import pymbolic as pmbl
import pyopencl as cl
import pyopencl.array  # noqa: F401

from volumential.tools import ScalarFieldExpressionEvaluation as Eval


#: Fixed rendering settings for the gallery figures. Together with omitting the
#: date and software metadata they make a rerun in the same environment
#: reproduce the SVG files byte for byte.
_GALLERY_DPI = 150
_GALLERY_RC = {
    "svg.hashsalt": "volumential-laplace2d",
    "svg.fonttype": "path",
    # The figures are shown about 600 px wide; keep their labels readable.
    "font.size": 12,
}
_GALLERY_SAVE_KWARGS = {
    "dpi": _GALLERY_DPI,
    "metadata": {"Date": None, "Creator": None},
}


def _write_gallery_figures(
    output_dir, *, points, source, reference, approx, tree, settings
):
    """Write the gallery figures from an already-computed Laplace run.

    The figures only display data the example has computed: the values at the
    quadrature nodes, shaded by linear interpolation over a Delaunay
    triangulation of the nodes, and the tree the FMM traversed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.tri import Triangulation
    except ImportError as exc:
        raise RuntimeError(
            "VOLUMENTIAL_GALLERY_OUTPUT_DIR is set, but matplotlib cannot be "
            "imported; install matplotlib to write the gallery figures"
        ) from exc

    from boxtree.visualization import TreePlotter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_coord, y_coord = points
    n_nodes = x_coord.size
    triangulation = Triangulation(x_coord, y_coord)
    abs_err = np.abs(approx - reference)
    # Differences below double-precision rounding of u carry no information,
    # and a log scale cannot show zero: draw them at eps * max|u|.
    err_floor = np.finfo(abs_err.dtype).eps * np.abs(reference).max()
    err_norm = LogNorm(vmin=err_floor, vmax=max(abs_err.max(), 10 * err_floor))
    source_scale = np.abs(source).max()
    potential_range = (
        min(approx.min(), reference.min()),
        max(approx.max(), reference.max()),
    )
    node_dot_area = float(np.clip(6000.0 / n_nodes, 0.1, 4.0))

    panels = (
        (
            source,
            r"Source $f = -\Delta u$",
            r"$f$",
            {"cmap": "RdBu_r", "vmin": -source_scale, "vmax": source_scale},
        ),
        (
            approx,
            r"Computed volume potential $u_h$",
            r"$u_h$",
            {"cmap": "viridis", "vmin": potential_range[0],
             "vmax": potential_range[1]},
        ),
        (
            reference,
            r"Reference $u = e^{-\alpha |x|^2}$ (whole space)",
            r"$u$",
            {"cmap": "viridis", "vmin": potential_range[0],
             "vmax": potential_range[1]},
        ),
        (
            np.maximum(abs_err, err_floor),
            r"Pointwise error $|u_h - u|$",
            r"$|u_h - u|$ (log scale, floor $\epsilon \max|u|$)",
            {"cmap": "magma", "norm": err_norm},
        ),
    )

    with plt.rc_context(_GALLERY_RC):
        figure, axes = plt.subplots(
            2, 2, figsize=(10.0, 9.0), constrained_layout=True
        )
        for axis, (values, title, label, color_kwargs) in zip(
            axes.flat, panels, strict=True
        ):
            artist = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                rasterized=True,
                **color_kwargs,
            )
            axis.set_title(title)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_xlim(*settings["domain"])
            axis.set_ylim(*settings["domain"])
            axis.set_aspect("equal", adjustable="box")
            figure.colorbar(artist, ax=axis, shrink=0.82, label=label)

        figure.suptitle(
            "examples/laplace2d.py, {mode} settings: "
            "q = {q_order}, {n_levels} mesh levels, multipole order {m_order}, "
            "{n_nodes} nodes\n"
            r"$\alpha$ = {alpha}; max $|u_h - u|$ = {max_err:.1e}".format(
                n_nodes=n_nodes, max_err=abs_err.max(), **settings
            )
        )
        overview_path = output_dir / "laplace2d_overview.svg"
        figure.savefig(overview_path, **_GALLERY_SAVE_KWARGS)
        plt.close(figure)

        tree_figure, tree_axis = plt.subplots(
            1, 1, figsize=(7.0, 7.4), constrained_layout=True
        )
        plt.sca(tree_axis)
        tree_axis.scatter(
            x_coord,
            y_coord,
            s=node_dot_area,
            color="tab:blue",
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        # Coarser levels get thicker, darker outlines, so the hierarchy reads
        # as nested boxes rather than as one flat grid. The finest level is
        # drawn first and the root last, on top.
        plotter = TreePlotter(tree)
        box_levels = np.asarray(tree.box_levels)
        level_widths = np.linspace(2.2, 0.4, max(tree.nlevels, 2))
        level_greys = np.linspace(0.0, 0.55, max(tree.nlevels, 2))
        for ibox in sorted(
            range(tree.nboxes), key=lambda ibox: -int(box_levels[ibox])
        ):
            level = int(box_levels[ibox])
            plotter.draw_box(
                ibox,
                fill=False,
                edgecolor=str(level_greys[level]),
                linewidth=level_widths[level],
                zorder=2,
            )
        plotter.set_bounding_box()
        tree_axis.set_aspect("equal", adjustable="box")
        tree_axis.set_title(
            f"Tree used by the volume FMM: {tree.nboxes} boxes, "
            f"{tree.nlevels} levels\n"
            "heavier outlines: coarser levels\n"
            f"dots: the {n_nodes} quadrature nodes ({settings['mode']} settings)"
        )
        tree_axis.set_xlabel("x")
        tree_axis.set_ylabel("y")
        tree_path = output_dir / "laplace2d_tree.svg"
        tree_figure.savefig(tree_path, **_GALLERY_SAVE_KWARGS)
        plt.close(tree_figure)

    return [overview_path, tree_path]


def main():

    print("*************************")
    print("* Setting up...")
    print("*************************")

    dim = 2

    smoke_mode = os.environ.get("VOLUMENTIAL_EXAMPLE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
    }

    # use local SQLite cache; nearfield tables are recomputed on cache miss
    table_filename = (
        "nft_laplace2d_smoke.sqlite" if smoke_mode else "nft_laplace2d.sqlite"
    )
    root_table_source_extent = 2

    print("Using table cache:", table_filename)

    if smoke_mode:
        q_order = 3
        n_levels = 2
        m_order = 8
    else:
        q_order = 9  # quadrature order
        n_levels = 6  # 2^(n_levels-1) subintervals in 1D
        m_order = 20  # multipole order

    use_multilevel_table = False

    dtype = np.float64
    force_direct_evaluation = False
    # Also evaluate by direct particle-to-particle summation and compare.
    compare_with_direct_p2p = False

    print("Multipole order =", m_order)

    alpha = 160

    x = pmbl.var("x")
    y = pmbl.var("y")
    expp = pmbl.var("exp")

    norm2 = x**2 + y**2
    source_expr = -(4 * alpha**2 * norm2 - 4 * alpha) * expp(-alpha * norm2)
    solu_expr = expp(-alpha * norm2)

    logger.info("Source expr: " + str(source_expr))
    logger.info("Solu expr: " + str(solu_expr))

    # bounding box
    a = -0.5
    b = 0.5

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

    q_points_host = np.ascontiguousarray(np.transpose(q_points))

    from pytools.obj_array import new_1d as obj_array_1d

    q_points = obj_array_1d(
        [cl.array.to_device(queue, q_points_host[i]) for i in range(dim)]
    )

    q_weights = cl.array.to_device(queue, q_weights)
    # q_radii = cl.array.to_device(queue, q_radii)

    # }}}

    # {{{ discretize the source field

    source_vals = cl.array.to_device(queue, source_eval(queue, q_points_host))

    # particle_weigt = source_val * q_weight

    # }}} End discretize the source field

    # {{{ build tree and traversals

    # Build the particle tree from the mesh's own box tree, so that every leaf
    # box is a mesh cell holding its q_order**2 Gauss nodes, which is the
    # geometry the near-field table assumes. A tree built from the particles
    # alone takes their extent, not [a, b]^2, as its root box, and its leaves
    # then do not coincide with the mesh cells.
    from boxtree.array_context import PyOpenCLArrayContext

    actx = PyOpenCLArrayContext(queue)
    _, _, tree, trav = mg.build_geometry_info(
        ctx, queue, dim, q_order, mesh,
        bbox=np.array([[a, b]] * dim, dtype=np.float64),
    )

    # }}} End build tree and traversals

    # {{{ build near field potential table

    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent, queue=queue
    )
    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=8 if smoke_mode else 50,
        radial_quad_order=21 if smoke_mode else 100,
    )

    if use_multilevel_table:
        assert (
            abs(
                int((b - a) / root_table_source_extent) * root_table_source_extent
                - (b - a)
            )
            < 1e-15
        )
        nftable = []
        for lev in range(0, tree.nlevels + 1):
            print("Getting table at level", lev)
            tb, _ = tm.get_table(
                dim,
                "Laplace",
                q_order,
                source_box_level=lev,
                queue=queue,
                build_config=build_config,
            )
            nftable.append(tb)

        print("Using table list of length", len(nftable))

    else:
        nftable, _ = tm.get_table(
            dim,
            "Laplace",
            q_order,
            force_recompute=False,
            queue=queue,
            build_config=build_config,
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
        FPNDTreeIndependentDataForWrangler,
    )

    tree_indep = FPNDTreeIndependentDataForWrangler(
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
        tree_indep=tree_indep,
        queue=queue,
        traversal=trav,
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

    import time

    from volumential.volume_fmm import drive_volume_fmm

    queue.finish()

    t0 = time.time()

    (pot,) = drive_volume_fmm(
        trav,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=force_direct_evaluation,
    )
    queue.finish()

    t1 = time.time()

    print("Finished in %.2f seconds." % (t1 - t0))
    print("(%e points per second)" % (len(q_weights) / (t1 - t0)))

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

    gallery_output_dir = os.environ.get("VOLUMENTIAL_GALLERY_OUTPUT_DIR")
    if gallery_output_dir:
        written = _write_gallery_figures(
            Path(gallery_output_dir),
            points=q_points_host,
            source=source_vals.get(),
            reference=ze,
            approx=zs,
            tree=actx.to_numpy(tree),
            settings={
                "mode": "smoke" if smoke_mode else "full",
                "alpha": alpha,
                "q_order": q_order,
                "n_levels": n_levels,
                "m_order": m_order,
                "domain": (a, b),
            },
        )
        for output_path in written:
            print(f"Wrote {output_path}")

    if compare_with_direct_p2p:
        print("Performing P2P")
        (pot_direct,) = drive_volume_fmm(
            trav,
            wrangler,
            source_vals * q_weights,
            source_vals,
            direct_evaluation=True,
        )
        zds = pot_direct.get()

        print("P2P-FMM diff =", np.max(np.abs(zs - zds)))

        print("P2P Error =", np.max(np.abs(ze - zds)))

    # }}} End postprocess and plot


if __name__ == "__main__":
    main()


# vim: filetype=pyopencl:foldmethod=marker
