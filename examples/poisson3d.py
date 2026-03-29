"""Manufactured Poisson solve in 3D with slice and volume visualizations.

This example evaluates a volume potential over [-0.5, 0.5]^3 using the Laplace
kernel and compares against a manufactured smooth solution.

Set ``VOLUMENTIAL_EXAMPLE_SMOKE=1`` for a lighter run.
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
import time
from functools import partial
from pathlib import Path

import numpy as np

import pymbolic as pmbl
import pyopencl as cl
import pyopencl.array  # noqa: F401

from volumential.tools import ScalarFieldExpressionEvaluation as Eval


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _is_smoke_mode():
    return os.environ.get("VOLUMENTIAL_EXAMPLE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _int_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%s", name, raw)
        return default


def _to_obj_array(queue, arrays):
    from pytools.obj_array import new_1d as obj_array_1d

    return obj_array_1d(
        [cl.array.to_device(queue, np.ascontiguousarray(a)) for a in arrays]
    )


def _build_manufactured_problem(dim):
    x = pmbl.var("x")
    y = pmbl.var("y")
    z = pmbl.var("z")
    expp = pmbl.var("exp")

    # Keep the manufactured field effectively zero at the box boundary so
    # refinement tracks discretization/FMM convergence instead of boundary
    # truncation error.
    alpha_1 = 120.0
    alpha_2 = 90.0
    coeff_2 = -0.65

    dx1 = x + 0.08
    dy1 = y - 0.06
    dz1 = z + 0.05
    r1_sq = dx1**2 + dy1**2 + dz1**2

    dx2 = x - 0.09
    dy2 = y + 0.07
    dz2 = z - 0.08
    r2_sq = dx2**2 + dy2**2 + dz2**2

    g1 = expp(-alpha_1 * r1_sq)
    g2 = expp(-alpha_2 * r2_sq)

    solution_expr = g1 + coeff_2 * g2
    source_expr = (2 * dim * alpha_1 - 4 * alpha_1**2 * r1_sq) * g1 + coeff_2 * (
        2 * dim * alpha_2 - 4 * alpha_2**2 * r2_sq
    ) * g2

    variables = [x, y, z]
    return source_expr, solution_expr, variables


def _plane_coordinates(plane_name, line, level):
    uu, vv = np.meshgrid(line, line, indexing="xy")
    if plane_name == "xy":
        xx = uu
        yy = vv
        zz = np.full_like(uu, level)
        plot_u = xx
        plot_v = yy
        labels = ("x", "y")
    elif plane_name == "xz":
        xx = uu
        yy = np.full_like(uu, level)
        zz = vv
        plot_u = xx
        plot_v = zz
        labels = ("x", "z")
    elif plane_name == "yz":
        xx = np.full_like(uu, level)
        yy = uu
        zz = vv
        plot_u = yy
        plot_v = zz
        labels = ("y", "z")
    else:
        raise ValueError(f"Unknown plane name '{plane_name}'")

    return labels, plot_u, plot_v, xx, yy, zz


def _evaluate_plane(
    queue,
    plane_name,
    line,
    level,
    trav,
    wrangler,
    potential,
    solution_eval,
):
    from volumential.volume_fmm import interpolate_volume_potential

    labels, plot_u, plot_v, xx, yy, zz = _plane_coordinates(plane_name, line, level)
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    flat_z = zz.ravel()

    targets = _to_obj_array(queue, [flat_x, flat_y, flat_z])
    approx = interpolate_volume_potential(targets, trav, wrangler, potential).get()
    exact = solution_eval(queue, np.array([flat_x, flat_y, flat_z]))

    approx = approx.reshape(xx.shape)
    exact = exact.reshape(xx.shape)

    return {
        "plane": plane_name,
        "labels": labels,
        "u": plot_u,
        "v": plot_v,
        "exact": exact,
        "approx": approx,
        "abs_err": np.abs(exact - approx),
    }


def _write_slice_figure(plane_data, output_file):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib is unavailable; skipping %s", output_file)
        return False

    plane_order = ["xy", "xz", "yz"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)

    for irow, plane_name in enumerate(plane_order):
        pdata = plane_data[plane_name]
        uu = pdata["u"]
        vv = pdata["v"]
        exact = pdata["exact"]
        approx = pdata["approx"]
        abs_err = pdata["abs_err"]

        vmin = float(min(exact.min(), approx.min()))
        vmax = float(max(exact.max(), approx.max()))
        extent = (uu.min(), uu.max(), vv.min(), vv.max())

        im_exact = axes[irow, 0].imshow(
            exact,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axes[irow, 1].imshow(
            approx,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        log_err = np.log10(abs_err + 1.0e-18)
        im_err = axes[irow, 2].imshow(
            log_err,
            origin="lower",
            extent=extent,
            cmap="magma",
            interpolation="nearest",
        )

        labels = pdata["labels"]
        axes[irow, 0].set_ylabel(labels[1])
        for icol in range(3):
            axes[irow, icol].set_xlabel(labels[0])

        axes[irow, 0].set_title(f"Exact ({plane_name} plane)")
        axes[irow, 1].set_title(f"FMM ({plane_name} plane)")
        axes[irow, 2].set_title(f"log10 abs error ({plane_name} plane)")

        fig.colorbar(im_exact, ax=[axes[irow, 0], axes[irow, 1]], shrink=0.8)
        fig.colorbar(im_err, ax=axes[irow, 2], shrink=0.8)

    fig.suptitle("Poisson 3D slices: exact vs FMM", fontsize=14)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    return True


def _write_point_cloud_figure(points, values, exact_values, output_file):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib is unavailable; skipping %s", output_file)
        return False

    npts = values.size
    max_plot_points = 40000
    if npts > max_plot_points:
        stride = int(np.ceil(npts / max_plot_points))
    else:
        stride = 1

    xx = points[0][::stride]
    yy = points[1][::stride]
    zz = points[2][::stride]
    err = np.abs(values[::stride] - exact_values[::stride])
    color_values = np.log10(err + 1.0e-18)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xx,
        yy,
        zz,
        c=color_values,
        cmap="inferno",
        s=2,
        alpha=0.45,
        linewidths=0,
    )

    ax.set_title("Poisson 3D point-cloud log10 abs error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    return True


def _write_plotly_isosurface(line, approx, exact, output_file):
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.info("plotly is unavailable; skipping %s", output_file)
        return False

    xx, yy, zz = np.meshgrid(line, line, line, indexing="ij")
    abs_err = np.abs(approx - exact)

    pot_min = float(np.percentile(approx, 35))
    pot_max = float(np.percentile(approx, 98))
    err_min = float(np.percentile(abs_err, 92))
    err_max = float(np.percentile(abs_err, 99.8))

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=approx.ravel(),
            isomin=pot_min,
            isomax=pot_max,
            surface_count=8,
            caps={"x_show": False, "y_show": False, "z_show": False},
            opacity=0.62,
            colorscale="Viridis",
            name="Potential",
        )
    )
    fig.add_trace(
        go.Isosurface(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=abs_err.ravel(),
            isomin=err_min,
            isomax=err_max,
            surface_count=3,
            caps={"x_show": False, "y_show": False, "z_show": False},
            opacity=0.18,
            colorscale="Reds",
            showscale=False,
            name="Error hot spots",
        )
    )

    fig.update_layout(
        title="Poisson 3D potential with error hot spots",
        template="plotly_white",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
        },
    )
    fig.write_html(str(output_file), include_plotlyjs="cdn")
    return True


def main():
    print("*************************")
    print("* Setting up...          ")
    print("*************************")

    smoke_mode = _is_smoke_mode()
    dim = 3
    dtype = np.float64

    if smoke_mode:
        q_order = 3
        n_levels = 2
        m_order = 10
        regular_quad_order = 10
        radial_quad_order = 35
        vis_side_points = 72
        volume_side_points = 20
    else:
        q_order = 7
        n_levels = 5
        m_order = 16
        regular_quad_order = 16
        radial_quad_order = 80
        vis_side_points = 180
        volume_side_points = 48

    q_order = _int_env("VOLUMENTIAL_POISSON3D_Q_ORDER", q_order)
    n_levels = _int_env("VOLUMENTIAL_POISSON3D_N_LEVELS", n_levels)
    m_order = _int_env("VOLUMENTIAL_POISSON3D_M_ORDER", m_order)
    vis_side_points = _int_env("VOLUMENTIAL_POISSON3D_VIS_SIDE_POINTS", vis_side_points)
    volume_side_points = _int_env(
        "VOLUMENTIAL_POISSON3D_VOLUME_SIDE_POINTS", volume_side_points
    )

    table_filename = (
        "nft_poisson3d_smoke.sqlite" if smoke_mode else "nft_poisson3d.sqlite"
    )
    root_table_source_extent = 2.0
    use_multilevel_table = False
    force_direct_evaluation = False
    exclude_self = True

    a = -0.5
    b = 0.5
    vis_plane_level = 0.0

    output_dir = Path(
        os.environ.get("VOLUMENTIAL_POISSON3D_OUTPUT_DIR", "poisson3d_output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Smoke mode: %s", smoke_mode)
    logger.info("Using table cache: %s", table_filename)
    logger.info(
        "Config q_order=%d n_levels=%d m_order=%d vis_side=%d volume_side=%d",
        q_order,
        n_levels,
        m_order,
        vis_side_points,
        volume_side_points,
    )

    source_expr, solu_expr, variables = _build_manufactured_problem(dim)
    logger.info("Source expr: %s", source_expr)
    logger.info("Solu expr: %s", solu_expr)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    source_eval = Eval(dim, source_expr, variables)
    solu_eval = Eval(dim, solu_expr, variables)

    print("Generating quadrature mesh...")
    import volumential.meshgen as mg

    mg.greet()
    mesh = mg.MeshGen3D(q_order, n_levels, a, b, queue=queue)
    mesh.print_info()
    q_points, q_weights, tree, trav = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        q_order,
        mesh,
        bbox=np.array([[a, b]] * dim, dtype=dtype),
    )
    assert tree.sources_are_targets
    q_weights_host = q_weights.get(queue)

    logger.info("Discretizing source field")
    source_vals = cl.array.to_device(
        queue,
        source_eval(queue, np.array([coords.get() for coords in q_points])),
    )

    print("Building near-field table...")
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    tm = NearFieldInteractionTableManager(
        table_filename, root_extent=root_table_source_extent, queue=queue
    )
    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )

    try:
        if use_multilevel_table:
            nftable = []
            for lev in range(0, tree.nlevels + 1):
                logger.info("Getting table at level %d", lev)
                tb_level, _ = tm.get_table(
                    dim,
                    "Laplace",
                    q_order,
                    source_box_level=lev,
                    queue=queue,
                    build_config=build_config,
                )
                nftable.append(tb_level)
        else:
            nftable, _ = tm.get_table(
                dim,
                "Laplace",
                q_order,
                force_recompute=False,
                queue=queue,
                build_config=build_config,
            )
    except NotImplementedError as exc:
        logger.warning("Skipping poisson3d example: %s", exc)
        return

    print("Building expansion wrangler...")
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    knl = LaplaceKernel(dim)
    out_kernels = [knl]

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(knl)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(knl)

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

    print("*************************")
    print("* Performing FMM ...    ")
    print("*************************")

    from volumential.volume_fmm import drive_volume_fmm, interpolate_volume_potential

    queue.finish()
    t0 = time.time()
    (pot,) = drive_volume_fmm(
        trav,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=force_direct_evaluation,
        list1_only=False,
    )
    queue.finish()
    t1 = time.time()

    elapsed = t1 - t0
    print(f"Finished in {elapsed:.2f} seconds")
    print(f"({len(q_weights_host) / max(elapsed, 1.0e-15):.6e} points per second)")

    q_coord_host = np.array([coords.get() for coords in q_points])
    exact_q = solu_eval(queue, q_coord_host)
    approx_q = pot.get()
    q_abs_err = np.abs(exact_q - approx_q)
    q_rel_l2 = np.linalg.norm(exact_q - approx_q) / max(
        np.linalg.norm(exact_q), 1.0e-15
    )
    print(f"Quadrature-node max abs error: {np.max(q_abs_err):.6e}")
    print(f"Quadrature-node relative L2 error: {q_rel_l2:.6e}")

    print("*************************")
    print("* Generating visuals... ")
    print("*************************")

    line = np.linspace(a, b, vis_side_points)
    plane_data = {}
    for plane_name in ("xy", "xz", "yz"):
        plane_data[plane_name] = _evaluate_plane(
            queue,
            plane_name,
            line,
            vis_plane_level,
            trav,
            wrangler,
            pot,
            solu_eval,
        )

    slice_png = output_dir / "poisson3d_slices.png"
    slice_written = _write_slice_figure(plane_data, slice_png)

    point_cloud_png = output_dir / "poisson3d_error_point_cloud.png"
    point_cloud_written = _write_point_cloud_figure(
        q_coord_host,
        approx_q,
        exact_q,
        point_cloud_png,
    )

    line_diag = np.linspace(a, b, vis_side_points)
    diag_targets = _to_obj_array(queue, [line_diag, line_diag, line_diag])
    approx_diag = interpolate_volume_potential(diag_targets, trav, wrangler, pot).get()
    exact_diag = solu_eval(queue, np.array([line_diag, line_diag, line_diag]))

    volume_line = np.linspace(a, b, volume_side_points)
    vol_x, vol_y, vol_z = np.meshgrid(
        volume_line, volume_line, volume_line, indexing="ij"
    )
    vol_targets = _to_obj_array(queue, [vol_x.ravel(), vol_y.ravel(), vol_z.ravel()])
    approx_volume = interpolate_volume_potential(vol_targets, trav, wrangler, pot).get()
    exact_volume = solu_eval(
        queue, np.array([vol_x.ravel(), vol_y.ravel(), vol_z.ravel()])
    )
    approx_volume = approx_volume.reshape(vol_x.shape)
    exact_volume = exact_volume.reshape(vol_x.shape)

    np.savez_compressed(
        output_dir / "poisson3d_diagnostics.npz",
        diagonal_t=line_diag,
        diagonal_exact=exact_diag,
        diagonal_approx=approx_diag,
        quadrature_exact=exact_q,
        quadrature_approx=approx_q,
        quadrature_abs_err=q_abs_err,
        volume_line=volume_line,
        volume_exact=exact_volume,
        volume_approx=approx_volume,
        volume_abs_err=np.abs(exact_volume - approx_volume),
    )

    isosurface_html = output_dir / "poisson3d_isosurface.html"
    isosurface_written = _write_plotly_isosurface(
        volume_line,
        approx_volume,
        exact_volume,
        isosurface_html,
    )

    if slice_written:
        print(f"Wrote {slice_png}")
    if point_cloud_written:
        print(f"Wrote {point_cloud_png}")
    if isosurface_written:
        print(f"Wrote {isosurface_html}")
    print(f"Wrote {output_dir / 'poisson3d_diagnostics.npz'}")


if __name__ == "__main__":
    main()


# vim: filetype=python.pyopencl:foldmethod=marker
