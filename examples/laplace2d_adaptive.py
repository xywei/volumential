"""Adaptive refinement for a source with local features.

This example evaluates the Laplace volume potential over [-0.5, 0.5]^2 twice,
on a uniform tree and on an adaptive tree with at most as many leaves, and
compares both against the same reference.

The source is ``f = -Laplacian(u)`` for the sum of two Gaussians of different
widths,

    u = exp(-400 |x - c1|^2) + 0.5 exp(-6400 |x - c2|^2),

c1 = (-0.15, -0.1), c2 = (0.22, 0.2), so ``u`` is the whole-space solution
that the computed potentials are compared against. Outside the box both
Gaussian factors are below ``exp(-40)``, so the source mass the box leaves out
is at rounding level.

The adaptive tree starts from the root box and is refined in passes. For every
leaf, a pass takes the Legendre coefficients of the polynomial that
interpolates ``f`` at the leaf's ``q x q`` Gauss nodes, which is how the volume
FMM represents the source on that leaf, and adds up the magnitudes of the
coefficients of degree ``q - 1`` in either variable. That sum times the leaf's
area is the leaf's indicator, and the pass refines every leaf whose indicator
is at least half of the largest; the tree is kept 2:1 balanced. Refinement
stops before the pass that would give the adaptive tree more leaves than the
uniform one.

Both trees use the same quadrature order, near-field table and multipole
order. For each tree the example prints the number of leaves and nodes, the
maximum of ``|u_h - u|`` over the tree's quadrature nodes, and the relative
L2 error ``sqrt(sum w (u_h - u)^2 / sum w u^2)`` with the tree's quadrature
weights ``w``.

Set ``VOLUMENTIAL_EXAMPLE_SMOKE=1`` for a small configuration, and
``VOLUMENTIAL_GALLERY_OUTPUT_DIR`` to write the documentation gallery figure
(SVG, requires matplotlib) from the computed data after both solves.
"""

__copyright__ = "Copyright (C) 2026 Xiaoyu Wei"

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
from functools import partial
from pathlib import Path

import numpy as np

import pymbolic as pmbl
import pyopencl as cl
import pyopencl.array

import volumential.meshgen as mg
from volumential.tools import ScalarFieldExpressionEvaluation as Eval


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DIM = 2
#: The box is [LOWER, UPPER]^2.
LOWER, UPPER = -0.5, 0.5
#: (amplitude, alpha, center) of each Gaussian term of u.
GAUSSIANS = (
    (1.0, 400.0, (-0.15, -0.1)),
    (0.5, 6400.0, (0.22, 0.2)),
)
#: A pass refines every leaf whose indicator is at least this fraction of the
#: largest indicator.
REFINE_FRACTION = 0.5

#: Fixed rendering settings for the gallery figure. Together with omitting the
#: date and software metadata they make a rerun in the same environment
#: reproduce the SVG file byte for byte.
_GALLERY_RC = {
    "svg.hashsalt": "volumential-laplace2d-adaptive",
    "svg.fonttype": "path",
    "font.size": 12,
}
_GALLERY_SAVE_KWARGS = {
    "dpi": 150,
    "metadata": {"Date": None, "Creator": None},
}


def manufactured_solution():
    """Return pymbolic expressions for ``f = -Laplacian(u)`` and ``u``."""
    x, y, exp = pmbl.var("x"), pmbl.var("y"), pmbl.var("exp")
    source_expr = 0
    solution_expr = 0
    for amplitude, alpha, (cx, cy) in GAUSSIANS:
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        bump = amplitude * exp(-alpha * r2)
        solution_expr = solution_expr + bump
        source_expr = source_expr - (4 * alpha**2 * r2 - 4 * alpha) * bump
    return source_expr, solution_expr, [x, y]


def resolution_indicator(queue, mesh, q_order, source_eval):
    """Return the refinement indicator of every leaf of *mesh*.

    It is the sum of the magnitudes of the Legendre coefficients of degree
    ``q_order - 1`` in either variable of the polynomial that interpolates the
    source at the leaf's Gauss nodes, times the leaf's area.
    """
    centers = mesh.get_cell_centers()
    sides = mesh.get_cell_measures() ** (1 / DIM)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(q_order)
    reference = np.stack(np.meshgrid(gauss_nodes, gauss_nodes, indexing="ij"), -1)
    points = centers[:, None, None, :] + 0.5 * sides[:, None, None, None] * reference
    values = source_eval(queue, np.ascontiguousarray(points.reshape(-1, DIM).T))
    values = values.reshape(-1, q_order, q_order)

    # The Gauss rule is exact for the Legendre coefficients of a polynomial of
    # degree q_order - 1: c_m = (m + 1/2) sum_k w_k P_m(x_k) f(x_k).
    vandermonde = np.polynomial.legendre.legvander(gauss_nodes, q_order - 1)
    transform = (vandermonde * gauss_weights[:, None]).T
    transform *= (np.arange(q_order) + 0.5)[:, None]
    coefficients = np.einsum("mk,lkj,nj->lmn", transform, values, transform)
    top_degree = np.abs(coefficients[:, -1, :]).sum(axis=1) + np.abs(
        coefficients[:, :-1, -1]
    ).sum(axis=1)
    return top_degree * sides**DIM


def build_adaptive_mesh(queue, q_order, max_leaves, source_eval):
    """Refine a mesh from the root box as the module docstring describes.

    Returns the mesh and the number of refinement passes applied to it.
    """

    def refine(max_passes):
        mesh = mg.MeshGen2D(q_order, 1, LOWER, UPPER, queue=queue)
        npasses = 0
        while max_passes is None or npasses < max_passes:
            indicator = resolution_indicator(queue, mesh, q_order, source_eval)
            marked = indicator >= REFINE_FRACTION * indicator.max()
            refine_flags = np.zeros(mesh.boxtree.nboxes, dtype=bool)
            refine_flags[mesh.boxtree.active_boxes.get()[marked]] = True
            mesh.boxtree.refine_and_coarsen(
                refine_flags=refine_flags,
                coarsen_flags=np.zeros_like(refine_flags),
            )
            if max_passes is None and mesh.n_active_cells() > max_leaves:
                break
            npasses += 1
        return mesh, npasses

    # A refinement cannot be undone, and keeping the tree 2:1 balanced can
    # refine more leaves than were marked, so first find how many passes stay
    # within the budget, then rebuild the mesh with that many.
    _, npasses = refine(None)
    return refine(npasses)


def solve(ctx, queue, mesh, *, q_order, m_order, nftable, tree_indep,
          source_eval, solution_eval):
    """Evaluate the volume potential on the quadrature nodes of *mesh*."""
    from volumential.expansion_wrangler_fpnd import FPNDExpansionWrangler
    from volumential.volume_fmm import drive_volume_fmm

    nodes = np.ascontiguousarray(mesh.get_q_points().T)
    source_vals = cl.array.to_device(queue, source_eval(queue, nodes))

    # As in laplace2d.py, the tree is built from the mesh's own boxes, so that
    # every leaf is a mesh cell holding its q_order**2 Gauss nodes.
    _, q_weights, tree, trav = mg.build_geometry_info(
        ctx, queue, DIM, q_order, mesh,
        bbox=np.array([[LOWER, UPPER]] * DIM, dtype=np.float64),
    )
    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=trav,
        near_field_table=nftable,
        dtype=np.float64,
        fmm_level_to_order=lambda *args: m_order,
        quad_order=q_order,
        self_extra_kwargs={
            "target_to_source": np.arange(tree.ntargets, dtype=np.int32)
        },
    )
    (pot,) = drive_volume_fmm(
        trav, wrangler, source_vals * q_weights, source_vals
    )

    approx = pot.get()
    reference = solution_eval(queue, nodes)
    weights = q_weights.get()
    error = approx - reference
    leaf_sides = mesh.get_cell_measures() ** (1 / DIM)
    leaf_levels = np.rint(np.log2((UPPER - LOWER) / leaf_sides)).astype(int)
    return {
        "nodes": nodes,
        "source": source_vals.get(),
        "approx": approx,
        "reference": reference,
        "leaf_centers": mesh.get_cell_centers(),
        "leaf_sides": leaf_sides,
        "n_leaves": mesh.n_active_cells(),
        "leaf_levels": (int(leaf_levels.min()), int(leaf_levels.max())),
        "n_boxes": tree.nboxes,
        "max_error": float(np.abs(error).max()),
        "rel_l2_error": float(
            np.sqrt(np.sum(weights * error**2) / np.sum(weights * reference**2))
        ),
    }


def _leaf_sides(run):
    """Describe the leaf side lengths of *run* as fractions of the box."""
    coarsest, finest = (2**level for level in run["leaf_levels"])
    return f"1/{coarsest}" if coarsest == finest else f"1/{coarsest} to 1/{finest}"


def _write_gallery_figure(output_dir, results, settings):
    """Write the gallery figure from the two computed runs.

    The figure only displays data the example has computed: each tree's leaf
    boxes over the source values at its quadrature nodes, and the pointwise
    error at those nodes, shaded by linear interpolation over a Delaunay
    triangulation of the nodes.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import LogNorm, SymLogNorm
        from matplotlib.patches import Rectangle
        from matplotlib.tri import Triangulation
    except ImportError as exc:
        raise RuntimeError(
            "VOLUMENTIAL_GALLERY_OUTPUT_DIR is set, but matplotlib cannot be "
            "imported; install matplotlib to write the gallery figure"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_scale = max(np.abs(run["source"]).max() for run in results.values())
    # The two Gaussian terms of f differ in size by an order of magnitude; a
    # symmetric log scale shows both, and the rings where f is negative.
    source_norm = SymLogNorm(
        linthresh=1e-2 * source_scale, vmin=-source_scale, vmax=source_scale
    )
    max_u = max(np.abs(run["reference"]).max() for run in results.values())
    # Differences below double-precision rounding of u carry no information,
    # and a log scale cannot show zero: draw them at eps * max|u|.
    err_floor = np.finfo(np.float64).eps * max_u
    err_top = max(run["max_error"] for run in results.values())
    err_norm = LogNorm(vmin=err_floor, vmax=max(err_top, 10 * err_floor))

    with plt.rc_context(_GALLERY_RC):
        figure, axes = plt.subplots(
            2, 2, figsize=(10.0, 9.6), constrained_layout=True
        )
        for column, (label, run) in enumerate(results.items()):
            triangulation = Triangulation(run["nodes"][0], run["nodes"][1])
            tree_axis = axes[0, column]
            source_artist = tree_axis.tripcolor(
                triangulation,
                run["source"],
                shading="gouraud",
                cmap="RdBu_r",
                norm=source_norm,
                rasterized=True,
            )
            corners = run["leaf_centers"] - 0.5 * run["leaf_sides"][:, None]
            tree_axis.add_collection(
                PatchCollection(
                    [
                        Rectangle(corner, side, side)
                        for corner, side in zip(
                            corners, run["leaf_sides"], strict=True
                        )
                    ],
                    facecolor="none",
                    edgecolor="0.15",
                    linewidth=0.5,
                )
            )
            tree_axis.set_title(
                f"{label.capitalize()} tree: {run['n_leaves']} leaves, "
                f"sides {_leaf_sides(run)}\n"
                f"source $f$ at its {run['nodes'].shape[1]} nodes"
            )

            error_axis = axes[1, column]
            error_artist = error_axis.tripcolor(
                triangulation,
                np.maximum(np.abs(run["approx"] - run["reference"]), err_floor),
                shading="gouraud",
                cmap="magma",
                norm=err_norm,
                rasterized=True,
            )
            error_axis.set_title(
                f"{label.capitalize()} tree: $|u_h - u|$ at the nodes\n"
                f"max {run['max_error']:.1e}, "
                f"relative $L^2$ {run['rel_l2_error']:.1e}"
            )

            for axis in (tree_axis, error_axis):
                axis.set_xlim(LOWER, UPPER)
                axis.set_ylim(LOWER, UPPER)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("x")
                axis.set_ylabel("y")

        figure.colorbar(
            source_artist, ax=axes[0, :], shrink=0.82,
            label="$f$ (symmetric log scale)",
        )
        figure.colorbar(
            error_artist, ax=axes[1, :], shrink=0.82,
            label=r"$|u_h - u|$ (log scale, floor $\epsilon \max|u|$)",
        )
        terms = " + ".join(
            ("" if amplitude == 1 else rf"{amplitude:g}\,")
            + rf"e^{{-{alpha:g}|x - c_{i}|^2}}"
            for i, (amplitude, alpha, _) in enumerate(GAUSSIANS, start=1)
        )
        centers = ", ".join(
            f"$c_{i}$ = ({cx:g}, {cy:g})"
            for i, (_, _, (cx, cy)) in enumerate(GAUSSIANS, start=1)
        )
        figure.suptitle(
            f"examples/laplace2d_adaptive.py, {settings['mode']} settings: "
            f"q = {settings['q_order']}, multipole order {settings['m_order']}\n"
            rf"$f = -\Delta u$, $u = {terms}$, {centers}"
        )
        path = output_dir / "laplace2d_adaptive.svg"
        figure.savefig(path, **_GALLERY_SAVE_KWARGS)
        plt.close(figure)

    return path


def main():
    smoke_mode = os.environ.get("VOLUMENTIAL_EXAMPLE_SMOKE", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if smoke_mode:
        q_order = 3
        n_levels = 3  # the uniform tree has 4 x 4 leaves
        m_order = 8
    else:
        q_order = 9
        n_levels = 5  # the uniform tree has 16 x 16 leaves
        m_order = 20

    # The same table as laplace2d.py, from the same cache file.
    table_filename = (
        "nft_laplace2d_smoke.sqlite" if smoke_mode else "nft_laplace2d.sqlite"
    )
    print("Using table cache:", table_filename)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    source_expr, solution_expr, variables = manufactured_solution()
    source_eval = Eval(DIM, source_expr, variables)
    solution_eval = Eval(DIM, solution_expr, variables)

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel

    from volumential.expansion_wrangler_fpnd import (
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    table_manager = NearFieldInteractionTableManager(
        table_filename, root_extent=2, queue=queue
    )
    nftable, _ = table_manager.get_table(
        DIM,
        "Laplace",
        q_order,
        queue=queue,
        build_config=DuffyBuildConfig(
            radial_rule="tanh-sinh-fast",
            regular_quad_order=8 if smoke_mode else 50,
            radial_quad_order=21 if smoke_mode else 100,
        ),
    )

    knl = LaplaceKernel(DIM)
    factory = DefaultExpansionFactory()
    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(factory.get_multipole_expansion_class(knl), knl),
        partial(factory.get_local_expansion_class(knl), knl),
        [knl],
        exclude_self=True,
    )

    uniform_mesh = mg.MeshGen2D(q_order, n_levels, LOWER, UPPER, queue=queue)
    adaptive_mesh, npasses = build_adaptive_mesh(
        queue, q_order, uniform_mesh.n_active_cells(), source_eval
    )

    results = {}
    for label, mesh in (("uniform", uniform_mesh), ("adaptive", adaptive_mesh)):
        run = solve(
            ctx, queue, mesh,
            q_order=q_order, m_order=m_order, nftable=nftable,
            tree_indep=tree_indep, source_eval=source_eval,
            solution_eval=solution_eval,
        )
        results[label] = run
        print(
            f"{label} tree: {run['n_leaves']} leaves with sides "
            f"{_leaf_sides(run)}, {run['n_boxes']} boxes, "
            f"{run['nodes'].shape[1]} nodes"
        )
        print(f"  max |u_h - u| over the nodes = {run['max_error']:.3e}")
        print(f"  relative L2 error            = {run['rel_l2_error']:.3e}")
    print(f"adaptive refinement passes: {npasses}")

    gallery_output_dir = os.environ.get("VOLUMENTIAL_GALLERY_OUTPUT_DIR")
    if gallery_output_dir:
        path = _write_gallery_figure(
            gallery_output_dir,
            results,
            {
                "mode": "smoke" if smoke_mode else "full",
                "q_order": q_order,
                "m_order": m_order,
            },
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
