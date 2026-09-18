"""Unit tests for the Experiment E geometry module (clipping, plane waves, polar rules).

Runs the checks the geometry module owes the rest of the two-dimensional
windowed-DMK assembly and prints a JSON summary:

1. clipped leaf areas of each domain summed against the closed-form area of
   ``Omega`` (``1e-13`` for the box and the L-shaped polygon, ``1e-12`` for the
   smooth star at the stated quadrature order), level by level, with every
   curved piece's area also cross-checked against an independent
   ``(1/2) oint (x dy - y dx)`` evaluation;
2. the two L-corner configurations -- the re-entrant corner at a leaf corner and
   strictly inside a leaf -- against exact rectangle areas;
3. a leaf of the smooth star whose boundary crosses one box edge twice, if the
   sweep finds one, checked the same way;
4. plane-wave coefficients: the closed forms against graded quadrature on box,
   polygon and curved pieces, and the polygon (divergence-theorem) route against
   the separable box route on the same rectangle, including ``k = 0`` and modes
   orthogonal to an edge;
5. the polar rules: the fan against the leaf-residual study's own polygon
   routine, against the separable heat-time rule on rectangles (including the
   rectangles of a cut L leaf), against a directly graded quadrature for targets
   off the piece, and against itself under order refinement on a cut star leaf;
   plus the arc fan's reproduction of the piece area with a constant kernel;
6. plots of the S and L leaf decompositions at ``L = 4``.

Exploratory campaign code: no warm-up, no repetition statistics, no claim that
any rule used here is optimal.

    python experiments/windowed_dmk/experiment_e_test_geometry.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # pragma: no cover - import shim for both invocation styles
    from experiments.windowed_dmk import experiment_c_leaf_residual as expc
    from experiments.windowed_dmk import experiment_e_common as ec
    from experiments.windowed_dmk import experiment_e_geometry as eg
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import experiment_c_leaf_residual as expc  # type: ignore[no-redef]
    import experiment_e_common as ec  # type: ignore[no-redef]
    import experiment_e_geometry as eg  # type: ignore[no-redef]


RHO_DECAY = 3.0
"""Decay of the smooth test density ``rho_1(x) = exp(-3 |x|^2)``."""

PROXY_ORDER = 5
"""Order ``q`` of the per-leaf tensor Lagrange proxy used by these tests."""


def rho_1(points: np.ndarray) -> np.ndarray:
    """The study's smooth non-polynomial test density ``exp(-3 |x|^2)``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return np.exp(-RHO_DECAY * (pts[:, 0] ** 2 + pts[:, 1] ** 2))


def _leaf_poly(box, order: int = PROXY_ORDER) -> ec.LeafPoly:
    """Tensor Lagrange proxy of ``rho_1`` on a leaf."""
    return ec.LeafPoly.from_callable(box, rho_1, int(order))


# ---------------------------------------------------------------------------
# check 1: clipped areas
# ---------------------------------------------------------------------------


def check_areas(levels: Sequence[int] = (3, 4, 5)) -> list[dict]:
    """Sum the clipped leaf areas of each domain and compare with the closed form."""
    rows = []
    for name in eg.DOMAIN_NAMES:
        domain = eg.make_domain(name)
        exact = domain.area()
        tol = 1e-13 if name in ("B", "L") else 1e-12
        for level in levels:
            tree = ec.build_quadtree(levels=int(level))
            total = 0.0
            n_cut = 0
            n_pieces = 0
            n_curved = 0
            green_err = 0.0
            neg_area = 0.0
            for box in tree.leaves():
                pieces = domain.clip_leaf(box)
                n_pieces += len(pieces)
                if pieces and not (len(pieces) == 1 and pieces[0].kind == "box"):
                    n_cut += 1
                for piece in pieces:
                    area = piece.area()
                    total += area
                    neg_area = min(neg_area, area)
                    if piece.kind == "curved":
                        n_curved += 1
                        green_err = max(
                            green_err, abs(area - eg.loop_area_green(piece))
                        )
            rows.append(
                {
                    "check": "clipped_area_sum",
                    "domain": name,
                    "level": int(level),
                    "h_leaf": ec.h_level(int(level)),
                    "area": total,
                    "exact_area": exact,
                    "abs_error": abs(total - exact),
                    "n_cut_leaves": n_cut,
                    "n_pieces": n_pieces,
                    "n_curved_pieces": n_curved,
                    "max_lune_vs_green": green_err,
                    "min_piece_area": neg_area,
                    "tolerance": tol,
                    "passed": bool(
                        abs(total - exact) <= tol
                        and green_err <= 1e-13
                        and neg_area >= 0.0
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# check 2: the two L-corner configurations
# ---------------------------------------------------------------------------


def check_l_corner() -> list[dict]:
    """Clip leaves with the re-entrant corner at a leaf corner and inside a leaf.

    On the uniform tree over ``[-1, 1]^2`` the corner of the plan's L-shaped
    domain sits on a grid line at every level below the root, so it is a leaf
    *corner*, and the four leaves around it are each wholly inside or wholly
    outside.  The genuinely interior-corner configuration is produced by moving
    the corner off the grid, which is the third and fourth case below; the
    clipping path is the same either way.
    """
    rows = []
    grid = eg.LDomain()
    off = eg.LDomain(corner=(0.03, 0.02))
    configs = {
        "corner_at_leaf_corner_inside": (grid, ec.Box((-0.125, -0.125), 0.125, 3),
                                         0.25 * 0.25, 1),
        "corner_at_leaf_corner_outside": (grid, ec.Box((0.125, 0.125), 0.125, 3),
                                          0.0, 0),
        "corner_on_leaf_edge": (grid, ec.Box((0.0, 0.125), 0.125, 3),
                                0.125 * 0.25, 1),
        "corner_inside_leaf": (off, ec.Box((0.0625, 0.0625), 0.0625, 4),
                               0.03 * 0.125 + 0.095 * 0.02, 2),
        "corner_inside_coarse_leaf": (off, ec.Box((0.0, 0.0), 0.25, 2),
                                      0.5 * 0.5 - 0.22 * 0.23, 2),
    }
    for label, (domain, box, expected, n_expected) in configs.items():
        pieces = domain.clip_leaf(box)
        area = float(sum(p.area() for p in pieces))
        rows.append(
            {
                "check": "l_corner_clipping",
                "case": label,
                "box_center": list(box.center),
                "box_half": box.half,
                "n_pieces": len(pieces),
                "expected_pieces": n_expected,
                "kinds": [p.kind for p in pieces],
                "area": area,
                "expected_area": expected,
                "abs_error": abs(area - expected),
                "tolerance": 1e-15,
                "passed": bool(
                    abs(area - expected) <= 1e-15 and len(pieces) == n_expected
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# check 3: a star leaf whose boundary crosses one box edge twice
# ---------------------------------------------------------------------------


def _double_edge_boxes(star, level: int, offsets: Sequence[float],
                       limit: int) -> list:
    """Boxes of a shifted grid at ``level`` whose crossings repeat a box edge."""
    h = ec.h_level(int(level))
    half = h / 2.0
    n = int(round(2.0 / h))
    out = []
    for ox in offsets:
        for oy in offsets:
            for i in range(n):
                for j in range(n):
                    box = ec.Box(
                        (-1.0 + (i + 0.5 + ox) * h, -1.0 + (j + 0.5 + oy) * h),
                        half,
                        int(level),
                    )
                    r_lo, r_hi = eg._box_radial_range(box)
                    if r_hi <= star.r0 * (1.0 - star.amp):
                        continue
                    if r_lo >= star.max_radius():
                        continue
                    crossings = star._box_crossings(box)
                    if len(crossings) < 4:
                        continue
                    edges = [int(math.floor(c["t"])) for c in crossings]
                    if len(set(edges)) < len(edges):
                        out.append((box, crossings))
                        if len(out) >= limit:
                            return out
    return out


def check_double_crossing(levels: Sequence[int] = (4, 5, 6)) -> list[dict]:
    """Validate leaves where the star boundary crosses one box edge twice.

    The uniform tree over ``[-1, 1]^2`` happens never to produce such a leaf for
    this star at levels 3 to 6, so the configuration is also sought on grids
    shifted by fractions of a leaf, which is a legitimate stress of the same
    clipping path.  Each hit is checked against an independent Green's-theorem
    area and against the sum over the box's four children.
    """
    star = eg.StarDomain()
    rows = []
    for level in levels:
        tree = ec.build_quadtree(levels=int(level))
        hits = []
        for box in tree.leaves():
            r_lo, r_hi = eg._box_radial_range(box)
            if r_hi <= star.r0 * (1.0 - star.amp) or r_lo >= star.max_radius():
                continue
            crossings = star._box_crossings(box)
            if len(crossings) < 4:
                continue
            edges = [int(math.floor(c["t"])) for c in crossings]
            if len(set(edges)) < len(edges):
                hits.append((box, crossings))
                break
        rows.append(
            {
                "check": "star_double_edge_crossing_on_tree",
                "level": int(level),
                "found": bool(hits),
                "passed": True,
            }
        )
    for box, crossings in _double_edge_boxes(star, 3, (0.25, 0.5), 3):
        pieces = star.clip_leaf(box)
        area = float(sum(p.area() for p in pieces))
        green = float(sum(eg.loop_area_green(p) for p in pieces))
        half = box.half / 2.0
        children = 0.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                kid = ec.Box(
                    (box.center[0] + sx * half, box.center[1] + sy * half),
                    half,
                    box.level + 1,
                )
                children += float(sum(p.area() for p in star.clip_leaf(kid)))
        err = max(abs(area - green), abs(area - children))
        rows.append(
            {
                "check": "star_double_edge_crossing_shifted_grid",
                "box_center": list(box.center),
                "box_half": box.half,
                "n_crossings": len(crossings),
                "edges_hit": sorted(int(math.floor(c["t"])) for c in crossings),
                "n_pieces": len(pieces),
                "area": area,
                "green_area": green,
                "children_area": children,
                "abs_error": err,
                "tolerance": 1e-14,
                "passed": bool(err <= 1e-14 and area > 0.0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# check 4: plane-wave coefficients
# ---------------------------------------------------------------------------


def find_cut_leaves(domain, level: int, kind: str | None = None,
                    n_pieces: int | None = None, limit: int = 4) -> list[tuple]:
    """Search a level for cut leaves matching a piece kind or piece count."""
    tree = ec.build_quadtree(levels=int(level))
    out = []
    for box in tree.leaves():
        pieces = domain.clip_leaf(box)
        if not pieces or (len(pieces) == 1 and pieces[0].kind == "box"):
            continue
        if kind is not None and pieces[0].kind != kind:
            continue
        if n_pieces is not None and len(pieces) != n_pieces:
            continue
        out.append((box, pieces))
        if len(out) >= limit:
            break
    return out


def _test_pieces() -> dict:
    """The pieces used by the plane-wave and polar checks, with their leaf boxes."""
    star = eg.StarDomain()
    ldom = eg.LDomain()
    ldom_off = eg.LDomain(corner=(0.03, 0.02))
    out = {}
    box = ec.Box((0.3125, -0.1875), 0.0625, 4)
    out["leaf_box"] = (ec.make_box_piece(box), box)
    lbox = ec.Box((-0.75, 0.75), 0.125, 3)
    out["L_rectangle"] = (ldom.clip_leaf(lbox)[0], lbox)
    corner_box = ec.Box((0.0625, 0.0625), 0.0625, 4)
    corner_pieces = ldom_off.clip_leaf(corner_box)
    if corner_pieces:
        out["L_corner_rectangle"] = (corner_pieces[0], corner_box)
    cbox = box
    out["triangle"] = (
        ec.make_polygon_piece(
            np.array([[0.26, -0.24], [0.375, -0.24], [0.30, -0.13]]), cbox
        ),
        cbox,
    )
    for level, label in ((3, "star_curved_coarse"), (4, "star_curved_fine")):
        found = find_cut_leaves(star, level, kind="curved", limit=1)
        if found:
            sbox, spieces = found[0]
            out[label] = (spieces[0], sbox)
    return out


def check_plane_waves() -> list[dict]:
    """Closed-form plane-wave coefficients against quadrature and against each other."""
    rows = []
    pieces = _test_pieces()
    for label, (piece, box) in pieces.items():
        poly = _leaf_poly(box)
        dk = 2.0 * math.pi / (6.0 * box.side)
        mults = [(0, 0), (1, 0), (0, 1), (2, -3), (7, 7), (13, -2), (0, 21), (24, 0)]
        kx = np.array([m[0] * dk for m in mults])
        ky = np.array([m[1] * dk for m in mults])
        closed = eg.plane_wave_coeffs(piece, poly, kx, ky)
        quad, n_nodes = eg.plane_wave_quadrature_reference(piece, poly, kx, ky)
        scale = float(np.max(np.abs(quad)))
        err = float(np.max(np.abs(closed - quad)))
        tol = 1e-13
        rows.append(
            {
                "check": "plane_wave_closed_vs_quadrature",
                "case": label,
                "kind": piece.kind,
                "n_modes": int(kx.size),
                "k_max_h": float(np.max(np.hypot(kx, ky)) * box.side),
                "max_abs_error": err,
                "rel_error": err / scale if scale else 0.0,
                "reference_nodes": int(n_nodes),
                "tolerance": tol,
                "passed": bool((err / scale if scale else 0.0) <= tol),
            }
        )
        if eg.is_axis_rectangle(piece):
            poly_route = eg.plane_wave_coeffs(piece, poly, kx, ky, method="polygon")
            err2 = float(np.max(np.abs(closed - poly_route)))
            rows.append(
                {
                    "check": "plane_wave_box_vs_polygon_closed_form",
                    "case": label,
                    "max_abs_error": err2,
                    "rel_error": err2 / scale if scale else 0.0,
                    "tolerance": 1e-13,
                    "passed": bool((err2 / scale if scale else 0.0) <= 1e-13),
                }
            )
    return rows


def check_plane_wave_grid_sample() -> list[dict]:
    """Closed form against quadrature on a genuine shell grid, sampled over the modes."""
    star = eg.StarDomain()
    found = find_cut_leaves(star, 4, kind="curved", limit=1)
    if not found:
        return []
    box, pieces = found[0]
    poly = _leaf_poly(box)
    grid = ec.plane_wave_grid(box.side, ec.t_l(3), ec.t_l(4))
    kx, ky = grid.kx, grid.ky
    take = np.linspace(0, kx.size - 1, 64).astype(int)
    rows = []
    cases = {"full_box_leaf": ec.make_box_piece(box), "curved_leaf": pieces[0]}
    for label, piece in cases.items():
        closed = eg.plane_wave_coeffs(piece, poly, kx[take], ky[take])
        quad, n_nodes = eg.plane_wave_quadrature_reference(
            piece, poly, kx[take], ky[take]
        )
        scale = float(np.max(np.abs(quad)))
        err = float(np.max(np.abs(closed - quad)))
        rows.append(
            {
                "check": "plane_wave_on_shell_grid",
                "case": label,
                "kind": piece.kind,
                "n_per_coordinate": int(grid.n_per_coordinate),
                "n_modes_total": int(grid.n_modes),
                "n_modes_sampled": int(take.size),
                "k_max_h": float(np.max(np.hypot(kx, ky)) * box.side),
                "max_abs_error": err,
                "rel_error": err / scale if scale else 0.0,
                "reference_nodes": int(n_nodes),
                "tolerance": 1e-13,
                "passed": bool((err / scale if scale else 0.0) <= 1e-13),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# check 5: polar quadrature
# ---------------------------------------------------------------------------


def _direct_prefix(piece, poly, targets, t, order=20, factor=3.0) -> np.ndarray:
    """Prefix potential by plain graded Gauss quadrature (valid off the piece only).

    The cell size is set by the distance from the target to the nearest node of a
    coarse rule on the piece, so that the log singularity sitting outside the
    piece stays several cells away; the caller checks convergence by refining.
    """
    pts = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    o = eg.piece_origin(piece)
    diam = eg.piece_diameter(piece)
    probe, _ = piece.quadrature(8, diam / 8.0)
    out = np.full(pts.shape[0], np.nan)
    for idx, target in enumerate(pts):
        dist = float(np.min(np.hypot(probe[:, 0] - target[0], probe[:, 1] - target[1])))
        if dist < 0.05 * diam:
            continue
        cell = max(min(dist, math.sqrt(t)) / factor, diam / 64.0)
        nodes, weights = piece.quadrature(order, cell)
        radius = np.hypot(nodes[:, 0] - target[0], nodes[:, 1] - target[1])
        values = eg.eval_proxy(poly, nodes, origin=tuple(o))
        out[idx] = float(
            np.sum(weights * values * expc.windowed_prefix(radius, t))
        )
    return out


def check_polar_basics() -> list[dict]:
    """The fan against the leaf-residual routine, the separable rule and piece areas."""
    rows = []
    box = ec.Box((0.3125, -0.1875), 0.0625, 4)
    poly = _leaf_poly(box)
    piece = ec.make_box_piece(box)
    o = eg.piece_origin(piece)
    dens = poly.monomial_coeffs(origin=tuple(o))
    t_leaf = ec.t_l(4)
    kernel = eg.prefix_radial_kernel(t_leaf)
    targets = np.array(
        [
            [0.3125, -0.1875],
            [0.2700, -0.2100],
            [0.3750, -0.1250],
            [0.3125, -0.1250],
            [0.4200, -0.0900],
        ]
    )
    mine = eg.polar_reference(piece, poly, targets, kernel)
    theirs = np.array(
        [
            expc.polar_polygon_potential(
                piece.vertices - o, x - o, dens, kernel,
                eg.DEFAULT_N_ANG, eg.DEFAULT_N_RAD,
            )
            for x in targets
        ]
    )
    err = float(np.max(np.abs(mine - theirs)))
    rows.append(
        {
            "check": "polar_vs_leaf_residual_routine",
            "max_abs_error": err,
            "scale": float(np.max(np.abs(theirs))),
            "tolerance": 1e-14,
            "passed": bool(err <= 1e-14),
        }
    )
    a1, b1, a2, b2 = box.bounds
    rect = (a1 - o[0], b1 - o[0], a2 - o[1], b2 - o[1])
    sep = np.array(
        [expc.separable_prefix_rect(rect, x - o, dens, t_leaf)[0] for x in targets]
    )
    err = float(np.max(np.abs(mine - sep)))
    rows.append(
        {
            "check": "polar_vs_separable_prefix_full_box",
            "max_abs_error": err,
            "scale": float(np.max(np.abs(sep))),
            "tolerance": 1e-12,
            "passed": bool(err <= 1e-12),
        }
    )
    for label, (test_piece, tbox) in _test_pieces().items():
        one = eg.smooth_radial_kernel(lambda r: np.ones_like(r), eg.piece_diameter(test_piece))
        centre = np.array([[tbox.center[0], tbox.center[1]]])
        area_polar = float(
            eg.polar_reference(test_piece, None, centre, one, n_ang=32, n_rad=24)[0]
        )
        area = test_piece.area()
        rows.append(
            {
                "check": "polar_constant_kernel_reproduces_area",
                "case": label,
                "kind": test_piece.kind,
                "polar_area": area_polar,
                "piece_area": area,
                "abs_error": abs(area_polar - area),
                "tolerance": 1e-13,
                "passed": bool(abs(area_polar - area) <= 1e-13),
            }
        )
    return rows


def check_polar_cut_leaves() -> list[dict]:
    """Leaf-closure polar quadrature on cut leaves against independent references."""
    rows = []
    star = eg.StarDomain()
    ldom = eg.LDomain(corner=(0.03, 0.02))
    t_leaf = ec.t_l(4)
    kernel = eg.prefix_radial_kernel(t_leaf)

    lbox = ec.Box((0.0625, 0.0625), 0.0625, 4)
    lpieces = ldom.clip_leaf(lbox)
    poly = _leaf_poly(lbox)
    if lpieces:
        targets = np.array(
            [
                [0.030, 0.020],
                [0.0625, 0.0625],
                [0.010, 0.010],
                [0.040, 0.090],
                [0.100, 0.015],
            ]
        )
        mine = np.zeros(targets.shape[0])
        sep = np.zeros(targets.shape[0])
        for piece in lpieces:
            o = eg.piece_origin(piece)
            dens = poly.monomial_coeffs(origin=tuple(o))
            mine += eg.polar_reference(piece, poly, targets, kernel)
            a1, b1, a2, b2 = piece.bounding_box()
            rect = (a1 - o[0], b1 - o[0], a2 - o[1], b2 - o[1])
            sep += np.array(
                [
                    expc.separable_prefix_rect(rect, x - o, dens, t_leaf)[0]
                    for x in targets
                ]
            )
        err = float(np.max(np.abs(mine - sep)))
        rows.append(
            {
                "check": "polar_vs_separable_on_cut_L_leaf",
                "n_pieces": len(lpieces),
                "n_targets": int(targets.shape[0]),
                "max_abs_error": err,
                "scale": float(np.max(np.abs(sep))),
                "tolerance": 1e-12,
                "passed": bool(err <= 1e-12),
            }
        )

    found = find_cut_leaves(star, 4, kind="curved", limit=1)
    if found:
        sbox, spieces = found[0]
        poly = _leaf_poly(sbox)
        piece = spieces[0]
        sd = star.signed_distance(np.array([sbox.center]))[0]
        phi, _ = star.closest_boundary(np.array([sbox.center]))
        foot = star.boundary_point(phi)[0]
        normal = star.boundary_normal(phi)[0]
        h = sbox.side
        targets = np.vstack(
            [
                np.array([sbox.center]),
                foot[None, :] - 0.25 * h * normal[None, :],
                foot[None, :] - 0.01 * h * normal[None, :],
                foot[None, :] + 0.01 * h * normal[None, :],
                foot[None, :] + 0.25 * h * normal[None, :],
            ]
        )
        base = eg.polar_reference(piece, poly, targets, kernel)
        fine = eg.polar_reference(piece, poly, targets, kernel, n_ang=40, n_rad=32)
        scale = float(np.max(np.abs(fine)))
        err = float(np.max(np.abs(base - fine)))
        rows.append(
            {
                "check": "polar_self_convergence_cut_star_leaf",
                "signed_distance_center": float(sd),
                "max_abs_error": err,
                "scale": scale,
                "rel_error": err / scale if scale else 0.0,
                "tolerance": 1e-13,
                "passed": bool(err <= 1e-13),
            }
        )
        outside = np.vstack(
            [
                foot[None, :] + 0.30 * h * normal[None, :],
                foot[None, :] + 0.60 * h * normal[None, :],
            ]
        )
        direct = _direct_prefix(piece, poly, outside, t_leaf)
        direct_fine = _direct_prefix(piece, poly, outside, t_leaf, order=24, factor=6.0)
        ref_drift = float(np.max(np.abs(direct - direct_fine)))
        polar_out = eg.polar_reference(
            piece, poly, outside, kernel, n_ang=40, n_rad=32
        )
        err = float(np.max(np.abs(polar_out - direct_fine)))
        rows.append(
            {
                "check": "polar_vs_direct_quadrature_off_piece",
                "max_abs_error": err,
                "reference_drift": ref_drift,
                "scale": float(np.max(np.abs(direct_fine))),
                "tolerance": 1e-12,
                "passed": bool(err <= 1e-12),
            }
        )
        whole = ec.make_box_piece(sbox)
        outside_piece_targets = targets
        polar_piece = eg.polar_reference(piece, poly, outside_piece_targets, kernel)
        polar_box = eg.polar_reference(whole, poly, outside_piece_targets, kernel)
        rows.append(
            {
                "check": "cut_vs_full_box_closure_gap",
                "note": "reported as context: the physical-side closure differs "
                        "from the box-extended one by the cut-away mass",
                "max_abs_gap": float(np.max(np.abs(polar_piece - polar_box))),
                "passed": True,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# check 6: plots
# ---------------------------------------------------------------------------


def make_plots(out: Path, level: int = 4) -> list[str]:
    """Draw the S and L leaf decompositions at the given level."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
    except Exception:  # pragma: no cover - plotting is optional
        return []
    names = []
    for name in ("S", "L"):
        domain = eg.make_domain(name)
        tree = ec.build_quadtree(levels=int(level))
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        for box in tree.leaves():
            a1, b1, a2, b2 = box.bounds
            ax.add_patch(
                MplPolygon(
                    [[a1, a2], [b1, a2], [b1, b2], [a1, b2]],
                    closed=True,
                    fill=False,
                    edgecolor="0.85",
                    linewidth=0.4,
                )
            )
            pieces = domain.clip_leaf(box)
            for piece in pieces:
                if piece.kind == "curved":
                    loop = []
                    for edge in piece.edges or []:
                        s = np.linspace(edge.s0, edge.s1, 33)
                        loop.append(edge.curve(s))
                    poly = np.vstack(loop)
                    color = "#c2410c"
                elif piece.kind == "box":
                    poly = piece.vertices
                    color = "#bfdbfe"
                else:
                    poly = piece.vertices
                    color = "#f59e0b"
                ax.add_patch(
                    MplPolygon(
                        poly, closed=True, facecolor=color, edgecolor="#1f2937",
                        linewidth=0.5, alpha=0.55,
                    )
                )
        if name == "S":
            phi = np.linspace(0.0, 2.0 * math.pi, 2001)
            curve = domain.boundary_point(phi)
            ax.plot(curve[:, 0], curve[:, 1], color="k", linewidth=1.2)
        else:
            verts = domain.polygon()
            ax.plot(
                np.append(verts[:, 0], verts[0, 0]),
                np.append(verts[:, 1], verts[0, 1]),
                color="k",
                linewidth=1.2,
            )
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"case {name}: leaf decomposition at L = {level}")
        fig.tight_layout()
        path = out / f"experiment_e_geometry_leaves_{name}_L{level}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        names.append(path.name)
    return names


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def environment_summary() -> dict:
    """Interpreter, numpy and platform identification for the run record."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check, write the JSON summary and the plots, and report the status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--levels", default="3,4,5", help="levels for the area sweep")
    parser.add_argument("--plot-level", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="draw the decomposition plots and skip the checks (for an "
             "interpreter that has matplotlib but is not the run's venv)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(v) for v in str(args.levels).split(",") if v.strip())

    if args.plots_only:
        names = make_plots(out, args.plot_level)
        print(json.dumps({"plots": names}, indent=2))
        return 0 if names else 1

    started = time.time()
    rows: list[dict] = []
    stages = []
    for label, func in (
        ("areas", lambda: check_areas(levels)),
        ("l_corner", check_l_corner),
        ("double_crossing",
         lambda: check_double_crossing(sorted(set(levels) | {6}))),
        ("plane_waves", check_plane_waves),
        ("plane_wave_grid", check_plane_wave_grid_sample),
        ("polar_basics", check_polar_basics),
        ("polar_cut_leaves", check_polar_cut_leaves),
    ):
        t0 = time.time()
        got = func()
        rows.extend(got)
        stages.append(
            {
                "stage": label,
                "rows": len(got),
                "seconds": time.time() - t0,
                "passed": all(bool(r.get("passed", True)) for r in got),
            }
        )
        print(f"[{label}] {len(got)} rows, {time.time() - t0:.1f} s", flush=True)

    plots = [] if args.no_plots else make_plots(out, args.plot_level)
    payload = {
        "environment": environment_summary(),
        "stages": stages,
        "rows": rows,
        "plots": plots,
        "all_passed": all(bool(r.get("passed", True)) for r in rows),
        "wall_seconds": time.time() - started,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=float)
    (out / "experiment_e_geometry_tests.json").write_text(text)
    print(text)
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
