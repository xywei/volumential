"""Experiment E study driver: the 2D windowed telescoping assembly on complex geometry.

This module runs the studies of the experiment E plan and writes them to disk
incrementally, one CSV per study plus one JSON block per study, so that an
interrupted run can be resumed study by study.  It computes nothing new: every
number comes from ``experiment_e_hierarchy`` (the three stages and their
Laplacians), ``experiment_e_geometry`` (domains, leaf clipping, polar rules)
and ``experiment_e_baseline`` (the asymptotic local part and the Nystrom
solver).  What this file adds is the experimental design: which probes, which
references, which sweeps, and how each result is reduced to a row.

Studies, in the order the plan asks for them:

``b``
    Case B (full root box, no cut), ``Theta = 8``, level counts 3 to 5:
    per-stage PDE residuals against the closed-form heat-smoothed right-hand
    sides, the assembled total against the direct polar reference at 24 probes,
    and the per-coordinate plane-wave mode count per level.
``l``
    Case L (re-entrant corner), ``Theta = 8``: the same per-stage residuals on
    the probe bands of ``experiment_e_common.probe_sets`` including the corner
    bisector inside and outside, the interface jumps of ``u`` and ``grad u``
    across the straight edges and at the corner, and the total against the
    polar reference.
``s``
    Case S (smooth star), ``Theta = 8``: per-stage residuals, interface jumps
    across the curve, total against the polar reference.
``jumps``
    The interface jumps alone, on cases L and S, at the plan's probe offsets
    ``10^{-3}``, ``10^{-4}`` and ``10^{-5}`` times ``h_L``.
``jumps_offsets``
    The same study at ten times those offsets.  It is the control that decides,
    site by site, which extrapolation model the jump columns may be read from,
    so it must cover the same cases and level counts as ``jumps``.
``baseline``
    The Fryklund Lemma 4.5 local part against the exact leaf closure: on case S
    versus distance to the boundary and versus ``h_L`` (with an order fit), and
    on case L at the corner-band probes with the nearest-edge closest point,
    where the expansion's hypotheses do not hold.
``bvp``
    The manufactured Dirichlet problem on case S: ``u = V[rho] + D[mu]`` with
    the assembled volume potential and the Nystrom double layer, measured
    against ``u_exact`` on an interior grid, versus the level count and versus
    the number of boundary nodes.
``theta12``
    Case S at one level count with ``Theta = 12``, the secondary declaration.
``volumential``
    The optional box-code cross-check on case B, which records itself as
    skipped unless it is explicitly enabled and a working OpenCL device is
    present.

Two conventions are worth stating because they decide how the numbers read.

First, the interface jump is measured as the difference of the two *one-sided
limits* at a boundary point, not as the difference of two nearby values.  The
volume potential is continuously differentiable across ``partial Omega`` while
its Laplacian jumps, so ``u(b + d n) - u(b - d n)`` is ``O(d)`` no matter how
accurate the scheme is, and the linear extrapolation to ``d = 0`` from the
plan's two distances ``d_1 = 10^{-3} h_L`` and ``d_2 = 10^{-4} h_L`` still
carries the geometric residue ``(1/2) [u_nn] d_1 d_2 = (1/2) rho d_1 d_2``
(measured maximum over the probe sites: ``2.5e-9`` at ``L = 3``, independent of
the scheme).  Each side is
therefore evaluated at a third distance ``d_3 = 10^{-5} h_L`` as well and
extrapolated quadratically, whose residue is ``O(d_1 d_2 d_3)``; that is the
number reported as the jump.  The raw differences at ``d_1`` and ``d_2`` and
the linear extrapolation are kept in the CSV beside it.

Second, the Fryklund comparison needs the closure over the whole of ``Omega``,
not the colleague-truncated closure the hierarchy assembles.  ``closure_exact``
sums ``prefix_potential`` over every leaf within ``cutoff sqrt(t_L)`` of the
target (default ``14``, where the ``chi_0`` kernel is below ``1e-21`` of its
value at the leaf scale), so the baseline column is not contaminated by the
window truncation that the residual column measures.  The difference between
the two is itself reported.

Invocation (with ``PYTHONPATH`` set to the directory holding these modules):

```bash
python experiment_e_run.py --out OUT --studies b,l,s,baseline,bvp,theta12
python experiment_e_run.py --out OUT --merge          # collect the JSON blocks
python experiment_e_run.py --out OUT --plots-only     # PNGs from the CSVs
```

Studies are independent processes by design: each writes its own
``experiment_e_<study>.json`` and its own CSV files, and ``--merge`` collects
them into ``experiment_e_summary.json``.  Running them in parallel is therefore
safe, and ``--resume`` skips any study whose JSON block already exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import experiment_e_baseline as eb  # noqa: E402
import experiment_e_common as ec  # noqa: E402
import experiment_e_geometry as eg  # noqa: E402
import experiment_e_hierarchy as eh  # noqa: E402


__all__ = [
    "IncrementalCsv",
    "boundary_sites",
    "closure_exact",
    "interior_grid",
    "probe_families",
    "run_geometry_case",
    "study_baseline",
    "study_bvp",
    "study_case_b",
    "study_fastpath",
    "study_case_l",
    "study_case_s",
    "study_jumps",
    "study_theta12",
    "study_volumential",
    "collect_subdirs",
    "main",
]


DEFAULT_PROXY_ORDER = 6
"""Tensor Lagrange order of the per-leaf density proxy (Paper 1's 2D order is 4)."""

DEFAULT_LEVELS = (3, 4, 5)
"""Level counts of the primary sweep."""

CLOSURE_CUTOFF = 14.0
"""Leaves within this many ``sqrt(t_L)`` enter ``closure_exact``; the rest are 1e-21."""

JUMP_FRACTIONS = (1.0e-3, 1.0e-4, 1.0e-5)
"""Probe offsets for the jump test, in units of ``h_L``.

The plan names the first two.  A third is added because linear extrapolation
from two offsets leaves a residue ``(1/2) [u_nn] d_1 d_2`` on a boundary where
``Delta u`` jumps, and that residue -- ``4e-10`` at ``L = 3``, a purely
geometric number -- would otherwise be reported as the scheme's jump.  Quadratic
extrapolation from three offsets leaves ``O(d_1 d_2 d_3)``, below ``10^{-14}``,
so the measured jump is a property of the scheme.  The two-offset value is kept
in the record alongside it."""

BVP_MARGIN = 0.12
"""Interior grid keeps this distance from the curve: the smooth double-layer rule
loses accuracy within a few node spacings of the boundary, which is a property of
the evaluation rule, not of the volume potential."""

BASELINE_DISTANCE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
"""Probe distances for the baseline column, in units of ``sqrt(t_L) = h_L / Theta``."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


class IncrementalCsv:
    """Append-only CSV writer that flushes every row, so a killed run keeps its rows."""

    def __init__(self, path, fieldnames: Sequence[str]):
        """Open ``path`` for appending and write the header when the file is new."""
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        fresh = not self.path.exists()
        self.handle = self.path.open("a", newline="")
        self.writer = csv.DictWriter(
            self.handle, fieldnames=self.fieldnames, extrasaction="ignore"
        )
        if fresh:
            self.writer.writeheader()
            self.handle.flush()

    def row(self, **kwargs) -> None:
        """Write one row and flush it to disk."""
        self.writer.writerow(kwargs)
        self.handle.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self.handle.close()


def _log(message: str) -> None:
    """Print a timestamped progress line, flushed, for the on-disk run log."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _write_block(out: Path, name: str, payload: dict) -> None:
    """Write one study's JSON block."""
    path = out / f"experiment_e_{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    _log(f"wrote {path.name}")


def _fit_order(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``log y`` against ``log x``; NaN when it is undefined."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    good = (xs > 0.0) & (ys > 0.0) & np.isfinite(xs) & np.isfinite(ys)
    if int(np.count_nonzero(good)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)[0])


def _extrapolation_weights(offsets: Sequence[float]) -> np.ndarray:
    """Lagrange weights that extrapolate values at ``offsets`` to offset zero."""
    nodes = np.asarray(offsets, dtype=np.float64)
    weights = np.ones(nodes.size, dtype=np.float64)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                weights[i] *= (0.0 - nodes[j]) / (nodes[i] - nodes[j])
    return weights


def _log_model_weights(offsets: Sequence[float]) -> np.ndarray:
    """Weights extracting the ``d -> 0`` limit under the model ``c0 + c1 d + c2 d log d``.

    The volume potential of a density that jumps across ``partial Omega`` is
    ``C^1`` but not ``C^{1,1}``: the one-sided expansion of ``grad u`` carries a
    ``d log d`` term, which no polynomial extrapolation can remove.  Measured on
    the star at ``L = 3``, the one-sided gradient difference falls by a factor
    ``7.4`` from ``d_1`` to ``d_2`` where a purely linear term would give ``10``
    and ``d log(1/d)`` gives ``7.8``.  Adding that basis function is what makes
    the gradient jump a measurement of the scheme rather than of the potential.
    """
    nodes = np.asarray(offsets, dtype=np.float64)
    basis = np.stack([np.ones_like(nodes), nodes, nodes * np.log(nodes)], axis=1)
    unit = np.zeros(nodes.size)
    unit[0] = 1.0
    return np.linalg.solve(basis.T, unit)


def _stats(values: np.ndarray) -> dict:
    """Max absolute value and RMS of an array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"max_abs": 0.0, "rms": 0.0}
    return {
        "max_abs": float(np.max(np.abs(arr))),
        "rms": float(np.sqrt(np.mean(arr * arr))),
    }


# ---------------------------------------------------------------------------
# probes, boundary sites and the untruncated closure
# ---------------------------------------------------------------------------


def probe_families(
    domain,
    tree: ec.QuadTree,
    levels: int,
    theta: float,
    n_per: int = 8,
    n_corner: int = 10,
    seed: int = 20260918,
) -> tuple[np.ndarray, list]:
    """Probe points and their family labels for one geometry case.

    Wraps ``experiment_e_common.probe_sets``: bulk points inside and outside, a
    band within one leaf of the boundary, a band within ``2 sqrt(t_L)`` of it,
    and, when the domain has a re-entrant corner, a geometric sequence along the
    interior bisector and its outward continuation.
    """
    h_leaf = tree.h(int(levels))
    t_leaf = ec.t_l(int(levels), float(theta), tree.root.half)
    kwargs: dict = {}
    corner = domain.corner()
    if corner is not None:
        kwargs["corner"] = corner[0]
        kwargs["bisector"] = corner[1]
        kwargs["n_corner"] = int(n_corner)
    sets = ec.probe_sets(
        domain.sdf(),
        h_leaf,
        t_leaf,
        root_half=tree.root.half,
        n_bulk=int(n_per),
        n_band=int(n_per),
        seed=int(seed),
        **kwargs,
    )
    points: list = []
    labels: list = []
    for name in sorted(sets):
        block = np.asarray(sets[name], dtype=np.float64).reshape(-1, 2)
        for point in block[: int(n_per) if "corner" not in name else int(n_corner)]:
            points.append(point)
            labels.append(name)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2), labels


def _outward(domain, point: np.ndarray, candidate: np.ndarray, eps: float) -> np.ndarray:
    """Orient a unit normal so that stepping along it leaves ``Omega``."""
    point = np.asarray(point, dtype=np.float64).reshape(2)
    nrm = np.asarray(candidate, dtype=np.float64).reshape(2)
    nrm = nrm / max(float(np.hypot(nrm[0], nrm[1])), 1e-300)
    plus = bool(domain.inside(point + eps * nrm)[0])
    minus = bool(domain.inside(point - eps * nrm)[0])
    if minus and not plus:
        return nrm
    if plus and not minus:
        return -nrm
    return nrm


def boundary_sites(domain, case: str, n_sites: int, eps: float) -> tuple:
    """Boundary points, outward unit normals and site kinds for the jump test.

    Case L takes interior samples of every polygon edge plus the re-entrant
    corner itself (where the outward direction is the outward bisector); case S
    takes equispaced parameters on the curve.  Case B has no interface inside
    the root box and returns nothing.
    """
    if case == "B":
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    points: list = []
    normals: list = []
    kinds: list = []
    if case == "L":
        verts = domain.polygon()
        n_edge = verts.shape[0]
        for i in range(n_edge):
            a = verts[i]
            b = verts[(i + 1) % n_edge]
            edge = b - a
            length = float(np.hypot(edge[0], edge[1]))
            if length < 10.0 * eps:
                continue
            tangent = edge / length
            cand = np.array([tangent[1], -tangent[0]])
            for frac in (0.35, 0.65):
                mid = a + frac * edge
                points.append(mid)
                normals.append(_outward(domain, mid, cand, eps))
                kinds.append("edge")
        corner, bisector = domain.corner()
        points.append(np.asarray(corner, dtype=np.float64))
        normals.append(-np.asarray(bisector, dtype=np.float64))
        kinds.append("corner")
    elif case == "S":
        phis = 2.0 * math.pi * np.arange(int(n_sites)) / float(n_sites) + 0.017
        pts = domain.boundary_point(phis)
        nrm = domain.boundary_normal(phis)
        for k in range(pts.shape[0]):
            points.append(pts[k])
            normals.append(_outward(domain, pts[k], nrm[k], eps))
            kinds.append("curve")
    else:
        raise ValueError(f"unknown case {case!r}")
    return (
        np.asarray(points, dtype=np.float64).reshape(-1, 2),
        np.asarray(normals, dtype=np.float64).reshape(-1, 2),
        kinds,
    )


def closure_exact(
    leaves: Sequence,
    targets,
    t_leaf: float,
    cutoff: float = CLOSURE_CUTOFF,
    n_ang: int = eh.DEFAULT_POLAR_ANG,
    n_rad: int = eh.DEFAULT_POLAR_RAD,
) -> np.ndarray:
    """``(4 pi)^{-1} int_Omega E_1(r^2 / 4 t_L) rho``, over every leaf within the cutoff.

    This is the quantity Fryklund's Lemma 4.5 approximates.  It differs from the
    hierarchy's ``closure_stage`` only in the source list: colleagues there, all
    leaves within ``cutoff sqrt(t_L)`` here, which removes the window truncation
    from the baseline comparison.
    """
    pts = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    out = np.zeros(pts.shape[0], dtype=np.float64)
    radius = float(cutoff) * math.sqrt(float(t_leaf))
    for leaf in leaves:
        if leaf.poly is None:
            continue
        centre = np.asarray(leaf.box.center, dtype=np.float64)
        reach = radius + leaf.box.half * math.sqrt(2.0)
        near = np.nonzero(
            np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) <= reach
        )[0]
        if near.size == 0:
            continue
        for piece in leaf.pieces:
            value, _, _ = eh.prefix_potential(
                piece, leaf.poly, pts[near], float(t_leaf),
                n_ang=n_ang, n_rad=n_rad,
            )
            out[near] += value
    return out


def interior_grid(
    domain, margin: float = BVP_MARGIN, n: int = 25, extent: float = 0.62,
    limit: int = 64
) -> np.ndarray:
    """Deterministic interior probe grid at least ``margin`` from the boundary."""
    axis = np.linspace(-float(extent), float(extent), int(n))
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    keep = pts[np.asarray(domain.signed_distance(pts)) <= -float(margin)]
    if keep.shape[0] > int(limit):
        idx = np.linspace(0, keep.shape[0] - 1, int(limit)).astype(np.int64)
        keep = keep[idx]
    return np.asarray(keep, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# one geometry case at one level count
# ---------------------------------------------------------------------------


def run_geometry_case(
    case: str,
    levels: int,
    theta: float,
    order: int = DEFAULT_PROXY_ORDER,
    n_per: int = 8,
    n_corner: int = 10,
    n_sites: int = 10,
    n_reference: int = 24,
    with_jumps: bool = True,
    jumps_only: bool = False,
    jump_fractions: Sequence[float] = JUMP_FRACTIONS,
    rho: Callable = eh.rho_gaussian,
    seed: int = 20260918,
) -> dict:
    """Assemble one case at one level count and reduce it to residuals, jumps, reference.

    A single ``assemble`` call covers the residual probes and the jump probes,
    because the cost is linear in the target count and the probe families are
    only a bookkeeping split of one target array.  One consequence has to be
    read with the residual table: the jump probes sit in the target array too,
    under the family label ``jump``, so the ``all`` rows of the residual CSV are
    maxima over probes at ``10^{-3}``, ``10^{-4}`` and ``10^{-5}`` leaf widths
    from the interface as well as over the named bands, and on the cut cases the
    ``all`` maximum is usually attained there.  The synthetic family
    ``all_except_jump`` carries the same maximum with those probes removed.
    """
    domain = eg.make_domain(case)
    tree = ec.build_quadtree(levels=int(levels))
    clock = time.perf_counter()
    leaves = domain.leaves(tree, rho, int(order))
    t_clip = time.perf_counter() - clock
    h_leaf = tree.h(int(levels))
    t_leaf = ec.t_l(int(levels), float(theta), tree.root.half)

    if jumps_only:
        probes, labels = np.zeros((0, 2)), []
    elif case == "B":
        probes, labels = eh.case_b_probes(int(levels), root_half=tree.root.half)
    else:
        probes, labels = probe_families(
            domain, tree, levels, theta, n_per=n_per, n_corner=n_corner, seed=seed
        )
    n_probes = int(probes.shape[0])

    sites_pts = np.zeros((0, 2))
    sites_nrm = np.zeros((0, 2))
    kinds: list = []
    jump_pts = np.zeros((0, 2))
    offsets = tuple(float(f) * h_leaf for f in jump_fractions)
    if with_jumps and case != "B":
        sites_pts, sites_nrm, kinds = boundary_sites(
            domain, case, n_sites, eps=1.0e-6 * h_leaf
        )
        block = []
        for k in range(sites_pts.shape[0]):
            for d in offsets:
                block.append(sites_pts[k] - d * sites_nrm[k])
                block.append(sites_pts[k] + d * sites_nrm[k])
        jump_pts = np.asarray(block, dtype=np.float64).reshape(-1, 2)

    targets = np.vstack([probes, jump_pts]) if jump_pts.size else probes
    all_labels = list(labels) + ["jump"] * int(jump_pts.shape[0])

    _log(
        f"case {case} L={levels} Theta={theta:g}: {len(leaves)} leaves, "
        f"{targets.shape[0]} targets (clip {t_clip:.1f}s)"
    )
    clock = time.perf_counter()
    result = eh.assemble(
        leaves, targets, theta=float(theta), tree=tree, levels=int(levels)
    )
    t_assemble = time.perf_counter() - clock
    _log(
        f"case {case} L={levels}: assembled in {t_assemble:.1f}s "
        f"(stages {result['seconds']})"
    )

    t_residual = 0.0
    t_reference = 0.0
    stage_rows: list = []
    reference_block: dict = {}
    indicator_mismatch = -1
    if not jumps_only:
        clock = time.perf_counter()
        report = eh.residual_report(
            leaves, targets, theta=float(theta), result=result, labels=all_labels,
            indicator=domain.inside,
        )
        t_residual = time.perf_counter() - clock
        _log(f"case {case} L={levels}: residual report in {t_residual:.1f}s")
        stage_rows = report["stages"]
        indicator_mismatch = int(report.get("indicator_mismatch", -1))
        if indicator_mismatch:
            _log(
                f"case {case} L={levels}: WARNING {indicator_mismatch} targets "
                f"where the stage indicator and the domain disagree"
            )

        # the assembled total against a direct polar reference
        ref_idx = np.unique(
            np.linspace(
                0, n_probes - 1, min(int(n_reference), n_probes)
            ).astype(np.int64)
        )
        clock = time.perf_counter()
        reference = eh.polar_reference_total(leaves, probes[ref_idx], t_leaf)
        t_reference = time.perf_counter() - clock
        diff = result["total"].u[ref_idx] - reference
        scale = max(float(np.max(np.abs(reference))), 1e-300)
        reference_block = {
            "n_probes": int(ref_idx.size),
            "max_abs": float(np.max(np.abs(diff))),
            "max_rel": float(np.max(np.abs(diff)) / scale),
            "rms": float(np.sqrt(np.mean(diff * diff))),
            "reference_max": scale,
        }

    jumps: list = []
    stage_jumps: dict = {}
    if jump_pts.size:
        base = n_probes
        n_off = len(offsets)
        lin = _extrapolation_weights(offsets[:2])
        quad = _extrapolation_weights(offsets)
        logw = _log_model_weights(offsets)
        for k in range(sites_pts.shape[0]):
            i0 = base + 2 * n_off * k
            inside = [i0 + 2 * m for m in range(n_off)]
            outside = [i0 + 2 * m + 1 for m in range(n_off)]
            u_in = result["total"].u[inside]
            u_out = result["total"].u[outside]
            g_in = result["total"].grad[inside]
            g_out = result["total"].grad[outside]
            u_in0 = float(quad @ u_in)
            u_out0 = float(quad @ u_out)
            g_in0 = quad @ g_in
            g_out0 = quad @ g_out
            u_in_log = float(logw @ u_in)
            u_out_log = float(logw @ u_out)
            g_in_log = logw @ g_in
            g_out_log = logw @ g_out
            jumps.append(
                {
                    "site": int(k),
                    "kind": kinds[k],
                    "x": float(sites_pts[k, 0]),
                    "y": float(sites_pts[k, 1]),
                    "d1": offsets[0],
                    "d2": offsets[1],
                    "d3": offsets[2] if n_off > 2 else "",
                    "jump_u_d1": float(abs(u_in[0] - u_out[0])),
                    "jump_u_d2": float(abs(u_in[1] - u_out[1])),
                    "jump_u_d3": float(abs(u_in[2] - u_out[2])) if n_off > 2 else "",
                    "jump_u_linear": float(abs(lin @ u_in[:2] - lin @ u_out[:2])),
                    "jump_u_extrap": float(abs(u_in0 - u_out0)),
                    "jump_u_log": float(abs(u_in_log - u_out_log)),
                    "jump_grad_d1": float(np.linalg.norm(g_in[0] - g_out[0])),
                    "jump_grad_d2": float(np.linalg.norm(g_in[1] - g_out[1])),
                    "jump_grad_d3": (
                        float(np.linalg.norm(g_in[2] - g_out[2]))
                        if n_off > 2 else ""
                    ),
                    "jump_grad_linear": float(
                        np.linalg.norm(lin @ g_in[:2] - lin @ g_out[:2])
                    ),
                    "jump_grad_extrap": float(np.linalg.norm(g_in0 - g_out0)),
                    "jump_grad_log": float(np.linalg.norm(g_in_log - g_out_log)),
                    "u_scale": float(max(abs(u_in0), abs(u_out0))),
                    "grad_scale": float(
                        max(np.linalg.norm(g_in0), np.linalg.norm(g_out0))
                    ),
                }
            )

        # which stage carries the residual jump: same estimators, per stage
        fields = {
            "w0": result["w0"],
            "shell_total": result["shell_total"],
            "closure": result["closure"],
            "total": result["total"],
        }
        for name, field in fields.items():
            worst = {"u_extrap": 0.0, "u_log": 0.0,
                     "grad_extrap": 0.0, "grad_log": 0.0}
            for k in range(sites_pts.shape[0]):
                i0 = n_probes + 2 * n_off * k
                ins = [i0 + 2 * m for m in range(n_off)]
                outs = [i0 + 2 * m + 1 for m in range(n_off)]
                worst["u_extrap"] = max(
                    worst["u_extrap"],
                    abs(float(quad @ field.u[ins] - quad @ field.u[outs])),
                )
                worst["u_log"] = max(
                    worst["u_log"],
                    abs(float(logw @ field.u[ins] - logw @ field.u[outs])),
                )
                worst["grad_extrap"] = max(
                    worst["grad_extrap"],
                    float(np.linalg.norm(
                        quad @ field.grad[ins] - quad @ field.grad[outs]
                    )),
                )
                worst["grad_log"] = max(
                    worst["grad_log"],
                    float(np.linalg.norm(
                        logw @ field.grad[ins] - logw @ field.grad[outs]
                    )),
                )
            stage_jumps[name] = worst

    family_counts: dict = {}
    for lab in all_labels:
        family_counts[lab] = family_counts.get(lab, 0) + 1
    if "jump" in family_counts and len(family_counts) > 1:
        family_counts["all_except_jump"] = len(all_labels) - family_counts["jump"]

    return {
        "case": case,
        "levels": int(levels),
        "theta": float(theta),
        "proxy_order": int(order),
        "h_leaf": float(h_leaf),
        "t_leaf": float(t_leaf),
        "n_leaves": len(leaves),
        "n_cut_leaves": int(sum(1 for lf in leaves if lf.is_cut())),
        "n_targets": int(targets.shape[0]),
        "n_probes": n_probes,
        "family_counts": family_counts,
        "indicator_mismatch": indicator_mismatch,
        "seconds": {
            "clip": t_clip,
            "assemble": t_assemble,
            "residual": t_residual,
            "reference": t_reference,
            "stages": result["seconds"],
        },
        "residuals": stage_rows,
        "modes": [] if jumps_only else [
            {
                "level": info["level"],
                "n_per_coordinate": info["n_per_coordinate"],
                "n_modes": info["n_modes"],
                "trunc_rel": info["trunc_rel"],
                "alias_rel": info["alias_rel"],
            }
            for info in result["modes"]
        ],
        "total_vs_polar_reference": reference_block,
        "jumps": jumps,
        "stage_jumps": stage_jumps,
    }


def _emit_case(out: Path, block: dict) -> None:
    """Append one case result to the residual, mode, reference and jump CSVs."""
    res_csv = IncrementalCsv(
        out / "experiment_e_residual.csv",
        [
            "case", "theta", "levels", "h_leaf", "t_leaf", "n_leaves",
            "n_cut_leaves", "n_targets", "stage", "family", "n_family",
            "max_abs", "rms", "max_rel",
        ],
    )
    common = {
        "case": block["case"],
        "theta": block["theta"],
        "levels": block["levels"],
        "h_leaf": block["h_leaf"],
        "t_leaf": block["t_leaf"],
        "n_leaves": block["n_leaves"],
        "n_cut_leaves": block["n_cut_leaves"],
        "n_targets": block["n_targets"],
    }
    for row in block["residuals"]:
        res_csv.row(
            stage=row["stage"], family="all", n_family=block["n_targets"],
            max_abs=row["max_abs"], rms=row["rms"], max_rel=row["max_rel"],
            **common,
        )
        for family, value in sorted(row.get("by_family", {}).items()):
            res_csv.row(
                stage=row["stage"], family=family,
                n_family=block["family_counts"].get(family, 0),
                max_abs=value, rms="", max_rel="", **common,
            )
    res_csv.close()

    mode_csv = IncrementalCsv(
        out / "experiment_e_modes.csv",
        ["case", "theta", "levels", "level", "n_per_coordinate", "n_modes",
         "trunc_rel", "alias_rel"],
    )
    for info in block["modes"]:
        mode_csv.row(
            case=block["case"], theta=block["theta"], levels=block["levels"],
            **info,
        )
    mode_csv.close()

    if not block["total_vs_polar_reference"]:
        _emit_jumps(out, block)
        return
    ref_csv = IncrementalCsv(
        out / "experiment_e_reference.csv",
        ["case", "theta", "levels", "n_probes", "max_abs", "max_rel", "rms",
         "reference_max", "assemble_seconds"],
    )
    ref_csv.row(
        case=block["case"], theta=block["theta"], levels=block["levels"],
        assemble_seconds=block["seconds"]["assemble"],
        **block["total_vs_polar_reference"],
    )
    ref_csv.close()

    _emit_jumps(out, block)


def _emit_jumps(out: Path, block: dict, stem: str = "jumps") -> None:
    """Append one case's interface-jump rows to the jump CSV of ``stem``."""
    if block["jumps"]:
        jump_csv = IncrementalCsv(
            out / f"experiment_e_{stem}.csv",
            ["case", "theta", "levels", "h_leaf", "site", "kind", "x", "y",
             "d1", "d2", "d3", "jump_u_d1", "jump_u_d2", "jump_u_d3",
             "jump_u_linear", "jump_u_extrap", "jump_u_log", "jump_grad_d1",
             "jump_grad_d2", "jump_grad_d3", "jump_grad_linear",
             "jump_grad_extrap", "jump_grad_log", "u_scale", "grad_scale"],
        )
        for row in block["jumps"]:
            jump_csv.row(
                case=block["case"], theta=block["theta"],
                levels=block["levels"], h_leaf=block["h_leaf"], **row,
            )
        jump_csv.close()


def _case_study(
    out: Path, name: str, case: str, levels_list: Sequence[int], theta: float,
    order: int, with_jumps: bool, budget_seconds: float = 0.0,
    extra_levels: Sequence[int] = (), **kwargs
) -> dict:
    """Run one geometry case over a list of level counts and write its block."""
    blocks = []
    elapsed_last = 0.0
    for levels in list(levels_list):
        clock = time.perf_counter()
        block = run_geometry_case(
            case, levels, theta, order=order, with_jumps=with_jumps, **kwargs
        )
        elapsed_last = time.perf_counter() - clock
        blocks.append(block)
        _emit_case(out, block)
        _write_block(
            out, name,
            {"study": name, "case": case, "theta": theta, "levels": blocks},
        )
    for levels in list(extra_levels):
        if budget_seconds > 0.0 and elapsed_last > budget_seconds:
            _log(
                f"case {case}: skipping L={levels}, the previous level took "
                f"{elapsed_last:.0f}s against a budget of {budget_seconds:.0f}s"
            )
            blocks.append(
                {
                    "case": case,
                    "levels": int(levels),
                    "skipped": True,
                    "reason": (
                        f"previous level took {elapsed_last:.0f}s; the extra "
                        f"level is run only under {budget_seconds:.0f}s"
                    ),
                }
            )
            continue
        clock = time.perf_counter()
        block = run_geometry_case(
            case, levels, theta, order=order, with_jumps=with_jumps, **kwargs
        )
        elapsed_last = time.perf_counter() - clock
        blocks.append(block)
        _emit_case(out, block)
    payload = {"study": name, "case": case, "theta": theta, "levels": blocks}
    _write_block(out, name, payload)
    return payload


def study_case_b(out: Path, levels_list=DEFAULT_LEVELS, theta: float = ec.THETA_PRIMARY,
                 order: int = DEFAULT_PROXY_ORDER) -> dict:
    """Study 1: case B, the full-box control."""
    return _case_study(out, "case_b", "B", levels_list, theta, order, with_jumps=False)


def study_case_l(out: Path, levels_list=DEFAULT_LEVELS, theta: float = ec.THETA_PRIMARY,
                 order: int = DEFAULT_PROXY_ORDER, extra_levels=(6,),
                 budget_seconds: float = 300.0) -> dict:
    """Study 2: case L, the re-entrant corner, with the extra level under a budget."""
    return _case_study(
        out, "case_l", "L", levels_list, theta, order, with_jumps=True,
        extra_levels=extra_levels, budget_seconds=budget_seconds,
    )


def study_case_s(out: Path, levels_list=DEFAULT_LEVELS, theta: float = ec.THETA_PRIMARY,
                 order: int = DEFAULT_PROXY_ORDER) -> dict:
    """Study 3a: case S, the smooth star."""
    return _case_study(out, "case_s", "S", levels_list, theta, order, with_jumps=True)


def study_theta12(out: Path, levels: int = 4, theta: float = ec.THETA_SECONDARY,
                  order: int = DEFAULT_PROXY_ORDER) -> dict:
    """Study 4: case S at one level count with the secondary window declaration."""
    return _case_study(out, "theta12", "S", (int(levels),), theta, order,
                       with_jumps=True)


# ---------------------------------------------------------------------------
# study: the asymptotic baseline against the exact closure
# ---------------------------------------------------------------------------


def _star_baseline_rows(domain, leaves, tree, levels: int, theta: float,
                        rho: Callable, n_sites: int = 6) -> list:
    """Fryklund ``V_L`` against ``closure_exact`` on case S, per site and distance."""
    t_leaf = ec.t_l(int(levels), float(theta), tree.root.half)
    h_leaf = tree.h(int(levels))
    sqrt_t = math.sqrt(t_leaf)
    phis = 2.0 * math.pi * np.arange(int(n_sites)) / float(n_sites) + 0.11
    base = domain.boundary_point(phis)
    nrm = domain.boundary_normal(phis)
    kappa = domain.curvature(phis)
    targets = []
    meta = []
    for k in range(base.shape[0]):
        for factor in BASELINE_DISTANCE_FACTORS:
            dist = float(factor) * sqrt_t
            targets.append(base[k] - dist * nrm[k])
            meta.append((k, float(factor), dist, float(kappa[k])))
    targets = np.asarray(targets, dtype=np.float64)
    exact = closure_exact(leaves, targets, t_leaf)
    rows = []
    for m, (site, factor, dist, kap) in enumerate(meta):
        r_loc, tangent, inward = eb.boundary_frame(targets[m], base[site], 1.0)
        jet = eb.jet_from_callable(rho, targets[m], tangent, inward)
        value = float(
            np.asarray(
                eb.fryklund_VL(targets[m], base[site], kap, jet, t_leaf, sign=1.0)
            ).ravel()[0]
        )
        err = value - float(exact[m])
        rows.append(
            {
                "case": "S",
                "site": "curve",
                "levels": int(levels),
                "theta": float(theta),
                "h_leaf": h_leaf,
                "t_leaf": t_leaf,
                "site_index": int(site),
                "distance": dist,
                "distance_over_sqrt_t": factor,
                "r_frame": float(r_loc),
                "kappa_b": kap,
                "closure_exact": float(exact[m]),
                "fryklund": value,
                "abs_err": float(abs(err)),
                "rel_err": float(abs(err) / max(abs(float(exact[m])), 1e-300)),
            }
        )
    return rows


def _l_corner_baseline_rows(domain, leaves, tree, levels: int, theta: float,
                            rho: Callable) -> list:
    """The same expansion at case L's corner band and, as a control, at an edge band."""
    t_leaf = ec.t_l(int(levels), float(theta), tree.root.half)
    h_leaf = tree.h(int(levels))
    sqrt_t = math.sqrt(t_leaf)
    corner, bisector = domain.corner()
    verts = domain.polygon()

    def _closest(point):
        """Nearest point of the polygon boundary and the number of edges attaining it."""
        best = (float("inf"), None)
        feet = []
        for i in range(verts.shape[0]):
            a = verts[i]
            b = verts[(i + 1) % verts.shape[0]]
            edge = b - a
            denom = float(edge @ edge)
            tau = float(np.clip(((point - a) @ edge) / denom, 0.0, 1.0))
            foot = a + tau * edge
            dist = float(np.hypot(*(point - foot)))
            feet.append((dist, foot))
            if dist < best[0]:
                best = (dist, foot)
        count = sum(1 for dist, _ in feet if dist <= best[0] * (1.0 + 1e-12))
        return best[1], best[0], count

    targets = []
    meta = []
    for factor in BASELINE_DISTANCE_FACTORS:
        dist = float(factor) * sqrt_t
        targets.append(np.asarray(corner) + dist * np.asarray(bisector))
        meta.append(("corner", factor, dist))
    edge_base = np.array([-0.4, -0.8])
    edge_normal = np.array([0.0, -1.0])
    for factor in BASELINE_DISTANCE_FACTORS:
        dist = float(factor) * sqrt_t
        targets.append(edge_base - dist * edge_normal)
        meta.append(("edge", factor, dist))
    targets = np.asarray(targets, dtype=np.float64)
    exact = closure_exact(leaves, targets, t_leaf)
    rows = []
    for m, (site, factor, dist) in enumerate(meta):
        foot, foot_dist, multiplicity = _closest(targets[m])
        _, tangent, inward = eb.boundary_frame(targets[m], foot, 1.0)
        jet = eb.jet_from_callable(rho, targets[m], tangent, inward)
        value = float(
            np.asarray(
                eb.fryklund_VL(targets[m], foot, 0.0, jet, t_leaf, sign=1.0)
            ).ravel()[0]
        )
        err = value - float(exact[m])
        rows.append(
            {
                "case": "L",
                "site": site,
                "levels": int(levels),
                "theta": float(theta),
                "h_leaf": h_leaf,
                "t_leaf": t_leaf,
                "site_index": int(multiplicity),
                "distance": dist,
                "distance_over_sqrt_t": factor,
                "r_frame": float(foot_dist),
                "kappa_b": 0.0,
                "closure_exact": float(exact[m]),
                "fryklund": value,
                "abs_err": float(abs(err)),
                "rel_err": float(abs(err) / max(abs(float(exact[m])), 1e-300)),
            }
        )
    return rows


def study_baseline(out: Path, levels_list=DEFAULT_LEVELS,
                   theta: float = ec.THETA_PRIMARY,
                   order: int = DEFAULT_PROXY_ORDER,
                   rho: Callable = eh.rho_gaussian) -> dict:
    """Study 4 of the plan's validation list: the asymptotic local part as a baseline.

    The jet is taken from the analytic density, not from the per-leaf proxy: the
    expansion is a statement about a smooth ``f``, and the proxy's interpolation
    error is orders of magnitude below the terms being tested.  The comparison
    value is ``closure_exact``, the untruncated closure.

    Two order fits are written.  ``order_fits`` fits the maximum over sites at
    each distance, which mixes sites as the level changes; ``order_fits_per_site``
    fits one site at one distance across the level counts, which is the estimator
    the expansion's stated order is about, and ``order_fits_per_site_summary``
    reduces those to a count, a median and a range per case and site kind.
    """
    fields = [
        "case", "site", "levels", "theta", "h_leaf", "t_leaf", "site_index",
        "distance", "distance_over_sqrt_t", "r_frame", "kappa_b",
        "closure_exact", "fryklund", "abs_err", "rel_err",
    ]
    csv_out = IncrementalCsv(out / "experiment_e_baseline_vs_exact.csv", fields)
    all_rows: list = []
    for levels in list(levels_list):
        star = eg.make_domain("S")
        tree = ec.build_quadtree(levels=int(levels))
        leaves = star.leaves(tree, rho, int(order))
        clock = time.perf_counter()
        rows = _star_baseline_rows(star, leaves, tree, levels, theta, rho)
        _log(f"baseline S L={levels}: {len(rows)} rows in "
             f"{time.perf_counter() - clock:.1f}s")
        for row in rows:
            csv_out.row(**row)
        all_rows.extend(rows)

        ldom = eg.make_domain("L")
        ltree = ec.build_quadtree(levels=int(levels))
        lleaves = ldom.leaves(ltree, rho, int(order))
        clock = time.perf_counter()
        lrows = _l_corner_baseline_rows(ldom, lleaves, ltree, levels, theta, rho)
        _log(f"baseline L L={levels}: {len(lrows)} rows in "
             f"{time.perf_counter() - clock:.1f}s")
        for row in lrows:
            csv_out.row(**row)
        all_rows.extend(lrows)
        _write_block(out, "baseline", {"study": "baseline", "rows": all_rows})
    csv_out.close()

    fits = {}
    for case, site in (("S", "curve"), ("L", "edge"), ("L", "corner")):
        for factor in BASELINE_DISTANCE_FACTORS:
            sel = [
                r for r in all_rows
                if r["case"] == case and r["site"] == site
                and abs(r["distance_over_sqrt_t"] - factor) < 1e-12
            ]
            if len(sel) < 2:
                continue
            key = f"{case}_{site}_c{factor:g}"
            grouped: dict = {}
            for r in sel:
                grouped.setdefault(r["h_leaf"], []).append(r["abs_err"])
            hs = sorted(grouped)
            errs = [max(grouped[h]) for h in hs]
            fits[key] = {
                "h_leaf": hs,
                "max_abs_err": errs,
                "order_in_h": _fit_order(hs, errs),
            }
    per_site: dict = {}
    for row in all_rows:
        key = (
            f"{row['case']}_{row['site']}_s{int(row['site_index'])}"
            f"_c{row['distance_over_sqrt_t']:g}"
        )
        per_site.setdefault(key, {"h_leaf": [], "abs_err": []})
        per_site[key]["h_leaf"].append(row["h_leaf"])
        per_site[key]["abs_err"].append(row["abs_err"])
    site_fits: dict = {}
    for key, data in per_site.items():
        pairs = sorted(zip(data["h_leaf"], data["abs_err"], strict=True))
        hs = [p[0] for p in pairs]
        errs = [p[1] for p in pairs]
        if len(hs) < 2:
            continue
        site_fits[key] = {
            "h_leaf": hs,
            "abs_err": errs,
            "order_in_h": _fit_order(hs, errs),
        }
    site_summary: dict = {}
    for case, site in (("S", "curve"), ("L", "edge"), ("L", "corner")):
        orders = [
            v["order_in_h"] for k, v in site_fits.items()
            if k.startswith(f"{case}_{site}_s")
            and np.isfinite(v["order_in_h"])
        ]
        if not orders:
            continue
        site_summary[f"{case}_{site}"] = {
            "n_pairs": len(orders),
            "median_order_in_h": float(np.median(orders)),
            "min_order_in_h": float(np.min(orders)),
            "max_order_in_h": float(np.max(orders)),
        }
    payload = {
        "study": "baseline",
        "rows": all_rows,
        "order_fits": fits,
        "order_fits_per_site": site_fits,
        "order_fits_per_site_summary": site_summary,
    }
    _write_block(out, "baseline", payload)
    return payload


# ---------------------------------------------------------------------------
# study: the manufactured boundary value problem on case S
# ---------------------------------------------------------------------------


def study_bvp(out: Path, levels_list=DEFAULT_LEVELS, theta: float = ec.THETA_PRIMARY,
              order: int = DEFAULT_PROXY_ORDER,
              boundary_nodes=(64, 128, 256, 512)) -> dict:
    """Study 3: ``-Delta u = rho`` in ``Omega``, ``u = u_exact`` on the curve.

    The volume potential is assembled once per level count at the union of the
    finest boundary-node set and the interior grid; the coarser node sets are
    subsets of the finest one because the parametrization is equispaced, so the
    boundary sweep costs nothing extra in volume-potential evaluations.
    """
    fields = [
        "levels", "theta", "h_leaf", "n_leaves", "n_boundary", "n_grid",
        "max_abs_err", "rms_err", "max_u_exact", "volume_seconds",
        "solve_seconds",
    ]
    csv_out = IncrementalCsv(out / "experiment_e_bvp.csv", fields)
    domain = eg.make_domain("S")
    nodes = sorted(int(n) for n in boundary_nodes)
    n_max = nodes[-1]
    curve_max = eb.star_curve(n_max)
    grid = interior_grid(domain)
    rows: list = []
    for levels in list(levels_list):
        tree = ec.build_quadtree(levels=int(levels))
        leaves = domain.leaves(tree, eb.rho_manufactured, int(order))
        targets = np.vstack([curve_max.points, grid])
        clock = time.perf_counter()
        result = eh.assemble(
            leaves, targets, theta=float(theta), tree=tree, levels=int(levels)
        )
        t_volume = time.perf_counter() - clock
        _log(f"bvp L={levels}: volume potential at {targets.shape[0]} targets "
             f"in {t_volume:.1f}s")
        v_boundary = result["total"].u[: curve_max.size]
        v_grid = result["total"].u[curve_max.size:]
        u_ref = eb.u_exact(grid)
        for n in nodes:
            stride = n_max // n
            if stride * n != n_max:
                continue
            curve = eb.star_curve(n)
            clock = time.perf_counter()
            data = eb.u_exact(curve.points) - v_boundary[::stride]
            mu = eb.solve_interior_dirichlet(curve, data)
            harmonic = eb.double_layer_eval(curve, mu, grid)
            t_solve = time.perf_counter() - clock
            err = v_grid + harmonic - u_ref
            row = {
                "levels": int(levels),
                "theta": float(theta),
                "h_leaf": tree.h(int(levels)),
                "n_leaves": len(leaves),
                "n_boundary": int(n),
                "n_grid": int(grid.shape[0]),
                "max_abs_err": float(np.max(np.abs(err))),
                "rms_err": float(np.sqrt(np.mean(err * err))),
                "max_u_exact": float(np.max(np.abs(u_ref))),
                "volume_seconds": t_volume,
                "solve_seconds": t_solve,
            }
            csv_out.row(**row)
            rows.append(row)
        _write_block(out, "bvp", {"study": "bvp", "rows": rows})
    csv_out.close()
    finest = max(int(n) for n in nodes)
    by_level = [r for r in rows if r["n_boundary"] == finest]
    payload = {
        "study": "bvp",
        "rows": rows,
        "u_exact": "sin(2 x) cosh(2 y) + exp(-2 (x^2 + y^2))",
        "rho": "-Laplacian u_exact = exp(-2 r^2) (8 - 16 r^2)",
        "grid_margin": BVP_MARGIN,
        "order_in_h": _fit_order(
            [r["h_leaf"] for r in by_level], [r["max_abs_err"] for r in by_level]
        ),
    }
    _write_block(out, "bvp", payload)
    return payload


# ---------------------------------------------------------------------------
# study: the optional box-code cross-check
# ---------------------------------------------------------------------------


def study_jumps(out: Path, levels_list=DEFAULT_LEVELS,
                theta: float = ec.THETA_PRIMARY, order: int = DEFAULT_PROXY_ORDER,
                theta2: float = ec.THETA_SECONDARY, theta2_levels: int = 4,
                cases=("L", "S"), jump_fractions=JUMP_FRACTIONS,
                with_theta2: bool = True, stem: str = "jumps") -> dict:
    """Interface jumps alone, at the three offsets and with the ``d log d`` model.

    Separated from the residual studies because it needs only the jump probes,
    which makes it cheap enough to rerun when the estimator changes.  Its rows
    supersede any jump rows written by the residual studies.

    ``stem`` names the artifacts and the JSON block.  The driver runs this twice:
    once as ``jumps`` at the plan's offsets, and once as ``jumps_offsets`` at ten
    times them.  The second run is the control that separates the scheme from the
    extrapolation estimator, so it must cover the same cases and the same level
    counts as the first, or the columns it certifies are not certified at every
    level.
    """
    blocks: list = []
    for case in list(cases):
        for levels in list(levels_list):
            block = run_geometry_case(
                case, levels, theta, order=order, with_jumps=True,
                jumps_only=True, jump_fractions=jump_fractions,
            )
            _emit_jumps(out, block, stem)
            blocks.append(block)
            _write_block(out, stem, {"study": stem, "cases": blocks})
    if with_theta2:
        block = run_geometry_case(
            "S", int(theta2_levels), float(theta2), order=order,
            with_jumps=True, jumps_only=True, jump_fractions=jump_fractions,
        )
        _emit_jumps(out, block, stem)
        blocks.append(block)
    payload = {
        "study": stem,
        "cases": blocks,
        "jump_fractions": list(jump_fractions),
    }
    _write_block(out, stem, payload)
    return payload


def study_fastpath(out: Path, levels: int = 4, theta: float = ec.THETA_PRIMARY,
                   order: int = DEFAULT_PROXY_ORDER) -> dict:
    """Check the tensor contraction of curved plane-wave coefficients against the flat route.

    The hierarchy's ``_curved_tensor_coeffs`` was added by this driver's author
    to make case S affordable; it must reproduce the flat route it replaces, on
    the same quadrature rule, to roundoff.  The comparison is run on a real cut
    star leaf and on the real shell grid, and on a deliberately small grid where
    the flat route is cheap enough to evaluate in full.
    """
    domain = eg.make_domain("S")
    tree = ec.build_quadtree(levels=int(levels))
    leaves = domain.leaves(tree, eh.rho_gaussian, int(order))
    cut = [lf for lf in leaves if any(p.kind == "curved" for p in lf.pieces)]
    rows = []
    for leaf in cut[:2]:
        piece = next(p for p in leaf.pieces if p.kind == "curved")
        for level, n_cap in ((int(levels) - 1, 21), (int(levels) - 1, 41)):
            t_c = ec.t_l(level, float(theta), tree.root.half)
            t_f = ec.t_l(level + 1, float(theta), tree.root.half)
            grid = ec.plane_wave_grid(tree.h(level), t_c, t_f)
            k1 = grid.k1
            if k1.size > n_cap:
                keep = np.linspace(0, k1.size - 1, n_cap).astype(np.int64)
                k1 = k1[keep]
            centre = np.asarray(leaf.box.center, dtype=np.float64)
            clock = time.perf_counter()
            fast = eh._curved_tensor_coeffs(piece, leaf.poly, k1, centre)
            t_fast = time.perf_counter() - clock
            n = k1.size
            kxf = np.repeat(k1, n)
            kyf = np.tile(k1, n)
            clock = time.perf_counter()
            flat = eh.plane_wave_coeffs(piece, leaf.poly, kxf, kyf)
            flat = (flat * np.exp(1j * (kxf * centre[0] + kyf * centre[1]))).reshape(
                n, n
            )
            t_flat = time.perf_counter() - clock
            scale = max(float(np.max(np.abs(flat))), 1e-300)
            rows.append(
                {
                    "leaf_center": [float(centre[0]), float(centre[1])],
                    "level": int(level),
                    "n_per_coordinate": int(n),
                    "max_abs_diff": float(np.max(np.abs(fast - flat))),
                    "max_rel_diff": float(np.max(np.abs(fast - flat)) / scale),
                    "coeff_scale": scale,
                    "seconds_tensor": t_fast,
                    "seconds_flat": t_flat,
                }
            )
            _log(
                f"fastpath n={n}: rel {rows[-1]['max_rel_diff']:.2e}, "
                f"{t_fast:.2f}s vs {t_flat:.2f}s"
            )
    payload = {"study": "fastpath", "rows": rows}
    _write_block(out, "fastpath", payload)
    return payload


def study_volumential(out: Path, enabled: bool = False, reason: str = "") -> dict:
    """Study 5: the optional cross-check, which records itself as skipped by default."""
    payload = {"study": "volumential", "ran": False, "reason": reason}
    if enabled:
        try:
            import pyopencl as cl  # noqa: F401
            from volumential.volume_fmm import drive_volume_fmm  # noqa: F401
            payload["reason"] = (
                "imports succeed; the cross-check itself was not implemented "
                "within the one-hour cap"
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            payload["reason"] = f"unavailable: {type(exc).__name__}: {exc}"
    _write_block(out, "volumential", payload)
    return payload


# ---------------------------------------------------------------------------
# plots (a second interpreter may run these from the CSVs alone)
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list:
    """Read a CSV into a list of dicts, or return an empty list when it is absent."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value, default=float("nan")) -> float:
    """Parse a CSV cell as a float, tolerating empty cells."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_plots(out: Path) -> list:
    """Render the study PNGs from the CSVs; returns the file names written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list = []

    rows = _read_csv(out / "experiment_e_residual.csv")
    if rows:
        cases = sorted({r["case"] for r in rows})
        fig, axes = plt.subplots(
            1, len(cases), figsize=(4.6 * len(cases), 4.0), squeeze=False
        )
        for ax, case in zip(axes[0], cases, strict=True):
            sel = [
                r for r in rows
                if r["case"] == case and r["family"] == "all"
                and abs(_as_float(r["theta"]) - 8.0) < 1e-9
            ]
            stages = sorted(
                {r["stage"] for r in sel},
                key=lambda s: (s == "total", s == "closure", s),
            )
            for stage in stages:
                pts = sorted(
                    (int(_as_float(r["levels"])), _as_float(r["max_abs"]))
                    for r in sel if r["stage"] == stage
                )
                if not pts:
                    continue
                ax.semilogy(
                    [p[0] for p in pts], [max(p[1], 1e-20) for p in pts],
                    marker="o" if stage == "total" else ".",
                    lw=2.0 if stage == "total" else 1.0,
                    label=stage,
                )
            ax.set_title(f"case {case}, Theta = 8")
            ax.set_xlabel("levels L")
            ax.set_ylabel("max |residual|")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        name = "experiment_e_residual_vs_levels.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    rows = _read_csv(out / "experiment_e_jumps.csv")
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        for ax, key, title in (
            (axes[0], "jump_u_extrap", "jump in u (quadratic model)"),
            (axes[1], "jump_grad_log", "jump in grad u (d log d model)"),
        ):
            for case in sorted({r["case"] for r in rows}):
                for kind in sorted({r["kind"] for r in rows if r["case"] == case}):
                    sel = [
                        r for r in rows
                        if r["case"] == case and r["kind"] == kind
                        and abs(_as_float(r["theta"]) - 8.0) < 1e-9
                    ]
                    grouped: dict = {}
                    for r in sel:
                        grouped.setdefault(
                            int(_as_float(r["levels"])), []
                        ).append(_as_float(r[key]))
                    if not grouped:
                        continue
                    levels = sorted(grouped)
                    ax.semilogy(
                        levels, [max(max(grouped[lv]), 1e-20) for lv in levels],
                        marker="o", label=f"{case}/{kind}",
                    )
            ax.set_xlabel("levels L")
            ax.set_ylabel(f"max one-sided {title}")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        name = "experiment_e_interface_jumps.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    rows = _read_csv(out / "experiment_e_bvp.csv")
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        finest = max(int(_as_float(r["n_boundary"])) for r in rows)
        sel = [r for r in rows if int(_as_float(r["n_boundary"])) == finest]
        pts = sorted(
            (_as_float(r["h_leaf"]), _as_float(r["max_abs_err"])) for r in sel
        )
        if pts:
            axes[0].loglog(
                [p[0] for p in pts], [max(p[1], 1e-20) for p in pts], marker="o"
            )
        axes[0].set_xlabel("h_L")
        axes[0].set_ylabel("max |u - u_exact|")
        axes[0].set_title(f"BVP error vs h_L ({finest} boundary nodes)")
        axes[0].grid(True, which="both", alpha=0.3)
        for levels in sorted({int(_as_float(r["levels"])) for r in rows}):
            pts = sorted(
                (int(_as_float(r["n_boundary"])), _as_float(r["max_abs_err"]))
                for r in rows if int(_as_float(r["levels"])) == levels
            )
            axes[1].loglog(
                [p[0] for p in pts], [max(p[1], 1e-20) for p in pts],
                marker="o", label=f"L = {levels}",
            )
        axes[1].set_xlabel("boundary nodes")
        axes[1].set_ylabel("max |u - u_exact|")
        axes[1].set_title("BVP error vs boundary discretization")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        name = "experiment_e_bvp_convergence.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    rows = _read_csv(out / "experiment_e_baseline_vs_exact.csv")
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        finest = max(int(_as_float(r["levels"])) for r in rows)
        for case, site in sorted({(r["case"], r["site"]) for r in rows}):
            pts = sorted(
                (_as_float(r["distance"]), _as_float(r["abs_err"]))
                for r in rows
                if r["case"] == case and r["site"] == site
                and int(_as_float(r["levels"])) == finest
            )
            if pts:
                axes[0].loglog(
                    [p[0] for p in pts], [max(p[1], 1e-30) for p in pts],
                    marker="o", ls="none", label=f"{case}/{site}",
                )
        axes[0].set_xlabel("distance to the boundary")
        axes[0].set_ylabel("|V_L(Lemma 4.5) - exact closure|")
        axes[0].set_title(f"baseline error vs distance (L = {finest})")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=7)
        for case, site in sorted({(r["case"], r["site"]) for r in rows}):
            grouped: dict = {}
            for r in rows:
                if r["case"] != case or r["site"] != site:
                    continue
                grouped.setdefault(_as_float(r["h_leaf"]), []).append(
                    _as_float(r["abs_err"])
                )
            hs = sorted(grouped)
            if len(hs) < 2:
                continue
            axes[1].loglog(
                hs, [max(max(grouped[h]), 1e-30) for h in hs],
                marker="o", label=f"{case}/{site}",
            )
        axes[1].set_xlabel("h_L")
        axes[1].set_ylabel("max |V_L - exact closure|")
        axes[1].set_title("baseline error vs h_L")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        name = "experiment_e_baseline_error.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(name)

    return written


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


STUDIES = (
    "fastpath", "b", "l", "s", "jumps", "jumps_offsets", "baseline", "bvp",
    "theta12", "volumential",
)
"""Study keys, in the order the plan lists them."""

_BLOCK_NAMES = {
    "fastpath": "fastpath",
    "b": "case_b",
    "l": "case_l",
    "s": "case_s",
    "jumps": "jumps",
    "jumps_offsets": "jumps_offsets",
    "baseline": "baseline",
    "bvp": "bvp",
    "theta12": "theta12",
    "volumential": "volumential",
}


def merge_summary(out: Path) -> dict:
    """Collect the per-study JSON blocks into ``experiment_e_summary.json``."""
    payload = {
        "experiment": "E: 2D windowed telescoping assembly on complex geometry",
        "environment": ec.environment_summary(),
        "studies": {},
    }
    for key, name in _BLOCK_NAMES.items():
        path = out / f"experiment_e_{name}.json"
        if path.exists():
            payload["studies"][key] = json.loads(path.read_text())
    (out / "experiment_e_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )
    _log(f"merged {len(payload['studies'])} studies into experiment_e_summary.json")
    return payload


def collect_subdirs(out: Path) -> dict:
    """Concatenate the CSVs and JSON blocks of per-study subdirectories into ``out``.

    Studies are run as independent processes, one output subdirectory each, so
    that no two of them append to the same file; this step joins them into the
    canonical flat set of artifacts.
    """
    subs = sorted(d for d in out.iterdir() if d.is_dir())
    names: set = set()
    for sub in subs:
        names |= {path.name for path in sub.glob("experiment_e_*.csv")}
    for name in sorted(names):
        header = None
        body: list = []
        for sub in subs:
            path = sub / name
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
            body.extend(line for line in lines[1:] if line.strip())
        if header is not None:
            (out / name).write_text("\n".join([header] + body) + "\n")
            _log(f"collected {name}: {len(body)} rows")
    for sub in subs:
        for path in sub.glob("experiment_e_*.json"):
            if path.name == "experiment_e_summary.json":
                continue
            (out / path.name).write_bytes(path.read_bytes())
    return merge_summary(out)


def main(argv=None) -> int:
    """Command-line entry point: run the requested studies, or merge, or plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--studies", default="fastpath,b,l,s,baseline,bvp,theta12",
        help=f"comma-separated subset of {','.join(STUDIES)}",
    )
    parser.add_argument("--levels", default="3,4,5", help="level counts of the sweep")
    parser.add_argument("--theta", type=float, default=ec.THETA_PRIMARY)
    parser.add_argument("--theta2", type=float, default=ec.THETA_SECONDARY)
    parser.add_argument("--theta2-levels", type=int, default=4)
    parser.add_argument("--order", type=int, default=DEFAULT_PROXY_ORDER)
    parser.add_argument(
        "--l-extra-levels", default="6",
        help="extra level counts for case L, run only under the time budget",
    )
    parser.add_argument("--l-budget-seconds", type=float, default=300.0)
    parser.add_argument("--boundary-nodes", default="64,128,256,512")
    parser.add_argument("--jump-fractions", default="1e-3,1e-4,1e-5",
                        help="jump probe offsets in units of h_L")
    parser.add_argument("--jump-offset-factor", type=float, default=10.0,
                        help="offset scaling of the jumps_offsets control study")
    parser.add_argument("--jump-cases", default="L,S")
    parser.add_argument("--no-jump-theta2", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip studies whose JSON block already exists")
    parser.add_argument("--merge", action="store_true", help="merge blocks and exit")
    parser.add_argument("--collect", action="store_true",
                        help="join per-study subdirectories into OUT and exit")
    parser.add_argument("--plots-only", action="store_true",
                        help="render the PNGs from the CSVs and exit")
    parser.add_argument("--volumential", action="store_true",
                        help="attempt the optional box-code cross-check")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        names = make_plots(out)
        _log(f"plots: {names}")
        return 0
    if args.collect:
        collect_subdirs(out)
        return 0
    if args.merge:
        merge_summary(out)
        return 0

    levels = tuple(int(v) for v in str(args.levels).split(",") if v.strip())
    extra = tuple(
        int(v) for v in str(args.l_extra_levels).split(",") if v.strip()
    )
    nodes = tuple(
        int(v) for v in str(args.boundary_nodes).split(",") if v.strip()
    )
    wanted = [s.strip() for s in str(args.studies).split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STUDIES]
    if unknown:
        raise SystemExit(f"unknown studies {unknown}; expected {STUDIES}")

    started = time.perf_counter()
    for key in wanted:
        block = out / f"experiment_e_{_BLOCK_NAMES[key]}.json"
        if args.resume and block.exists():
            _log(f"study {key}: already on disk, skipped")
            continue
        clock = time.perf_counter()
        _log(f"study {key}: start")
        if key == "fastpath":
            study_fastpath(out, args.theta2_levels, args.theta, args.order)
        elif key == "b":
            study_case_b(out, levels, args.theta, args.order)
        elif key == "l":
            study_case_l(
                out, levels, args.theta, args.order,
                extra_levels=extra, budget_seconds=args.l_budget_seconds,
            )
        elif key == "s":
            study_case_s(out, levels, args.theta, args.order)
        elif key == "jumps":
            study_jumps(
                out, levels, args.theta, args.order, args.theta2,
                args.theta2_levels,
                cases=tuple(
                    c.strip() for c in str(args.jump_cases).split(",") if c.strip()
                ),
                jump_fractions=tuple(
                    float(v) for v in str(args.jump_fractions).split(",") if v.strip()
                ),
                with_theta2=not args.no_jump_theta2,
            )
        elif key == "jumps_offsets":
            study_jumps(
                out, levels, args.theta, args.order, args.theta2,
                args.theta2_levels,
                cases=tuple(
                    c.strip() for c in str(args.jump_cases).split(",") if c.strip()
                ),
                jump_fractions=tuple(
                    float(args.jump_offset_factor) * float(v)
                    for v in str(args.jump_fractions).split(",") if v.strip()
                ),
                with_theta2=not args.no_jump_theta2,
                stem="jumps_offsets",
            )
        elif key == "baseline":
            study_baseline(out, levels, args.theta, args.order)
        elif key == "bvp":
            study_bvp(out, levels, args.theta, args.order, boundary_nodes=nodes)
        elif key == "theta12":
            study_theta12(out, args.theta2_levels, args.theta2, args.order)
        elif key == "volumential":
            study_volumential(
                out, enabled=bool(args.volumential),
                reason=(
                    "" if args.volumential else
                    "skipped: the plan caps this optional cross-check at one "
                    "hour of agent time. A CPU OpenCL device and a working "
                    "volumential import were reported on another host of the "
                    "pool, but no driver for a 2D Laplace volume potential on "
                    "this study's tree and density exists there; writing and "
                    "debugging one (tree build, near-field table parameters, "
                    "source-node convention) does not fit the cap, so the "
                    "cross-check was not run and no number is claimed for it."
                ),
            )
        _log(f"study {key}: done in {time.perf_counter() - clock:.1f}s")

    merge_summary(out)
    _log(f"all requested studies done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
