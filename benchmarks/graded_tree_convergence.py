#!/usr/bin/env python3
"""Emit graded-tree manufactured-solution convergence CSVs for Paper 1 (E9).

Every committed continuum-accuracy study runs on uniform leaf grids; this
driver measures continuum convergence, error against degrees of freedom, on
GENUINELY graded 2:1 trees.  A tight off-center Gaussian source drives
source-adapted refinement (refine leaf boxes whose local interpolation-error
proxy ``|f(center)| * h**q_order`` is within ``--adapt-fraction`` of the
maximum, then close to 2:1 balance without refining same-level colleagues),
so the adaptive ladder produces graded trees whose grading is verified
in-driver and recorded per row (leaf-level histogram, 2:1 balance, cross-level
List 1 fractions).

The problem is the 3D Laplace free-space volume potential of the Gaussian at
``q_order = 3`` in full mode, evaluated through the canonical rescaled-table
path (the level-reuse mechanism) against the closed-form analytic potential
``mass * erf(sqrt(alpha) r) / (4 pi r)``; the Gaussian mass omitted outside
the box bounds the modeling gap and is gated far below the achievable errors.
Two refinement ladders share the same source, table, FMM order, and table
quadrature: a uniform ladder over ``--uniform-nlevels`` and an adaptive
ladder over ``--adapt-steps`` from ``--base-nlevels``.  Observed convergence
orders in DOF are recorded between consecutive rungs; each ladder's verdict
(two consecutive rungs in the asymptotic regime, or an explicit limitation
asking for a longer ladder) lands in the CSV and the JSON metadata sidecar,
together with the matched-error DOF advantage of the adaptive ladder.

Smoke mode is a small ``q_order = 2`` ladder for CI/local validation; full
mode is the manuscript ladder for a controlled remote host.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from adaptive_timing import (  # noqa: E402
    _leaf_diagnostics,
    _list1_diagnostics,
    _populated_source_levels,
    _select_opencl_device,
    _table_payload_bytes,
    _table_phase_seconds,
)
from adaptive_timing_3d import (  # noqa: E402
    _balance_closure,
    _build_wrangler,
)
from volumential.gaussian import (  # noqa: E402
    GaussianComponent,
    GaussianMixture,
    evaluate_gaussian_mixture,
    gaussian_mixture_tail_report,
    laplace3d_gaussian_potential,
    write_json_metadata,
)


FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "kernel_normalization",
    "table_path",
    "reference_kind",
    "refinement",
    "ladder_rung",
    "q_order",
    "base_nlevels",
    "adapt_steps",
    "adapt_fraction",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "source_alpha",
    "source_center_json",
    "root_extent",
    "n_active_boxes",
    "n_total_boxes",
    "n_targets",
    "min_leaf_level",
    "max_leaf_level",
    "leaf_level_histogram_json",
    "max_adjacent_leaf_level_difference",
    "n_list1_interactions",
    "n_cross_level_list1_interactions",
    "cross_level_list1_fraction",
    "list1_source_target_level_pair_histogram_json",
    "populated_source_levels_json",
    "grading_status",
    "source_omitted_abs_fraction",
    "quadrature_source_mass",
    "table_get_s",
    "table_build_s",
    "table_load_s",
    "table_payload_bytes",
    "mesh_build_s",
    "geometry_s",
    "fmm_wall_s",
    "error_norm_definition",
    "rel_l2_vs_analytic",
    "weighted_rel_l2_vs_analytic",
    "linf_vs_analytic",
    "observed_order_dof",
    "in_asymptotic_regime",
    "ladder_verdict",
)

ROOT_BOUNDS = (-0.5, 0.5)
KERNEL_SCALE = 1.0 / (4.0 * math.pi)

DEFAULT_SOURCE_ALPHA = 200.0
DEFAULT_SOURCE_CENTER = (0.11, 0.07, -0.05)
DEFAULT_ADAPT_FRACTION = 0.3

# The analytic reference is a full-space potential; the omitted Gaussian
# tail outside the box must sit far below any continuum error the ladder
# can reach, or the study measures the modeling gap instead.
OMITTED_MASS_GATE = 1.0e-10

# Two consecutive observed orders within this relative tolerance of each
# other count as "in the asymptotic regime".
ORDER_AGREEMENT_RTOL = 0.25

SMOKE_CONFIG = {
    "q_order": 2,
    "uniform_nlevels": (3, 4),
    "base_nlevels": 3,
    "adapt_steps": (0, 1, 2),
    "fmm_order": 8,
    "regular_quad_order": 8,
    "radial_quad_order": 25,
}

FULL_CONFIG = {
    "q_order": 3,
    "uniform_nlevels": (3, 4, 5),
    "base_nlevels": 3,
    "adapt_steps": (0, 1, 2, 3, 4, 5, 6),
    "fmm_order": 14,
    "regular_quad_order": 14,
    "radial_quad_order": 55,
}

ERROR_NORM_DEFINITION = (
    "rel_l2=norm(u-u_exact)/max(norm(u_exact),1e-300);"
    "weighted_rel_l2=sqrt(sum_i w_i*(u_i-u_exact_i)**2)/"
    "max(sqrt(sum_i w_i*u_exact_i**2),1e-300);"
    "linf=max_i abs(u_i-u_exact_i)"
)


# {{{ pure helpers (queue-free, unit-tested)

def _source_mixture(alpha: float, center: tuple[float, ...]) -> GaussianMixture:
    return GaussianMixture(
        name="graded-tree-convergence-source",
        components=(GaussianComponent(1.0, tuple(center), float(alpha)),),
    )


def _run_configuration_token(
    *,
    mode: str,
    base_nlevels: int,
    nlevels: int,
    adapt_steps: int,
    adapt_fraction: float,
    fmm_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    mixture: GaussianMixture,
) -> str:
    """Eight hex digits identifying the run configuration behind a rung.

    ``q_order``, the refinement and the rung are already in the case id,
    but two campaigns differing only in ``--source-alpha``,
    ``--source-center``, ``--adapt-fraction``, the FMM order or the table
    quadrature orders measure different problems under otherwise
    identical ids, so a collection keyed on the id merges or overwrites
    them.  The values enter through a sorted JSON dump, so the token is
    stable across runs and Python versions.
    """
    import hashlib

    payload = json.dumps(
        {
            "mode": str(mode),
            "base_nlevels": int(base_nlevels),
            "nlevels": int(nlevels),
            "adapt_steps": int(adapt_steps),
            # repr round-trips a float64, so two adapt fractions that
            # differ beyond six digits cannot share a token
            "adapt_fraction": repr(float(adapt_fraction)),
            "fmm_order": int(fmm_order),
            "regular_quad_order": int(regular_quad_order),
            "radial_quad_order": int(radial_quad_order),
            "source": mixture.as_metadata(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.blake2s(
        payload.encode("utf-8"), digest_size=4
    ).hexdigest()


def _refinement_eta(
    values: np.ndarray,
    levels: np.ndarray,
    root_extent: float,
    q_order: int,
) -> np.ndarray:
    """Local interpolation-error proxy ``|f(center)| * h**q_order``."""
    h = float(root_extent) * np.exp2(-np.asarray(levels, dtype=np.float64))
    return np.abs(np.asarray(values, dtype=np.float64)) * h**int(q_order)


def _observed_order(
    error_prev: float,
    error: float,
    dof_prev: int,
    dof: int,
    dim: int,
) -> float | str:
    """Observed order p with ``error ~ DOF**(-p/dim)`` between two rungs."""
    if not (
        math.isfinite(error_prev) and error_prev > 0.0
        and math.isfinite(error) and error > 0.0
        and dof > dof_prev > 0
    ):
        return ""
    return float(
        dim * math.log(error_prev / error) / math.log(dof / dof_prev)
    )


def _ladder_summary(
    errors: list[float],
    dofs: list[int],
    dim: int,
    *,
    agreement_rtol: float = ORDER_AGREEMENT_RTOL,
) -> dict[str, Any]:
    """Observed orders between consecutive rungs plus the asymptotic-regime
    verdict the manuscript slot requires: either two consecutive rungs agree
    in observed order, or the output states the limitation explicitly."""
    if len(errors) != len(dofs):
        raise ValueError("errors and dofs must have equal length")
    orders: list[float | str] = [""]
    for i in range(1, len(errors)):
        orders.append(
            _observed_order(errors[i - 1], errors[i], dofs[i - 1], dofs[i], dim)
        )
    numeric = [order for order in orders if isinstance(order, float)]
    in_asymptotic_regime = False
    if len(numeric) >= 2:
        last, previous = numeric[-1], numeric[-2]
        if last > 0.0 and previous > 0.0:
            in_asymptotic_regime = abs(last - previous) <= (
                agreement_rtol * max(abs(last), abs(previous))
            )
    if in_asymptotic_regime:
        verdict = (
            "asymptotic: last two observed orders "
            f"{numeric[-2]:.2f} and {numeric[-1]:.2f} agree within "
            f"{agreement_rtol:.0%}"
        )
    elif len(numeric) >= 2:
        verdict = (
            "limitation: last two observed orders "
            f"{numeric[-2]:.2f} and {numeric[-1]:.2f} disagree beyond "
            f"{agreement_rtol:.0%}; extend the ladder"
        )
    else:
        verdict = (
            "limitation: fewer than three rungs with computable orders; "
            "extend the ladder"
        )
    return {
        "observed_orders": orders,
        "in_asymptotic_regime": in_asymptotic_regime,
        "verdict": verdict,
    }


def _interpolate_dof_at_error(
    curve: list[tuple[float, int]], target_error: float
) -> float | str:
    """Log-interpolate DOF at ``target_error`` on an error-vs-DOF curve of
    ``(error, dof)`` rungs with increasing DOF; "" if not bracketed."""
    for (error_hi, dof_lo), (error_lo, dof_hi) in zip(
        curve[:-1], curve[1:], strict=False
    ):
        if not error_lo <= target_error <= error_hi:
            continue
        if error_lo == error_hi:
            # A plateau at exactly the target -- an FMM or table-error
            # floor, say.  The cheaper rung already achieves the target,
            # so reporting the expensive one would move the matched-error
            # DOF advantage by the whole jump between the two rungs.
            return float(min(dof_lo, dof_hi))
        fraction = (
            math.log(error_hi / target_error)
            / math.log(error_hi / error_lo)
        )
        return float(
            math.exp(math.log(dof_lo) + fraction * math.log(dof_hi / dof_lo))
        )
    return ""


def _matched_error_dof_advantage(
    uniform_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
) -> float | str:
    """DOF ratio (uniform / adaptive) at a matched error, log-interpolating
    whichever curve brackets the other's finest error: first the uniform
    curve at the finest adaptive error, otherwise the adaptive curve at the
    finest uniform error; "" if neither error is bracketed."""
    if not uniform_rows or not adaptive_rows:
        return ""

    def curve(rows):
        return [
            (float(row["weighted_rel_l2_vs_analytic"]), int(row["n_targets"]))
            for row in rows
        ]

    uniform_curve, adaptive_curve = curve(uniform_rows), curve(adaptive_rows)
    uniform_dof = _interpolate_dof_at_error(
        uniform_curve, adaptive_curve[-1][0]
    )
    if isinstance(uniform_dof, float):
        return uniform_dof / adaptive_curve[-1][1]
    adaptive_dof = _interpolate_dof_at_error(
        adaptive_curve, uniform_curve[-1][0]
    )
    if isinstance(adaptive_dof, float):
        return uniform_curve[-1][1] / adaptive_dof
    return ""


class _BenchmarkGateError(RuntimeError):
    """A post-run gate failure that carries the rows it was measured on.

    The gates of :func:`_validate_rows` run on complete rows, so a failure
    means every expensive solve of the ladder already happened and only
    the verdict on them is negative -- exactly when the diagnostics are
    needed to investigate the failure.  :func:`main` writes them out
    before re-raising, the same idiom ``split_parameter_sweep`` uses.
    """

    def __init__(self, message: str, rows: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.rows = rows


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    """Hard gates: adaptive rungs must be genuinely graded, references must
    dominate the modeling gap, and each ladder must actually converge."""
    by_ladder: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ladder.setdefault(row["refinement"], []).append(row)
        error = float(row["weighted_rel_l2_vs_analytic"])
        if not (math.isfinite(error) and error > 0.0):
            raise RuntimeError(
                f"non-finite or non-positive error for {row['case_id']}"
            )
        if float(row["source_omitted_abs_fraction"]) > OMITTED_MASS_GATE:
            raise RuntimeError(
                "omitted Gaussian mass fraction "
                f"{row['source_omitted_abs_fraction']:.3e} exceeds the "
                f"{OMITTED_MASS_GATE:.0e} modeling-gap gate for "
                f"{row['case_id']}"
            )
        if int(row["max_adjacent_leaf_level_difference"]) > 1:
            raise RuntimeError(
                f"tree is not 2:1 balanced for {row['case_id']}"
            )
        adaptive = row["refinement"] == "adaptive"
        graded = int(row["min_leaf_level"]) < int(row["max_leaf_level"])
        if adaptive and int(row["adapt_steps"]) >= 1:
            if not graded:
                raise RuntimeError(
                    "adaptive refinement produced a uniform leaf level for "
                    f"{row['case_id']}: the graded-tree study requires "
                    "genuine grading"
                )
            if int(row["n_cross_level_list1_interactions"]) <= 0:
                raise RuntimeError(
                    "adaptive rung has no cross-level List 1 work for "
                    f"{row['case_id']}"
                )
            if row["grading_status"] != "graded":
                raise RuntimeError(
                    f"inconsistent grading status for {row['case_id']}"
                )

    for refinement, ladder in by_ladder.items():
        if len(ladder) < 2:
            continue
        first = float(ladder[0]["weighted_rel_l2_vs_analytic"])
        last = float(ladder[-1]["weighted_rel_l2_vs_analytic"])
        if not last < first:
            raise RuntimeError(
                f"the {refinement} ladder did not converge: first rung "
                f"error {first:.3e}, last rung error {last:.3e}"
            )

# }}}


def _build_uniform_mesh(queue, q_order: int, nlevels: int):
    import volumential.meshgen as mg

    return mg.MeshGen3D(
        q_order, nlevels, ROOT_BOUNDS[0], ROOT_BOUNDS[1], queue=queue
    )  # pyright: ignore[reportArgumentType]


def _refine_adaptively(
    mesh,
    mixture: GaussianMixture,
    q_order: int,
    adapt_steps: int,
    adapt_fraction: float,
) -> None:
    """Source-adapted refinement with per-step 2:1 balance closure.

    MeshGen's public update path closes same-level colleagues, which turns
    compact cases uniform; this uses ``refine_and_coarsen_tree_of_boxes``
    plus the colleague-preserving balance closure so grading survives.
    """
    from boxtree import refine_and_coarsen_tree_of_boxes

    for _ in range(adapt_steps):
        tree_of_boxes = mesh.boxtree._tree
        leaf_boxes = np.asarray(tree_of_boxes.leaf_boxes)
        levels = np.asarray(tree_of_boxes.box_levels)[leaf_boxes]
        centers = np.asarray(tree_of_boxes.box_centers)[:, leaf_boxes].T
        values = evaluate_gaussian_mixture(mixture, centers)
        eta = _refinement_eta(
            values, levels, float(tree_of_boxes.root_extent), q_order
        )
        refine = eta >= adapt_fraction * float(np.max(eta))
        if not np.any(refine):
            raise RuntimeError(
                "adaptive refinement selected no leaf boxes; the source "
                "indicator is degenerate"
            )
        refine_flags = np.zeros(tree_of_boxes.nboxes, dtype=bool)
        refine_flags[leaf_boxes[refine]] = True
        mesh.boxtree._tree = refine_and_coarsen_tree_of_boxes(
            tree_of_boxes, refine_flags=refine_flags
        )
        mesh.boxtree._tree = _balance_closure(mesh.boxtree._tree)


def _build_geometry(ctx, queue, q_order: int, mesh):
    import volumential.meshgen as mg

    bbox = np.array([list(ROOT_BOUNDS)] * 3, dtype=np.float64)
    return mg.build_geometry_info(ctx, queue, 3, q_order, mesh, bbox=bbox)


def _get_table(
    queue,
    cache_path: Path,
    q_order: int,
    *,
    regular_quad_order: int,
    radial_quad_order: int,
    force_recompute: bool,
):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=DuffyBuildConfig(
                radial_rule="tanh-sinh-fast",
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
            ),
        )
        timings = table_manager.last_get_table_timings
    return table, timings


def _run_fmm(
    ctx,
    queue,
    traversal,
    table,
    *,
    q_order: int,
    fmm_order: int,
    q_weights,
    source_host: np.ndarray,
):
    import pyopencl.array as cla

    from volumential.volume_fmm import drive_volume_fmm

    source_vals = cla.to_device(
        queue, np.ascontiguousarray(source_host.astype(np.float64))
    )
    wrangler = _build_wrangler(ctx, queue, traversal, table, q_order, fmm_order)
    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    queue.finish()
    return potential.get(queue), time.perf_counter() - start


def _run_rung(
    ctx,
    queue,
    *,
    mode: str,
    refinement: str,
    ladder_rung: int,
    q_order: int,
    base_nlevels: int,
    nlevels: int,
    adapt_steps: int,
    adapt_fraction: float,
    fmm_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    mixture: GaussianMixture,
    tail: dict[str, Any],
    table,
    table_timings,
) -> dict[str, Any]:
    mesh_start = time.perf_counter()
    mesh = _build_uniform_mesh(queue, q_order, nlevels)
    if refinement == "adaptive" and adapt_steps:
        _refine_adaptively(mesh, mixture, q_order, adapt_steps, adapt_fraction)
    mesh_build_s = time.perf_counter() - mesh_start

    geometry_start = time.perf_counter()
    q_points, q_weights, tree, traversal = _build_geometry(
        ctx, queue, q_order, mesh
    )
    geometry_s = time.perf_counter() - geometry_start

    leaf_diagnostics = _leaf_diagnostics(mesh)
    list1_diagnostics = _list1_diagnostics(queue, tree, traversal)
    source_levels = _populated_source_levels(queue, tree, traversal)
    graded = (
        leaf_diagnostics["min_leaf_level"] < leaf_diagnostics["max_leaf_level"]
    )

    coords = np.array([axis.get(queue) for axis in q_points])
    weights = q_weights.get(queue)
    source = evaluate_gaussian_mixture(mixture, coords)
    reference = laplace3d_gaussian_potential(
        mixture, coords, kernel_scale=KERNEL_SCALE
    )

    potential, fmm_wall_s = _run_fmm(
        ctx,
        queue,
        traversal,
        table,
        q_order=q_order,
        fmm_order=fmm_order,
        q_weights=q_weights,
        source_host=source,
    )
    error = potential - reference
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-300)
    weighted_reference_norm = max(
        float(np.sqrt(np.sum(weights * reference**2))), 1.0e-300
    )

    if refinement == "adaptive":
        case_tail = f"a{adapt_steps}"
    else:
        case_tail = f"l{nlevels}"
    configuration_token = _run_configuration_token(
        mode=mode,
        base_nlevels=base_nlevels,
        nlevels=nlevels,
        adapt_steps=adapt_steps,
        adapt_fraction=adapt_fraction,
        fmm_order=fmm_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        mixture=mixture,
    )
    row = {
        "case_id": (
            f"graded-laplace3d-q{q_order}-{refinement}-{case_tail}"
            f"-cfg{configuration_token}"
        ),
        "mode": mode,
        "problem": "gaussian-free-space-manufactured-solution",
        "dim": 3,
        "kernel": "Laplace",
        "kernel_normalization": "sumpy_global_scaling_1_over_4pi_r",
        "table_path": "canonical_rescaled",
        "reference_kind": (
            "analytic_full_space_gaussian_potential_with_omitted_tail_gate"
        ),
        "refinement": refinement,
        "ladder_rung": ladder_rung,
        "q_order": q_order,
        "base_nlevels": base_nlevels,
        "adapt_steps": adapt_steps if refinement == "adaptive" else "",
        "adapt_fraction": adapt_fraction if refinement == "adaptive" else "",
        "fmm_order": fmm_order,
        "regular_quad_order": regular_quad_order,
        "radial_quad_order": radial_quad_order,
        "source_alpha": mixture.components[0].alpha,
        "source_center_json": json.dumps(
            list(mixture.components[0].center), separators=(",", ":")
        ),
        "root_extent": float(tree.root_extent),
        "n_active_boxes": int(mesh.n_active_cells()),
        "n_total_boxes": int(mesh.n_cells()),
        "n_targets": int(tree.ntargets),
        **leaf_diagnostics,
        **list1_diagnostics,
        "populated_source_levels_json": json.dumps(
            source_levels, separators=(",", ":")
        ),
        "grading_status": "graded" if graded else "uniform",
        "source_omitted_abs_fraction": tail["omitted_abs_fraction"],
        "quadrature_source_mass": float(np.sum(weights * source)),
        "table_get_s": (
            table_timings.get("total_s", "") if table_timings else ""
        ),
        "table_build_s": _table_phase_seconds(table_timings, "compute"),
        "table_load_s": _table_phase_seconds(table_timings, "load"),
        "table_payload_bytes": _table_payload_bytes(table_timings),
        "mesh_build_s": mesh_build_s,
        "geometry_s": geometry_s,
        "fmm_wall_s": fmm_wall_s,
        "error_norm_definition": ERROR_NORM_DEFINITION,
        "rel_l2_vs_analytic": float(np.linalg.norm(error) / reference_norm),
        "weighted_rel_l2_vs_analytic": float(
            np.sqrt(np.sum(weights * error**2)) / weighted_reference_norm
        ),
        "linf_vs_analytic": float(np.max(np.abs(error))),
        "observed_order_dof": "",
        "in_asymptotic_regime": "",
        "ladder_verdict": "",
    }
    print(
        f"[{row['case_id']}] n_targets={row['n_targets']} "
        f"levels={row['leaf_level_histogram_json']} "
        f"weighted_rel_l2={row['weighted_rel_l2_vs_analytic']:.3e}",
        flush=True,
    )
    return row


def _annotate_ladder(rows: list[dict[str, Any]], dim: int) -> dict[str, Any]:
    errors = [float(row["weighted_rel_l2_vs_analytic"]) for row in rows]
    dofs = [int(row["n_targets"]) for row in rows]
    summary = _ladder_summary(errors, dofs, dim)
    for row, order in zip(rows, summary["observed_orders"], strict=True):
        row["observed_order_dof"] = order
        row["in_asymptotic_regime"] = int(summary["in_asymptotic_regime"])
        row["ladder_verdict"] = summary["verdict"]
    return summary


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    uniform_nlevels: list[int],
    base_nlevels: int,
    adapt_steps: list[int],
    adapt_fraction: float,
    fmm_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    source_alpha: float,
    source_center: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = _select_opencl_device(backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    mixture = _source_mixture(source_alpha, source_center)
    bbox = np.array([list(ROOT_BOUNDS)] * 3, dtype=np.float64)
    tail = gaussian_mixture_tail_report(mixture, bbox)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"graded-laplace3d-q{q_order}-r{regular_quad_order}"
        f"-rad{radial_quad_order}.sqlite"
    )
    if cache_path.exists():
        cache_path.unlink()
    table, table_timings = _get_table(
        queue,
        cache_path,
        q_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        force_recompute=True,
    )

    uniform_rows = []
    for rung, nlevels in enumerate(uniform_nlevels):
        uniform_rows.append(
            _run_rung(
                ctx,
                queue,
                mode=mode,
                refinement="uniform",
                ladder_rung=rung,
                q_order=q_order,
                base_nlevels=nlevels,
                nlevels=nlevels,
                adapt_steps=0,
                adapt_fraction=adapt_fraction,
                fmm_order=fmm_order,
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
                mixture=mixture,
                tail=tail,
                table=table,
                table_timings=table_timings,
            )
        )

    adaptive_rows = []
    for rung, steps in enumerate(adapt_steps):
        adaptive_rows.append(
            _run_rung(
                ctx,
                queue,
                mode=mode,
                refinement="adaptive",
                ladder_rung=rung,
                q_order=q_order,
                base_nlevels=base_nlevels,
                nlevels=base_nlevels,
                adapt_steps=steps,
                adapt_fraction=adapt_fraction,
                fmm_order=fmm_order,
                regular_quad_order=regular_quad_order,
                radial_quad_order=radial_quad_order,
                mixture=mixture,
                tail=tail,
                table=table,
                table_timings=table_timings,
            )
        )

    uniform_summary = _annotate_ladder(uniform_rows, 3)
    adaptive_summary = _annotate_ladder(adaptive_rows, 3)
    rows = uniform_rows + adaptive_rows
    # The gates run on complete rows and carry them out on the exception,
    # so main can write the CSV before reporting the failed verdict.
    try:
        _validate_rows(rows)
    except RuntimeError as exc:
        raise _BenchmarkGateError(str(exc), rows) from exc

    dof_advantage = _matched_error_dof_advantage(uniform_rows, adaptive_rows)
    metadata = {
        "case": "graded-tree-convergence",
        "mode": mode,
        "backend": backend,
        "problem": {
            "dim": 3,
            "kernel": "Laplace",
            "kernel_scale": KERNEL_SCALE,
            "source": mixture.as_metadata(),
            "tail_report": {
                key: tail[key]
                for key in (
                    "total_abs_component_mass",
                    "in_box_abs_component_mass",
                    "omitted_abs_component_mass",
                    "omitted_abs_fraction",
                )
            },
        },
        "configuration": {
            "q_order": q_order,
            "uniform_nlevels": list(uniform_nlevels),
            "base_nlevels": base_nlevels,
            "adapt_steps": list(adapt_steps),
            "adapt_fraction": adapt_fraction,
            "fmm_order": fmm_order,
            "regular_quad_order": regular_quad_order,
            "radial_quad_order": radial_quad_order,
        },
        "ladders": {
            "uniform": {
                "dofs": [int(row["n_targets"]) for row in uniform_rows],
                "weighted_rel_l2": [
                    float(row["weighted_rel_l2_vs_analytic"])
                    for row in uniform_rows
                ],
                **uniform_summary,
            },
            "adaptive": {
                "dofs": [int(row["n_targets"]) for row in adaptive_rows],
                "weighted_rel_l2": [
                    float(row["weighted_rel_l2_vs_analytic"])
                    for row in adaptive_rows
                ],
                "leaf_level_histograms": [
                    row["leaf_level_histogram_json"] for row in adaptive_rows
                ],
                **adaptive_summary,
            },
        },
        "matched_error_dof_advantage_uniform_over_adaptive": dof_advantage,
    }
    print(
        f"[graded-tree-convergence] uniform verdict: "
        f"{uniform_summary['verdict']}",
        flush=True,
    )
    print(
        f"[graded-tree-convergence] adaptive verdict: "
        f"{adaptive_summary['verdict']}",
        flush=True,
    )
    print(
        "[graded-tree-convergence] matched-error DOF advantage "
        f"(uniform/adaptive): {dof_advantage}",
        flush=True,
    )
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_ints(raw: str, *, minimum: int) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    if any(value < minimum for value in values):
        raise ValueError(f"values must be >= {minimum}")
    if len(set(values)) != len(values):
        raise ValueError("values must be unique")
    if values != sorted(values):
        raise ValueError("values must be increasing")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--q-order",
        type=int,
        default=None,
        help="source quadrature order (default: 2 smoke, 3 full)",
    )
    parser.add_argument(
        "--uniform-nlevels",
        default=None,
        help=(
            "comma-separated uniform-ladder mesh level counts "
            "(default: '3,4' smoke, '3,4,5' full)"
        ),
    )
    parser.add_argument(
        "--base-nlevels",
        type=int,
        default=None,
        help="adaptive-ladder base mesh level count (default: 3)",
    )
    parser.add_argument(
        "--adapt-steps",
        default=None,
        help=(
            "comma-separated adaptive-ladder refinement step counts "
            "(default: '0,1,2' smoke, '0,1,2,3,4,5,6' full)"
        ),
    )
    parser.add_argument(
        "--adapt-fraction",
        type=float,
        default=DEFAULT_ADAPT_FRACTION,
        help=(
            "refine leaves whose error proxy is within this fraction of "
            "the maximum (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--fmm-order",
        type=int,
        default=None,
        help="fixed FMM order along both ladders (default: 8 smoke, 14 full)",
    )
    parser.add_argument(
        "--regular-quad-order",
        type=int,
        default=None,
        help="table regular quadrature order (default: 8 smoke, 14 full)",
    )
    parser.add_argument(
        "--radial-quad-order",
        type=int,
        default=None,
        help="table radial quadrature order (default: 25 smoke, 55 full)",
    )
    parser.add_argument(
        "--source-alpha",
        type=float,
        default=DEFAULT_SOURCE_ALPHA,
        help="Gaussian sharpness alpha (default: %(default)s)",
    )
    parser.add_argument(
        "--source-center",
        default=",".join(str(value) for value in DEFAULT_SOURCE_CENTER),
        help="comma-separated off-center Gaussian center",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/graded-tree-convergence.csv"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="JSON metadata sidecar (default: <out stem>-metadata.json)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/graded-tree-convergence-cache"),
    )
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.mode == "smoke" else FULL_CONFIG
    q_order = args.q_order if args.q_order is not None else config["q_order"]
    try:
        uniform_nlevels = (
            _parse_csv_ints(args.uniform_nlevels, minimum=2)
            if args.uniform_nlevels is not None
            else list(config["uniform_nlevels"])
        )
        adapt_steps = (
            _parse_csv_ints(args.adapt_steps, minimum=0)
            if args.adapt_steps is not None
            else list(config["adapt_steps"])
        )
    except ValueError as exc:
        parser.error(str(exc))
    base_nlevels = (
        args.base_nlevels
        if args.base_nlevels is not None
        else config["base_nlevels"]
    )
    fmm_order = (
        args.fmm_order if args.fmm_order is not None else config["fmm_order"]
    )
    regular_quad_order = (
        args.regular_quad_order
        if args.regular_quad_order is not None
        else config["regular_quad_order"]
    )
    radial_quad_order = (
        args.radial_quad_order
        if args.radial_quad_order is not None
        else config["radial_quad_order"]
    )
    if not 0.0 < args.adapt_fraction <= 1.0:
        parser.error("--adapt-fraction must be in (0, 1]")
    source_center = tuple(
        float(part.strip())
        for part in args.source_center.split(",")
        if part.strip()
    )
    if len(source_center) != 3:
        parser.error("--source-center must have three coordinates")

    try:
        rows, metadata = run_benchmark(
            mode=args.mode,
            backend=args.backend,
            cache_dir=args.cache_dir,
            q_order=q_order,
            uniform_nlevels=uniform_nlevels,
            base_nlevels=base_nlevels,
            adapt_steps=adapt_steps,
            adapt_fraction=args.adapt_fraction,
            fmm_order=fmm_order,
            regular_quad_order=regular_quad_order,
            radial_quad_order=radial_quad_order,
            source_alpha=args.source_alpha,
            source_center=source_center,  # pyright: ignore[reportArgumentType]
        )
    except _BenchmarkGateError as exc:
        # The rows are measured; only the verdict on them failed.  Write
        # them before re-raising, so a multi-hour ladder keeps the very
        # diagnostics the failed gate has to be investigated with.
        write_csv(args.out, exc.rows)
        print(f"GATE-FAILED (CSV written to {args.out}): {exc}")
        raise
    write_csv(args.out, rows)
    metadata_out = args.metadata_out
    if metadata_out is None:
        metadata_out = args.out.parent / f"{args.out.stem}-metadata.json"
    write_json_metadata(metadata_out, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
