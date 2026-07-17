#!/usr/bin/env python3
"""Emit accuracy-preservation CSVs for Paper 1.

The benchmark uses a smooth manufactured 3D Poisson/Laplace volume-potential
problem and compares three evaluator paths on the same mesh:

* canonical: one level-0 near-field table with runtime scaling,
* per_level: direct per-level near-field tables,
* direct_p2p: direct point-particle diagnostic through the same volume-FMM driver.

The finite-box reference follows from Green's identity: the analytic full-space
Gaussian solution is corrected by an overresolved boundary integral. The
quadrature-weighted L2 norm is ``sqrt(sum_i w_i |v_i|**2)``. The output also
retains unweighted nodal relative L2 errors and pairwise path differences. The
canonical/per-level table comparison is the per-level row's difference from
``canonical_rescaled``; the direct P2P path is diagnostic and not a near-field
table reference.

Smoke mode is small enough for CI; full mode is intended for a controlled remote
paper run. ``--q-orders`` and ``--n-levels`` form a Cartesian product so either
q- or h-convergence can be measured while the FMM and table quadrature orders
are held fixed with their corresponding CLI options.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymbolic as pmbl
import pyopencl as cl

from volumential.tools import ScalarFieldExpressionEvaluation as Eval


@dataclass(frozen=True)
class AccuracyCase:
    q_order: int
    n_levels: int
    fmm_order: int
    regular_quad_order: int
    radial_quad_order: int


ROOT_BOUNDS = (-0.5, 0.5)
GAUSSIAN_COMPONENTS = (
    (1.0, 120.0, (-0.08, 0.06, -0.05)),
    (-0.65, 90.0, (0.09, -0.07, 0.08)),
)


SMOKE_CASES = (
    AccuracyCase(
        q_order=2,
        n_levels=2,
        fmm_order=8,
        regular_quad_order=8,
        radial_quad_order=25,
    ),
    AccuracyCase(
        q_order=3,
        n_levels=2,
        fmm_order=10,
        regular_quad_order=10,
        radial_quad_order=35,
    ),
)

FULL_CASES = (
    AccuracyCase(
        q_order=2,
        n_levels=3,
        fmm_order=14,
        regular_quad_order=14,
        radial_quad_order=55,
    ),
    AccuracyCase(
        q_order=3,
        n_levels=2,
        fmm_order=14,
        regular_quad_order=14,
        radial_quad_order=55,
    ),
    AccuracyCase(
        q_order=3,
        n_levels=3,
        fmm_order=14,
        regular_quad_order=14,
        radial_quad_order=55,
    ),
    AccuracyCase(
        q_order=3,
        n_levels=4,
        fmm_order=14,
        regular_quad_order=14,
        radial_quad_order=55,
    ),
    AccuracyCase(
        q_order=4,
        n_levels=3,
        fmm_order=14,
        regular_quad_order=14,
        radial_quad_order=55,
    ),
)

FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "kernel_normalization",
    "path",
    "reference_path",
    "reference_method",
    "weighted_l2_definition",
    "q_order",
    "n_levels",
    "h_max",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "reference_quad_order",
    "reference_check_quad_order",
    "n_targets",
    "wall_s",
    "reference_boundary_weighted_l2",
    "reference_boundary_linf",
    "reference_quad_delta_weighted_l2",
    "reference_quad_delta_linf",
    "rel_l2_vs_exact",
    "rms_vs_exact",
    "weighted_l2_vs_exact",
    "weighted_rel_l2_vs_exact",
    "linf_vs_exact",
    "h_observed_order_vs_exact",
    "q_error_ratio_vs_previous",
    "q_log_error_slope_vs_exact",
    "rel_l2_vs_direct",
    "rms_vs_direct",
    "weighted_l2_vs_direct",
    "weighted_rel_l2_vs_direct",
    "linf_vs_direct",
    "rel_l2_vs_canonical",
    "rms_vs_canonical",
    "weighted_l2_vs_canonical",
    "weighted_rel_l2_vs_canonical",
    "linf_vs_canonical",
)


def _build_manufactured_problem(dim):
    if dim != 3:
        raise ValueError("the manufactured problem is 3D")

    variables = [pmbl.var(name) for name in ("x", "y", "z")]
    expp = pmbl.var("exp")

    solution_expr = 0
    source_expr = 0
    for amplitude, alpha, center in GAUSSIAN_COMPONENTS:
        radius_sq = sum(
            (variable - center_i) ** 2
            for variable, center_i in zip(variables, center, strict=True)
        )
        gaussian = amplitude * expp(-alpha * radius_sq)
        solution_expr += gaussian
        # LaplaceKernel(3) already has global scaling 1/(4*pi), so its Green's
        # function must be convolved with f=-Delta u, without another 1/(4*pi).
        source_expr += (2 * dim * alpha - 4 * alpha**2 * radius_sq) * gaussian

    return source_expr, solution_expr, variables


def _evaluate_manufactured_solution(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (npoints, 3)")

    values = np.zeros(points.shape[0], dtype=np.float64)
    gradients = np.zeros_like(points)
    for amplitude, alpha, center in GAUSSIAN_COMPONENTS:
        offsets = points - np.asarray(center, dtype=np.float64)
        component = amplitude * np.exp(-alpha * np.sum(offsets**2, axis=1))
        values += component
        gradients += (-2.0 * alpha * component)[:, np.newaxis] * offsets
    return values, gradients


def _finite_box_reference(targets, bbox, quad_order, *, chunk_size=256):
    """Evaluate the finite-box potential using analytic Green representation.

    For ``f=-Delta u`` and ``G=1/(4*pi*r)``, Green's second identity gives

    ``integral_Omega G f = u + integral_boundary (u d_n G - G d_n u)``.

    The six smooth face integrals are evaluated by overresolved tensor-product
    Gauss-Legendre quadrature. Volume quadrature targets are strictly interior,
    so the boundary integrands are nonsingular.
    """

    targets = np.asarray(targets, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (npoints, 3)")
    if bbox.shape != (3, 2):
        raise ValueError("bbox must have shape (3, 2)")
    if quad_order < 1:
        raise ValueError("quad_order must be positive")

    solution, _ = _evaluate_manufactured_solution(targets)
    boundary_correction = np.zeros(targets.shape[0], dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    kernel_scale = 1.0 / (4.0 * np.pi)

    for normal_axis in range(3):
        tangential_axes = [axis for axis in range(3) if axis != normal_axis]
        mapped_nodes = []
        mapped_weights = []
        for axis in tangential_axes:
            lo, hi = bbox[axis]
            mapped_nodes.append(0.5 * ((hi - lo) * nodes + hi + lo))
            mapped_weights.append(0.5 * (hi - lo) * weights)
        face_grid = np.meshgrid(*mapped_nodes, indexing="ij")
        weight_grid = np.meshgrid(*mapped_weights, indexing="ij")
        face_weights = np.prod(np.asarray(weight_grid), axis=0).ravel()

        for side, normal_sign in ((0, -1.0), (1, 1.0)):
            face_points = np.empty((quad_order**2, 3), dtype=np.float64)
            face_points[:, normal_axis] = bbox[normal_axis, side]
            for grid, axis in zip(face_grid, tangential_axes, strict=True):
                face_points[:, axis] = grid.ravel()

            face_solution, face_gradient = _evaluate_manufactured_solution(face_points)
            normal_derivative = normal_sign * face_gradient[:, normal_axis]

            for start in range(0, targets.shape[0], chunk_size):
                stop = min(start + chunk_size, targets.shape[0])
                displacement = (
                    targets[start:stop, np.newaxis, :] - face_points[np.newaxis, :, :]
                )
                radius_sq = np.sum(displacement**2, axis=2)
                green = kernel_scale / np.sqrt(radius_sq)
                green_normal_derivative = (
                    kernel_scale
                    * normal_sign
                    * displacement[:, :, normal_axis]
                    / radius_sq**1.5
                )
                integrand = (
                    face_solution[np.newaxis, :] * green_normal_derivative
                    - green * normal_derivative[np.newaxis, :]
                )
                boundary_correction[start:stop] += integrand @ face_weights

    return solution + boundary_correction, solution, boundary_correction


def _build_geometry(ctx, queue, case):
    import volumential.meshgen as mg

    dim = 3
    a, b = ROOT_BOUNDS
    mesh = mg.MeshGen3D(case.q_order, case.n_levels, a, b, queue=queue)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        case.q_order,
        mesh,
        bbox=np.array([[a, b]] * dim, dtype=np.float64),
    )
    h_max = float(np.max(mesh.get_cell_measures()) ** (1.0 / dim))
    return q_points, q_weights, tree, traversal, h_max


def _build_tables(queue, tree, case, cache_dir, path):
    from volumential.nearfield_potential_table import DuffyBuildConfig
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"accuracy-{path}-q{case.q_order}-l{case.n_levels}.sqlite"
    if cache_path.exists():
        cache_path.unlink()

    build_config = DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=case.regular_quad_order,
        radial_quad_order=case.radial_quad_order,
    )

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=1.0, queue=queue
    ) as tm:
        if path == "per_level":
            tables = []
            for level in range(tree.nlevels + 1):
                table, _ = tm.get_table(
                    3,
                    "Laplace",
                    case.q_order,
                    source_box_level=level,
                    queue=queue,
                    build_config=build_config,
                )
                tables.append(table)
            return tables

        table, _ = tm.get_table(
            3,
            "Laplace",
            case.q_order,
            queue=queue,
            build_config=build_config,
        )
        return table


def _build_wrangler(ctx, queue, traversal, table, case):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    kernel = LaplaceKernel(3)
    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(kernel)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, kernel),
        partial(local_expn_class, kernel),
        [kernel],
        exclude_self=True,
    )
    target_to_source = np.arange(traversal.tree.ntargets, dtype=np.int32)
    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: case.fmm_order,
        quad_order=case.q_order,
        self_extra_kwargs={"target_to_source": target_to_source},
    )


def _run_path(
    *,
    ctx,
    queue,
    traversal,
    table,
    case,
    source_strengths,
    source_vals,
    direct_evaluation,
):
    from volumential.volume_fmm import drive_volume_fmm

    wrangler = _build_wrangler(ctx, queue, traversal, table, case)
    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        source_strengths,
        source_vals,
        direct_evaluation=direct_evaluation,
        list1_only=False,
    )
    queue.finish()
    return potential.get(), time.perf_counter() - start


def _norm_row(
    *,
    mode,
    case,
    path,
    reference_path,
    reference_method,
    values,
    exact,
    direct,
    canonical,
    weights,
    h_max,
    reference_quad_order,
    reference_check_quad_order,
    reference_boundary,
    reference_quad_delta,
    wall_s,
):
    def metrics(diff, reference):
        reference_norm = max(float(np.linalg.norm(reference)), 1.0e-300)
        weighted_reference_norm = max(
            float(np.sqrt(np.sum(weights * np.abs(reference) ** 2))),
            1.0e-300,
        )
        weighted_error = float(np.sqrt(np.sum(weights * np.abs(diff) ** 2)))
        return {
            "rel_l2": float(np.linalg.norm(diff) / reference_norm),
            "rms": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
            "weighted_l2": weighted_error,
            "weighted_rel_l2": weighted_error / weighted_reference_norm,
            "linf": float(np.max(np.abs(diff))),
        }

    exact_metrics = metrics(values - exact, exact)
    direct_metrics = metrics(values - direct, direct)
    canonical_metrics = metrics(values - canonical, canonical)

    def weighted_l2(values):
        return float(np.sqrt(np.sum(weights * np.abs(values) ** 2)))

    return {
        "case_id": f"poisson3d-q{case.q_order}-l{case.n_levels}",
        "mode": mode,
        "problem": "smooth-gaussian-poisson3d",
        "dim": 3,
        "kernel": "Laplace",
        "kernel_normalization": "sumpy_global_scaling_1_over_4pi_r",
        "path": path,
        "reference_path": reference_path,
        "reference_method": reference_method,
        "weighted_l2_definition": "sqrt(sum_i quadrature_weight_i * abs(value_i)**2)",
        "q_order": case.q_order,
        "n_levels": case.n_levels,
        "h_max": h_max,
        "fmm_order": case.fmm_order,
        "regular_quad_order": case.regular_quad_order,
        "radial_quad_order": case.radial_quad_order,
        "reference_quad_order": reference_quad_order,
        "reference_check_quad_order": reference_check_quad_order,
        "n_targets": values.size,
        "wall_s": wall_s,
        "reference_boundary_weighted_l2": weighted_l2(reference_boundary),
        "reference_boundary_linf": float(np.max(np.abs(reference_boundary))),
        "reference_quad_delta_weighted_l2": weighted_l2(reference_quad_delta),
        "reference_quad_delta_linf": float(np.max(np.abs(reference_quad_delta))),
        "rel_l2_vs_exact": exact_metrics["rel_l2"],
        "rms_vs_exact": exact_metrics["rms"],
        "weighted_l2_vs_exact": exact_metrics["weighted_l2"],
        "weighted_rel_l2_vs_exact": exact_metrics["weighted_rel_l2"],
        "linf_vs_exact": exact_metrics["linf"],
        "h_observed_order_vs_exact": "",
        "q_error_ratio_vs_previous": "",
        "q_log_error_slope_vs_exact": "",
        "rel_l2_vs_direct": direct_metrics["rel_l2"],
        "rms_vs_direct": direct_metrics["rms"],
        "weighted_l2_vs_direct": direct_metrics["weighted_l2"],
        "weighted_rel_l2_vs_direct": direct_metrics["weighted_rel_l2"],
        "linf_vs_direct": direct_metrics["linf"],
        "rel_l2_vs_canonical": canonical_metrics["rel_l2"],
        "rms_vs_canonical": canonical_metrics["rms"],
        "weighted_l2_vs_canonical": canonical_metrics["weighted_l2"],
        "weighted_rel_l2_vs_canonical": canonical_metrics["weighted_rel_l2"],
        "linf_vs_canonical": canonical_metrics["linf"],
    }


def _add_convergence_rates(rows):
    h_groups = {}
    q_groups = {}
    for row in rows:
        fixed_orders = (
            row["fmm_order"],
            row["regular_quad_order"],
            row["radial_quad_order"],
        )
        h_groups.setdefault(
            (row["path"], row["q_order"], *fixed_orders), []
        ).append(row)
        q_groups.setdefault(
            (row["path"], row["n_levels"], *fixed_orders), []
        ).append(row)

    for group in h_groups.values():
        group.sort(key=lambda row: row["h_max"], reverse=True)
        for coarse, fine in zip(group, group[1:], strict=False):
            coarse_h = float(coarse["h_max"])
            fine_h = float(fine["h_max"])
            coarse_error = float(coarse["weighted_rel_l2_vs_exact"])
            fine_error = float(fine["weighted_rel_l2_vs_exact"])
            if coarse_h > fine_h and coarse_error > 0.0 and fine_error > 0.0:
                fine["h_observed_order_vs_exact"] = float(
                    np.log(coarse_error / fine_error) / np.log(coarse_h / fine_h)
                )

    for group in q_groups.values():
        group.sort(key=lambda row: row["q_order"])
        for previous, current in zip(group, group[1:], strict=False):
            previous_q = int(previous["q_order"])
            current_q = int(current["q_order"])
            previous_error = float(previous["weighted_rel_l2_vs_exact"])
            current_error = float(current["weighted_rel_l2_vs_exact"])
            if current_q > previous_q and previous_error > 0.0 and current_error > 0.0:
                ratio = current_error / previous_error
                current["q_error_ratio_vs_previous"] = float(ratio)
                current["q_log_error_slope_vs_exact"] = float(
                    -np.log(ratio) / (current_q - previous_q)
                )


def _make_case(
    q_order,
    n_levels,
    *,
    mode,
    fmm_order=None,
    regular_quad_order=None,
    radial_quad_order=None,
):
    if q_order < 1:
        raise ValueError("q_order must be positive")
    if n_levels < 1:
        raise ValueError("n_levels must be positive")

    if mode == "smoke":
        default_fmm_order = 2 * q_order + 4
        default_regular_order = 2 * q_order + 4
        default_radial_order = 10 * q_order + 5
    else:
        default_fmm_order = 2 * q_order + 6
        default_regular_order = 2 * q_order + 6
        default_radial_order = 10 * q_order + 15

    return AccuracyCase(
        q_order=q_order,
        n_levels=n_levels,
        fmm_order=default_fmm_order if fmm_order is None else fmm_order,
        regular_quad_order=(
            default_regular_order if regular_quad_order is None else regular_quad_order
        ),
        radial_quad_order=(
            default_radial_order if radial_quad_order is None else radial_quad_order
        ),
    )


def _select_cases(
    *,
    mode,
    q_orders=None,
    n_levels=None,
    fmm_order=None,
    regular_quad_order=None,
    radial_quad_order=None,
):
    preset = SMOKE_CASES if mode == "smoke" else FULL_CASES
    if q_orders is None and n_levels is None:
        selected = preset
    else:
        selected_q_orders = (
            sorted({case.q_order for case in preset})
            if q_orders is None
            else list(dict.fromkeys(q_orders))
        )
        selected_n_levels = (
            sorted({case.n_levels for case in preset})
            if n_levels is None
            else list(dict.fromkeys(n_levels))
        )
        selected = tuple(
            _make_case(q_order, level, mode=mode)
            for q_order in selected_q_orders
            for level in selected_n_levels
        )

    if any(
        order is not None
        for order in (fmm_order, regular_quad_order, radial_quad_order)
    ):
        selected = tuple(
            AccuracyCase(
                q_order=case.q_order,
                n_levels=case.n_levels,
                fmm_order=case.fmm_order if fmm_order is None else fmm_order,
                regular_quad_order=(
                    case.regular_quad_order
                    if regular_quad_order is None
                    else regular_quad_order
                ),
                radial_quad_order=(
                    case.radial_quad_order
                    if radial_quad_order is None
                    else radial_quad_order
                ),
            )
            for case in selected
        )
    return selected


def run_benchmark(
    *,
    mode: str,
    cache_dir: Path,
    cases=None,
    reference_quad_order=None,
):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    if cases is None:
        cases = SMOKE_CASES if mode == "smoke" else FULL_CASES
    if reference_quad_order is None:
        reference_quad_order = 24 if mode == "smoke" else 40
    if reference_quad_order < 2:
        raise ValueError("reference_quad_order must be at least 2")
    reference_check_quad_order = max(1, reference_quad_order // 2)
    bbox = np.array([ROOT_BOUNDS] * 3, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for case in cases:
        q_points, q_weights, tree, traversal, h_max = _build_geometry(ctx, queue, case)
        source_expr, _, variables = _build_manufactured_problem(3)
        source_eval = Eval(3, source_expr, variables)
        q_coords = np.array([coords.get() for coords in q_points])
        q_weights_host = q_weights.get(queue)
        source_vals = cl.array.to_device(queue, source_eval(queue, q_coords))
        source_strengths = source_vals * q_weights
        exact, _, reference_boundary = _finite_box_reference(
            q_coords.T,
            bbox,
            reference_quad_order,
        )
        reference_check, _, _ = _finite_box_reference(
            q_coords.T,
            bbox,
            reference_check_quad_order,
        )
        reference_quad_delta = exact - reference_check

        canonical_table = _build_tables(queue, tree, case, cache_dir, "canonical")
        per_level_tables = _build_tables(queue, tree, case, cache_dir, "per_level")

        canonical, canonical_wall_s = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            table=canonical_table,
            case=case,
            source_strengths=source_strengths,
            source_vals=source_vals,
            direct_evaluation=False,
        )
        per_level, per_level_wall_s = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            table=per_level_tables,
            case=case,
            source_strengths=source_strengths,
            source_vals=source_vals,
            direct_evaluation=False,
        )
        direct, direct_wall_s = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            table=per_level_tables,
            case=case,
            source_strengths=source_strengths,
            source_vals=source_vals,
            direct_evaluation=True,
        )

        common_norm_args = {
            "mode": mode,
            "case": case,
            "reference_path": "finite_box_green_identity",
            "reference_method": (
                "analytic Gaussian u plus overresolved Gauss-Legendre "
                "boundary correction"
            ),
            "exact": exact,
            "direct": direct,
            "canonical": canonical,
            "weights": q_weights_host,
            "h_max": h_max,
            "reference_quad_order": reference_quad_order,
            "reference_check_quad_order": reference_check_quad_order,
            "reference_boundary": reference_boundary,
            "reference_quad_delta": reference_quad_delta,
        }
        rows.append(
            _norm_row(
                path="canonical_rescaled",
                values=canonical,
                wall_s=canonical_wall_s,
                **common_norm_args,
            )
        )
        rows.append(
            _norm_row(
                path="per_level_tables",
                values=per_level,
                wall_s=per_level_wall_s,
                **common_norm_args,
            )
        )
        rows.append(
            _norm_row(
                path="direct_p2p",
                values=direct,
                wall_s=direct_wall_s,
                **common_norm_args,
            )
        )

    _add_convergence_rates(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--q-orders",
        type=_positive_int,
        nargs="+",
        help="q orders; forms a Cartesian product with --n-levels",
    )
    parser.add_argument(
        "--n-levels",
        type=_positive_int,
        nargs="+",
        help="uniform mesh level counts; forms a Cartesian product with --q-orders",
    )
    parser.add_argument(
        "--fmm-order",
        type=_positive_int,
        help="fixed FMM order for every selected h/q case",
    )
    parser.add_argument(
        "--regular-quad-order",
        type=_positive_int,
        help="fixed regular near-field quadrature order for every selected case",
    )
    parser.add_argument(
        "--radial-quad-order",
        type=_positive_int,
        help="fixed radial near-field quadrature order for every selected case",
    )
    parser.add_argument(
        "--reference-quad-order",
        type=_positive_int,
        help="Gauss-Legendre order per boundary-face axis (default: 24/40)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/accuracy-preservation.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/accuracy-cache"),
    )
    args = parser.parse_args()
    cases = _select_cases(
        mode=args.mode,
        q_orders=args.q_orders,
        n_levels=args.n_levels,
        fmm_order=args.fmm_order,
        regular_quad_order=args.regular_quad_order,
        radial_quad_order=args.radial_quad_order,
    )
    rows = run_benchmark(
        mode=args.mode,
        cache_dir=args.cache_dir,
        cases=cases,
        reference_quad_order=args.reference_quad_order,
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
