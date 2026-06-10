#!/usr/bin/env python3
"""Emit accuracy-preservation CSVs for Paper 1.

The benchmark uses a smooth manufactured 3D Poisson/Laplace volume-potential
problem and compares three evaluator paths on the same mesh:

* canonical: one level-0 near-field table with runtime scaling,
* per_level: direct per-level near-field tables,
* direct_p2p: direct point-particle diagnostic through the same volume-FMM driver.

The output reports exact-solution errors and pairwise path differences by order.
The canonical/per-level table comparison is the per-level row's difference from
``canonical_rescaled``; the direct P2P path is diagnostic and not a near-field
table reference.
Smoke mode is small enough for CI; full mode is intended for controlled paper
runs on ``ipa``.
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


SMOKE_CASES = (
    AccuracyCase(q_order=2, n_levels=2, fmm_order=8, regular_quad_order=8, radial_quad_order=25),
    AccuracyCase(q_order=3, n_levels=2, fmm_order=10, regular_quad_order=10, radial_quad_order=35),
)

FULL_CASES = (
    AccuracyCase(q_order=2, n_levels=3, fmm_order=10, regular_quad_order=10, radial_quad_order=35),
    AccuracyCase(q_order=3, n_levels=3, fmm_order=12, regular_quad_order=12, radial_quad_order=45),
    AccuracyCase(q_order=4, n_levels=3, fmm_order=14, regular_quad_order=14, radial_quad_order=55),
)

FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "path",
    "reference_path",
    "reference_method",
    "q_order",
    "n_levels",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "n_targets",
    "wall_s",
    "rel_l2_vs_exact",
    "linf_vs_exact",
    "rel_l2_vs_direct",
    "linf_vs_direct",
    "rel_l2_vs_canonical",
    "linf_vs_canonical",
)


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

    return source_expr, solution_expr, [x, y, z]


def _build_geometry(ctx, queue, case):
    import volumential.meshgen as mg

    dim = 3
    a = -0.5
    b = 0.5
    mesh = mg.MeshGen3D(case.q_order, case.n_levels, a, b, queue=queue)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        dim,
        case.q_order,
        mesh,
        bbox=np.array([[a, b]] * dim, dtype=np.float64),
    )
    return q_points, q_weights, tree, traversal


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
    wall_s,
):
    exact_diff = values - exact
    direct_diff = values - direct
    canonical_diff = values - canonical
    exact_norm = max(float(np.linalg.norm(exact)), 1.0e-300)
    direct_norm = max(float(np.linalg.norm(direct)), 1.0e-300)
    canonical_norm = max(float(np.linalg.norm(canonical)), 1.0e-300)
    return {
        "case_id": f"poisson3d-q{case.q_order}-l{case.n_levels}",
        "mode": mode,
        "problem": "smooth-gaussian-poisson3d",
        "dim": 3,
        "kernel": "Laplace",
        "path": path,
        "reference_path": reference_path,
        "reference_method": reference_method,
        "q_order": case.q_order,
        "n_levels": case.n_levels,
        "fmm_order": case.fmm_order,
        "regular_quad_order": case.regular_quad_order,
        "radial_quad_order": case.radial_quad_order,
        "n_targets": values.size,
        "wall_s": wall_s,
        "rel_l2_vs_exact": float(np.linalg.norm(exact_diff) / exact_norm),
        "linf_vs_exact": float(np.max(np.abs(exact_diff))),
        "rel_l2_vs_direct": float(np.linalg.norm(direct_diff) / direct_norm),
        "linf_vs_direct": float(np.max(np.abs(direct_diff))),
        "rel_l2_vs_canonical": float(np.linalg.norm(canonical_diff) / canonical_norm),
        "linf_vs_canonical": float(np.max(np.abs(canonical_diff))),
    }


def run_benchmark(*, mode: str, cache_dir: Path):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    cases = SMOKE_CASES if mode == "smoke" else FULL_CASES

    rows: list[dict[str, Any]] = []
    for case in cases:
        q_points, q_weights, tree, traversal = _build_geometry(ctx, queue, case)
        source_expr, solution_expr, variables = _build_manufactured_problem(3)
        source_eval = Eval(3, source_expr, variables)
        solution_eval = Eval(3, solution_expr, variables)
        q_coords = np.array([coords.get() for coords in q_points])
        source_vals = cl.array.to_device(queue, source_eval(queue, q_coords))
        exact = solution_eval(queue, q_coords)
        source_strengths = source_vals * q_weights

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

        rows.append(
            _norm_row(
                mode=mode,
                case=case,
                path="canonical_rescaled",
                reference_path="exact",
                reference_method="manufactured analytic solution",
                values=canonical,
                exact=exact,
                direct=direct,
                canonical=canonical,
                wall_s=canonical_wall_s,
            )
        )
        rows.append(
            _norm_row(
                mode=mode,
                case=case,
                path="per_level_tables",
                reference_path="exact",
                reference_method="manufactured analytic solution",
                values=per_level,
                exact=exact,
                direct=direct,
                canonical=canonical,
                wall_s=per_level_wall_s,
            )
        )
        rows.append(
            _norm_row(
                mode=mode,
                case=case,
                path="direct_p2p",
                reference_path="exact",
                reference_method="manufactured analytic solution; P2P diagnostic path",
                values=direct,
                exact=exact,
                direct=direct,
                canonical=canonical,
                wall_s=direct_wall_s,
            )
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
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
    rows = run_benchmark(mode=args.mode, cache_dir=args.cache_dir)
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
