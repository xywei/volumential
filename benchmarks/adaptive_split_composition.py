#!/usr/bin/env python3
"""Emit three-mechanism composition CSVs for Paper 1.

The benchmark evaluates 2D Yukawa volume potentials on the deterministically
graded 2:1 trees of the adaptive-timing benchmark, so all three reuse
mechanisms compose in one run:

* symmetry reuse: every table is stored in orbit-reduced form;
* level reuse: the RKE path uses one canonical base table plus canonical
  channel tables scaled across the mixed leaf levels;
* parameter reuse: one fixed channel family serves every screening parameter
  through online coefficients.

The direct path builds per-level, per-parameter fixed-``lambda`` Yukawa tables
and evaluates the same potential.  The benchmark reports cross-level List 1
diagnostics, path mismatches between the RKE and direct potentials, and
build/payload comparisons.  Smoke mode is sized for CI; full mode reproduces
the graded-tree cases of the adaptive-timing benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    _build_adaptive_geometry,
    _leaf_diagnostics,
    _list1_diagnostics,
    _populated_source_levels,
    _validate_adaptive_diagnostics,
)
from split_parameter_sweep import (  # noqa: E402
    _build_config,
    _build_path,
    _gaussian_source_host,
    _coords_host,
    _get_laplace_2d_table,
    _select_opencl_device,
)


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "q_order",
    "initial_nlevels",
    "adapt_steps",
    "parameter",
    "split_order",
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
    "max_theta",
    "path_mismatch_norm_definition",
    "rke_vs_direct_weighted_rel_l2",
    "rke_vs_direct_linf",
    "direct_wall_s",
    "rke_wall_s",
    "direct_table_count",
    "direct_table_build_s",
    "direct_table_payload_bytes",
    "rke_base_table_build_s",
    "rke_channel_table_count",
    "rke_channel_table_build_s",
    "rke_channel_table_payload_bytes",
)


SMOKE_CASES = ((3, 2, 1),)
FULL_CASES = ((3, 4, 2), (4, 4, 2), (4, 4, 3))

SMOKE_PARAMETERS = (2.0,)
FULL_PARAMETERS = (2.0, 4.0, 8.0)

SMOKE_SPLIT_ORDERS = (2,)
FULL_SPLIT_ORDERS = (2, 3)


def _get_yukawa_2d_table_with_timings(
    queue,
    cache_path: Path,
    q_order: int,
    lam: float,
    level: int,
    *,
    tree_root_extent: float,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=tree_root_extent, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=True,
            queue=queue,
            build_config=_build_config(q_order),
            lam=float(lam),
        )
        timings = dict(table_manager.last_get_table_timings)
    compute = timings.get("compute") or {}
    build_s = float(compute.get("table_build_s", compute.get("total_s", 0.0)))
    payload_bytes = int(compute.get("payload_bytes", 0))
    return table, build_s, payload_bytes


def _drive(queue, traversal, wrangler, weighted_sources, source_vals):
    from volumential.volume_fmm import drive_volume_fmm

    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        weighted_sources,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    queue.finish()
    return potential.get(queue), time.perf_counter() - start


def run_case(
    ctx,
    queue,
    *,
    mode: str,
    cache_dir: Path,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    parameters,
    split_orders,
):
    mesh, q_points, q_weights, tree, traversal, _, _, _ = _build_adaptive_geometry(
        ctx, queue, q_order, initial_nlevels, adapt_steps
    )
    leaf_diagnostics = _leaf_diagnostics(mesh)
    list1_diagnostics = _list1_diagnostics(queue, tree, traversal)
    _validate_adaptive_diagnostics(leaf_diagnostics, list1_diagnostics)
    source_levels = _populated_source_levels(queue, tree, traversal)

    fmm_order = max(8, 4 * q_order)
    weights_host = q_weights.get(queue)
    source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    tree_root_extent = float(tree.root_extent)
    h_max_leaf = tree_root_extent * 0.5 ** leaf_diagnostics["min_leaf_level"]

    rows = []
    for split_order in split_orders:
        # One fixed channel family per retained order; built once, reused for
        # every parameter below.
        rke_cache_path = cache_dir / (
            f"composition-rke-q{q_order}-l{initial_nlevels}-a{adapt_steps}-"
            f"p{split_order}.sqlite"
        )
        if rke_cache_path.exists():
            rke_cache_path.unlink()

        channel_build_start = time.perf_counter()
        base_table = _get_laplace_2d_table(queue, rke_cache_path, q_order)
        seed_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=float(parameters[0]),
            table=base_table,
            source_weights=q_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
        )
        rke_channel_build_s = time.perf_counter() - channel_build_start
        split_term_tables = dict(seed_wrangler.helmholtz_split_term_tables)
        rke_channel_payload_bytes = 0
        for term_tables in split_term_tables.values():
            tables = (
                term_tables if isinstance(term_tables, (list, tuple))
                else [term_tables]
            )
            for term_table in tables:
                data = np.asarray(term_table.data)
                rke_channel_payload_bytes += int(data.nbytes)

        for parameter in parameters:
            direct_cache_path = cache_dir / (
                f"composition-direct-q{q_order}-l{initial_nlevels}-"
                f"a{adapt_steps}-lam{parameter:g}.sqlite"
            )
            if direct_cache_path.exists():
                direct_cache_path.unlink()

            direct_tables = []
            direct_build_s = 0.0
            direct_payload_bytes = 0
            for level in source_levels:
                table, build_s, payload_bytes = (
                    _get_yukawa_2d_table_with_timings(
                        queue,
                        direct_cache_path,
                        q_order,
                        parameter,
                        level,
                        tree_root_extent=tree_root_extent,
                    )
                )
                direct_tables.append(table)
                direct_build_s += build_s
                direct_payload_bytes += payload_bytes

            direct_wrangler, weighted_sources, source_vals = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel="Yukawa",
                parameter=float(parameter),
                table=direct_tables,
                source_weights=q_weights,
                q_points=q_points,
                source_values_host=source_values_host,
                split=False,
                split_order=split_order,
            )
            direct_potential, direct_wall_s = _drive(
                queue, traversal, direct_wrangler, weighted_sources, source_vals
            )

            rke_wrangler, weighted_sources, source_vals = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel="Yukawa",
                parameter=float(parameter),
                table=base_table,
                source_weights=q_weights,
                q_points=q_points,
                source_values_host=source_values_host,
                split=True,
                split_order=split_order,
                split_term_tables=split_term_tables,
            )
            rke_potential, rke_wall_s = _drive(
                queue, traversal, rke_wrangler, weighted_sources, source_vals
            )

            difference = rke_potential - direct_potential
            weighted_error = float(
                np.sqrt(np.sum(weights_host * np.abs(difference) ** 2))
            )
            weighted_reference = max(
                float(
                    np.sqrt(
                        np.sum(weights_host * np.abs(direct_potential) ** 2)
                    )
                ),
                1.0e-300,
            )

            rows.append(
                {
                    "case_id": (
                        f"yukawa2d-q{q_order}-l{initial_nlevels}-"
                        f"a{adapt_steps}-lam{parameter:g}-p{split_order}"
                    ),
                    "mode": mode,
                    "kernel": "Yukawa",
                    "q_order": q_order,
                    "initial_nlevels": initial_nlevels,
                    "adapt_steps": adapt_steps,
                    "parameter": parameter,
                    "split_order": split_order,
                    "n_targets": int(tree.ntargets),
                    **leaf_diagnostics,
                    **list1_diagnostics,
                    "populated_source_levels_json": json.dumps(
                        source_levels, separators=(",", ":")
                    ),
                    "max_theta": parameter * h_max_leaf,
                    "path_mismatch_norm_definition": (
                        "weighted_rel_l2=sqrt(sum_i quadrature_weight_i*"
                        "abs(rke_i-direct_i)**2)/max(sqrt(sum_i "
                        "quadrature_weight_i*abs(direct_i)**2),1e-300);"
                        "linf=max_i abs(rke_i-direct_i)"
                    ),
                    "rke_vs_direct_weighted_rel_l2": (
                        weighted_error / weighted_reference
                    ),
                    "rke_vs_direct_linf": float(np.max(np.abs(difference))),
                    "direct_wall_s": direct_wall_s,
                    "rke_wall_s": rke_wall_s,
                    "direct_table_count": len(direct_tables),
                    "direct_table_build_s": direct_build_s,
                    "direct_table_payload_bytes": direct_payload_bytes,
                    "rke_base_table_build_s": rke_channel_build_s,
                    "rke_channel_table_count": len(split_term_tables),
                    "rke_channel_table_build_s": rke_channel_build_s,
                    "rke_channel_table_payload_bytes": (
                        rke_channel_payload_bytes
                    ),
                }
            )
            print(
                f"[{rows[-1]['case_id']}] rke_vs_direct_weighted_rel_l2="
                f"{rows[-1]['rke_vs_direct_weighted_rel_l2']:.3e} "
                f"cross_level_fraction="
                f"{list1_diagnostics['cross_level_list1_fraction']:.3f}",
                flush=True,
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
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--parameters",
        type=float,
        nargs="+",
        help="Yukawa screening parameters (default: 2 smoke; 2,4,8 full)",
    )
    parser.add_argument(
        "--split-orders",
        type=int,
        nargs="+",
        help="retained channel orders (default: 2 smoke; 2,3 full)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/adaptive-split-composition.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-split-composition-cache"),
    )
    args = parser.parse_args()

    device = _select_opencl_device(cl, args.backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    cases = SMOKE_CASES if args.mode == "smoke" else FULL_CASES
    parameters = args.parameters
    if parameters is None:
        parameters = SMOKE_PARAMETERS if args.mode == "smoke" else FULL_PARAMETERS
    split_orders = args.split_orders
    if split_orders is None:
        split_orders = (
            SMOKE_SPLIT_ORDERS if args.mode == "smoke" else FULL_SPLIT_ORDERS
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_order, initial_nlevels, adapt_steps in cases:
        rows.extend(
            run_case(
                ctx,
                queue,
                mode=args.mode,
                cache_dir=args.cache_dir,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                parameters=parameters,
                split_orders=split_orders,
            )
        )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
