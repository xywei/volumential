#!/usr/bin/env python3
"""Emit 3D three-mechanism composition CSVs for Paper 1.

This is the 3D companion of ``adaptive_split_composition.py``.  It evaluates
3D Yukawa volume potentials on the deterministically graded 2:1 trees of the
``adaptive_timing_3d`` benchmark, so all three reuse mechanisms compose in one
run:

* symmetry reuse: every table is stored in orbit-reduced form;
* level reuse: the RKE path uses one canonical ``1/r`` base table plus
  canonical odd-power channel tables scaled across the mixed leaf levels;
* parameter reuse: one fixed channel family serves every screening parameter
  through online coefficients.

The direct path builds per-level, per-parameter fixed-``lambda`` 3D Yukawa
tables and evaluates the same potential.  The benchmark reports cross-level
List 1 diagnostics, path mismatches between the RKE and direct potentials, and
build/payload comparisons.  Smoke mode is sized for CI; full mode reproduces
the graded-tree cases of the 3D adaptive-timing benchmark.
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
    _leaf_diagnostics,
    _list1_diagnostics,
    _populated_source_levels,
    _validate_adaptive_diagnostics,
)
from adaptive_timing_3d import (  # noqa: E402
    _build_adaptive_geometry,
)
from rke_field_demo_3d import (  # noqa: E402
    _field_build_configs,
    _field_smooth_quad_order,
)
from split_parameter_sweep import (  # noqa: E402
    _capture_table_get_timings,
    _clear_sqlite_cache,
    _coords_host,
    _gaussian_source_host,
    _select_opencl_device,
    _summarize_table_get_timings,
)


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "q_order",
    "initial_nlevels",
    "adapt_steps",
    "parameter",
    "split_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
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
    "rke_base_table_payload_bytes",
    "rke_channel_table_count",
    "rke_channel_table_build_s",
    "rke_channel_table_payload_bytes",
    "rke_total_table_payload_bytes",
)

LEAF_DIAGNOSTIC_FIELDS = (
    "min_leaf_level",
    "max_leaf_level",
    "leaf_level_histogram_json",
    "max_adjacent_leaf_level_difference",
)

LIST1_DIAGNOSTIC_FIELDS = (
    "n_list1_interactions",
    "n_cross_level_list1_interactions",
    "cross_level_list1_fraction",
    "list1_source_target_level_pair_histogram_json",
)


SMOKE_CASES = ((2, 3, 1),)
FULL_CASES = ((3, 4, 2), (3, 4, 3))

SMOKE_PARAMETERS = (2.0,)
FULL_PARAMETERS = (2.0, 4.0, 8.0)

SMOKE_SPLIT_ORDERS = (2,)
FULL_SPLIT_ORDERS = (1, 2, 3)


def _get_laplace_3d_base_table(
    queue,
    cache_path: Path,
    q_order: int,
    *,
    build_config,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Laplace",
            q_order,
            force_recompute=True,
            queue=queue,
            build_config=build_config,
        )
    return table


def _get_yukawa_3d_table_with_timings(
    queue,
    cache_path: Path,
    q_order: int,
    lam: float,
    level: int,
    *,
    tree_root_extent: float,
    build_config,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=tree_root_extent, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=True,
            queue=queue,
            build_config=build_config,
            lam=float(lam),
        )
    timings = dict(table_manager.last_get_table_timings)
    compute = timings.get("compute") or {}
    build_s = float(timings.get("total_s", compute.get("total_s", 0.0)))
    payload_bytes = int(compute.get("payload_bytes", 0))
    return table, build_s, payload_bytes


def _build_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    lam: float,
    table,
    source_weights,
    source_values_host,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_smooth_quad_order=None,
):
    from functools import partial

    import pyopencl.array as cla
    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    out_kernel = YukawaKernel(3)
    kernel_kwargs = {out_kernel.yukawa_lambda_name: float(lam)}
    # The shared split evaluator emits complex intermediates; keep direct and
    # split paths on the same dtype.
    dtype = np.complex128

    source_vals = cla.to_device(
        queue, np.ascontiguousarray(source_values_host.astype(dtype))
    )
    weighted_sources = source_vals * source_weights.astype(dtype)

    expn_factory = DefaultExpansionFactory()
    local_expn_class = expn_factory.get_local_expansion_class(out_kernel)
    mpole_expn_class = expn_factory.get_multipole_expansion_class(out_kernel)

    tree_indep = FPNDTreeIndependentDataForWrangler(
        ctx,
        partial(mpole_expn_class, out_kernel),
        partial(local_expn_class, out_kernel),
        [out_kernel],
        exclude_self=True,
    )
    self_extra_kwargs = {}
    if traversal.tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            traversal.tree.ntargets, dtype=np.int32
        )

    wrangler = FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=dtype,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        kernel_extra_kwargs=kernel_kwargs,
        self_extra_kwargs=self_extra_kwargs,
        helmholtz_split=split,
        helmholtz_split_order=split_order,
        helmholtz_split_term_tables=split_term_tables,
        helmholtz_split_smooth_quad_order=split_smooth_quad_order,
    )

    return wrangler, weighted_sources, source_vals


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


def _validate_diagnostic_fields(name, diagnostics, expected_fields) -> None:
    expected = set(expected_fields)
    actual = set(diagnostics)
    if not expected.issubset(FIELDS):
        raise RuntimeError(f"{name} fields are missing from the CSV schema")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            f"{name} diagnostics do not match the CSV schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


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
    _validate_diagnostic_fields(
        "leaf", leaf_diagnostics, LEAF_DIAGNOSTIC_FIELDS
    )
    _validate_diagnostic_fields(
        "List 1", list1_diagnostics, LIST1_DIAGNOSTIC_FIELDS
    )
    _validate_adaptive_diagnostics(leaf_diagnostics, list1_diagnostics)
    source_levels = _populated_source_levels(queue, tree, traversal)

    fmm_order = max(8, 4 * q_order)
    weights_host = q_weights.get(queue)
    source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    tree_root_extent = float(tree.root_extent)
    h_max_leaf = tree_root_extent * 0.5 ** leaf_diagnostics["min_leaf_level"]
    high_accuracy = mode == "full"
    direct_build_config, rke_channel_build_config = _field_build_configs(
        q_order, high_accuracy=high_accuracy
    )

    direct_results = {}
    for parameter in parameters:
        direct_cache_path = cache_dir / (
            f"composition3d-direct-q{q_order}-l{initial_nlevels}-"
            f"a{adapt_steps}-lam{parameter:g}.sqlite"
        )
        _clear_sqlite_cache(direct_cache_path)

        direct_tables = []
        direct_build_s = 0.0
        direct_payload_bytes = 0
        for level in source_levels:
            table, build_s, payload_bytes = _get_yukawa_3d_table_with_timings(
                queue,
                direct_cache_path,
                q_order,
                parameter,
                level,
                tree_root_extent=tree_root_extent,
                build_config=direct_build_config,
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
            lam=float(parameter),
            table=direct_tables,
            source_weights=q_weights,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
        )
        direct_potential, direct_wall_s = _drive(
            queue, traversal, direct_wrangler, weighted_sources, source_vals
        )
        direct_results[parameter] = {
            "potential": direct_potential,
            "wall_s": direct_wall_s,
            "table_count": len(direct_tables),
            "build_s": direct_build_s,
            "payload_bytes": direct_payload_bytes,
        }

    rows = []
    for split_order in split_orders:
        smooth_quad_order = _field_smooth_quad_order(
            q_order, split_order, high_accuracy=high_accuracy
        )
        # One fixed channel family per retained order; built once, reused for
        # every parameter below.
        rke_cache_path = cache_dir / (
            f"composition3d-rke-q{q_order}-l{initial_nlevels}-a{adapt_steps}-"
            f"p{split_order}.sqlite"
        )
        _clear_sqlite_cache(rke_cache_path)

        with _capture_table_get_timings() as base_records:
            base_table = _get_laplace_3d_base_table(
                queue,
                rke_cache_path,
                q_order,
                build_config=rke_channel_build_config,
            )
        with _capture_table_get_timings() as channel_records:
            seed_wrangler, _, _ = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                lam=float(parameters[0]),
                table=base_table,
                source_weights=q_weights,
                source_values_host=source_values_host,
                split=True,
                split_order=split_order,
                split_smooth_quad_order=smooth_quad_order,
            )
        base_costs = _summarize_table_get_timings(base_records)
        channel_costs = _summarize_table_get_timings(channel_records)
        rke_base_table_build_s = float(base_costs["build_s"])
        rke_base_table_payload_bytes = int(
            base_costs["build_cache_payload_bytes"]
        )
        rke_channel_build_s = float(channel_costs["build_s"])
        rke_channel_payload_bytes = int(
            channel_costs["build_cache_payload_bytes"]
        )
        split_term_tables = dict(seed_wrangler.helmholtz_split_term_tables)

        for parameter in parameters:
            direct_result = direct_results[parameter]
            direct_potential = direct_result["potential"]

            rke_wrangler, weighted_sources, source_vals = _build_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                lam=float(parameter),
                table=base_table,
                source_weights=q_weights,
                source_values_host=source_values_host,
                split=True,
                split_order=split_order,
                split_term_tables=split_term_tables,
                split_smooth_quad_order=smooth_quad_order,
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
                        f"yukawa3d-q{q_order}-l{initial_nlevels}-"
                        f"a{adapt_steps}-lam{parameter:g}-p{split_order}"
                    ),
                    "mode": mode,
                    "kernel": "Yukawa",
                    "dim": 3,
                    "q_order": q_order,
                    "initial_nlevels": initial_nlevels,
                    "adapt_steps": adapt_steps,
                    "parameter": parameter,
                    "split_order": split_order,
                    "direct_regular_quad_order": (
                        direct_build_config.regular_quad_order
                    ),
                    "direct_radial_quad_order": (
                        direct_build_config.radial_quad_order
                    ),
                    "rke_channel_regular_quad_order": (
                        rke_channel_build_config.regular_quad_order
                    ),
                    "rke_channel_radial_quad_order": (
                        rke_channel_build_config.radial_quad_order
                    ),
                    "split_smooth_quad_order": (
                        "" if smooth_quad_order is None else smooth_quad_order
                    ),
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
                    "direct_wall_s": direct_result["wall_s"],
                    "rke_wall_s": rke_wall_s,
                    "direct_table_count": direct_result["table_count"],
                    "direct_table_build_s": direct_result["build_s"],
                    "direct_table_payload_bytes": (
                        direct_result["payload_bytes"]
                    ),
                    "rke_base_table_build_s": rke_base_table_build_s,
                    "rke_base_table_payload_bytes": (
                        rke_base_table_payload_bytes
                    ),
                    "rke_channel_table_count": len(split_term_tables),
                    "rke_channel_table_build_s": rke_channel_build_s,
                    "rke_channel_table_payload_bytes": (
                        rke_channel_payload_bytes
                    ),
                    "rke_total_table_payload_bytes": (
                        rke_base_table_payload_bytes
                        + rke_channel_payload_bytes
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
        help="retained channel orders (default: 2 smoke; 1,2,3 full)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/adaptive-split-composition-3d.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-split-composition-3d-cache"),
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
