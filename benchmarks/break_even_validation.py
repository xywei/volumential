#!/usr/bin/env python3
"""Empirically validate the Yukawa RKE break-even repeat count.

Section 6 reports a linear amortization model for the repeated-solve tradeoff
between the direct per-parameter-per-level Yukawa table strategy and the RKE
channel strategy, with a modeled ``p = 3`` break-even of roughly 340 repeats
per parameter on the reference host.  This driver measures the crossing
directly:

* cold phase: build the direct fixed-``lambda`` tables and the RKE channel
  family once, both timed through the table-manager timing hooks.  The
  direct provisioning strategy is selectable (``--direct-provisioning``):
  ``eager`` builds every anticipated level per parameter (the historical
  Table 3 policy and the committed-artifact default), while ``lazy`` builds
  only the level the priced workload actually touches — on the uniform
  benchmark tree the leaf level owns all List 1 work, so the lazy baseline
  executes the provisioning policy the cost model previously carried only
  as an arithmetic projection (experiment E3);
* repeat phase: run interleaved warm level-``nlevels`` solves for both
  strategies, recording every solve wall time individually, until past the
  modeled break-even;
* accounting: accumulate the two cumulative cost curves
  ``C(n) = build_total + sum over parameters of the first n solve times``
  and report the measured crossing (first ``n`` with the RKE curve above the
  direct curve, plus a linearly interpolated fractional crossing), next to
  the linear-model prediction recomputed from this run's own measured means;
* operation counters (experiment E3): alongside the timing columns, the
  summary row reports the operation counts of the cost model, computed at
  the driver level from the executed configuration — symmetry-reduced
  entries per table, singular-quadrature node evaluations per entry (from
  the executed node builders and the actual build routing),
  special-function evaluations by function, near-field point pairs per
  solve, split-remainder series lengths, and table-apply FMA counts.  They
  are computed after the timed phases, so timings are unaffected.

The repeat phase alternates strategies within each repeat index so slow host
drift affects both curves equally.  This is a timing benchmark: full mode
must run on an otherwise quiet host.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from split_parameter_sweep import (  # noqa: E402
    _build_geometry,
    _build_path,
    _coords_host,
    _gaussian_source_host,
    _prepare_direct_tables,
    _prepare_rke_channels,
    _select_opencl_device,
    _split_channel_build_config,
    _split_smooth_quad_order,
    _yukawa_reference_build_config,
)

SOLVE_FIELDS = (
    "mode",
    "kernel",
    "parameter",
    "strategy",
    "split_order",
    "repeat_index",
    "solve_wall_s",
)

SUMMARY_FIELDS = (
    "mode",
    "kernel",
    "q_order",
    "nlevels",
    "fmm_order",
    "split_order",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "parameter_count",
    "parameters",
    "level_count",
    "direct_levels",
    "repeat_count",
    "n_targets",
    "direct_build_total_s",
    "rke_build_total_s",
    "direct_warmup_total_s",
    "rke_warmup_total_s",
    "direct_solve_mean_s",
    "direct_solve_std_s",
    "rke_solve_mean_s",
    "rke_solve_std_s",
    "measured_break_even_repeat",
    "measured_break_even_repeat_interpolated",
    "measured_crossing_cumulative_cost_s",
    "modeled_break_even_repeat_from_this_run",
    "cumulative_cost_definition",
    "max_rel_l2_rke_vs_direct",
    "benchmark_total_s",
    # provisioning strategy of the direct baseline (E3): "eager" builds all
    # anticipated levels, "lazy" only the level the priced workload touches
    "direct_provisioning",
    # operation counters (E3), computed from the executed configuration
    "ops_reduced_entries_per_table",
    "ops_direct_build_routing",
    "ops_direct_tables_built",
    "ops_direct_entries_built",
    "ops_direct_singular_nodes_per_entry",
    "ops_direct_singular_node_evals",
    "ops_direct_special_function",
    "ops_direct_special_function_evals",
    "ops_rke_channel_tables_built",
    "ops_rke_channel_entries_built",
    "ops_rke_channel_singular_nodes_per_entry",
    "ops_rke_channel_singular_node_evals",
    "ops_rke_channel_special_function_evals",
    "ops_nearfield_point_pairs_per_solve",
    "ops_direct_table_fmas_per_solve",
    "ops_split_table_count",
    "ops_split_table_fmas_per_solve",
    "ops_split_series_nmax_per_parameter",
    "ops_split_remainder_pair_evals_per_solve",
    "ops_split_remainder_term_flops_per_solve_per_parameter",
)


def _solve_wall_s(queue, traversal, wrangler, weighted_sources, source_vals):
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
    return potential, time.perf_counter() - start


def _solve_statistics(solve_rows, strategy):
    samples = [
        row["solve_wall_s"] for row in solve_rows
        if row["strategy"] == strategy
    ]
    if not samples:
        raise ValueError(f"no solve samples recorded for strategy {strategy!r}")
    return float(np.mean(samples)), float(np.std(samples))


def _modeled_break_even(
    *,
    parameter_count,
    direct_build_s,
    rke_build_s,
    direct_solve_mean_s,
    rke_solve_mean_s,
):
    setup_advantage = direct_build_s - rke_build_s
    per_repeat_gap = parameter_count * (
        rke_solve_mean_s - direct_solve_mean_s
    )
    if setup_advantage > 0.0 and per_repeat_gap > 0.0:
        return setup_advantage / per_repeat_gap
    return ""


def _resolve_direct_levels(
    *, smoke: bool, provisioning: str, nlevels: int
) -> list[int]:
    """Levels the direct baseline provisions under the chosen strategy.

    ``eager`` reproduces the committed-artifact policy (all anticipated
    levels).  ``lazy`` provisions only what the priced workload touches: the
    warm solves run on a uniform level-``nlevels`` grid whose List 1 work is
    owned entirely by the leaf level, so exactly one table per parameter.
    """
    if provisioning == "lazy":
        return [nlevels]
    if provisioning != "eager":
        raise ValueError(
            f"unknown direct provisioning strategy: {provisioning!r}"
        )
    return [1, 2] if smoke else [0, 1, 2, 3, 4, 5]


def _operation_counters(
    *,
    queue,
    traversal,
    parameters,
    direct_levels,
    direct_tables,
    direct_build_config,
    rke_base_table,
    split_term_tables,
    rke_channel_build_config,
    rke_wranglers,
    split_order,
):
    """Operation counts of the cost model, from the executed configuration.

    Every number is derived from executed objects — the built tables'
    symmetry-reduced entry sets, the node builders at the requested orders,
    the actual build routing predicate, the FMM traversal's List 1, and the
    wrangler's own series-length rule — not from hardcoded constants, so the
    emitted columns confirm (or refute) the analytic counts of the ops cost
    model in situ.
    """
    import volumential.opcounters as opcounters

    sample_direct = direct_tables[parameters[0]]
    n_rep = opcounters.reduced_entry_count(sample_direct)
    routing = (
        "batched"
        if sample_direct._supports_batched_duffy_builder()
        else "scalar"
    )
    if routing == "batched":
        direct_nodes_per_entry = opcounters.batched_duffy_nodes_per_entry(
            int(sample_direct.dim),
            direct_build_config.regular_quad_order,
            direct_build_config.radial_quad_order,
        )
        direct_special_function = "hankel1_imaginary_ray"
    else:
        geometry = opcounters.duffy_block_geometry(sample_direct)
        direct_nodes_per_entry = opcounters.scalar_duffy_singular_nodes(
            sample_direct,
            direct_build_config.regular_quad_order,
            direct_build_config.radial_quad_order,
            geometry=geometry,
        ) / max(geometry["n_reduced_entries"], 1)
        direct_special_function = "kv0"
    direct_tables_built = len(parameters) * len(direct_levels)
    direct_entries_built = direct_tables_built * n_rep
    direct_node_evals = int(
        round(direct_entries_built * direct_nodes_per_entry)
    )

    # split_term_tables maps each term key to a per-level list of tables (the
    # wrangler normalizes a bare table to a one-element list), so flatten
    # before counting: every listed table is separately built and its entries
    # separately quadratured.
    rke_tables = [rke_base_table]
    for term_tables in split_term_tables.values():
        if isinstance(term_tables, list):
            rke_tables.extend(term_tables)
        else:
            rke_tables.append(term_tables)
    rke_entry_counts = [
        opcounters.reduced_entry_count(table) for table in rke_tables
    ]
    rke_nodes_per_entry = opcounters.batched_duffy_nodes_per_entry(
        int(rke_base_table.dim),
        rke_channel_build_config.regular_quad_order,
        rke_channel_build_config.radial_quad_order,
    )
    rke_entries_built = int(np.sum(rke_entry_counts))

    n_nf = opcounters.nearfield_point_pairs(queue, traversal)
    nmax_by_parameter = [
        int(rke_wranglers[parameter]._helmholtz_split_series_nmax(
            split_order
        ))
        for parameter in parameters
    ]
    split_table_count = len(rke_tables)

    return {
        "ops_reduced_entries_per_table": n_rep,
        "ops_direct_build_routing": routing,
        "ops_direct_tables_built": direct_tables_built,
        "ops_direct_entries_built": direct_entries_built,
        "ops_direct_singular_nodes_per_entry": direct_nodes_per_entry,
        "ops_direct_singular_node_evals": direct_node_evals,
        "ops_direct_special_function": direct_special_function,
        "ops_direct_special_function_evals": direct_node_evals,
        "ops_rke_channel_tables_built": len(rke_tables),
        "ops_rke_channel_entries_built": rke_entries_built,
        "ops_rke_channel_singular_nodes_per_entry": rke_nodes_per_entry,
        "ops_rke_channel_singular_node_evals": (
            rke_entries_built * rke_nodes_per_entry
        ),
        # power/power-log channel integrands are elementary (log and radial
        # powers): no special-function quadrature anywhere in the family
        "ops_rke_channel_special_function_evals": 0,
        "ops_nearfield_point_pairs_per_solve": n_nf,
        "ops_direct_table_fmas_per_solve": n_nf,
        "ops_split_table_count": split_table_count,
        "ops_split_table_fmas_per_solve": split_table_count * n_nf,
        "ops_split_series_nmax_per_parameter": ";".join(
            str(nmax) for nmax in nmax_by_parameter
        ),
        "ops_split_remainder_pair_evals_per_solve": n_nf,
        "ops_split_remainder_term_flops_per_solve_per_parameter": ";".join(
            str(n_nf * nmax) for nmax in nmax_by_parameter
        ),
    }


def run_validation(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    split_order: int,
    parameters: list[float],
    direct_levels: list[int],
    repeat_count: int,
    warmup_count: int,
    direct_provisioning: str = "eager",
):
    import pyopencl as cl

    benchmark_start = time.perf_counter()
    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    q_points, source_weights, tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    high_accuracy = mode == "full"
    direct_build_config = _yukawa_reference_build_config(
        q_order, high_accuracy=high_accuracy
    )
    rke_channel_build_config = _split_channel_build_config(
        q_order, high_accuracy=high_accuracy
    )
    smooth_quad_order = _split_smooth_quad_order(
        q_order, split_order, high_accuracy=high_accuracy
    )

    # cold phase: direct provisioning per parameter (eager: all anticipated
    # levels; lazy: only the level the priced workload touches)
    direct_tables = {}
    direct_build_total_s = 0.0
    for parameter in parameters:
        table, costs = _prepare_direct_tables(
            kernel="Yukawa",
            queue=queue,
            cache_dir=cache_dir,
            q_order=q_order,
            parameter=parameter,
            direct_levels=direct_levels,
            active_level=nlevels,
            build_config=direct_build_config,
        )
        direct_tables[parameter] = table
        direct_build_total_s += costs["build_s"]
        print(
            f"direct cold build lam={parameter:g}: {costs['build_s']:.1f} s",
            flush=True,
        )

    # cold phase: one RKE channel family for all parameters
    rke_table, split_term_tables, rke_costs = _prepare_rke_channels(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        q_order=q_order,
        fmm_order=fmm_order,
        kernel="Yukawa",
        parameter=parameters[0],
        split_order=split_order,
        source_weights=source_weights,
        q_points=q_points,
        source_values_host=source_values_host,
        cache_dir=cache_dir,
        build_config=rke_channel_build_config,
        split_smooth_quad_order=smooth_quad_order,
    )
    rke_build_total_s = rke_costs["build_s"]
    print(f"rke cold build (p={split_order}): {rke_build_total_s:.1f} s",
          flush=True)

    # wranglers built once per parameter and strategy (steady-state reuse)
    paths = {}
    for parameter in parameters:
        direct_wrangler, weighted_sources, source_vals = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=parameter,
            table=direct_tables[parameter],
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=split_order,
        )
        rke_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel="Yukawa",
            parameter=parameter,
            table=rke_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_term_tables=split_term_tables,
            split_smooth_quad_order=smooth_quad_order,
        )
        paths[parameter] = {
            "direct": direct_wrangler,
            "rke": rke_wrangler,
            "weighted_sources": weighted_sources,
            "source_vals": source_vals,
        }

    # warmup solves (JIT compilation etc.), excluded from the curves
    warmup_totals = {"direct": 0.0, "rke": 0.0}
    max_rel_l2 = 0.0
    for parameter in parameters:
        path = paths[parameter]
        results = {}
        for _ in range(warmup_count):
            for strategy in ("direct", "rke"):
                potential, wall_s = _solve_wall_s(
                    queue,
                    traversal,
                    path[strategy],
                    path["weighted_sources"],
                    path["source_vals"],
                )
                warmup_totals[strategy] += wall_s
                results[strategy] = potential.get(queue)
        rel_l2 = float(
            np.linalg.norm(results["rke"] - results["direct"])
            / max(np.linalg.norm(results["direct"]), 1e-300)
        )
        max_rel_l2 = max(max_rel_l2, rel_l2)
        print(
            f"warmup lam={parameter:g}: rke vs direct rel_l2={rel_l2:.3e}",
            flush=True,
        )

    # repeat phase: alternate strategies within each repeat index
    solve_rows = []
    per_repeat_cost = {"direct": np.zeros(repeat_count),
                       "rke": np.zeros(repeat_count)}
    for repeat_index in range(repeat_count):
        for parameter in parameters:
            path = paths[parameter]
            for strategy in ("direct", "rke"):
                _, wall_s = _solve_wall_s(
                    queue,
                    traversal,
                    path[strategy],
                    path["weighted_sources"],
                    path["source_vals"],
                )
                per_repeat_cost[strategy][repeat_index] += wall_s
                solve_rows.append(
                    {
                        "mode": mode,
                        "kernel": "Yukawa",
                        "parameter": parameter,
                        "strategy": strategy,
                        "split_order": split_order,
                        "repeat_index": repeat_index,
                        "solve_wall_s": wall_s,
                    }
                )
        if (repeat_index + 1) % 25 == 0 or repeat_index + 1 == repeat_count:
            direct_cum = direct_build_total_s + float(
                np.sum(per_repeat_cost["direct"][: repeat_index + 1])
            )
            rke_cum = rke_build_total_s + float(
                np.sum(per_repeat_cost["rke"][: repeat_index + 1])
            )
            print(
                f"repeat {repeat_index + 1}/{repeat_count}: "
                f"cumulative direct={direct_cum:.1f} s rke={rke_cum:.1f} s",
                flush=True,
            )

    # cumulative curves and measured crossing
    direct_curve = direct_build_total_s + np.cumsum(per_repeat_cost["direct"])
    rke_curve = rke_build_total_s + np.cumsum(per_repeat_cost["rke"])
    above = np.nonzero(rke_curve >= direct_curve)[0]
    if len(above):
        n_cross = int(above[0]) + 1
        crossing_cost = float(direct_curve[above[0]])
        if above[0] == 0:
            interpolated = float(n_cross)
        else:
            i = above[0]
            gap_before = float(direct_curve[i - 1] - rke_curve[i - 1])
            gap_after = float(rke_curve[i] - direct_curve[i])
            interpolated = float(i) + gap_before / max(
                gap_before + gap_after, 1e-300
            )
        measured_break_even = n_cross
    else:
        measured_break_even = ""
        interpolated = ""
        crossing_cost = ""

    direct_mean, direct_std = _solve_statistics(solve_rows, "direct")
    rke_mean, rke_std = _solve_statistics(solve_rows, "rke")
    modeled = _modeled_break_even(
        parameter_count=len(parameters),
        direct_build_s=direct_build_total_s,
        rke_build_s=rke_build_total_s,
        direct_solve_mean_s=direct_mean,
        rke_solve_mean_s=rke_mean,
    )

    # operation counters (E3): computed after every timed phase, so the
    # timing columns are unaffected by the counting itself
    operation_counters = _operation_counters(
        queue=queue,
        traversal=traversal,
        parameters=parameters,
        direct_levels=direct_levels,
        direct_tables=direct_tables,
        direct_build_config=direct_build_config,
        rke_base_table=rke_table,
        split_term_tables=split_term_tables,
        rke_channel_build_config=rke_channel_build_config,
        rke_wranglers={
            parameter: paths[parameter]["rke"] for parameter in parameters
        },
        split_order=split_order,
    )
    print(
        "ops: direct "
        f"{operation_counters['ops_direct_entries_built']} entries x "
        f"{operation_counters['ops_direct_singular_nodes_per_entry']:g} "
        f"{operation_counters['ops_direct_build_routing']} nodes "
        f"({operation_counters['ops_direct_special_function']}); rke "
        f"{operation_counters['ops_rke_channel_entries_built']} entries x "
        f"{operation_counters['ops_rke_channel_singular_nodes_per_entry']} "
        "nodes (elementary); near-field pairs/solve "
        f"{operation_counters['ops_nearfield_point_pairs_per_solve']}, "
        "series nmax "
        f"{operation_counters['ops_split_series_nmax_per_parameter']}",
        flush=True,
    )

    summary_row = {
        "mode": mode,
        "kernel": "Yukawa",
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "split_order": split_order,
        "direct_regular_quad_order": direct_build_config.regular_quad_order,
        "direct_radial_quad_order": direct_build_config.radial_quad_order,
        "rke_channel_regular_quad_order": (
            rke_channel_build_config.regular_quad_order
        ),
        "rke_channel_radial_quad_order": (
            rke_channel_build_config.radial_quad_order
        ),
        "split_smooth_quad_order": (
            "" if smooth_quad_order is None else smooth_quad_order
        ),
        "parameter_count": len(parameters),
        "parameters": ";".join(f"{p:g}" for p in parameters),
        "level_count": len(direct_levels),
        "direct_levels": ";".join(str(level) for level in direct_levels),
        "repeat_count": repeat_count,
        "n_targets": int(tree.ntargets),
        "direct_build_total_s": direct_build_total_s,
        "rke_build_total_s": rke_build_total_s,
        "direct_warmup_total_s": warmup_totals["direct"],
        "rke_warmup_total_s": warmup_totals["rke"],
        "direct_solve_mean_s": direct_mean,
        "direct_solve_std_s": direct_std,
        "rke_solve_mean_s": rke_mean,
        "rke_solve_std_s": rke_std,
        "measured_break_even_repeat": measured_break_even,
        "measured_break_even_repeat_interpolated": interpolated,
        "measured_crossing_cumulative_cost_s": crossing_cost,
        "modeled_break_even_repeat_from_this_run": modeled,
        "cumulative_cost_definition": (
            "C(n)=strategy_build_total_s+sum_parameters sum_{k<=n} "
            "solve_wall_s;warmup_excluded;interleaved_strategies_per_repeat"
        ),
        "max_rel_l2_rke_vs_direct": max_rel_l2,
        "benchmark_total_s": time.perf_counter() - benchmark_start,
        "direct_provisioning": direct_provisioning,
        **operation_counters,
    }
    return solve_rows, summary_row


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/benchmarks/break-even-validation"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/break-even-cache"),
    )
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--split-order", type=int, default=3)
    parser.add_argument(
        "--direct-provisioning",
        choices=("eager", "lazy"),
        default="eager",
        help="direct-baseline provisioning strategy: 'eager' builds every "
        "anticipated level per parameter (the committed-artifact default); "
        "'lazy' builds only the leaf level the priced workload touches "
        "(the executed lazy baseline of experiment E3)",
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = 2 if smoke else 4
    nlevels = 2 if smoke else 5
    fmm_order = 8 if smoke else 16
    parameters = [4.0] if smoke else [4.0, 8.0, 12.0]
    direct_levels = _resolve_direct_levels(
        smoke=smoke,
        provisioning=args.direct_provisioning,
        nlevels=nlevels,
    )
    repeat_count = args.repeat_count
    if repeat_count is None:
        repeat_count = 6 if smoke else 400
    warmup_count = 1 if smoke else 2

    solve_rows, summary_row = run_validation(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        fmm_order=fmm_order,
        split_order=args.split_order,
        parameters=parameters,
        direct_levels=direct_levels,
        repeat_count=repeat_count,
        warmup_count=warmup_count,
        direct_provisioning=args.direct_provisioning,
    )

    _write_csv(args.out_dir / "break_even_solves.csv", SOLVE_FIELDS, solve_rows)
    _write_csv(
        args.out_dir / "break_even_summary.csv", SUMMARY_FIELDS, [summary_row]
    )
    print(
        "measured break-even repeat: "
        f"{summary_row['measured_break_even_repeat']} "
        f"(modeled from this run: "
        f"{summary_row['modeled_break_even_repeat_from_this_run']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
