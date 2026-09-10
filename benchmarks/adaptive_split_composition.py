#!/usr/bin/env python3
"""Emit three-mechanism composition CSVs for Paper 1.

The benchmark evaluates 2D Yukawa (and optionally Helmholtz) volume
potentials on the deterministically graded 2:1 trees of the adaptive-timing
benchmark, so all three reuse mechanisms compose in one run:

* symmetry reuse: every table is stored in orbit-reduced form;
* level reuse: the RKE path uses one canonical base table plus canonical
  channel tables scaled across the mixed leaf levels;
* parameter reuse: one fixed channel family serves every screening parameter
  through online coefficients.

The direct path builds per-level, per-parameter fixed-parameter tables and
evaluates the same potential.  The benchmark reports cross-level List 1
diagnostics, path mismatches between the RKE and direct potentials, and
build/payload comparisons.  Smoke mode is sized for CI; full mode reproduces
the graded-tree cases of the adaptive-timing benchmark.

Three evidence extensions (E5) sit behind flags with committed-run defaults
unchanged:

* ``--kernels Yukawa Helmholtz`` adds Helmholtz composition rows on the same
  graded trees (the same parameter list is read as wave numbers ``k``);
* ``--quadrature-policy high-accuracy`` forces the split-parameter sweep's
  high-accuracy Yukawa quadrature policy (direct 80/320, channels 32/80,
  smooth remainder ``2q`` above retained order one) regardless of mode, so
  the 2D composed path's split-order convergence is not floored by the
  default table quadrature (``auto`` keeps the historical behavior:
  default policy in smoke, high accuracy in full);
* ``--include-windowed`` appends one windowed-assembled composition row per
  kernel and parameter (``table_strategy=windowed_assembled``): the
  parameter-independent windowed channel family is built per populated
  source level, a fixed-parameter table is offline-assembled per level,
  registered through the standard table manager
  (``register_external_table``), reloaded through the ordinary cache path,
  and run through the identical graded-tree evaluator against the same
  direct per-level reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import sqlite3
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
    DEFAULT_WINDOW_THETA,
    DEFAULT_WINDOWED_CHAN_ORDERS_2D,
    DEFAULT_WINDOWED_P_STAR,
    WINDOWED_SMALL_THETA_AGREEMENT,
    WINDOWED_SMALL_THETA_MAX,
    _build_config,
    _build_path,
    _capture_table_get_timings,
    _clear_sqlite_cache,
    _configure_logging,
    _coords_host,
    _gaussian_source_host,
    _get_laplace_2d_table,
    _prepare_windowed_family,
    _register_and_load_windowed_table,
    _select_opencl_device,
    _split_channel_build_config,
    _split_smooth_quad_order,
    _summarize_table_get_timings,
    _table_build_routing,
    _yukawa_reference_build_config,
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
    # E5 extension columns (appended; the committed column prefix above is
    # byte-position-identical).  ``online_split`` rows are the historical
    # composed RKE path; ``windowed_assembled`` rows provision per-level
    # fixed-parameter tables by windowed offline assembly and run them
    # through the same graded-tree evaluator.
    "quadrature_policy",
    "table_strategy",
    "window_theta",
    "windowed_p_star",
    "windowed_chan_regular_order",
    "windowed_chan_radial_order",
    "windowed_status",
    "windowed_refusal",
    "windowed_theta_by_level_json",
    "windowed_max_condition_number",
    "windowed_smooth_quad_orders_json",
    "windowed_channel_build_s",
    "windowed_channel_build_was_cold",
    "windowed_assemble_s",
    "windowed_register_s",
    "windowed_table_load_s",
    "windowed_table_count",
    "windowed_register_payload_bytes",
    "windowed_wall_s",
    "windowed_vs_direct_weighted_rel_l2",
    "windowed_vs_direct_linf",
    # which DuffyRadial builder produced the per-level direct reference tables
    # of this row (';'-joined over levels); "scalar-fallback" marks a row whose
    # reference came from the slower per-entry builder after a batched failure
    "direct_build_routing",
)

WINDOWED_FIELDS = tuple(
    field for field in FIELDS
    if field.startswith("windowed_") or field == "window_theta"
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


SMOKE_CASES = ((3, 2, 1),)
FULL_CASES = ((3, 4, 2), (4, 4, 2), (4, 4, 3))

SMOKE_PARAMETERS = (2.0,)
FULL_PARAMETERS = (2.0, 4.0, 8.0)

SMOKE_SPLIT_ORDERS = (2,)
FULL_SPLIT_ORDERS = (2, 3)

KERNEL_PARAMETER_TAGS = {"Yukawa": "lam", "Helmholtz": "k"}

# Full-mode split-order convergence gates under the high-accuracy policy
# (E5), calibrated like the split-parameter sweep's Yukawa gate: with the
# quadrature floor lifted, p=2 must improve on p=1 by three orders of
# magnitude wherever both are present, and no consecutive order pair may
# materially degrade (p=3 can legitimately sit at the p=2 floor).
ORDER_CONVERGENCE_P2_OVER_P1 = 1.0e-3
ORDER_CONVERGENCE_DEGRADATION = 1.1

PATH_MISMATCH_NORM_DEFINITION = (
    "weighted_rel_l2=sqrt(sum_i quadrature_weight_i*"
    "abs({path}_i-direct_i)**2)/max(sqrt(sum_i "
    "quadrature_weight_i*abs(direct_i)**2),1e-300);"
    "linf=max_i abs({path}_i-direct_i)"
)


def _resolve_quadrature_policy(policy: str, mode: str) -> str:
    if policy == "auto":
        return "high-accuracy" if mode == "full" else "default"
    if policy in ("default", "high-accuracy"):
        return policy
    raise ValueError(f"unknown quadrature policy: {policy}")


def _kernel_build_configs(kernel: str, q_order: int, *, high_accuracy: bool):
    """Per-kernel direct and channel build configs, mirroring the
    split-parameter sweep: the high-accuracy policy is the Yukawa
    logarithmic-singularity policy; Helmholtz uses the default rule."""
    if kernel == "Yukawa":
        return (
            _yukawa_reference_build_config(q_order, high_accuracy=high_accuracy),
            _split_channel_build_config(q_order, high_accuracy=high_accuracy),
        )
    if kernel == "Helmholtz":
        return _build_config(q_order), _build_config(q_order)
    raise ValueError(f"unknown kernel: {kernel}")


def _get_direct_2d_table_with_timings(
    queue,
    cache_path: Path,
    kernel: str,
    q_order: int,
    parameter: float,
    level: int,
    *,
    tree_root_extent: float,
    build_config,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    manager_kwargs: dict[str, Any] = {}
    get_kwargs: dict[str, Any] = {}
    if kernel == "Yukawa":
        kernel_request = "Yukawa"
        get_kwargs["lam"] = float(parameter)
    elif kernel == "Helmholtz":
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(2)
        kernel_request = "Helmholtz-Reference"
        manager_kwargs["dtype"] = np.complex128
        get_kwargs["sumpy_knl"] = knl
        get_kwargs[knl.helmholtz_k_name] = float(parameter)
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=tree_root_extent, queue=queue,
        **manager_kwargs,
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            kernel_request,
            q_order,
            source_box_level=int(level),
            force_recompute=True,
            queue=queue,
            build_config=build_config,
            **get_kwargs,
        )
    timings = dict(table_manager.last_get_table_timings)
    compute = timings.get("compute") or {}
    build_s = float(timings.get("total_s", compute.get("total_s", 0.0)))
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


def _weighted_mismatch(weights_host, candidate, reference) -> tuple[float, float]:
    difference = candidate - reference
    weighted_error = float(
        np.sqrt(np.sum(weights_host * np.abs(difference) ** 2))
    )
    weighted_reference = max(
        float(np.sqrt(np.sum(weights_host * np.abs(reference) ** 2))),
        1.0e-300,
    )
    return (
        weighted_error / weighted_reference,
        float(np.max(np.abs(difference))),
    )


def _run_windowed_composition(
    queue,
    *,
    cache_dir: Path,
    kernel: str,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    parameter: float,
    source_levels: list[int],
    tree_root_extent: float,
    window_theta: float,
    windowed_p_star: int,
    windowed_chan_orders: tuple[int, int],
) -> dict[str, Any]:
    """Provision per-level windowed-assembled tables for one parameter.

    Returns a partial row dict (windowed_* columns plus loaded tables under
    the private ``_tables`` key on success) following the ``ok`` /
    ``refused`` / ``failed`` taxonomy.
    """
    from volumential.rke_table_assembly import (
        RKEWindowConditioningError,
        RKEWindowCoverageError,
        assemble_windowed_parameterized_table,
    )

    family_cache = cache_dir / (
        f"composition-windowed-family-q{q_order}-l{initial_nlevels}-"
        f"a{adapt_steps}-Theta{window_theta:g}.sqlite"
    )

    result: dict[str, Any] = {
        "window_theta": window_theta,
        "windowed_p_star": windowed_p_star,
        "windowed_chan_regular_order": windowed_chan_orders[0],
        "windowed_chan_radial_order": windowed_chan_orders[1],
        "windowed_theta_by_level_json": json.dumps(
            {
                str(level): parameter * tree_root_extent * 0.5**level
                for level in source_levels
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "windowed_channel_build_s": 0.0,
        "windowed_channel_build_was_cold": 0,
        "windowed_assemble_s": 0.0,
        "windowed_register_s": 0.0,
        "windowed_table_load_s": 0.0,
        "windowed_table_count": 0,
        "windowed_register_payload_bytes": 0,
    }

    channel_build_s = 0.0
    was_cold = False
    for level in source_levels:
        family = _prepare_windowed_family(
            cache_path=family_cache,
            q_order=q_order,
            source_box_level=int(level),
            window_theta=window_theta,
            p_star=windowed_p_star,
            chan_regular_order=windowed_chan_orders[0],
            chan_radial_order=windowed_chan_orders[1],
            root_extent=tree_root_extent,
        )
        channel_build_s += family["build_s"]
        was_cold = was_cold or family["was_cold"]
    result["windowed_channel_build_s"] = channel_build_s
    result["windowed_channel_build_was_cold"] = int(was_cold)

    parameter_tag = f"{parameter:.17g}".replace("-", "m").replace(".", "p")
    tables = []
    condition_numbers = []
    smooth_quad_orders: dict[str, int] = {}
    assemble_s = 0.0
    register_s = 0.0
    load_s = 0.0
    register_payload_bytes = 0
    for level in source_levels:
        assemble_start = time.perf_counter()
        try:
            assembled_table, certificate = (
                assemble_windowed_parameterized_table(
                    family_cache,
                    2,
                    kernel,
                    q_order,
                    parameter,
                    source_box_level=int(level),
                    root_extent=tree_root_extent,
                    window_theta=window_theta,
                    p_star=windowed_p_star,
                    chan_regular_order=windowed_chan_orders[0],
                    chan_radial_order=windowed_chan_orders[1],
                )
            )
        except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
            result["windowed_status"] = "refused"
            result["windowed_refusal"] = (
                f"level {int(level)}: {type(exc).__name__}: {exc}"
            )
            result["windowed_assemble_s"] = (
                assemble_s + time.perf_counter() - assemble_start
            )
            return result
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            result["windowed_status"] = "failed"
            result["windowed_refusal"] = (
                f"level {int(level)}: {type(exc).__name__}: {exc}"
            )
            result["windowed_assemble_s"] = (
                assemble_s + time.perf_counter() - assemble_start
            )
            return result
        assemble_s += time.perf_counter() - assemble_start
        condition_numbers.append(float(certificate["condition_number"]))
        smooth_quad_orders[str(int(level))] = int(
            certificate["smooth_quad_order"]
        )

        registered_cache = cache_dir / (
            f"composition-windowed-registered-{kernel.lower()}-"
            f"q{q_order}-l{initial_nlevels}-a{adapt_steps}-"
            f"parameter{parameter_tag}-lev{int(level)}.sqlite"
        )
        register_start = time.perf_counter()
        try:
            loaded_table, transfer = _register_and_load_windowed_table(
                queue=queue,
                cache_path=registered_cache,
                kernel=kernel,
                q_order=q_order,
                parameter=parameter,
                source_box_level=int(level),
                table=assembled_table,
                certificate=certificate,
                root_extent=tree_root_extent,
            )
        except (
            ValueError, RuntimeError, NotImplementedError, OSError, KeyError,
            # sqlite3's exceptions descend from Exception, not OSError, and
            # this phase is a SQLite round trip
            sqlite3.Error,
        ) as exc:
            # Registration and the pure-cache reload belong to the same
            # taxonomy as the assembly above.  main() writes the CSV only
            # after every case completes, so an exception escaping here
            # discards every measurement already taken, not just this row.
            result["windowed_status"] = "failed"
            result["windowed_refusal"] = (
                f"level {int(level)}: {type(exc).__name__}: {exc}"
            )
            result["windowed_assemble_s"] = assemble_s
            # whatever the helper got through before it raised
            partial = getattr(exc, "partial_windowed_transfer", None)
            if partial is not None:
                register_s += float(partial["register_s"])
                load_s += float(partial["load_s"])
                register_payload_bytes += int(
                    partial["register_payload_bytes"]
                )
            else:
                register_s += time.perf_counter() - register_start
            result["windowed_register_s"] = register_s
            result["windowed_table_load_s"] = load_s
            result["windowed_register_payload_bytes"] = register_payload_bytes
            result["windowed_table_count"] = len(tables)
            return result
        tables.append(loaded_table)
        register_s += transfer["register_s"]
        load_s += transfer["load_s"]
        register_payload_bytes += int(transfer["register_payload_bytes"])

    result.update(
        {
            "windowed_status": "ok",
            "windowed_refusal": "",
            "windowed_max_condition_number": max(condition_numbers),
            "windowed_smooth_quad_orders_json": json.dumps(
                smooth_quad_orders, sort_keys=True, separators=(",", ":")
            ),
            "windowed_assemble_s": assemble_s,
            "windowed_register_s": register_s,
            "windowed_table_load_s": load_s,
            "windowed_table_count": len(tables),
            "windowed_register_payload_bytes": register_payload_bytes,
            "_tables": tables,
        }
    )
    return result


def _validate_windowed_composition_rows(rows: list[dict[str, Any]]) -> None:
    """Taxonomy and agreement gates for windowed-assembled composition rows.

    A ``failed`` row, or a certificate refusal while every populated source
    level sits inside the declaration, is a driver failure.  At small theta
    the windowed-assembled path must agree with the direct per-level
    reference through the same graded-tree evaluator.
    """
    for row in rows:
        if row.get("table_strategy") != "windowed_assembled":
            continue
        status = row["windowed_status"]
        max_theta = float(row["max_theta"])
        window_theta = float(row["window_theta"])
        if status == "failed":
            raise RuntimeError(
                f"windowed assembly failed for {row['case_id']}: "
                f"{row['windowed_refusal']}"
            )
        if status == "refused" and max_theta <= window_theta * (1.0 + 1.0e-9):
            raise RuntimeError(
                f"windowed assembly refused inside the declaration for "
                f"{row['case_id']} (max theta={max_theta:g} <= Theta="
                f"{window_theta:g}): {row['windowed_refusal']}"
            )
        if status == "ok" and max_theta <= WINDOWED_SMALL_THETA_MAX:
            gate = WINDOWED_SMALL_THETA_AGREEMENT[row["mode"]]
            rel_l2 = float(row["windowed_vs_direct_weighted_rel_l2"])
            if not rel_l2 <= gate:
                raise RuntimeError(
                    "windowed-assembled composition path disagrees with the "
                    f"direct reference at small theta for {row['case_id']}: "
                    f"weighted_rel_l2={rel_l2:.3e} > {gate:.1e}"
                )


def _validate_split_order_convergence(rows: list[dict[str, Any]]) -> None:
    """Full-mode, high-accuracy-policy gate: the 2D composed Yukawa path
    must resolve its split-order dependence once the quadrature floor is
    lifted (the committed default-policy run could not)."""
    errors: dict[tuple[str, float], dict[int, float]] = {}
    for row in rows:
        if row["mode"] != "full" or row["kernel"] != "Yukawa":
            continue
        if row.get("table_strategy", "online_split") != "online_split":
            continue
        if row.get("quadrature_policy") != "high-accuracy":
            continue
        case_key = (
            f"q{row['q_order']}-l{row['initial_nlevels']}-"
            f"a{row['adapt_steps']}"
        )
        errors.setdefault(
            (case_key, float(row["parameter"])), {}
        )[int(row["split_order"])] = float(
            row["rke_vs_direct_weighted_rel_l2"]
        )

    for (case_key, parameter), by_order in errors.items():
        orders = sorted(by_order)
        if len(orders) < 2:
            continue
        if 1 in by_order and 2 in by_order and (
            by_order[2] > ORDER_CONVERGENCE_P2_OVER_P1 * by_order[1]
        ):
            raise RuntimeError(
                "full high-accuracy 2D Yukawa composition p=2 mismatch did "
                "not improve by three orders of magnitude at "
                f"{case_key} lambda={parameter:g}: p=1 gives "
                f"{by_order[1]:.3e}, p=2 gives {by_order[2]:.3e}"
            )
        for p_low, p_high in zip(orders[:-1], orders[1:], strict=False):
            if by_order[p_high] > ORDER_CONVERGENCE_DEGRADATION * by_order[p_low]:
                raise RuntimeError(
                    "full high-accuracy 2D Yukawa composition mismatch "
                    f"materially degraded from p={p_low} to p={p_high} at "
                    f"{case_key} lambda={parameter:g}: "
                    f"{by_order[p_low]:.3e} -> {by_order[p_high]:.3e}"
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
    kernels=("Yukawa",),
    quadrature_policy: str | None = None,
    include_windowed: bool = False,
    window_theta: float = DEFAULT_WINDOW_THETA,
    windowed_p_star: int = DEFAULT_WINDOWED_P_STAR,
    windowed_chan_orders: tuple[int, int] = DEFAULT_WINDOWED_CHAN_ORDERS_2D,
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
    if quadrature_policy is None:
        quadrature_policy = _resolve_quadrature_policy("auto", mode)
    high_accuracy = quadrature_policy == "high-accuracy"

    common_columns = {
        "mode": mode,
        "q_order": q_order,
        "initial_nlevels": initial_nlevels,
        "adapt_steps": adapt_steps,
        "n_targets": int(tree.ntargets),
        **leaf_diagnostics,
        **list1_diagnostics,
        "populated_source_levels_json": json.dumps(
            source_levels, separators=(",", ":")
        ),
        "quadrature_policy": quadrature_policy,
    }

    rows = []
    for kernel in kernels:
        parameter_tag = KERNEL_PARAMETER_TAGS[kernel]
        direct_build_config, rke_channel_build_config = _kernel_build_configs(
            kernel, q_order, high_accuracy=high_accuracy
        )

        direct_results = {}
        for parameter in parameters:
            direct_cache_path = cache_dir / (
                f"composition-direct-{kernel.lower()}-q{q_order}-"
                f"l{initial_nlevels}-a{adapt_steps}-"
                f"{parameter_tag}{parameter:g}.sqlite"
            )
            _clear_sqlite_cache(direct_cache_path)

            direct_tables = []
            direct_build_s = 0.0
            direct_payload_bytes = 0
            for level in source_levels:
                table, build_s, payload_bytes = (
                    _get_direct_2d_table_with_timings(
                        queue,
                        direct_cache_path,
                        kernel,
                        q_order,
                        parameter,
                        level,
                        tree_root_extent=tree_root_extent,
                        build_config=direct_build_config,
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
                kernel=kernel,
                parameter=float(parameter),
                table=direct_tables,
                source_weights=q_weights,
                q_points=q_points,
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
                # the routing recorded by the builder for every per-level
                # direct table of this parameter (';'-joined when they differ)
                "build_routing": _table_build_routing(direct_tables),
            }

        for split_order in split_orders:
            smooth_quad_order = _split_smooth_quad_order(
                q_order, split_order, high_accuracy=high_accuracy
            )
            # One fixed channel family per retained order; built once, reused
            # for every parameter below.
            rke_cache_path = cache_dir / (
                f"composition-rke-{kernel.lower()}-q{q_order}-"
                f"l{initial_nlevels}-a{adapt_steps}-p{split_order}.sqlite"
            )
            _clear_sqlite_cache(rke_cache_path)

            with _capture_table_get_timings() as base_records:
                base_table = _get_laplace_2d_table(
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
                    kernel=kernel,
                    parameter=float(parameters[0]),
                    table=base_table,
                    source_weights=q_weights,
                    q_points=q_points,
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
                    kernel=kernel,
                    parameter=float(parameter),
                    table=base_table,
                    source_weights=q_weights,
                    q_points=q_points,
                    source_values_host=source_values_host,
                    split=True,
                    split_order=split_order,
                    split_term_tables=split_term_tables,
                    split_smooth_quad_order=smooth_quad_order,
                )
                rke_potential, rke_wall_s = _drive(
                    queue, traversal, rke_wrangler, weighted_sources, source_vals
                )

                weighted_rel_l2, linf = _weighted_mismatch(
                    weights_host, rke_potential, direct_potential
                )

                rows.append(
                    {
                        "case_id": (
                            f"{kernel.lower()}2d-q{q_order}-l{initial_nlevels}-"
                            f"a{adapt_steps}-{parameter_tag}{parameter:g}-"
                            f"p{split_order}"
                        ),
                        **common_columns,
                        "kernel": kernel,
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
                        "max_theta": parameter * h_max_leaf,
                        "path_mismatch_norm_definition": (
                            PATH_MISMATCH_NORM_DEFINITION.format(path="rke")
                        ),
                        "rke_vs_direct_weighted_rel_l2": weighted_rel_l2,
                        "rke_vs_direct_linf": linf,
                        "direct_wall_s": direct_result["wall_s"],
                        "rke_wall_s": rke_wall_s,
                        "direct_table_count": direct_result["table_count"],
                        "direct_table_build_s": direct_result["build_s"],
                        "direct_table_payload_bytes": (
                            direct_result["payload_bytes"]
                        ),
                        "direct_build_routing": direct_result[
                            "build_routing"
                        ],
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
                        "table_strategy": "online_split",
                        **{field: "" for field in WINDOWED_FIELDS},
                    }
                )
                print(
                    f"[{rows[-1]['case_id']}] rke_vs_direct_weighted_rel_l2="
                    f"{rows[-1]['rke_vs_direct_weighted_rel_l2']:.3e} "
                    f"cross_level_fraction="
                    f"{list1_diagnostics['cross_level_list1_fraction']:.3f}",
                    flush=True,
                )

        if not include_windowed:
            continue

        for parameter in parameters:
            direct_result = direct_results[parameter]
            windowed = _run_windowed_composition(
                queue,
                cache_dir=cache_dir,
                kernel=kernel,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                parameter=float(parameter),
                source_levels=source_levels,
                tree_root_extent=tree_root_extent,
                window_theta=window_theta,
                windowed_p_star=windowed_p_star,
                windowed_chan_orders=windowed_chan_orders,
            )
            windowed_tables = windowed.pop("_tables", None)

            row = {
                "case_id": (
                    f"{kernel.lower()}2d-q{q_order}-l{initial_nlevels}-"
                    f"a{adapt_steps}-{parameter_tag}{parameter:g}-windowed"
                ),
                **common_columns,
                "kernel": kernel,
                "parameter": parameter,
                "split_order": "",
                "direct_regular_quad_order": (
                    direct_build_config.regular_quad_order
                ),
                "direct_radial_quad_order": (
                    direct_build_config.radial_quad_order
                ),
                "rke_channel_regular_quad_order": "",
                "rke_channel_radial_quad_order": "",
                "split_smooth_quad_order": "",
                "max_theta": parameter * h_max_leaf,
                "path_mismatch_norm_definition": (
                    PATH_MISMATCH_NORM_DEFINITION.format(path="windowed")
                ),
                "rke_vs_direct_weighted_rel_l2": "",
                "rke_vs_direct_linf": "",
                "direct_wall_s": direct_result["wall_s"],
                "rke_wall_s": "",
                "direct_table_count": direct_result["table_count"],
                "direct_table_build_s": direct_result["build_s"],
                "direct_table_payload_bytes": direct_result["payload_bytes"],
                "direct_build_routing": direct_result["build_routing"],
                "rke_base_table_build_s": "",
                "rke_base_table_payload_bytes": "",
                "rke_channel_table_count": "",
                "rke_channel_table_build_s": "",
                "rke_channel_table_payload_bytes": "",
                "rke_total_table_payload_bytes": "",
                "table_strategy": "windowed_assembled",
                **{field: "" for field in WINDOWED_FIELDS},
                **windowed,
            }

            if windowed_tables is not None:
                windowed_wrangler, weighted_sources, source_vals = _build_path(
                    ctx=ctx,
                    queue=queue,
                    traversal=traversal,
                    q_order=q_order,
                    fmm_order=fmm_order,
                    kernel=kernel,
                    parameter=float(parameter),
                    table=windowed_tables,
                    source_weights=q_weights,
                    q_points=q_points,
                    source_values_host=source_values_host,
                    split=False,
                    split_order=1,
                )
                windowed_potential, windowed_wall_s = _drive(
                    queue,
                    traversal,
                    windowed_wrangler,
                    weighted_sources,
                    source_vals,
                )
                weighted_rel_l2, linf = _weighted_mismatch(
                    weights_host,
                    windowed_potential,
                    direct_result["potential"],
                )
                row.update(
                    {
                        "windowed_wall_s": windowed_wall_s,
                        "windowed_vs_direct_weighted_rel_l2": weighted_rel_l2,
                        "windowed_vs_direct_linf": linf,
                    }
                )

            rows.append(row)
            print(
                f"[{row['case_id']}] windowed_status={row['windowed_status']}"
                " windowed_vs_direct_weighted_rel_l2="
                f"{row['windowed_vs_direct_weighted_rel_l2']}",
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
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=("Yukawa", "Helmholtz"),
        default=None,
        help=(
            "kernels to compose on the same graded trees "
            "(default: Yukawa only, the committed configuration)"
        ),
    )
    parser.add_argument(
        "--parameters",
        type=float,
        nargs="+",
        help=(
            "kernel parameters: Yukawa screening lambda and/or Helmholtz "
            "wave number k (default: 2 smoke; 2,4,8 full)"
        ),
    )
    parser.add_argument(
        "--split-orders",
        type=int,
        nargs="+",
        help="retained channel orders (default: 2 smoke; 2,3 full)",
    )
    parser.add_argument(
        "--quadrature-policy",
        choices=("auto", "default", "high-accuracy"),
        default="auto",
        help=(
            "table quadrature policy; 'auto' (default) keeps the historical "
            "behavior (default policy in smoke, the split-parameter sweep's "
            "high-accuracy Yukawa policy in full), the other values force "
            "one policy in either mode"
        ),
    )
    parser.add_argument(
        "--include-windowed",
        action="store_true",
        help=(
            "append one windowed-assembled composition row per kernel and "
            "parameter (per-level offline assembly through "
            "register_external_table, run through the same evaluator)"
        ),
    )
    parser.add_argument(
        "--window-theta",
        type=float,
        default=DEFAULT_WINDOW_THETA,
        help="declared window Theta for the windowed-assembled rows",
    )
    parser.add_argument(
        "--windowed-p-star",
        type=int,
        default=DEFAULT_WINDOWED_P_STAR,
        help="number of tabulated windowed channels",
    )
    parser.add_argument(
        "--windowed-chan-orders",
        default=None,
        help=(
            "'regular,radial' windowed channel quadrature orders "
            "(default 48,61: the assembler's tested 2D orders)"
        ),
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
    kernels = tuple(args.kernels) if args.kernels else ("Yukawa",)
    if len(set(kernels)) != len(kernels):
        parser.error("--kernels must be unique")
    quadrature_policy = _resolve_quadrature_policy(
        args.quadrature_policy, args.mode
    )
    if args.windowed_chan_orders is None:
        windowed_chan_orders = DEFAULT_WINDOWED_CHAN_ORDERS_2D
    else:
        parts = [
            int(part.strip())
            for part in args.windowed_chan_orders.split(",")
            if part.strip()
        ]
        if len(parts) != 2:
            parser.error(
                "--windowed-chan-orders must be a 'regular,radial' pair"
            )
        windowed_chan_orders = (parts[0], parts[1])
    if args.windowed_p_star < 1:
        parser.error("--windowed-p-star must be >= 1")
    if not (args.window_theta > 0.0):
        parser.error("--window-theta must be positive")

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
                kernels=kernels,
                quadrature_policy=quadrature_policy,
                include_windowed=args.include_windowed,
                window_theta=args.window_theta,
                windowed_p_star=args.windowed_p_star,
                windowed_chan_orders=windowed_chan_orders,
            )
        )
    _validate_windowed_composition_rows(rows)
    _validate_split_order_convergence(rows)
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
