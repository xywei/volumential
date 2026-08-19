#!/usr/bin/env python3
"""Emit Helmholtz/Yukawa split-parameter sweep CSVs for Paper 1.

The benchmark sweeps the Helmholtz wave number and Yukawa screening parameter
while recording split-table accounting. Rows compare against fixed-parameter
direct near-field tables and separate direct-table setup/apply costs from RKE
channel setup, coefficient, residual, and full split costs. Cold/warm strategy
totals and break-even roots expose the parameter/level/repeat amortization model.

Smoke mode is intended for CI/local validation. Full mode is intended for
metadata-wrapped runs on a controlled remote compute host.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "parameter_name",
    "parameter_value",
    "split_order",
    "power_log_beta_mode",
    "direct_regular_quad_order",
    "direct_radial_quad_order",
    "rke_channel_regular_quad_order",
    "rke_channel_radial_quad_order",
    "split_smooth_quad_order",
    "q_order",
    "nlevels",
    "fmm_order",
    "n_targets",
    "reference_path",
    "rel_l2_error",
    "linf_error",
    "reference_warm_s",
    "split_warm_s",
    "online_remainder_s",
    "online_remainder_time_kind",
    "parameter_count",
    "base_table_count",
    "split_term_table_count",
    "basis_table_count",
    "base_table_payload_bytes",
    "split_term_table_payload_bytes",
    "total_table_payload_bytes",
    "split_term_keys",
    "uses_online_coefficients",
    "uses_online_remainder",
    "level_count",
    "direct_levels",
    "repeat_count",
    "solve_count",
    "direct_table_count",
    "direct_table_payload_bytes",
    "direct_table_cache_payload_bytes",
    "direct_table_build_s",
    "direct_table_quadrature_build_s",
    "direct_table_load_s",
    "rke_channel_build_s",
    "rke_channel_quadrature_build_s",
    "rke_channel_load_s",
    "rke_channel_cache_payload_bytes",
    "direct_table_apply_total_s",
    "direct_table_apply_mean_s",
    "split_full_apply_total_s",
    "split_full_apply_mean_s",
    "split_correction_total_s",
    "split_correction_mean_s",
    "split_correction_time_kind",
    "smooth_residual_total_s",
    "smooth_residual_mean_s",
    "smooth_residual_time_kind",
    "coefficient_eval_mean_s",
    "coefficient_eval_time_kind",
    "reference_solve_total_s",
    "split_solve_total_s",
    "direct_strategy_cold_total_s",
    "rke_strategy_cold_total_s",
    "direct_strategy_warm_total_s",
    "rke_strategy_warm_total_s",
    "direct_amortized_cold_s_per_solve",
    "rke_amortized_cold_s_per_solve",
    "direct_amortized_warm_s_per_solve",
    "rke_amortized_warm_s_per_solve",
    "modeled_cold_savings_s",
    "direct_table_build_mean_s_per_parameter_level",
    "direct_solve_mean_s_per_solve",
    "split_solve_mean_s_per_solve",
    "amortization_time_kind",
    "break_even_parameter_count",
    "break_even_level_count",
    "break_even_repeat_count",
    "break_even_time_kind",
    "benchmark_total_s",
    "benchmark_total_time_kind",
    # table-provisioning strategy columns (E1).  "online_split" rows are the
    # historical online split-evaluator comparison; "windowed_assembled" rows
    # provision the near-field table by windowed offline assembly
    # (rke_table_assembly.assemble_windowed_parameterized_table), register it
    # through the standard table manager, and run the identical evaluator
    # path against the same direct fixed-parameter reference.
    "table_strategy",
    "theta",
    "window_theta",
    "windowed_p_star",
    "windowed_smooth_quad_order",
    "windowed_chan_regular_order",
    "windowed_chan_radial_order",
    "windowed_status",
    "windowed_refusal",
    "windowed_condition_number",
    "windowed_channel_build_s",
    "windowed_channel_build_was_cold",
    "windowed_assemble_s",
    "windowed_register_s",
    "windowed_register_payload_bytes",
    "windowed_table_load_s",
    "windowed_table_load_payload_bytes",
    "windowed_solve_warm_s",
    "windowed_table_apply_mean_s",
    "classical_probe_kind",
    "classical_probe_status",
    "classical_probe_detail",
    "classical_probe_n_terms",
    "classical_probe_condition_number",
    "classical_probe_s",
)

# Root extent of the table-manager convention used throughout this driver
# (the [-0.5, 0.5]^2 unit tree maps to table level ell + 1).
TABLE_ROOT_EXTENT = 2.0

# Windowed strategy defaults (E1): declared window and per-mode theta ladders.
# The smoke ladder holds one theta inside the polynomial-certified range, one
# in the polynomial refusal band, and the declaration edge itself.
DEFAULT_WINDOW_THETA = 16.0
DEFAULT_WINDOWED_P_STAR = 6
DEFAULT_WINDOWED_CHAN_ORDERS_2D = (48, 61)
DEFAULT_SMOKE_WINDOWED_THETAS = (1.0, 6.0, 16.0)
DEFAULT_FULL_WINDOWED_THETAS = (
    0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0,
)
CLASSICAL_PROBE_TOLERANCE = 1.0e-11
# Evaluator-level agreement gates for the windowed-assembled strategy at
# small theta (theta <= 2), where the direct reference is well resolved.
# The smoke gate tolerates the loose smoke-mode direct quadrature; the full
# gate sits above the ~2e-7 channel-quadrature accuracy of the default
# 48/61 orders (see rke_table_assembly._resolve_channel_orders) with margin,
# while still catching any structural (sign/normalization/registration)
# error, which would show up at O(1).
WINDOWED_SMALL_THETA_AGREEMENT = {"smoke": 1.0e-2, "full": 1.0e-5}
WINDOWED_SMALL_THETA_MAX = 2.0


def _parse_csv_floats(raw: str, *, allow_empty: bool = False) -> list[float]:
    if allow_empty and raw.strip().lower() in {"", "none", "skip"}:
        return []
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        if allow_empty:
            return []
        raise ValueError("expected at least one float value")
    return values


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    if any(value < 1 for value in values):
        raise ValueError("integer values must be >= 1")
    return values


def _parse_csv_levels(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one source-box level")
    if any(value < 0 for value in values):
        raise ValueError("source-box levels must be >= 0")
    if len(set(values)) != len(values):
        raise ValueError("source-box levels must be unique")
    return sorted(values)


def _limit_parameter_values(values: list[float], count: int | None) -> list[float]:
    if count is None or not values:
        return values
    if count < 1:
        raise ValueError("parameter_count must be >= 1")
    if count > len(values):
        raise ValueError(
            f"parameter_count={count} exceeds the {len(values)} supplied values"
        )
    return values[:count]


def _table_payload_bytes(table) -> int:
    if bool(getattr(table, "table_data_is_symmetry_reduced", False)):
        if hasattr(table, "get_reduced_table_data"):
            _, values = table.get_reduced_table_data()
            return int(np.asarray(values).nbytes)
        data = np.asarray(table.data)
        return int(np.count_nonzero(np.isfinite(data)) * data.dtype.itemsize)
    return int(np.asarray(table.data).nbytes)


def _clear_sqlite_cache(path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


@contextmanager
def _capture_table_get_timings():
    """Capture table-manager timings, including split-channel auto-builds."""
    from volumential.table_manager import NearFieldInteractionTableManager

    records = []
    original_get_table = NearFieldInteractionTableManager.get_table

    def timed_get_table(table_manager, *args, **kwargs):
        result = original_get_table(table_manager, *args, **kwargs)
        records.append(dict(table_manager.last_get_table_timings or {}))
        return result

    NearFieldInteractionTableManager.get_table = timed_get_table
    try:
        yield records
    finally:
        NearFieldInteractionTableManager.get_table = original_get_table


def _summarize_table_get_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    build_records = [record for record in records if record.get("is_recomputed")]
    load_records = [record for record in records if not record.get("is_recomputed")]
    return {
        "build_s": float(
            sum(record.get("total_s", 0.0) for record in build_records)
        ),
        "quadrature_build_s": float(
            sum((record.get("compute") or {}).get("table_build_s", 0.0)
                for record in build_records)
        ),
        "load_s": float(sum(record.get("total_s", 0.0) for record in load_records)),
        "cache_payload_bytes": int(
            sum((record.get("load") or {}).get("payload_bytes", 0)
                for record in load_records)
        ),
        "build_cache_payload_bytes": int(
            sum((record.get("compute") or {}).get("payload_bytes", 0)
                for record in build_records)
        ),
        "build_count": len(build_records),
        "load_count": len(load_records),
    }


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(
        getattr(device, "double_fp_config", 0)
    )


def _select_opencl_device(cl, backend: str):
    backend = backend.lower()
    platforms = cl.get_platforms()

    if backend == "pocl-cpu":
        for platform in platforms:
            if "portable computing language" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("PoCL CPU device with fp64 support not found")

    if backend == "cuda-gpu":
        for platform in platforms:
            if "nvidia cuda" in platform.name.lower():
                for dev in platform.get_devices():
                    if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                        return dev
        raise RuntimeError("NVIDIA CUDA GPU device with fp64 support not found")

    if backend != "auto":
        raise ValueError("backend must be one of: auto, pocl-cpu, cuda-gpu")

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.GPU and _device_supports_fp64(dev):
                return dev

    for platform in platforms:
        for dev in platform.get_devices():
            if dev.type & cl.device_type.CPU and _device_supports_fp64(dev):
                return dev

    raise RuntimeError("No OpenCL GPU/CPU device with fp64 support found")


def _build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
    )


def _yukawa_reference_build_config(q_order: int, *, high_accuracy: bool):
    if not high_accuracy:
        return _build_config(q_order)

    from volumential.nearfield_potential_table import DuffyBuildConfig

    # The default scalar Duffy rule under-resolves the 2D Yukawa logarithmic
    # singularity and can hide split-order convergence behind table noise.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(80, 20 * q_order),
        radial_quad_order=max(320, 80 * q_order),
    )


def _split_channel_build_config(q_order: int, *, high_accuracy: bool):
    if not high_accuracy:
        return _build_config(q_order)

    from volumential.nearfield_potential_table import DuffyBuildConfig

    # Auto-built log-power channels inherit the canonical Laplace table's rule.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(32, 12 * q_order),
        radial_quad_order=max(80, 40 * q_order),
    )


def _split_smooth_quad_order(
        q_order: int, split_order: int, *, high_accuracy: bool):
    if split_order <= 1:
        return q_order
    return (2 if high_accuracy else 1) * q_order


def _get_laplace_2d_table(
    queue,
    cache_path: Path,
    q_order: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
        )
    return table


def _get_yukawa_2d_table(
    queue,
    cache_path: Path,
    q_order: int,
    lam: float,
    level: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
            lam=float(lam),
        )
    return table


def _build_helmholtz_2d_table(
    queue,
    cache_path: Path,
    q_order: int,
    wave_number: float,
    level: int,
    *,
    force_recompute: bool = False,
    build_config=None,
):
    from sumpy.kernel import HelmholtzKernel
    from volumential.table_manager import NearFieldInteractionTableManager

    kernel = HelmholtzKernel(2)
    kernel_kwargs = {kernel.helmholtz_k_name: float(wave_number)}
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, dtype=np.complex128, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Helmholtz-Reference",
            q_order,
            source_box_level=int(level),
            force_recompute=force_recompute,
            queue=queue,
            build_config=(
                _build_config(q_order) if build_config is None else build_config
            ),
            sumpy_knl=kernel,
            **kernel_kwargs,
        )
    return table


def _build_geometry(ctx, queue, q_order: int, nlevels: int):
    import volumential.meshgen as mg

    mesh = mg.MeshGen2D(q_order, nlevels, -0.5, 0.5, queue=queue)
    return mg.build_geometry_info(
        ctx,
        queue,
        2,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 2, dtype=np.float64),
    )


def _coords_host(queue, q_points):
    return np.array([axis.get(queue) for axis in q_points])


def _gaussian_source_host(coords):
    x = coords[0]
    y = coords[1]
    return np.exp(-35.0 * ((x + 0.11) ** 2 + (y - 0.07) ** 2))


def _helmholtz_manufactured_source_and_exact(coords, wave_number: float):
    alpha = 80.0
    r2 = coords[0] * coords[0] + coords[1] * coords[1]
    exact = np.exp(-alpha * r2)
    source = (4 * alpha - 4 * alpha * alpha * r2 - wave_number * wave_number) * exact
    return source, exact


def _source_values(queue, q_points, dtype, source_values_host=None):
    import pyopencl.array as cla

    if source_values_host is None:
        source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
    return cla.to_device(
        queue, np.ascontiguousarray(source_values_host.astype(dtype))
    )


def _build_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    table,
    source_weights,
    q_points,
    source_values_host=None,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_auto_config=None,
    split_smooth_quad_order: int | None = None,
):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel, YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    if kernel == "Helmholtz":
        out_kernel = HelmholtzKernel(2)
        kernel_kwargs = {out_kernel.helmholtz_k_name: float(parameter)}
        dtype = np.complex128
    elif kernel == "Yukawa":
        out_kernel = YukawaKernel(2)
        kernel_kwargs = {out_kernel.yukawa_lambda_name: float(parameter)}
        # The shared split evaluator currently emits complex intermediates for
        # both kernel families. Keep direct and split paths on the same dtype.
        dtype = np.complex128
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    source_vals = _source_values(queue, q_points, dtype, source_values_host)
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
        helmholtz_split_smooth_quad_order=split_smooth_quad_order,
        helmholtz_split_auto_config=split_auto_config,
        helmholtz_split_term_tables=split_term_tables,
    )

    return wrangler, weighted_sources, source_vals


def _time_repeated(queue, repeat_count: int, operation):
    _ = operation()
    queue.finish()
    start = time.perf_counter()
    result = None
    for _ in range(repeat_count):
        result = operation()
    queue.finish()
    total_s = time.perf_counter() - start
    return result, total_s, total_s / repeat_count


def _run_path(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    table,
    source_weights,
    q_points,
    source_values_host=None,
    split: bool,
    split_order: int,
    split_term_tables=None,
    split_auto_config=None,
    split_smooth_quad_order: int | None = None,
    repeat_count: int,
):
    from volumential.volume_fmm import drive_volume_fmm

    wrangler, weighted_sources, source_vals = _build_path(
        ctx=ctx,
        queue=queue,
        traversal=traversal,
        q_order=q_order,
        fmm_order=fmm_order,
        kernel=kernel,
        parameter=parameter,
        table=table,
        source_weights=source_weights,
        q_points=q_points,
        source_values_host=source_values_host,
        split=split,
        split_order=split_order,
        split_term_tables=split_term_tables,
        split_auto_config=split_auto_config,
        split_smooth_quad_order=split_smooth_quad_order,
    )

    def solve():
        return drive_volume_fmm(
            traversal,
            wrangler,
            weighted_sources,
            source_vals,
            direct_evaluation=False,
            list1_only=False,
        )

    potentials, solve_total_s, solve_mean_s = _time_repeated(
        queue, repeat_count, solve
    )
    (potential,) = potentials

    reordered_source_vals = wrangler.reorder_sources(source_vals)
    reordered_weighted_sources = wrangler.reorder_sources(weighted_sources)

    def table_apply():
        values, _ = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            reordered_source_vals,
        )
        return values

    _, table_apply_total_s, table_apply_mean_s = _time_repeated(
        queue, repeat_count, table_apply
    )

    timing = {
        "solve_total_s": solve_total_s,
        "solve_mean_s": solve_mean_s,
        "table_apply_total_s": table_apply_total_s,
        "table_apply_mean_s": table_apply_mean_s,
        "split_full_apply_total_s": 0.0,
        "split_full_apply_mean_s": 0.0,
        "split_correction_total_s": 0.0,
        "split_correction_mean_s": 0.0,
        "split_correction_time_kind": "not_applicable",
        "smooth_residual_total_s": 0.0,
        "smooth_residual_mean_s": 0.0,
        "smooth_residual_time_kind": "not_applicable",
        "coefficient_eval_mean_s": 0.0,
        "coefficient_eval_time_kind": "not_applicable_no_retained_channels",
    }

    if split:
        correction_args = (
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            reordered_weighted_sources,
        )

        def split_correction():
            correction, _ = wrangler.eval_direct_helmholtz_split_correction(
                *correction_args,
                src_func=reordered_source_vals,
            )
            return correction

        def split_full_apply():
            base_values = table_apply()
            correction = split_correction()
            return base_values[0] + correction[0]

        _, split_full_total_s, split_full_mean_s = _time_repeated(
            queue, repeat_count, split_full_apply
        )
        _, correction_total_s, correction_mean_s = _time_repeated(
            queue, repeat_count, split_correction
        )

        method_name = "_helmholtz_split_extra_terms"
        setattr(wrangler, method_name, MethodType(lambda self: [], wrangler))
        try:
            _, residual_total_s, residual_mean_s = _time_repeated(
                queue, repeat_count, split_correction
            )
        finally:
            delattr(wrangler, method_name)

        timing.update(
            {
                "split_full_apply_total_s": split_full_total_s,
                "split_full_apply_mean_s": split_full_mean_s,
                "split_correction_total_s": correction_total_s,
                "split_correction_mean_s": correction_mean_s,
                "split_correction_time_kind": "isolated_list1_split_correction",
                "smooth_residual_total_s": residual_total_s,
                "smooth_residual_mean_s": residual_mean_s,
                "smooth_residual_time_kind": (
                    "isolated_implemented_residual_correction_without_"
                    "retained_channels"
                ),
            }
        )

        if split_order > 1:
            coefficient_repeats = 1000
            _ = wrangler._helmholtz_split_extra_terms()
            start = time.perf_counter()
            for _ in range(coefficient_repeats):
                _ = wrangler._helmholtz_split_extra_terms()
            coefficient_total_s = time.perf_counter() - start
            timing["coefficient_eval_mean_s"] = (
                coefficient_total_s / coefficient_repeats
            )
            timing["coefficient_eval_time_kind"] = (
                "isolated_warm_python_scalar_coefficients"
            )

    return potential.get(queue), timing, wrangler


def _get_direct_table(
    *,
    kernel: str,
    queue,
    cache_path: Path,
    q_order: int,
    parameter: float,
    level: int,
    build_config=None,
):
    if kernel == "Helmholtz":
        return _build_helmholtz_2d_table(
            queue,
            cache_path,
            q_order,
            parameter,
            level,
            build_config=build_config,
        )
    if kernel == "Yukawa":
        return _get_yukawa_2d_table(
            queue,
            cache_path,
            q_order,
            parameter,
            level,
            build_config=build_config,
        )
    raise ValueError(f"unknown kernel: {kernel}")


def _prepare_direct_tables(
    *,
    kernel: str,
    queue,
    cache_dir: Path,
    q_order: int,
    parameter: float,
    direct_levels: list[int],
    active_level: int,
    build_config=None,
):
    parameter_tag = f"{parameter:.17g}".replace("-", "m").replace(".", "p")
    cache_path = cache_dir / (
        f"cost-direct-{kernel.lower()}-parameter{parameter_tag}-q{q_order}.sqlite"
    )
    _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        for level in direct_levels:
            _get_direct_table(
                kernel=kernel,
                queue=queue,
                cache_path=cache_path,
                q_order=q_order,
                parameter=parameter,
                level=level,
                build_config=build_config,
            )

    warm_tables = {}
    with _capture_table_get_timings() as warm_records:
        for level in direct_levels:
            warm_tables[level] = _get_direct_table(
                kernel=kernel,
                queue=queue,
                cache_path=cache_path,
                q_order=q_order,
                parameter=parameter,
                level=level,
                build_config=build_config,
            )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if cold["build_count"] != len(direct_levels):
        raise RuntimeError("direct-table cold pass did not build every requested level")
    if warm["load_count"] != len(direct_levels) or warm["build_count"]:
        raise RuntimeError("direct-table warm pass was not a pure cache load")
    return warm_tables[active_level], {
        "build_s": cold["build_s"],
        "quadrature_build_s": cold["quadrature_build_s"],
        "load_s": warm["load_s"],
        "cache_payload_bytes": warm["cache_payload_bytes"],
        "table_count": len(warm_tables),
        "payload_bytes": sum(_table_payload_bytes(table)
                             for table in warm_tables.values()),
    }


def _prepare_rke_channels(
    *,
    ctx,
    queue,
    traversal,
    q_order: int,
    fmm_order: int,
    kernel: str,
    parameter: float,
    split_order: int,
    source_weights,
    q_points,
    source_values_host,
    cache_dir: Path,
    split_auto_config=None,
    build_config=None,
    split_smooth_quad_order: int | None = None,
    cache_path: Path | None = None,
    clear_cache: bool = True,
):
    if cache_path is None:
        cache_path = cache_dir / (
            f"cost-rke-{kernel.lower()}-q{q_order}-p{split_order}.sqlite"
        )
    if clear_cache:
        _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        cold_base_table = _get_laplace_2d_table(
            queue, cache_path, q_order, build_config=build_config
        )
        _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=cold_base_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_auto_config=split_auto_config,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    with _capture_table_get_timings() as warm_records:
        warm_base_table = _get_laplace_2d_table(
            queue, cache_path, q_order, build_config=build_config
        )
        warm_wrangler, _, _ = _build_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=warm_base_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=True,
            split_order=split_order,
            split_auto_config=split_auto_config,
            split_smooth_quad_order=split_smooth_quad_order,
        )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if cold["build_count"] < 1:
        raise RuntimeError("RKE cold pass did not build a new table")
    if cold["build_count"] + cold["load_count"] != warm["load_count"]:
        raise RuntimeError("RKE cold and warm channel request counts differ")
    if warm["build_count"]:
        raise RuntimeError("RKE warm pass unexpectedly rebuilt a channel table")
    return (
        warm_base_table,
        dict(warm_wrangler.helmholtz_split_term_tables),
        {
            "build_s": cold["build_s"],
            "quadrature_build_s": cold["quadrature_build_s"],
            "load_s": warm["load_s"],
            "cache_payload_bytes": warm["cache_payload_bytes"],
        },
    )


# {{{ windowed-assembled table-provisioning strategy (E1)

def _parse_windowed_thetas(raw: str | None, mode: str) -> list[float]:
    if raw is None:
        defaults = (
            DEFAULT_SMOKE_WINDOWED_THETAS
            if mode == "smoke"
            else DEFAULT_FULL_WINDOWED_THETAS
        )
        return list(defaults)
    if raw.strip().lower() in {"", "none", "skip"}:
        return []
    values = _parse_csv_floats(raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("windowed thetas must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("windowed thetas must be unique")
    return values


def _prepare_windowed_family(
    *,
    cache_path: Path,
    q_order: int,
    source_box_level: int,
    window_theta: float,
    p_star: int,
    chan_regular_order: int,
    chan_radial_order: int,
    root_extent: float = TABLE_ROOT_EXTENT,
) -> dict[str, Any]:
    """Build or reload the parameter-independent windowed channel family."""
    from volumential.rke_table_assembly import get_windowed_channel_table

    was_cold = False
    start = time.perf_counter()
    for m in range(p_star):
        channel = get_windowed_channel_table(
            cache_path,
            2,
            q_order,
            m,
            source_box_level=source_box_level,
            root_extent=root_extent,
            window_theta=window_theta,
            chan_regular_order=chan_regular_order,
            chan_radial_order=chan_radial_order,
        )
        disposition = getattr(channel, "_windowed_cache_disposition", None)
        if disposition not in ("hit", "rebuilt"):
            raise RuntimeError(
                "windowed channel table did not report a valid cache "
                "disposition"
            )
        was_cold = was_cold or disposition == "rebuilt"
    return {
        "build_s": time.perf_counter() - start,
        "was_cold": was_cold,
    }


def _classical_certificate_probe(
    *,
    queue,
    cache_path: Path,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    tolerance: float,
    probe_kind: str,
) -> dict[str, Any]:
    """Certificate status of the polynomial-completion assembler at this
    parameter: ``certified`` / ``refused`` / ``failed`` (plus ``skipped``).

    ``probe_kind == "truncation"`` checks only the (cheap) series-tail
    certificate; ``"full"`` runs the complete certified assembly, exposing
    both refusal modes (term budget and recombination conditioning).
    """
    empty = {
        "kind": probe_kind,
        "status": "skipped",
        "detail": "",
        "n_terms": "",
        "condition_number": "",
        "probe_s": "",
    }
    if probe_kind == "off":
        return empty

    from volumential.rke_table_assembly import (
        RKEConditioningError,
        RKETruncationError,
        assemble_parameterized_table,
        choose_truncation_order,
    )

    start = time.perf_counter()
    if probe_kind == "truncation":
        box_extent = TABLE_ROOT_EXTENT * 0.5**source_box_level
        radius = 3.0 * math.sqrt(2.0) * box_extent
        k = (
            complex(parameter)
            if kernel == "Helmholtz"
            else complex(1j * parameter)
        )
        try:
            n_terms, _ = choose_truncation_order(2, k, radius, tolerance)
        except RKETruncationError as exc:
            return {
                **empty,
                "status": "refused",
                "detail": f"{type(exc).__name__}: {exc}",
                "probe_s": time.perf_counter() - start,
            }
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            return {
                **empty,
                "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "probe_s": time.perf_counter() - start,
            }
        return {
            **empty,
            "status": "certified-truncation-only",
            "n_terms": int(n_terms),
            "probe_s": time.perf_counter() - start,
        }

    if probe_kind != "full":
        raise ValueError(f"unknown classical probe kind: {probe_kind}")
    try:
        _, certificate = assemble_parameterized_table(
            queue,
            cache_path,
            2,
            kernel,
            q_order,
            parameter,
            source_box_level=source_box_level,
            root_extent=TABLE_ROOT_EXTENT,
            tolerance=tolerance,
        )
    except (RKETruncationError, RKEConditioningError) as exc:
        return {
            **empty,
            "status": "refused",
            "detail": f"{type(exc).__name__}: {exc}",
            "probe_s": time.perf_counter() - start,
        }
    except Exception as exc:  # noqa: BLE001 - recorded for the taxonomy
        return {
            **empty,
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "probe_s": time.perf_counter() - start,
        }
    return {
        **empty,
        "status": "certified",
        "n_terms": int(certificate["n_series_terms"]),
        "condition_number": float(certificate["condition_number"]),
        "probe_s": time.perf_counter() - start,
    }


def _register_and_load_windowed_table(
    *,
    queue,
    cache_path: Path,
    kernel: str,
    q_order: int,
    parameter: float,
    source_box_level: int,
    table,
    certificate: dict[str, Any],
    root_extent: float = TABLE_ROOT_EXTENT,
) -> tuple[Any, dict[str, Any]]:
    """Register the assembled table under the standard cache slot, then load
    it back through the ordinary ``get_table`` path (asserting a pure cache
    load), so the evaluator consumes it exactly like a direct-built table."""
    from volumential.table_manager import NearFieldInteractionTableManager

    manager_kwargs: dict[str, Any] = {}
    get_kwargs: dict[str, Any] = {}
    if kernel == "Helmholtz":
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(2)
        manager_kwargs["dtype"] = np.complex128
        get_kwargs["sumpy_knl"] = knl
        get_kwargs[knl.helmholtz_k_name] = float(parameter)
        kernel_request = "Helmholtz-Reference"
    elif kernel == "Yukawa":
        get_kwargs["lam"] = float(parameter)
        kernel_request = "Yukawa"
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    _clear_sqlite_cache(cache_path)
    provenance = {
        "kind": "windowed_rke_assembly",
        "window_theta": float(certificate["window_theta"]),
        "p_star": int(certificate["p_star"]),
        "smooth_quad_order": int(certificate["smooth_quad_order"]),
        "condition_number": float(certificate["condition_number"]),
    }
    register_start = time.perf_counter()
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=root_extent, queue=queue,
        **manager_kwargs,
    ) as table_manager:
        table_manager.register_external_table(
            2,
            kernel_request,
            q_order,
            table,
            source_box_level=source_box_level,
            provenance=provenance,
            **get_kwargs,
        )
        register_payload_bytes = int(
            table_manager.last_register_timings["payload_bytes"]
        )
    register_s = time.perf_counter() - register_start

    with _capture_table_get_timings() as load_records:
        with NearFieldInteractionTableManager(
            str(cache_path), root_extent=root_extent, queue=queue,
            **manager_kwargs,
        ) as table_manager:
            loaded_table, is_recomputed = table_manager.get_table(
                2,
                kernel_request,
                q_order,
                source_box_level=source_box_level,
                queue=queue,
                **get_kwargs,
            )
    if is_recomputed:
        raise RuntimeError(
            "registered windowed table did not load as a pure cache hit"
        )
    load_summary = _summarize_table_get_timings(load_records)
    if load_summary["build_count"] or load_summary["load_count"] != 1:
        raise RuntimeError(
            "registered windowed table load pass was not a single pure load"
        )
    return loaded_table, {
        "register_s": register_s,
        "register_payload_bytes": register_payload_bytes,
        "load_s": load_summary["load_s"],
        "load_payload_bytes": load_summary["cache_payload_bytes"],
    }


def _windowed_row_base(
    *,
    mode: str,
    kernel: str,
    parameter_name: str,
    parameter: float,
    theta: float,
    window_theta: float,
    p_star: int,
    chan_orders: tuple[int, int],
    direct_build_config,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    repeat_count: int,
    classical_probe: dict[str, Any],
) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "case_id": (
                f"{kernel.lower()}2d-{parameter_name}{parameter:g}"
                f"-windowed-theta{theta:g}"
            ),
            "mode": mode,
            "kernel": kernel,
            "dim": 2,
            "parameter_name": parameter_name,
            "parameter_value": parameter,
            "direct_regular_quad_order": direct_build_config.regular_quad_order,
            "direct_radial_quad_order": direct_build_config.radial_quad_order,
            "q_order": q_order,
            "nlevels": nlevels,
            "fmm_order": fmm_order,
            "reference_path": "direct_fixed_parameter_table",
            "repeat_count": repeat_count,
            "table_strategy": "windowed_assembled",
            "theta": theta,
            "window_theta": window_theta,
            "windowed_p_star": p_star,
            "windowed_chan_regular_order": chan_orders[0],
            "windowed_chan_radial_order": chan_orders[1],
            "classical_probe_kind": classical_probe["kind"],
            "classical_probe_status": classical_probe["status"],
            "classical_probe_detail": classical_probe["detail"],
            "classical_probe_n_terms": classical_probe["n_terms"],
            "classical_probe_condition_number": classical_probe[
                "condition_number"
            ],
            "classical_probe_s": classical_probe["probe_s"],
        }
    )
    return row


def _run_windowed_strategy(
    *,
    mode: str,
    ctx,
    queue,
    traversal,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    kernel: str,
    parameter_name: str,
    thetas: list[float],
    window_theta: float,
    p_star: int,
    chan_orders: tuple[int, int],
    classical_probe_kind: str,
    classical_probe_tolerance: float,
    direct_build_config,
    repeat_count: int,
    source_weights,
    q_points,
    coords_host,
) -> list[dict[str, Any]]:
    from volumential.rke_table_assembly import (
        RKEWindowConditioningError,
        RKEWindowCoverageError,
        assemble_windowed_parameterized_table,
    )

    box_extent = TABLE_ROOT_EXTENT * 0.5**nlevels
    family_cache = cache_dir / (
        f"windowed-channels-q{q_order}-l{nlevels}-Theta{window_theta:g}.sqlite"
    )
    classical_cache = cache_dir / f"classical-probe-q{q_order}-l{nlevels}.sqlite"

    family = _prepare_windowed_family(
        cache_path=family_cache,
        q_order=q_order,
        source_box_level=nlevels,
        window_theta=window_theta,
        p_star=p_star,
        chan_regular_order=chan_orders[0],
        chan_radial_order=chan_orders[1],
    )

    rows: list[dict[str, Any]] = []
    for theta in thetas:
        parameter = theta / box_extent
        classical_probe = _classical_certificate_probe(
            queue=queue,
            cache_path=classical_cache,
            kernel=kernel,
            q_order=q_order,
            parameter=parameter,
            source_box_level=nlevels,
            tolerance=classical_probe_tolerance,
            probe_kind=classical_probe_kind,
        )
        row = _windowed_row_base(
            mode=mode,
            kernel=kernel,
            parameter_name=parameter_name,
            parameter=parameter,
            theta=theta,
            window_theta=window_theta,
            p_star=p_star,
            chan_orders=chan_orders,
            direct_build_config=direct_build_config,
            q_order=q_order,
            nlevels=nlevels,
            fmm_order=fmm_order,
            repeat_count=repeat_count,
            classical_probe=classical_probe,
        )
        row["windowed_channel_build_s"] = family["build_s"]
        row["windowed_channel_build_was_cold"] = int(family["was_cold"])

        assemble_start = time.perf_counter()
        try:
            assembled_table, certificate = (
                assemble_windowed_parameterized_table(
                    family_cache,
                    2,
                    kernel,
                    q_order,
                    parameter,
                    source_box_level=nlevels,
                    root_extent=TABLE_ROOT_EXTENT,
                    window_theta=window_theta,
                    p_star=p_star,
                    chan_regular_order=chan_orders[0],
                    chan_radial_order=chan_orders[1],
                )
            )
        except (RKEWindowCoverageError, RKEWindowConditioningError) as exc:
            row["windowed_status"] = "refused"
            row["windowed_refusal"] = f"{type(exc).__name__}: {exc}"
            row["windowed_assemble_s"] = time.perf_counter() - assemble_start
            rows.append(row)
            continue
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            row["windowed_status"] = "failed"
            row["windowed_refusal"] = f"{type(exc).__name__}: {exc}"
            row["windowed_assemble_s"] = time.perf_counter() - assemble_start
            rows.append(row)
            continue
        row["windowed_assemble_s"] = time.perf_counter() - assemble_start
        row["windowed_status"] = "ok"
        row["windowed_condition_number"] = float(
            certificate["condition_number"]
        )
        row["windowed_smooth_quad_order"] = int(
            certificate["smooth_quad_order"]
        )

        parameter_tag = (
            f"{parameter:.17g}".replace("-", "m").replace(".", "p")
        )
        registered_cache = cache_dir / (
            f"windowed-registered-{kernel.lower()}-parameter{parameter_tag}"
            f"-q{q_order}.sqlite"
        )
        loaded_table, transfer = _register_and_load_windowed_table(
            queue=queue,
            cache_path=registered_cache,
            kernel=kernel,
            q_order=q_order,
            parameter=parameter,
            source_box_level=nlevels,
            table=assembled_table,
            certificate=certificate,
        )
        row["windowed_register_s"] = transfer["register_s"]
        row["windowed_register_payload_bytes"] = transfer[
            "register_payload_bytes"
        ]
        row["windowed_table_load_s"] = transfer["load_s"]
        row["windowed_table_load_payload_bytes"] = transfer[
            "load_payload_bytes"
        ]

        if kernel == "Helmholtz":
            source_values_host, _exact = (
                _helmholtz_manufactured_source_and_exact(
                    coords_host, parameter
                )
            )
        else:
            source_values_host = _gaussian_source_host(coords_host)

        direct_table, direct_costs = _prepare_direct_tables(
            kernel=kernel,
            queue=queue,
            cache_dir=cache_dir,
            q_order=q_order,
            parameter=parameter,
            direct_levels=[nlevels],
            active_level=nlevels,
            build_config=direct_build_config,
        )
        row.update(
            {
                "level_count": 1,
                "direct_levels": str(nlevels),
                "direct_table_count": direct_costs["table_count"],
                "direct_table_payload_bytes": direct_costs["payload_bytes"],
                "direct_table_cache_payload_bytes": direct_costs[
                    "cache_payload_bytes"
                ],
                "direct_table_build_s": direct_costs["build_s"],
                "direct_table_quadrature_build_s": direct_costs[
                    "quadrature_build_s"
                ],
                "direct_table_load_s": direct_costs["load_s"],
            }
        )

        reference_values, reference_timing, _ = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=direct_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
            repeat_count=repeat_count,
        )
        windowed_values, windowed_timing, _ = _run_path(
            ctx=ctx,
            queue=queue,
            traversal=traversal,
            q_order=q_order,
            fmm_order=fmm_order,
            kernel=kernel,
            parameter=parameter,
            table=loaded_table,
            source_weights=source_weights,
            q_points=q_points,
            source_values_host=source_values_host,
            split=False,
            split_order=1,
            repeat_count=repeat_count,
        )

        diff = windowed_values - reference_values
        reference_norm = max(
            float(np.linalg.norm(reference_values)), 1.0e-300
        )
        row.update(
            {
                "n_targets": int(reference_values.size),
                "rel_l2_error": float(np.linalg.norm(diff) / reference_norm),
                "linf_error": float(np.max(np.abs(diff))),
                "reference_warm_s": reference_timing["solve_mean_s"],
                "direct_table_apply_total_s": reference_timing[
                    "table_apply_total_s"
                ],
                "direct_table_apply_mean_s": reference_timing[
                    "table_apply_mean_s"
                ],
                "reference_solve_total_s": reference_timing["solve_total_s"],
                "windowed_solve_warm_s": windowed_timing["solve_mean_s"],
                "windowed_table_apply_mean_s": windowed_timing[
                    "table_apply_mean_s"
                ],
            }
        )
        rows.append(row)
    return rows


def _validate_windowed_rows(rows: list[dict[str, Any]]) -> None:
    """Taxonomy and agreement gates for windowed-assembled strategy rows.

    A ``failed`` row, or a certificate refusal at a theta the declaration
    covers, is a driver failure.  At small theta the windowed-assembled
    evaluator path must agree with the direct fixed-parameter reference.
    """
    for row in rows:
        if row.get("table_strategy") != "windowed_assembled":
            continue
        theta = float(row["theta"])
        window_theta = float(row["window_theta"])
        status = row["windowed_status"]
        if status == "failed":
            raise RuntimeError(
                f"windowed assembly failed for {row['case_id']}: "
                f"{row['windowed_refusal']}"
            )
        if row["classical_probe_status"] == "failed":
            raise RuntimeError(
                f"classical certificate probe failed for {row['case_id']}: "
                f"{row['classical_probe_detail']}"
            )
        if status == "refused" and theta <= window_theta * (1.0 + 1.0e-9):
            raise RuntimeError(
                f"windowed assembly refused inside the declaration for "
                f"{row['case_id']} (theta={theta:g} <= Theta="
                f"{window_theta:g}): {row['windowed_refusal']}"
            )
        if (
            status == "ok"
            and theta <= WINDOWED_SMALL_THETA_MAX
        ):
            gate = WINDOWED_SMALL_THETA_AGREEMENT[row["mode"]]
            rel_l2 = float(row["rel_l2_error"])
            if not rel_l2 <= gate:
                raise RuntimeError(
                    "windowed-assembled evaluator path disagrees with the "
                    f"direct reference at small theta for {row['case_id']}: "
                    f"rel_l2={rel_l2:.3e} > {gate:.1e}"
                )

# }}}


def _positive_root(value: float) -> float | str:
    if math.isfinite(value) and value > 0.0:
        return float(value)
    return ""


def _amortization_accounting(
    *,
    parameter_count: int,
    level_count: int,
    repeat_count: int,
    direct_build_s: float,
    direct_load_s: float,
    rke_build_s: float,
    rke_load_s: float,
    direct_solve_total_s: float,
    split_solve_total_s: float,
) -> dict[str, Any]:
    solve_count = parameter_count * repeat_count
    direct_apply_mean_s = direct_solve_total_s / solve_count
    split_apply_mean_s = split_solve_total_s / solve_count
    direct_build_per_parameter_level_s = direct_build_s / (
        parameter_count * level_count
    )

    direct_cold_total_s = direct_build_s + direct_solve_total_s
    rke_cold_total_s = rke_build_s + split_solve_total_s
    direct_warm_total_s = direct_load_s + direct_solve_total_s
    rke_warm_total_s = rke_load_s + split_solve_total_s

    parameter_denominator = (
        level_count * direct_build_per_parameter_level_s
        + repeat_count * (direct_apply_mean_s - split_apply_mean_s)
    )
    level_root = (
        rke_build_s
        + parameter_count * repeat_count
        * (split_apply_mean_s - direct_apply_mean_s)
    ) / (parameter_count * direct_build_per_parameter_level_s)
    repeat_denominator = parameter_count * (
        direct_apply_mean_s - split_apply_mean_s
    )

    if parameter_denominator == 0.0:
        parameter_root = math.inf
    else:
        parameter_root = rke_build_s / parameter_denominator
    if repeat_denominator == 0.0:
        repeat_root = math.inf
    else:
        repeat_root = (
            rke_build_s
            - parameter_count * level_count
            * direct_build_per_parameter_level_s
        ) / repeat_denominator

    return {
        "solve_count": solve_count,
        "direct_strategy_cold_total_s": direct_cold_total_s,
        "rke_strategy_cold_total_s": rke_cold_total_s,
        "direct_strategy_warm_total_s": direct_warm_total_s,
        "rke_strategy_warm_total_s": rke_warm_total_s,
        "direct_amortized_cold_s_per_solve": direct_cold_total_s / solve_count,
        "rke_amortized_cold_s_per_solve": rke_cold_total_s / solve_count,
        "direct_amortized_warm_s_per_solve": direct_warm_total_s / solve_count,
        "rke_amortized_warm_s_per_solve": rke_warm_total_s / solve_count,
        "modeled_cold_savings_s": direct_cold_total_s - rke_cold_total_s,
        "direct_table_build_mean_s_per_parameter_level": (
            direct_build_per_parameter_level_s
        ),
        "direct_solve_mean_s_per_solve": direct_apply_mean_s,
        "split_solve_mean_s_per_solve": split_apply_mean_s,
        "amortization_time_kind": (
            "measured_table_setup_plus_warm_full_solve;"
            "linear_projection_uses_observed_mean_costs"
        ),
        "break_even_parameter_count": _positive_root(parameter_root),
        "break_even_level_count": _positive_root(level_root),
        "break_even_repeat_count": _positive_root(repeat_root),
        "break_even_time_kind": (
            "positive_root_of_cold_setup_plus_warm_full_solve_linear_model;"
            "blank_means_no_positive_finite_root"
        ),
    }


def _split_term_keys(accounting) -> str:
    return ";".join(f"{kind}:{power}" for kind, power in accounting.split_term_keys)


def _validate_yukawa_order_convergence(rows: list[dict[str, Any]]) -> None:
    errors_by_parameter: dict[float, dict[int, float]] = {}
    for row in rows:
        if row["mode"] != "full" or row["kernel"] != "Yukawa":
            continue
        if row.get("table_strategy", "online_split") != "online_split":
            continue
        errors_by_parameter.setdefault(float(row["parameter_value"]), {})[
            int(row["split_order"])
        ] = float(row["rel_l2_error"])

    for parameter, errors in errors_by_parameter.items():
        if 1 in errors and 2 in errors and errors[2] > 1.0e-3 * errors[1]:
            raise RuntimeError(
                "full 2D Yukawa RKE p=2 error did not improve by three "
                f"orders of magnitude at lambda={parameter:g}: "
                f"p=1 gives {errors[1]:.3e}, p=2 gives {errors[2]:.3e}"
            )
        if 2 in errors and 3 in errors and errors[3] > 1.1 * errors[2]:
            raise RuntimeError(
                "full 2D Yukawa RKE p=3 error materially degraded from p=2 at "
                f"lambda={parameter:g}: p=2 gives {errors[2]:.3e}, "
                f"p=3 gives {errors[3]:.3e}"
            )


def _row_from_result(
    *,
    mode: str,
    kernel: str,
    parameter_name: str,
    parameter: float,
    split_order: int,
    power_log_beta_mode: str,
    direct_build_config,
    rke_channel_build_config,
    split_smooth_quad_order: int | None,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    reference_path: str,
    reference_values,
    split_values,
    reference_timing: dict[str, Any],
    split_timing: dict[str, Any],
    accounting,
    direct_costs: dict[str, Any],
    rke_costs: dict[str, Any],
    amortization: dict[str, Any],
    direct_levels: list[int],
    repeat_count: int,
) -> dict[str, Any]:
    diff = split_values - reference_values
    reference_norm = max(float(np.linalg.norm(reference_values)), 1.0e-300)
    accounting_dict = asdict(accounting)
    return {
        "case_id": (
            f"{kernel.lower()}2d-{parameter_name}{parameter:g}-p{split_order}"
        ),
        "mode": mode,
        "kernel": kernel,
        "dim": 2,
        "parameter_name": parameter_name,
        "parameter_value": parameter,
        "split_order": split_order,
        "power_log_beta_mode": power_log_beta_mode,
        "direct_regular_quad_order": direct_build_config.regular_quad_order,
        "direct_radial_quad_order": direct_build_config.radial_quad_order,
        "rke_channel_regular_quad_order": (
            rke_channel_build_config.regular_quad_order
        ),
        "rke_channel_radial_quad_order": (
            rke_channel_build_config.radial_quad_order
        ),
        "split_smooth_quad_order": (
            "" if split_smooth_quad_order is None else split_smooth_quad_order
        ),
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "n_targets": int(reference_values.size),
        "reference_path": reference_path,
        "rel_l2_error": float(np.linalg.norm(diff) / reference_norm),
        "linf_error": float(np.max(np.abs(diff))),
        "reference_warm_s": reference_timing["solve_mean_s"],
        "split_warm_s": split_timing["solve_mean_s"],
        "online_remainder_s": split_timing["smooth_residual_mean_s"],
        "online_remainder_time_kind": split_timing["smooth_residual_time_kind"],
        "split_term_keys": _split_term_keys(accounting),
        **{
            key: value
            for key, value in accounting_dict.items()
            if key not in {"split_enabled", "split_order", "split_term_keys"}
        },
        "level_count": len(direct_levels),
        "direct_levels": ";".join(str(level) for level in direct_levels),
        "repeat_count": repeat_count,
        "direct_table_count": direct_costs["table_count"],
        "direct_table_payload_bytes": direct_costs["payload_bytes"],
        "direct_table_cache_payload_bytes": direct_costs["cache_payload_bytes"],
        "direct_table_build_s": direct_costs["build_s"],
        "direct_table_quadrature_build_s": direct_costs["quadrature_build_s"],
        "direct_table_load_s": direct_costs["load_s"],
        "rke_channel_build_s": rke_costs["build_s"],
        "rke_channel_quadrature_build_s": rke_costs["quadrature_build_s"],
        "rke_channel_load_s": rke_costs["load_s"],
        "rke_channel_cache_payload_bytes": rke_costs["cache_payload_bytes"],
        "direct_table_apply_total_s": reference_timing["table_apply_total_s"],
        "direct_table_apply_mean_s": reference_timing["table_apply_mean_s"],
        "split_full_apply_total_s": split_timing["split_full_apply_total_s"],
        "split_full_apply_mean_s": split_timing["split_full_apply_mean_s"],
        "split_correction_total_s": split_timing["split_correction_total_s"],
        "split_correction_mean_s": split_timing["split_correction_mean_s"],
        "split_correction_time_kind": split_timing["split_correction_time_kind"],
        "smooth_residual_total_s": split_timing["smooth_residual_total_s"],
        "smooth_residual_mean_s": split_timing["smooth_residual_mean_s"],
        "smooth_residual_time_kind": split_timing["smooth_residual_time_kind"],
        "coefficient_eval_mean_s": split_timing["coefficient_eval_mean_s"],
        "coefficient_eval_time_kind": split_timing["coefficient_eval_time_kind"],
        "reference_solve_total_s": reference_timing["solve_total_s"],
        "split_solve_total_s": split_timing["solve_total_s"],
        **amortization,
        "table_strategy": "online_split",
        "theta": parameter * TABLE_ROOT_EXTENT * 0.5**nlevels,
    }


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    split_orders: list[int],
    helmholtz_k: list[float],
    yukawa_lam: list[float],
    direct_levels: list[int],
    repeat_count: int,
    power_log_beta_mode: str = "p2p",
    windowed_thetas: list[float] | None = None,
    window_theta: float = DEFAULT_WINDOW_THETA,
    windowed_p_star: int = DEFAULT_WINDOWED_P_STAR,
    windowed_chan_orders: tuple[int, int] = DEFAULT_WINDOWED_CHAN_ORDERS_2D,
    classical_probe_kind: str | None = None,
    classical_probe_tolerance: float = CLASSICAL_PROBE_TOLERANCE,
) -> list[dict[str, Any]]:
    import pyopencl as cl

    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1")
    if len(set(split_orders)) != len(split_orders):
        raise ValueError("split_orders must be unique")
    if nlevels not in direct_levels:
        raise ValueError("direct_levels must include nlevels")
    if power_log_beta_mode not in {"p2p", "table"}:
        raise ValueError("power_log_beta_mode must be 'p2p' or 'table'")
    if windowed_thetas is None:
        windowed_thetas = []
    if classical_probe_kind is None:
        classical_probe_kind = "truncation" if mode == "smoke" else "full"
    if classical_probe_kind not in {"off", "truncation", "full"}:
        raise ValueError(
            "classical_probe_kind must be 'off', 'truncation', or 'full'"
        )
    if windowed_thetas:
        if not math.isfinite(window_theta) or window_theta <= 0.0:
            raise ValueError("window_theta must be finite and positive")
        if windowed_p_star < 1:
            raise ValueError("windowed_p_star must be >= 1")

    split_auto_config = {
        "power_log_single_table_beta_mode": power_log_beta_mode,
    }

    benchmark_start = time.perf_counter()
    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    q_points, source_weights, _tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    coords_host = _coords_host(queue, q_points)

    sweep_specs = [
        ("Helmholtz", "k", helmholtz_k),
        ("Yukawa", "lambda", yukawa_lam),
    ]

    for kernel, parameter_name, parameters in sweep_specs:
        if not parameters and not windowed_thetas:
            continue

        high_accuracy = mode == "full"
        direct_build_config = (
            _yukawa_reference_build_config(
                q_order, high_accuracy=high_accuracy
            )
            if kernel == "Yukawa"
            else _build_config(q_order)
        )
        rke_channel_build_config = (
            _split_channel_build_config(q_order, high_accuracy=high_accuracy)
            if kernel == "Yukawa"
            else _build_config(q_order)
        )
        parameter_cases = []
        direct_costs = {
            "build_s": 0.0,
            "quadrature_build_s": 0.0,
            "load_s": 0.0,
            "cache_payload_bytes": 0,
            "table_count": 0,
            "payload_bytes": 0,
        }

        for parameter in parameters:
            if kernel == "Helmholtz":
                source_values_host, _exact_values = (
                    _helmholtz_manufactured_source_and_exact(
                        coords_host, parameter
                    )
                )
            else:
                source_values_host = _gaussian_source_host(coords_host)

            direct_table, parameter_direct_costs = _prepare_direct_tables(
                kernel=kernel,
                queue=queue,
                cache_dir=cache_dir,
                q_order=q_order,
                parameter=parameter,
                direct_levels=direct_levels,
                active_level=nlevels,
                build_config=direct_build_config,
            )
            for key in direct_costs:
                direct_costs[key] += parameter_direct_costs[key]

            reference_values, reference_timing, _ = _run_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel=kernel,
                parameter=parameter,
                table=direct_table,
                source_weights=source_weights,
                q_points=q_points,
                source_values_host=source_values_host,
                split=False,
                split_order=1,
                repeat_count=repeat_count,
            )
            parameter_cases.append(
                {
                    "parameter": parameter,
                    "source_values_host": source_values_host,
                    "reference_values": reference_values,
                    "reference_timing": reference_timing,
                }
            )

        direct_solve_total_s = sum(
            case["reference_timing"]["solve_total_s"] for case in parameter_cases
        )

        # Keep one cache across orders so each pass measures only newly required
        # tables; _prepare_rke_channels otherwise defaults to a per-order file.
        rke_cache_path = cache_dir / f"cost-rke-{kernel.lower()}-q{q_order}.sqlite"
        cumulative_rke_build_s = 0.0
        cumulative_rke_quadrature_build_s = 0.0
        for split_index, split_order in enumerate(
            sorted(split_orders) if parameter_cases else []
        ):
            smooth_quad_order = _split_smooth_quad_order(
                q_order, split_order, high_accuracy=high_accuracy
            )
            representative_case = parameter_cases[0]
            split_table, split_term_tables, rke_costs = _prepare_rke_channels(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel=kernel,
                parameter=representative_case["parameter"],
                split_order=split_order,
                source_weights=source_weights,
                q_points=q_points,
                source_values_host=representative_case["source_values_host"],
                cache_dir=cache_dir,
                split_auto_config=split_auto_config,
                build_config=rke_channel_build_config,
                split_smooth_quad_order=smooth_quad_order,
                cache_path=rke_cache_path,
                clear_cache=split_index == 0,
            )
            cumulative_rke_build_s += rke_costs["build_s"]
            cumulative_rke_quadrature_build_s += rke_costs[
                "quadrature_build_s"
            ]
            rke_costs["build_s"] = cumulative_rke_build_s
            rke_costs["quadrature_build_s"] = (
                cumulative_rke_quadrature_build_s
            )

            split_results = []
            for case in parameter_cases:
                split_values, split_timing, split_wrangler = _run_path(
                    ctx=ctx,
                    queue=queue,
                    traversal=traversal,
                    q_order=q_order,
                    fmm_order=fmm_order,
                    kernel=kernel,
                    parameter=case["parameter"],
                    table=split_table,
                    source_weights=source_weights,
                    q_points=q_points,
                    source_values_host=case["source_values_host"],
                    split=True,
                    split_order=split_order,
                    split_term_tables=split_term_tables,
                    split_auto_config=split_auto_config,
                    split_smooth_quad_order=smooth_quad_order,
                    repeat_count=repeat_count,
                )
                accounting = split_wrangler.get_helmholtz_split_cache_accounting(
                    parameter_count=len(parameters)
                )
                split_results.append(
                    (case, split_values, split_timing, accounting)
                )

            split_solve_total_s = sum(
                split_timing["solve_total_s"]
                for _, _, split_timing, _ in split_results
            )
            amortization = _amortization_accounting(
                parameter_count=len(parameters),
                level_count=len(direct_levels),
                repeat_count=repeat_count,
                direct_build_s=direct_costs["build_s"],
                direct_load_s=direct_costs["load_s"],
                rke_build_s=rke_costs["build_s"],
                rke_load_s=rke_costs["load_s"],
                direct_solve_total_s=direct_solve_total_s,
                split_solve_total_s=split_solve_total_s,
            )

            for case, split_values, split_timing, accounting in split_results:
                rows.append(
                    _row_from_result(
                        mode=mode,
                        kernel=kernel,
                        parameter_name=parameter_name,
                        parameter=case["parameter"],
                        split_order=split_order,
                        power_log_beta_mode=power_log_beta_mode,
                        direct_build_config=direct_build_config,
                        rke_channel_build_config=rke_channel_build_config,
                        split_smooth_quad_order=smooth_quad_order,
                        q_order=q_order,
                        nlevels=nlevels,
                        fmm_order=fmm_order,
                        reference_path="direct_fixed_parameter_table",
                        reference_values=case["reference_values"],
                        split_values=split_values,
                        reference_timing=case["reference_timing"],
                        split_timing=split_timing,
                        accounting=accounting,
                        direct_costs=direct_costs,
                        rke_costs=rke_costs,
                        amortization=amortization,
                        direct_levels=direct_levels,
                        repeat_count=repeat_count,
                    )
                )

        if windowed_thetas:
            rows.extend(
                _run_windowed_strategy(
                    mode=mode,
                    ctx=ctx,
                    queue=queue,
                    traversal=traversal,
                    cache_dir=cache_dir,
                    q_order=q_order,
                    nlevels=nlevels,
                    fmm_order=fmm_order,
                    kernel=kernel,
                    parameter_name=parameter_name,
                    thetas=windowed_thetas,
                    window_theta=window_theta,
                    p_star=windowed_p_star,
                    chan_orders=windowed_chan_orders,
                    classical_probe_kind=classical_probe_kind,
                    classical_probe_tolerance=classical_probe_tolerance,
                    direct_build_config=direct_build_config,
                    repeat_count=repeat_count,
                    source_weights=source_weights,
                    q_points=q_points,
                    coords_host=coords_host,
                )
            )

    _validate_yukawa_order_convergence(rows)
    _validate_windowed_rows(rows)

    benchmark_total_s = time.perf_counter() - benchmark_start
    for row in rows:
        row["benchmark_total_s"] = benchmark_total_s
        row["benchmark_total_time_kind"] = (
            "driver_wall_including_all_cases_setup_and_isolated_diagnostics"
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
        "--out",
        type=Path,
        default=Path("build/benchmarks/split-parameter-sweep.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/split-parameter-cache"),
    )
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--split-orders")
    parser.add_argument(
        "--power-log-beta-mode",
        choices=("p2p", "table"),
        default="p2p",
    )
    parser.add_argument("--helmholtz-k")
    parser.add_argument("--yukawa-lambda")
    parser.add_argument(
        "--parameter-count",
        type=int,
        help="use the first COUNT values from each non-empty parameter list",
    )
    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--level-count",
        type=int,
        help="number of direct-table levels ending at --nlevels",
    )
    level_group.add_argument(
        "--direct-levels",
        help=(
            "comma-separated direct-table source-box levels; "
            "must include --nlevels"
        ),
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        help="number of timed applications per parameter",
    )
    parser.add_argument(
        "--windowed-thetas",
        help=(
            "comma-separated theta = parameter * leaf-table-extent values "
            "for the windowed-assembled strategy (E1); 'none' disables; "
            "default smoke '1,6,16', full "
            "'0.25,0.5,1,2,4,6,8,12,16'"
        ),
    )
    parser.add_argument(
        "--window-theta",
        type=float,
        default=DEFAULT_WINDOW_THETA,
        help="declared window Theta for the windowed-assembled strategy",
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
            "'regular,radial' channel quadrature orders "
            "(default 48,61: the assembler's tested 2D orders)"
        ),
    )
    parser.add_argument(
        "--classical-probe",
        choices=("off", "truncation", "full"),
        default=None,
        help=(
            "polynomial-completion certificate probe per windowed theta; "
            "default 'truncation' (cheap) in smoke, 'full' in full mode"
        ),
    )
    parser.add_argument(
        "--classical-probe-tolerance",
        type=float,
        default=CLASSICAL_PROBE_TOLERANCE,
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else (2 if smoke else 4)
    nlevels = args.nlevels if args.nlevels is not None else (2 if smoke else 3)
    fmm_order = (
        args.fmm_order if args.fmm_order is not None else (8 if smoke else 16)
    )
    split_orders = _parse_csv_ints(
        args.split_orders or ("1,2" if smoke else "1,2,3")
    )
    helmholtz_k = _parse_csv_floats(
        args.helmholtz_k
        if args.helmholtz_k is not None
        else ("4" if smoke else "4,8,12"),
        allow_empty=True,
    )
    yukawa_lam = _parse_csv_floats(
        args.yukawa_lambda
        if args.yukawa_lambda is not None
        else ("4" if smoke else "4,8,12"),
        allow_empty=True,
    )
    helmholtz_k = _limit_parameter_values(helmholtz_k, args.parameter_count)
    yukawa_lam = _limit_parameter_values(yukawa_lam, args.parameter_count)

    repeat_count = (
        args.repeat_count if args.repeat_count is not None else (1 if smoke else 5)
    )
    if repeat_count < 1:
        parser.error("--repeat-count must be >= 1")

    if args.direct_levels:
        direct_levels = _parse_csv_levels(args.direct_levels)
    else:
        level_count = (
            args.level_count
            if args.level_count is not None
            else (1 if smoke else nlevels + 1)
        )
        if not 1 <= level_count <= nlevels + 1:
            parser.error("--level-count must be between 1 and nlevels + 1")
        direct_levels = list(range(nlevels - level_count + 1, nlevels + 1))
    if nlevels not in direct_levels:
        parser.error(
            "--direct-levels must include --nlevels for the direct reference"
        )
    try:
        windowed_thetas = _parse_windowed_thetas(args.windowed_thetas, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.windowed_chan_orders is None:
        windowed_chan_orders = DEFAULT_WINDOWED_CHAN_ORDERS_2D
    else:
        parts = _parse_csv_ints(args.windowed_chan_orders)
        if len(parts) != 2:
            parser.error(
                "--windowed-chan-orders must be a 'regular,radial' pair"
            )
        windowed_chan_orders = (parts[0], parts[1])

    if not helmholtz_k and not yukawa_lam and not windowed_thetas:
        parser.error(
            "at least one of --helmholtz-k, --yukawa-lambda, or "
            "--windowed-thetas must be non-empty"
        )

    rows = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        fmm_order=fmm_order,
        split_orders=split_orders,
        helmholtz_k=helmholtz_k,
        yukawa_lam=yukawa_lam,
        direct_levels=direct_levels,
        repeat_count=repeat_count,
        power_log_beta_mode=args.power_log_beta_mode,
        windowed_thetas=windowed_thetas,
        window_theta=args.window_theta,
        windowed_p_star=args.windowed_p_star,
        windowed_chan_orders=windowed_chan_orders,
        classical_probe_kind=args.classical_probe,
        classical_probe_tolerance=args.classical_probe_tolerance,
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
