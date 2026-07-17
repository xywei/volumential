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
)


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


def _get_laplace_2d_table(
    queue, cache_path: Path, q_order: int, *, force_recompute: bool = False
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
            build_config=_build_config(q_order),
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
            build_config=_build_config(q_order),
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
            build_config=_build_config(q_order),
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
):
    if kernel == "Helmholtz":
        return _build_helmholtz_2d_table(
            queue,
            cache_path,
            q_order,
            parameter,
            level,
        )
    if kernel == "Yukawa":
        return _get_yukawa_2d_table(
            queue,
            cache_path,
            q_order,
            parameter,
            level,
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
):
    cache_path = cache_dir / (
        f"cost-rke-{kernel.lower()}-q{q_order}-p{split_order}.sqlite"
    )
    _clear_sqlite_cache(cache_path)

    with _capture_table_get_timings() as cold_records:
        cold_base_table = _get_laplace_2d_table(queue, cache_path, q_order)
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
        )

    with _capture_table_get_timings() as warm_records:
        warm_base_table = _get_laplace_2d_table(queue, cache_path, q_order)
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
        )

    cold = _summarize_table_get_timings(cold_records)
    warm = _summarize_table_get_timings(warm_records)
    if cold["build_count"] < 1 or cold["build_count"] != warm["load_count"]:
        raise RuntimeError("RKE cold-build and warm-load channel counts differ")
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


def _row_from_result(
    *,
    mode: str,
    kernel: str,
    parameter_name: str,
    parameter: float,
    split_order: int,
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
) -> list[dict[str, Any]]:
    import pyopencl as cl

    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1")
    if nlevels not in direct_levels:
        raise ValueError("direct_levels must include nlevels")

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
        if not parameters:
            continue

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

        for split_order in split_orders:
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
    if not helmholtz_k and not yukawa_lam:
        parser.error(
            "at least one of --helmholtz-k or --yukawa-lambda must be non-empty"
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
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
