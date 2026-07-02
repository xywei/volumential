#!/usr/bin/env python3
"""Emit Helmholtz/Yukawa split-parameter sweep CSVs for Paper 1.

The benchmark sweeps the Helmholtz wave number and Yukawa screening parameter
while recording split-table accounting. For each kernel parameter, rows compare
against a fixed-parameter direct near-field table, giving an independent
reference for the split/RKE channel family.

Smoke mode is intended for CI/local validation. Full mode is intended for
metadata-wrapped runs on controlled machines such as ``ipa``.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


TABLE_ROOT_EXTENT = 2.0


FIELDS = (
    "case_id",
    "mode",
    "kernel",
    "dim",
    "parameter_name",
    "parameter_value",
    "table_root_extent",
    "source_box_level",
    "table_side_length",
    "effective_parameter",
    "resolution_metric_name",
    "resolution_metric",
    "effective_parameter_regime",
    "split_order",
    "split_smooth_quad_order",
    "q_order",
    "nlevels",
    "fmm_order",
    "n_targets",
    "reference_path",
    "reference_l2_norm",
    "diff_l2_norm",
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


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(getattr(device, "double_fp_config", 0))


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


def _table_side_length(source_box_level: int, table_root_extent: float) -> float:
    return float(table_root_extent) / float(2**int(source_box_level))


def _effective_parameter_regime(theta: float) -> str:
    if theta <= 1.0:
        return "resolved"
    if theta <= 2.0:
        return "transition"
    return "stress"


def _resolution_diagnostics(
    *,
    kernel: str,
    parameter: float,
    q_order: int,
    source_box_level: int,
    table_root_extent: float,
) -> dict[str, Any]:
    side_length = _table_side_length(source_box_level, table_root_extent)
    theta = abs(float(parameter)) * side_length
    if theta == 0.0:
        metric = math.inf
    elif kernel == "Helmholtz":
        metric = 2.0 * math.pi * float(q_order) / theta
    elif kernel == "Yukawa":
        metric = float(q_order) / theta
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    metric_name = (
        "points_per_wavelength"
        if kernel == "Helmholtz"
        else "nodes_per_decay_length"
    )
    return {
        "table_root_extent": float(table_root_extent),
        "source_box_level": int(source_box_level),
        "table_side_length": side_length,
        "effective_parameter": theta,
        "resolution_metric_name": metric_name,
        "resolution_metric": metric,
        "effective_parameter_regime": _effective_parameter_regime(theta),
    }


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

    # 2D Yukawa direct tables use the scalar Duffy path. The default order is
    # enough for smoke runs but can under-integrate the logarithmic singularity
    # in paper-facing fixed-parameter references.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(32, 12 * q_order),
        radial_quad_order=max(80, 40 * q_order),
    )


def _split_channel_build_config(q_order: int, *, high_accuracy: bool):
    if not high_accuracy:
        return _build_config(q_order)

    from volumential.nearfield_potential_table import DuffyBuildConfig

    # The Laplace table's build config is reused for auto-built log-power split
    # channel tables. Use the same paper-facing accuracy target as the 2D
    # Yukawa direct reference so the split-order trend is not table-noise-limited.
    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(32, 12 * q_order),
        radial_quad_order=max(80, 40 * q_order),
    )


def _get_laplace_2d_table(
    queue,
    cache_path: Path,
    q_order: int,
    table_root_extent: float,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=table_root_extent, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Laplace",
            q_order,
            force_recompute=False,
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
    table_root_extent: float,
    build_config=None,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=table_root_extent, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=False,
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
    table_root_extent: float,
):
    from sumpy.kernel import HelmholtzKernel
    from volumential.table_manager import NearFieldInteractionTableManager

    kernel = HelmholtzKernel(2)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=table_root_extent, dtype=np.complex128, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Helmholtz-Reference",
            q_order,
            source_box_level=int(level),
            force_recompute=False,
            queue=queue,
            build_config=_build_config(q_order),
            sumpy_knl=kernel,
            **{kernel.helmholtz_k_name: float(wave_number)},
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
    return cla.to_device(queue, np.ascontiguousarray(source_values_host.astype(dtype)))


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
    split_smooth_quad_order: int | None,
):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import HelmholtzKernel, YukawaKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )
    from volumential.volume_fmm import drive_volume_fmm

    if kernel == "Helmholtz":
        out_kernel = HelmholtzKernel(2)
        kernel_kwargs = {out_kernel.helmholtz_k_name: float(parameter)}
        dtype = np.complex128
    elif kernel == "Yukawa":
        out_kernel = YukawaKernel(2)
        kernel_kwargs = {out_kernel.yukawa_lambda_name: float(parameter)}
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
    )

    # Warm compilation/caches first, then time the second application.
    _ = drive_volume_fmm(
        traversal,
        wrangler,
        weighted_sources,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
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
    return potential.get(queue), time.perf_counter() - start, wrangler


def _split_term_keys(accounting) -> str:
    return ";".join(f"{kind}:{power}" for kind, power in accounting.split_term_keys)


def _row_from_result(
    *,
    mode: str,
    kernel: str,
    parameter_name: str,
    parameter: float,
    split_order: int,
    split_smooth_quad_order: int | None,
    table_root_extent: float,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    reference_path: str,
    reference_values,
    split_values,
    reference_warm_s: float | str,
    split_warm_s: float,
    accounting,
) -> dict[str, Any]:
    diff = split_values - reference_values
    reference_norm = max(float(np.linalg.norm(reference_values)), 1.0e-300)
    diff_norm = float(np.linalg.norm(diff))
    accounting_dict = asdict(accounting)
    resolution = _resolution_diagnostics(
        kernel=kernel,
        parameter=parameter,
        q_order=q_order,
        source_box_level=nlevels,
        table_root_extent=table_root_extent,
    )
    smooth_order_suffix = (
        "default"
        if split_smooth_quad_order is None
        else f"s{int(split_smooth_quad_order)}"
    )
    return {
        "case_id": (
            f"{kernel.lower()}2d-q{q_order}-l{nlevels}-"
            f"{parameter_name}{parameter:g}-p{split_order}-{smooth_order_suffix}"
        ),
        "mode": mode,
        "kernel": kernel,
        "dim": 2,
        "parameter_name": parameter_name,
        "parameter_value": parameter,
        **resolution,
        "split_order": split_order,
        "split_smooth_quad_order": (
            "" if split_smooth_quad_order is None else int(split_smooth_quad_order)
        ),
        "q_order": q_order,
        "nlevels": nlevels,
        "fmm_order": fmm_order,
        "n_targets": int(reference_values.size),
        "reference_path": reference_path,
        "reference_l2_norm": float(reference_norm),
        "diff_l2_norm": diff_norm,
        "rel_l2_error": float(diff_norm / reference_norm),
        "linf_error": float(np.max(np.abs(diff))),
        "reference_warm_s": reference_warm_s,
        "split_warm_s": float(split_warm_s),
        "online_remainder_s": float(split_warm_s if accounting.uses_online_remainder else 0.0),
        "online_remainder_time_kind": "warm_solve_upper_bound",
        "split_term_keys": _split_term_keys(accounting),
        **{
            key: value
            for key, value in accounting_dict.items()
            if key not in {"split_enabled", "split_order", "split_term_keys"}
        },
    }


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    fmm_order: int,
    table_root_extent: float,
    split_orders: list[int],
    split_smooth_quad_orders: list[int] | None,
    helmholtz_k: list[float],
    yukawa_lam: list[float],
) -> list[dict[str, Any]]:
    import pyopencl as cl

    device = _select_opencl_device(cl, backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    q_points, source_weights, _tree, traversal = _build_geometry(
        ctx, queue, q_order, nlevels
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_table = _get_laplace_2d_table(
        queue,
        cache_dir / f"split-laplace-q{q_order}.sqlite",
        q_order,
        table_root_extent,
        build_config=_split_channel_build_config(
            q_order,
            high_accuracy=mode == "full",
        ),
    )
    rows: list[dict[str, Any]] = []

    sweep_specs = [
        ("Helmholtz", "k", helmholtz_k),
        ("Yukawa", "lambda", yukawa_lam),
    ]
    parameter_count_by_kernel = {
        "Helmholtz": len(helmholtz_k),
        "Yukawa": len(yukawa_lam),
    }

    for kernel, parameter_name, parameters in sweep_specs:
        for parameter in parameters:
            if kernel == "Helmholtz":
                source_values_host, _exact_values = _helmholtz_manufactured_source_and_exact(
                    _coords_host(queue, q_points), parameter
                )
                direct_reference_table = _build_helmholtz_2d_table(
                    queue,
                    cache_dir
                    / f"direct-helmholtz-k{parameter:g}-q{q_order}-l{nlevels}.sqlite",
                    q_order,
                    parameter,
                    nlevels,
                    table_root_extent,
                )
            else:
                source_values_host = _gaussian_source_host(_coords_host(queue, q_points))
                direct_reference_table = _get_yukawa_2d_table(
                    queue,
                    cache_dir
                    / f"direct-yukawa-lambda{parameter:g}-q{q_order}-l{nlevels}.sqlite",
                    q_order,
                    parameter,
                    nlevels,
                    table_root_extent,
                    build_config=_yukawa_reference_build_config(
                        q_order,
                        high_accuracy=mode == "full",
                    ),
                )

            reference_values, reference_warm_s, _ = _run_path(
                ctx=ctx,
                queue=queue,
                traversal=traversal,
                q_order=q_order,
                fmm_order=fmm_order,
                kernel=kernel,
                parameter=parameter,
                table=direct_reference_table,
                source_weights=source_weights,
                q_points=q_points,
                source_values_host=source_values_host,
                split=False,
                split_order=1,
                split_smooth_quad_order=None,
            )
            reference_path = "direct_fixed_parameter_table"

            split_results = []
            for split_order in split_orders:
                smooth_orders = (
                    (split_smooth_quad_orders or [q_order])
                    if split_order > 1
                    else [None]
                )
                for split_smooth_quad_order in smooth_orders:
                    effective_smooth_order = split_smooth_quad_order
                    split_values, split_warm_s, split_wrangler = _run_path(
                        ctx=ctx,
                        queue=queue,
                        traversal=traversal,
                        q_order=q_order,
                        fmm_order=fmm_order,
                        kernel=kernel,
                        parameter=parameter,
                        table=split_table,
                        source_weights=source_weights,
                        q_points=q_points,
                        source_values_host=source_values_host,
                        split=True,
                        split_order=split_order,
                        split_smooth_quad_order=effective_smooth_order,
                    )
                    accounting = split_wrangler.get_helmholtz_split_cache_accounting(
                        parameter_count=parameter_count_by_kernel[kernel]
                    )
                    split_results.append(
                        (
                            split_order,
                            effective_smooth_order,
                            split_values,
                            split_warm_s,
                            accounting,
                        )
                    )

            for (
                split_order,
                split_smooth_quad_order,
                split_values,
                split_warm_s,
                accounting,
            ) in split_results:
                rows.append(
                    _row_from_result(
                        mode=mode,
                        kernel=kernel,
                        parameter_name=parameter_name,
                        parameter=parameter,
                        split_order=split_order,
                        split_smooth_quad_order=split_smooth_quad_order,
                        table_root_extent=table_root_extent,
                        q_order=q_order,
                        nlevels=nlevels,
                        fmm_order=fmm_order,
                        reference_path=reference_path,
                        reference_values=reference_values,
                        split_values=split_values,
                        reference_warm_s=reference_warm_s,
                        split_warm_s=split_warm_s,
                        accounting=accounting,
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
    parser.add_argument("--q-orders")
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--nlevels-list")
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--table-root-extent", type=float, default=TABLE_ROOT_EXTENT)
    parser.add_argument("--split-orders")
    parser.add_argument("--split-smooth-quad-orders")
    parser.add_argument("--helmholtz-k")
    parser.add_argument("--yukawa-lambda")
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else (2 if smoke else 4)
    nlevels = args.nlevels if args.nlevels is not None else (2 if smoke else 3)
    q_orders = _parse_csv_ints(args.q_orders) if args.q_orders else [q_order]
    nlevels_list = (
        _parse_csv_ints(args.nlevels_list) if args.nlevels_list else [nlevels]
    )
    fmm_order = args.fmm_order if args.fmm_order is not None else (8 if smoke else 16)
    split_orders = _parse_csv_ints(args.split_orders or ("1,2" if smoke else "1,2,3"))
    split_smooth_quad_orders = (
        _parse_csv_ints(args.split_smooth_quad_orders)
        if args.split_smooth_quad_orders
        else None
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

    rows = []
    for q_order_i in q_orders:
        for nlevels_i in nlevels_list:
            rows.extend(
                run_benchmark(
                    mode=args.mode,
                    backend=args.backend,
                    cache_dir=args.cache_dir,
                    q_order=q_order_i,
                    nlevels=nlevels_i,
                    fmm_order=fmm_order,
                    table_root_extent=args.table_root_extent,
                    split_orders=split_orders,
                    split_smooth_quad_orders=split_smooth_quad_orders,
                    helmholtz_k=helmholtz_k,
                    yukawa_lam=yukawa_lam,
                )
            )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
