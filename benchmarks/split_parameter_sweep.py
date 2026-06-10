#!/usr/bin/env python3
"""Emit Helmholtz/Yukawa split-parameter sweep CSVs for Paper 1.

The benchmark sweeps the Helmholtz wave number and Yukawa screening parameter
while recording split-table accounting. For each kernel parameter, rows compare
against the highest split order in the same run, giving a direct split-order
convergence measurement without changing the pretabulated basis-table family.

Smoke mode is intended for CI/local validation. Full mode is intended for
metadata-wrapped runs on controlled machines such as ``ipa``.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
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
)


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
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


def _build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
    )


def _get_laplace_2d_table(queue, cache_path: Path, q_order: int):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Laplace",
            q_order,
            force_recompute=False,
            queue=queue,
            build_config=_build_config(q_order),
        )
    return table


def _get_yukawa_2d_table(queue, cache_path: Path, q_order: int, lam: float, level: int):
    from volumential.table_manager import NearFieldInteractionTableManager

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2.0, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Yukawa",
            q_order,
            source_box_level=int(level),
            force_recompute=False,
            queue=queue,
            build_config=_build_config(q_order),
            lam=float(lam),
        )
    return table


def _build_helmholtz_2d_table(
    queue, cache_path: Path, q_order: int, wave_number: float, level: int
):
    from sumpy.kernel import HelmholtzKernel

    import volumential.nearfield_potential_table as npt

    kernel = HelmholtzKernel(2)
    kernel_kwargs = {kernel.helmholtz_k_name: float(wave_number)}
    table = npt.NearFieldInteractionTable(
        quad_order=q_order,
        dim=2,
        build_method="DuffyRadial",
        kernel_func=npt.sumpy_kernel_to_lambda(kernel, parameter_values=kernel_kwargs),
        kernel_type="helmholtz-direct",
        sumpy_kernel=kernel,
        source_box_extent=2.0 * (2 ** (-int(level))),
        dtype=np.complex128,
        progress_bar=False,
    )
    table.source_box_level = int(level)
    table.table_root_extent = 2.0
    table._table_cache_filename = str(cache_path)
    table._table_cache_root_extent = 2.0
    table.build_table_via_duffy_radial(
        queue=queue,
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
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
        dtype = np.complex128 if split else np.float64
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
    accounting_dict = asdict(accounting)
    return {
        "case_id": f"{kernel.lower()}2d-{parameter_name}{parameter:g}-p{split_order}",
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
    split_orders: list[int],
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
        queue, cache_dir / f"split-laplace-q{q_order}.sqlite", q_order
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
            else:
                source_values_host = _gaussian_source_host(_coords_host(queue, q_points))

            split_results = []
            for split_order in split_orders:
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
                )
                accounting = split_wrangler.get_helmholtz_split_cache_accounting(
                    parameter_count=parameter_count_by_kernel[kernel]
                )
                split_results.append(
                    (split_order, split_values, split_warm_s, accounting)
                )

            reference_split_order, reference_values, reference_warm_s, _ = max(
                split_results, key=lambda result: result[0]
            )
            reference_path = f"split_order_{reference_split_order}"
            for split_order, split_values, split_warm_s, accounting in split_results:
                rows.append(
                    _row_from_result(
                        mode=mode,
                        kernel=kernel,
                        parameter_name=parameter_name,
                        parameter=parameter,
                        split_order=split_order,
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
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--split-orders")
    parser.add_argument("--helmholtz-k")
    parser.add_argument("--yukawa-lambda")
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else (2 if smoke else 4)
    nlevels = args.nlevels if args.nlevels is not None else (2 if smoke else 3)
    fmm_order = args.fmm_order if args.fmm_order is not None else (8 if smoke else 16)
    split_orders = _parse_csv_ints(args.split_orders or ("1,2" if smoke else "1,2,3"))
    helmholtz_k = _parse_csv_floats(args.helmholtz_k or ("4" if smoke else "4,8,12"))
    yukawa_lam = _parse_csv_floats(args.yukawa_lambda or ("4" if smoke else "4,8,12"))

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
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
