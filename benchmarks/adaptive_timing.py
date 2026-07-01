#!/usr/bin/env python3
"""Emit adaptive end-to-end timing CSVs for Paper 1.

The benchmark runs 2D Laplace volume-potential evaluations on meshes refined by
a fixed Gaussian indicator. It records cold-cache and warm-cache timing stacks
for geometry construction, table build/load, and FMM phases exposed by
``drive_volume_fmm``. It also records repeated FMM timings on the same adapted
tree/table so table build or load overhead can be amortized across evaluations.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl


FIELDS = (
    "case_id",
    "mode",
    "cache_state",
    "q_order",
    "initial_nlevels",
    "adapt_steps",
    "n_active_boxes",
    "n_total_boxes",
    "n_targets",
    "min_leaf_level",
    "max_leaf_level",
    "mesh_init_s",
    "adapt_s",
    "geometry_s",
    "table_get_s",
    "table_build_s",
    "table_load_s",
    "table_payload_bytes",
    "fmm_wall_s",
    "fmm_repeat_count",
    "fmm_repeat_wall_s",
    "fmm_repeat_mean_s",
    "table_fmm_amortized_s",
    "geometry_table_fmm_amortized_s",
    "timing_form_multipoles_s",
    "timing_coarsen_multipoles_s",
    "timing_eval_direct_s",
    "timing_multipole_to_local_s",
    "timing_eval_multipoles_s",
    "timing_form_locals_s",
    "timing_refine_locals_s",
    "timing_eval_locals_s",
)


SMOKE_CASES = ((3, 2, 1),)
FULL_CASES = ((3, 2, 2), (4, 2, 2), (4, 3, 2))
SMOKE_FMM_REPEATS = 2
FULL_FMM_REPEATS = 5


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(getattr(device, "double_fp_config", 0))


def _select_opencl_device(backend: str):
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


def _indicator(mesh) -> np.ndarray:
    centers = mesh.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]
    return np.exp(-90.0 * ((x - 0.12) ** 2 + (y + 0.08) ** 2))


def _build_adaptive_geometry(ctx, queue, q_order: int, initial_nlevels: int, adapt_steps: int):
    import volumential.meshgen as mg

    start = time.perf_counter()
    mesh = mg.MeshGen2D(q_order, initial_nlevels, -0.5, 0.5, queue=queue)  # pyright: ignore[reportArgumentType]
    mesh_init_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(adapt_steps):
        mesh.update_mesh(
            _indicator(mesh),
            top_fraction_of_cells=0.35,
            bottom_fraction_of_cells=0.0,
        )
    adapt_s = time.perf_counter() - start

    start = time.perf_counter()
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        2,
        q_order,
        mesh,
        bbox=np.array([[-0.5, 0.5]] * 2, dtype=np.float64),
    )
    geometry_s = time.perf_counter() - start
    return mesh, q_points, q_weights, tree, traversal, mesh_init_s, adapt_s, geometry_s


def _leaf_level_stats(mesh):
    levels = mesh.boxtree.box_levels.get()[mesh.boxtree.active_boxes.get()]
    return int(np.min(levels)), int(np.max(levels))


def _build_config(q_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=max(8, 4 * q_order),
        radial_quad_order=max(21, 10 * q_order),
    )


def _get_table(queue, cache_path: Path, q_order: int, *, force_recompute: bool):
    from volumential.table_manager import NearFieldInteractionTableManager
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=2, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            2,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=_build_config(q_order),
        )
        timings = table_manager.last_get_table_timings
    return table, timings


def _source_values(queue, q_points):
    import pyopencl.array as cla

    coords = np.array([axis.get(queue) for axis in q_points])
    x = coords[0]
    y = coords[1]
    values = np.exp(-35.0 * ((x + 0.10) ** 2 + (y - 0.05) ** 2))
    return cla.to_device(queue, np.ascontiguousarray(values.astype(np.float64)))


def _build_wrangler(ctx, queue, traversal, table, q_order: int, fmm_order: int):
    from functools import partial

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    from volumential.expansion_wrangler_fpnd import (
        FPNDExpansionWrangler,
        FPNDTreeIndependentDataForWrangler,
    )

    kernel = LaplaceKernel(2)
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
    self_extra_kwargs = {}
    if traversal.tree.sources_are_targets:
        self_extra_kwargs["target_to_source"] = np.arange(
            traversal.tree.ntargets, dtype=np.int32
        )
    return FPNDExpansionWrangler(
        tree_indep=tree_indep,
        queue=queue,
        traversal=traversal,
        near_field_table=table,
        dtype=np.float64,
        fmm_level_to_order=lambda kernel, kernel_args, tree, lev: fmm_order,
        quad_order=q_order,
        self_extra_kwargs=self_extra_kwargs,
    )


def _timing_seconds(value) -> float | str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return float(value)
    for name in ("wall_elapsed", "process_elapsed"):
        attr = getattr(value, name, None)
        if attr is None:
            continue
        try:
            elapsed = attr() if callable(attr) else attr
            return float(elapsed)  # pyright: ignore[reportArgumentType]
        except TypeError:
            pass
    return ""


def _run_fmm(ctx, queue, traversal, table, q_order: int, fmm_order: int, q_weights, q_points):
    from volumential.volume_fmm import drive_volume_fmm

    source_vals = _source_values(queue, q_points)
    wrangler = _build_wrangler(ctx, queue, traversal, table, q_order, fmm_order)
    timing_data: dict[str, Any] = {}
    queue.finish()
    start = time.perf_counter()
    _ = drive_volume_fmm(
        traversal,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
        timing_data=timing_data,
    )
    queue.finish()
    return time.perf_counter() - start, timing_data


def _run_fmm_repeats(
    ctx,
    queue,
    traversal,
    table,
    q_order: int,
    fmm_order: int,
    q_weights,
    q_points,
    repeat_count: int,
):
    wall_times = []
    first_timings: dict[str, Any] | None = None
    for _ in range(repeat_count):
        fmm_wall_s, fmm_timings = _run_fmm(
            ctx, queue, traversal, table, q_order, fmm_order, q_weights, q_points
        )
        wall_times.append(fmm_wall_s)
        if first_timings is None:
            first_timings = fmm_timings
    return wall_times, first_timings or {}


def _table_phase_seconds(timings, phase: str) -> float | str:
    if not timings:
        return ""
    details = timings.get(phase)
    if not details:
        return ""
    return details.get("total_s", "")


def _table_payload_bytes(timings) -> int | str:
    if not timings:
        return ""
    details = timings.get("compute") or timings.get("load") or {}
    return int(details.get("payload_bytes", 0)) if details else ""


def _row(
    *,
    mode: str,
    cache_state: str,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    mesh,
    tree,
    mesh_init_s: float,
    adapt_s: float,
    geometry_s: float,
    table_timings,
    fmm_wall_s: float,
    fmm_repeat_wall_s: float,
    fmm_repeat_count: int,
    fmm_timings,
) -> dict[str, Any]:
    min_level, max_level = _leaf_level_stats(mesh)
    table_get_s = table_timings.get("total_s", "") if table_timings else ""
    table_get_float = float(table_get_s) if table_get_s != "" else 0.0
    return {
        "case_id": f"laplace2d-q{q_order}-l{initial_nlevels}-a{adapt_steps}",
        "mode": mode,
        "cache_state": cache_state,
        "q_order": q_order,
        "initial_nlevels": initial_nlevels,
        "adapt_steps": adapt_steps,
        "n_active_boxes": int(mesh.n_active_cells()),
        "n_total_boxes": int(mesh.n_cells()),
        "n_targets": int(tree.ntargets),
        "min_leaf_level": min_level,
        "max_leaf_level": max_level,
        "mesh_init_s": mesh_init_s,
        "adapt_s": adapt_s,
        "geometry_s": geometry_s,
        "table_get_s": table_get_s,
        "table_build_s": _table_phase_seconds(table_timings, "compute"),
        "table_load_s": _table_phase_seconds(table_timings, "load"),
        "table_payload_bytes": _table_payload_bytes(table_timings),
        "fmm_wall_s": fmm_wall_s,
        "fmm_repeat_count": fmm_repeat_count,
        "fmm_repeat_wall_s": fmm_repeat_wall_s,
        "fmm_repeat_mean_s": fmm_repeat_wall_s / fmm_repeat_count,
        "table_fmm_amortized_s": (
            table_get_float + fmm_repeat_wall_s
        )
        / fmm_repeat_count,
        "geometry_table_fmm_amortized_s": (
            geometry_s + table_get_float + fmm_repeat_wall_s
        )
        / fmm_repeat_count,
        "timing_form_multipoles_s": _timing_seconds(fmm_timings.get("form_multipoles")),
        "timing_coarsen_multipoles_s": _timing_seconds(fmm_timings.get("coarsen_multipoles")),
        "timing_eval_direct_s": _timing_seconds(fmm_timings.get("eval_direct")),
        "timing_multipole_to_local_s": _timing_seconds(fmm_timings.get("multipole_to_local")),
        "timing_eval_multipoles_s": _timing_seconds(fmm_timings.get("eval_multipoles")),
        "timing_form_locals_s": _timing_seconds(fmm_timings.get("form_locals")),
        "timing_refine_locals_s": _timing_seconds(fmm_timings.get("refine_locals")),
        "timing_eval_locals_s": _timing_seconds(fmm_timings.get("eval_locals")),
    }


def run_case(
    ctx,
    queue,
    *,
    mode: str,
    cache_dir: Path,
    q_order: int,
    initial_nlevels: int,
    adapt_steps: int,
    fmm_repeats: int,
):
    cache_path = cache_dir / f"adaptive-laplace2d-q{q_order}.sqlite"
    if cache_path.exists():
        cache_path.unlink()

    fmm_order = max(8, 4 * q_order)
    rows = []
    for cache_state, force_recompute in (("cold", True), ("warm", False)):
        mesh, q_points, q_weights, tree, traversal, mesh_init_s, adapt_s, geometry_s = (
            _build_adaptive_geometry(ctx, queue, q_order, initial_nlevels, adapt_steps)
        )
        table, table_timings = _get_table(
            queue, cache_path, q_order, force_recompute=force_recompute
        )
        fmm_wall_times, fmm_timings = _run_fmm_repeats(
            ctx,
            queue,
            traversal,
            table,
            q_order,
            fmm_order,
            q_weights,
            q_points,
            fmm_repeats,
        )
        rows.append(
            _row(
                mode=mode,
                cache_state=cache_state,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                mesh=mesh,
                tree=tree,
                mesh_init_s=mesh_init_s,
                adapt_s=adapt_s,
                geometry_s=geometry_s,
                table_timings=table_timings,
                fmm_wall_s=fmm_wall_times[0],
                fmm_repeat_wall_s=sum(fmm_wall_times),
                fmm_repeat_count=fmm_repeats,
                fmm_timings=fmm_timings,
            )
        )
    return rows


def run_benchmark(
    *, mode: str, backend: str, cache_dir: Path, fmm_repeats: int | None
):
    device = _select_opencl_device(backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    cases = SMOKE_CASES if mode == "smoke" else FULL_CASES
    if fmm_repeats is None:
        fmm_repeats = SMOKE_FMM_REPEATS if mode == "smoke" else FULL_FMM_REPEATS
    if fmm_repeats < 1:
        raise ValueError("fmm_repeats must be at least 1")
    rows = []
    for q_order, initial_nlevels, adapt_steps in cases:
        rows.extend(
            run_case(
                ctx,
                queue,
                mode=mode,
                cache_dir=cache_dir,
                q_order=q_order,
                initial_nlevels=initial_nlevels,
                adapt_steps=adapt_steps,
                fmm_repeats=fmm_repeats,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)  # pyright: ignore[reportArgumentType]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/adaptive-timing-cache"),
    )
    parser.add_argument(
        "--fmm-repeats",
        type=int,
        default=None,
        help="FMM evaluations per cold/warm row; defaults depend on --mode.",
    )
    args = parser.parse_args()
    if args.fmm_repeats is not None and args.fmm_repeats < 1:
        parser.error("--fmm-repeats must be at least 1")
    rows = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        fmm_repeats=args.fmm_repeats,
    )
    write_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
