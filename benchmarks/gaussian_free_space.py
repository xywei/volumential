#!/usr/bin/env python3
"""Run a deterministic Gaussian free-space 3D Laplace smoke demo.

The benchmark evaluates an overlapping Gaussian mixture on a Volumential box
mesh, compares against the analytic full-space Gaussian/Laplace reference, and
emits a summary CSV plus NPZ/JSON artifacts suitable for paper plotting.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from volumential.gaussian import (
    axis_aligned_slice_grid,
    default_overlapping_gaussian_mixture,
    evaluate_gaussian_mixture,
    gaussian_mixture_tail_report,
    laplace3d_gaussian_potential,
    mesh_leaf_box_arrays,
    nearest_axis_slice,
    write_json_metadata,
    write_npz,
)
from volumential.version import VERSION_TEXT


SUMMARY_FIELDS = (
    "case_id",
    "mode",
    "problem",
    "dim",
    "kernel",
    "kernel_normalization",
    "q_order",
    "nlevels",
    "adapt_steps",
    "fmm_order",
    "regular_quad_order",
    "radial_quad_order",
    "root_extent",
    "n_targets",
    "n_active_boxes",
    "n_total_boxes",
    "min_leaf_level",
    "max_leaf_level",
    "source_signed_mass_full_space",
    "source_signed_mass_box",
    "source_omitted_signed_mass",
    "source_abs_mass_full_space",
    "source_abs_mass_box",
    "source_omitted_abs_mass",
    "source_omitted_abs_fraction",
    "quadrature_source_mass",
    "table_get_s",
    "table_build_s",
    "table_load_s",
    "table_payload_bytes",
    "fmm_wall_s",
    "rel_l2_vs_analytic_full_space",
    "weighted_rel_l2_vs_analytic_full_space",
    "linf_vs_analytic_full_space",
)


def _device_supports_fp64(device) -> bool:
    extensions = set(getattr(device, "extensions", "").split())
    return "cl_khr_fp64" in extensions or bool(getattr(device, "double_fp_config", 0))


def _select_opencl_device(backend: str):
    backend = backend.lower()
    platforms = cl.get_platforms()

    if backend == "pocl-cpu":
        for platform_ in platforms:
            if "portable computing language" in platform_.name.lower():
                for device in platform_.get_devices():
                    if device.type & cl.device_type.CPU and _device_supports_fp64(device):
                        return device
        raise RuntimeError("PoCL CPU device with fp64 support not found")

    if backend == "cuda-gpu":
        for platform_ in platforms:
            if "nvidia cuda" in platform_.name.lower():
                for device in platform_.get_devices():
                    if device.type & cl.device_type.GPU and _device_supports_fp64(device):
                        return device
        raise RuntimeError("NVIDIA CUDA GPU device with fp64 support not found")

    if backend != "auto":
        raise ValueError("backend must be one of: auto, pocl-cpu, cuda-gpu")

    for platform_ in platforms:
        for device in platform_.get_devices():
            if device.type & cl.device_type.GPU and _device_supports_fp64(device):
                return device
    for platform_ in platforms:
        for device in platform_.get_devices():
            if device.type & cl.device_type.CPU and _device_supports_fp64(device):
                return device
    raise RuntimeError("No OpenCL GPU/CPU device with fp64 support found")


def _build_config(regular_quad_order: int, radial_quad_order: int):
    from volumential.nearfield_potential_table import DuffyBuildConfig

    return DuffyBuildConfig(
        radial_rule="tanh-sinh-fast",
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
    )


def _build_mesh(queue, *, q_order: int, nlevels: int, root_radius: float, adapt_steps: int, adapt_fraction: float):
    import volumential.meshgen as mg

    mixture = default_overlapping_gaussian_mixture(3)
    mesh = mg.MeshGen3D(q_order, nlevels, -root_radius, root_radius, queue=queue)
    if adapt_steps and not (0.0 < adapt_fraction <= 1.0):
        raise ValueError("adapt_fraction must be in (0, 1]")
    for _ in range(adapt_steps):
        indicator = np.abs(evaluate_gaussian_mixture(mixture, mesh.get_cell_centers()))
        mesh.update_mesh(
            indicator,
            top_fraction_of_cells=adapt_fraction,
            bottom_fraction_of_cells=0.0,
        )
    return mesh


def _build_geometry(ctx, queue, *, q_order: int, nlevels: int, root_radius: float, adapt_steps: int, adapt_fraction: float):
    import volumential.meshgen as mg

    mesh = _build_mesh(
        queue,
        q_order=q_order,
        nlevels=nlevels,
        root_radius=root_radius,
        adapt_steps=adapt_steps,
        adapt_fraction=adapt_fraction,
    )
    bbox = np.array([[-root_radius, root_radius]] * 3, dtype=np.float64)
    q_points, q_weights, tree, traversal = mg.build_geometry_info(
        ctx,
        queue,
        3,
        q_order,
        mesh,
        bbox=bbox,
    )
    return mesh, bbox, q_points, q_weights, tree, traversal


def _get_table(
    queue,
    *,
    cache_path: Path,
    root_extent: float,
    q_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    force_recompute: bool,
):
    from volumential.table_manager import NearFieldInteractionTableManager

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NearFieldInteractionTableManager(
        str(cache_path), root_extent=root_extent, queue=queue
    ) as table_manager:
        table, _ = table_manager.get_table(
            3,
            "Laplace",
            q_order,
            force_recompute=force_recompute,
            queue=queue,
            build_config=_build_config(regular_quad_order, radial_quad_order),
        )
        timings = table_manager.last_get_table_timings
    return table, timings


def _build_wrangler(ctx, queue, traversal, table, *, q_order: int, fmm_order: int):
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


def _coords_host(queue, q_points) -> np.ndarray:
    return np.array([axis.get(queue) for axis in q_points], dtype=np.float64).T


def _run_fmm(
    ctx,
    queue,
    traversal,
    table,
    *,
    q_order: int,
    fmm_order: int,
    q_weights,
    source_host: np.ndarray,
):
    from volumential.volume_fmm import drive_volume_fmm

    source_vals = cla.to_device(queue, np.ascontiguousarray(source_host.astype(np.float64)))
    wrangler = _build_wrangler(
        ctx,
        queue,
        traversal,
        table,
        q_order=q_order,
        fmm_order=fmm_order,
    )
    queue.finish()
    start = time.perf_counter()
    (potential,) = drive_volume_fmm(
        traversal,
        wrangler,
        source_vals * q_weights,
        source_vals,
        direct_evaluation=False,
        list1_only=False,
    )
    queue.finish()
    return potential.get(queue), time.perf_counter() - start


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


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _device_metadata(device) -> dict[str, Any]:
    return {
        "platform": device.platform.name,
        "device": device.name,
        "vendor": device.vendor,
        "version": device.version,
        "type": int(device.type),
    }


def _leaf_stats(leaf_arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    levels = leaf_arrays["leaf_levels"]
    return int(np.min(levels)), int(np.max(levels))


def run_benchmark(
    *,
    mode: str,
    backend: str,
    cache_dir: Path,
    q_order: int,
    nlevels: int,
    adapt_steps: int,
    adapt_fraction: float,
    fmm_order: int,
    regular_quad_order: int,
    radial_quad_order: int,
    root_radius: float,
    force_recompute: bool,
    slice_size: int,
) -> dict[str, Any]:
    mixture = default_overlapping_gaussian_mixture(3)
    device = _select_opencl_device(backend)
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    mesh, bbox, q_points, q_weights, tree, traversal = _build_geometry(
        ctx,
        queue,
        q_order=q_order,
        nlevels=nlevels,
        root_radius=root_radius,
        adapt_steps=adapt_steps,
        adapt_fraction=adapt_fraction,
    )
    coords = _coords_host(queue, q_points)
    weights = q_weights.get(queue)
    source = evaluate_gaussian_mixture(mixture, coords)
    reference = laplace3d_gaussian_potential(mixture, coords)

    table, table_timings = _get_table(
        queue,
        cache_path=cache_dir / f"gaussian-free-space-laplace3d-q{q_order}.sqlite",
        root_extent=2.0 * root_radius,
        q_order=q_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        force_recompute=force_recompute,
    )
    potential, fmm_wall_s = _run_fmm(
        ctx,
        queue,
        traversal,
        table,
        q_order=q_order,
        fmm_order=fmm_order,
        q_weights=q_weights,
        source_host=source,
    )
    error = potential - reference
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-300)
    weighted_reference_norm = max(float(np.sqrt(np.sum(weights * reference**2))), 1.0e-300)
    tail = gaussian_mixture_tail_report(mixture, bbox)
    leaf_arrays = mesh_leaf_box_arrays(mesh)
    min_leaf_level, max_leaf_level = _leaf_stats(leaf_arrays)

    case_id = f"gaussian-laplace3d-q{q_order}-l{nlevels}-a{adapt_steps}"
    row = {
        "case_id": case_id,
        "mode": mode,
        "problem": "gaussian-mixture-free-space",
        "dim": 3,
        "kernel": "Laplace",
        "kernel_normalization": "sumpy_1_over_r",
        "q_order": q_order,
        "nlevels": nlevels,
        "adapt_steps": adapt_steps,
        "fmm_order": fmm_order,
        "regular_quad_order": regular_quad_order,
        "radial_quad_order": radial_quad_order,
        "root_extent": 2.0 * root_radius,
        "n_targets": int(tree.ntargets),
        "n_active_boxes": int(mesh.n_active_cells()),
        "n_total_boxes": int(mesh.n_cells()),
        "min_leaf_level": min_leaf_level,
        "max_leaf_level": max_leaf_level,
        "source_signed_mass_full_space": tail["total_signed_mass"],
        "source_signed_mass_box": tail["in_box_signed_mass"],
        "source_omitted_signed_mass": tail["omitted_signed_mass"],
        "source_abs_mass_full_space": tail["total_abs_component_mass"],
        "source_abs_mass_box": tail["in_box_abs_component_mass"],
        "source_omitted_abs_mass": tail["omitted_abs_component_mass"],
        "source_omitted_abs_fraction": tail["omitted_abs_fraction"],
        "quadrature_source_mass": float(np.sum(weights * source)),
        "table_get_s": table_timings.get("total_s", "") if table_timings else "",
        "table_build_s": _table_phase_seconds(table_timings, "compute"),
        "table_load_s": _table_phase_seconds(table_timings, "load"),
        "table_payload_bytes": _table_payload_bytes(table_timings),
        "fmm_wall_s": fmm_wall_s,
        "rel_l2_vs_analytic_full_space": float(np.linalg.norm(error) / reference_norm),
        "weighted_rel_l2_vs_analytic_full_space": float(
            np.sqrt(np.sum(weights * error**2)) / weighted_reference_norm
        ),
        "linf_vs_analytic_full_space": float(np.max(np.abs(error))),
    }

    node_slice = nearest_axis_slice(
        coords,
        {
            "source": source,
            "potential": potential,
            "reference": reference,
            "error": error,
            "weights": weights,
        },
        axis=2,
        value=0.0,
    )
    analytic_slice = axis_aligned_slice_grid(
        bbox,
        fixed_axis=2,
        fixed_value=0.0,
        shape=(slice_size, slice_size),
    )
    analytic_slice_source = evaluate_gaussian_mixture(mixture, analytic_slice.points)
    analytic_slice_reference = laplace3d_gaussian_potential(mixture, analytic_slice.points)

    arrays = {
        "node_coords": coords,
        "node_weights": weights,
        "node_source": source,
        "node_potential": potential,
        "node_reference_full_space": reference,
        "node_error": error,
        "slice_node_coords": node_slice["coords"],
        "slice_node_indices": node_slice["indices"],
        "slice_node_axis_distances": node_slice["axis_distances"],
        "slice_node_source": node_slice["source"],
        "slice_node_potential": node_slice["potential"],
        "slice_node_reference": node_slice["reference"],
        "slice_node_error": node_slice["error"],
        "slice_node_weights": node_slice["weights"],
        "analytic_slice_coords": analytic_slice.points,
        "analytic_slice_shape": np.asarray(analytic_slice.shape, dtype=np.int32),
        "analytic_slice_axis0": analytic_slice.axis_values[0],
        "analytic_slice_axis1": analytic_slice.axis_values[1],
        "analytic_slice_source": analytic_slice_source,
        "analytic_slice_reference": analytic_slice_reference,
        **{f"tree_{name}": values for name, values in leaf_arrays.items()},
    }
    metadata = {
        "case_id": case_id,
        "mode": mode,
        "problem": "gaussian-mixture-free-space",
        "kernel": {
            "name": "Laplace",
            "dimension": 3,
            "normalization": "sumpy_1_over_r",
            "analytic_reference": "sum_i mass_i * erf(sqrt(alpha_i) r_i) / r_i",
        },
        "fixture": mixture.as_metadata(),
        "bbox": bbox,
        "tail": tail,
        "discretization": {
            "q_order": q_order,
            "nlevels": nlevels,
            "adapt_steps": adapt_steps,
            "adapt_fraction": adapt_fraction,
            "fmm_order": fmm_order,
            "regular_quad_order": regular_quad_order,
            "radial_quad_order": radial_quad_order,
        },
        "tree": {
            "n_targets": int(tree.ntargets),
            "n_active_boxes": int(mesh.n_active_cells()),
            "n_total_boxes": int(mesh.n_cells()),
            "min_leaf_level": min_leaf_level,
            "max_leaf_level": max_leaf_level,
        },
        "slice": {
            "node_slice_axis": int(node_slice["axis"]),
            "node_slice_value": float(node_slice["value"]),
            "node_slice_max_selected_distance": float(
                node_slice["max_selected_distance"]
            ),
            "node_slice_count": int(node_slice["coords"].shape[0]),
            "analytic_slice_shape": analytic_slice.shape,
            "analytic_slice_axes": analytic_slice.axes,
            "analytic_slice_fixed_axis": analytic_slice.fixed_axis,
            "analytic_slice_fixed_value": analytic_slice.fixed_value,
        },
        "errors": {
            "rel_l2_vs_analytic_full_space": row["rel_l2_vs_analytic_full_space"],
            "weighted_rel_l2_vs_analytic_full_space": row[
                "weighted_rel_l2_vs_analytic_full_space"
            ],
            "linf_vs_analytic_full_space": row["linf_vs_analytic_full_space"],
        },
        "timing": {
            "table_get_s": row["table_get_s"],
            "table_build_s": row["table_build_s"],
            "table_load_s": row["table_load_s"],
            "table_payload_bytes": row["table_payload_bytes"],
            "fmm_wall_s": row["fmm_wall_s"],
        },
        "cache": {
            "cache_dir": str(cache_dir),
            "force_recompute": force_recompute,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencl_device": _device_metadata(device),
        },
        "volumential": {
            "version": VERSION_TEXT,
            "git_commit": _git_commit(),
        },
    }
    return {"rows": [row], "arrays": arrays, "metadata": metadata}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--q-order", type=int)
    parser.add_argument("--nlevels", type=int)
    parser.add_argument("--adapt-steps", type=int)
    parser.add_argument("--adapt-fraction", type=float, default=0.35)
    parser.add_argument("--fmm-order", type=int)
    parser.add_argument("--regular-quad-order", type=int)
    parser.add_argument("--radial-quad-order", type=int)
    parser.add_argument("--root-radius", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=64)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/benchmarks/gaussian-free-space.csv"),
    )
    parser.add_argument(
        "--arrays-out",
        type=Path,
        default=Path("build/benchmarks/gaussian-free-space-arrays.npz"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("build/benchmarks/gaussian-free-space-metadata.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("build/benchmarks/gaussian-free-space-cache"),
    )
    args = parser.parse_args()

    smoke = args.mode == "smoke"
    q_order = args.q_order if args.q_order is not None else (2 if smoke else 3)
    nlevels = args.nlevels if args.nlevels is not None else (2 if smoke else 3)
    adapt_steps = args.adapt_steps if args.adapt_steps is not None else (0 if smoke else 1)
    fmm_order = args.fmm_order if args.fmm_order is not None else (8 if smoke else 12)
    regular_quad_order = (
        args.regular_quad_order if args.regular_quad_order is not None else max(8, 4 * q_order)
    )
    radial_quad_order = (
        args.radial_quad_order if args.radial_quad_order is not None else max(25, 10 * q_order)
    )

    result = run_benchmark(
        mode=args.mode,
        backend=args.backend,
        cache_dir=args.cache_dir,
        q_order=q_order,
        nlevels=nlevels,
        adapt_steps=adapt_steps,
        adapt_fraction=args.adapt_fraction,
        fmm_order=fmm_order,
        regular_quad_order=regular_quad_order,
        radial_quad_order=radial_quad_order,
        root_radius=args.root_radius,
        force_recompute=args.force_recompute,
        slice_size=args.slice_size,
    )
    metadata = result["metadata"]
    metadata["command"] = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
    }
    metadata["outputs"] = {
        "summary_csv": str(args.out),
        "arrays_npz": str(args.arrays_out),
        "metadata_json": str(args.metadata_out),
    }
    write_csv(args.out, result["rows"])
    write_npz(args.arrays_out, **result["arrays"])
    write_json_metadata(args.metadata_out, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
